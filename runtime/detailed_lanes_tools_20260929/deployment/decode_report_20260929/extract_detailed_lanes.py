#!/usr/bin/env python3
"""Export categorized events from the existing two-rank profiler caches."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import re


CATEGORIES = [
    ("attention", "Attention / sparse indexer", "#007f86"),
    ("moe_gemm", "MoE expert GEMM", "#2167ac"),
    ("dense_gemm", "Other GEMM / projections", "#669bd1"),
    ("engram", "Engram", "#aa4499"),
    ("mhc_norm", "mHC / normalization", "#8d6ab8"),
    ("route_quant_act", "Routing / quant / activation", "#b68b24"),
    ("kv_meta", "KV / RoPE / attention metadata", "#6a8450"),
    ("graph_other", "Other graph kernels", "#7d8794"),
    ("output", "Sampling / outside-graph GEMM", "#ce6679"),
    ("runtime", "Runtime / input preparation", "#4d8c9a"),
    ("memcpy", "Memory copy", "#646464"),
    ("memset", "Memory set", "#aaaaaa"),
    ("allreduce", "NCCL AllReduce", "#30944f"),
    ("allgather", "NCCL AllGather / other collective", "#90b65e"),
    ("pp", "PP SendRecv (includes wait)", "#c46736"),
    ("feedback", "Feedback Broadcast (includes wait)", "#cc8e7e"),
]


def classify(event, in_graph, labels):
    name = event["name"]
    lower = name.lower()
    purpose = labels.get(name, "").lower()
    source = event.get("cat", "")
    if source == "gpu_memcpy":
        return "memcpy"
    if source == "gpu_memset":
        return "memset"
    for token, group in (("broadcast", "feedback"), ("sendrecv", "pp"),
                         ("allreduce", "allreduce"), ("nccl", "allgather")):
        if token in lower:
            return group
    if "engram" in lower or "engram" in purpose:
        return "engram"
    if any(x in lower for x in ("sparse_mla", "paged_mqa_logits<")):
        return "attention"
    if any(x in lower for x in ("qnormrope", "kvrope", "kv_cache", "swa_indices",
                               "paged_mqa_logits_metadata", "cache_insert")):
        return "kv_meta"
    if "mhc" in lower or "rmsnorm" in lower or "rms_norm" in lower:
        return "mhc_norm"
    if any(x in lower for x in ("_dsv4_topk", "_fwd_kernel_ep_", "silu", "swiglu",
                               "act_and_mul", "quant", "ue8m0", "expert_routing")):
        return "route_quant_act"
    if "routed expert" in purpose and "projection" in purpose:
        return "moe_gemm"
    if "sm120_fp8_fp4_gemm_1d1d_impl" in lower and "GemmType)1" in name:
        return "moe_gemm"
    if any(x in lower for x in ("sample", "gumbel", "argmax")):
        return "output" if not in_graph else "graph_other"
    if any(x in lower for x in ("gemm", "gemv", "cublas", "splitkreduce")):
        return "dense_gemm" if in_graph else "output"
    return "graph_other" if in_graph else "runtime"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    with gzip.open(args.root / "analysis/report_data.json.gz", "rt") as stream:
        original = json.load(stream)
    label_data = json.loads(args.labels.read_text())
    labels = {r["full_kernel_symbol"]: r.get("purpose", "") for r in label_data["rows"]}
    caches = {}
    source_files = []
    for rank in (0, 4):
        trace, = (args.root / "profile_trace").glob(f"*rank{rank}.*.gz")
        key = hashlib.sha256(str(trace.resolve()).encode()).hexdigest()[:20]
        path = args.root / "analysis_gpu_cache" / f"rank_{key}.gpu_events.pkl"
        with path.open("rb") as stream:
            cache = pickle.load(stream)
        st = trace.stat()
        assert cache["rank"] == rank
        assert cache["source"]["size"] == st.st_size
        assert cache["source"]["mtime_ns"] == st.st_mtime_ns
        assert len(cache["runs"]) == 9
        caches[rank] = cache
        source_files.append(cache["source"])
    symbols, symbol_index = [], {}
    batches = []
    for ordinal, item in enumerate(original["batches"]):
        origin = item["timeline"]["origin_us"]
        phases = item["timeline"]["phases"]
        stop = origin + item["timeline"]["span_us"]
        events = []
        graph_audit = []
        for rank, cache in caches.items():
            replays = cache["runs"][ordinal]["replays"][:5]
            assert len(replays) == 5
            graph_steps = {id(event): step for step, replay in enumerate(replays) for event in replay}
            for step, replay in enumerate(replays):
                expected = next(p for p in phases if p["rank"] == rank and p["step"] == step)
                assert abs(min(e["ts"] for e in replay) - origin - expected["start_us"]) < .001
                graph_audit.append({"rank": rank, "step": step, "events": len(replay),
                                    "source_cats": dict(Counter(e["cat"] for e in replay)),
                                    "same_kernel_multiset_as_step0": Counter((e["cat"], e["name"]) for e in replay)
                                    == Counter((e["cat"], e["name"]) for e in replays[0])})
            for event in cache["events"]:
                start, end = event["ts"], event["ts"] + event.get("dur", 0)
                if start >= stop or end <= origin:
                    continue
                step = graph_steps.get(id(event), -1)
                category = classify(event, step >= 0, labels)
                key = (event["name"], category, event["cat"])
                if key not in symbol_index:
                    symbol_index[key] = len(symbols)
                    symbols.append(dict(name=event["name"], category=category,
                                        source_cat=event["cat"], purpose=labels.get(event["name"], "")))
                events.append([rank, round(max(start, origin) - origin, 3),
                               round(min(end, stop) - max(start, origin), 3), symbol_index[key],
                               event.get("args", {}).get("stream", -1), step])
        events.sort(key=lambda e: (e[0], e[1], e[3]))
        assert all(e[1] >= 0 and e[2] >= 0 for e in events)
        batches.append(dict(batch=item["batch"], origin_us=origin, span_us=stop-origin,
                            phases=phases, events=events, graph_audit=graph_audit))
    result = dict(schema="decode_detailed_lanes_v1", resolution_us=.001,
                  categories=[dict(id=i, label=l, color=c) for i, l, c in CATEGORIES],
                  symbols=symbols, batches=batches, source_files=source_files,
                  notes=["GPU events only; not a CPU activity trace",
                         "Rank0 and rank4 are representatives, not all eight ranks",
                         "Event boundaries rounded to 0.001 us, not a claim of timing accuracy",
                         "Names and graph membership classify events; generic GEMMs are not assigned to a model module",
                         "Memory copy direction is retained in the full event name",
                         "Feedback Broadcast is retained in data but hidden by default",
                         "PP and collective durations include waiting; colored occupancy is not utilization"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        stream.write(gzip.compress(json.dumps(result, separators=(",", ":")).encode(), mtime=0))
    print(json.dumps(dict(path=str(args.output), bytes=args.output.stat().st_size,
                          sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                          symbols=len(symbols), events={b["batch"]:len(b["events"]) for b in batches})))


if __name__ == "__main__":
    main()
