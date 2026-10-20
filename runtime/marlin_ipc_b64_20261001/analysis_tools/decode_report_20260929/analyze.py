#!/usr/bin/env python3
"""Analyze a serialized multi-batch decode capture without loading raw traces at once.

The collector is expected to write nine consecutive graph-id runs per rank, with
15 replays in each run.  The middle nine replays are passed to the existing
observed dependency-chain analyzer.  This intentionally fails on mixed prefill
or ambiguous replay boundaries instead of assigning events by position.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import os
import pickle
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from analyze_critical_path import GPU_CATS, backtrace, step_graph  # noqa: E402


EXPECTED_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
REPLAYS_PER_BATCH = 15
STABLE_REPLAYS = 9
STABLE_START = (REPLAYS_PER_BATCH - STABLE_REPLAYS) // 2


def _json(path: Path):
    return json.loads(path.read_text())


def _trace_paths(root: Path):
    paths = sorted((root / "graph" / "timelines").rglob("*.trace.json.gz"))
    if not paths:
        paths = sorted(root.rglob("*.trace.json.gz"))
    if not paths:
        raise RuntimeError(f"no compressed trace files below {root}; refusing to infer from sidecars")
    return paths


def _graph_id(event):
    args = event.get("args", {})
    return args.get("graph id", args.get("graph_id", args.get("graphId")))


def _is_graph(event):
    return _graph_id(event) not in (None, 0, "0")


def _replay_key(event):
    args = event.get("args", {})
    gid = _graph_id(event)
    correlation = args.get("correlation", args.get("correlation_id", args.get("correlationId")))
    if gid is None or correlation is None:
        raise RuntimeError("graph event lacks graph id or correlation; replay boundaries are ambiguous")
    return str(gid), str(correlation)


def _rank_from(data, path):
    rank = data.get("distributedInfo", {}).get("rank")
    if rank is None:
        name = path.name
        if "rank_" in name:
            rank = int(name.split("rank_", 1)[1].split("_", 1)[0])
    if rank is None:
        raise RuntimeError(f"cannot determine rank for {path}")
    return int(rank)


def _cache_path(path, cache_dir):
    key = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:20]
    return cache_dir / f"rank_{key}.gpu_events.pkl"


def _read_rank(path, cache_dir=None):
    stat = path.stat()
    cache = _cache_path(path, cache_dir) if cache_dir is not None else None
    source = {"filename": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if cache is not None and cache.exists():
        try:
            with cache.open("rb") as stream:
                value = pickle.load(stream)
            if value.get("source") == source:
                return value["rank"], value["events"], value["runs"], value["base"]
        except (OSError, EOFError, pickle.PickleError, ValueError, KeyError):
            pass
    with gzip.open(path, "rt") as stream:
        data = json.load(stream)
    rank = _rank_from(data, path)
    events = sorted(
        (e for e in data.get("traceEvents", []) if e.get("cat") in GPU_CATS),
        key=lambda e: (e.get("ts", 0), e.get("pid", 0)),
    )
    grouped = {}
    for event in events:
        if not _is_graph(event):
            continue
        grouped.setdefault(_replay_key(event), []).append(event)
    replay_groups = sorted(grouped.values(), key=lambda es: min(e["ts"] for e in es))
    if not replay_groups:
        raise RuntimeError(f"rank {rank}: no graph replay events")
    runs = []
    for replay in replay_groups:
        gid = str(_graph_id(replay[0]))
        if not runs or runs[-1]["graph_id"] != gid:
            runs.append({"graph_id": gid, "replays": []})
        runs[-1]["replays"].append(replay)
    if len(runs) != len(EXPECTED_BATCHES):
        raise RuntimeError(
            f"rank {rank}: expected {len(EXPECTED_BATCHES)} consecutive graph-id runs, "
            f"got {len(runs)} ({[len(r['replays']) for r in runs]}); "
            "prefill or interleaved graph capture is not safely separable"
        )
    for i, run in enumerate(runs):
        if len(run["replays"]) != REPLAYS_PER_BATCH:
            raise RuntimeError(
                f"rank {rank}, run {i} graph_id={run['graph_id']}: expected "
                f"{REPLAYS_PER_BATCH} replays, got {len(run['replays'])}"
            )
    base = data.get("baseTimeNanoseconds")
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        value = {"source": source, "rank": rank, "base": base, "events": events, "runs": runs}
        temporary = cache.with_name(cache.name + f".tmp-{os.getpid()}")
        try:
            with temporary.open("wb") as stream:
                pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary, cache)
        finally:
            if temporary.exists():
                temporary.unlink()
    return rank, events, runs, base


def _load_graphs(root):
    graphs = {}
    paths = sorted((root / "graph_dags").glob("graph_pid*.json"))
    if not paths:
        raise RuntimeError(f"no graph DAGs below {root / 'graph_dags'}")
    for path in paths:
        record = _json(path)
        graphs[int(record["torch_rank"]), int(record["graph"]["exec_graph_id"])] = record
    return graphs


def _cases(root):
    candidates = [root / "profile_cases.json", root / "cases.json"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise RuntimeError("missing profile_cases.json; batch labels cannot be inferred from graph order")
    value = _json(path)
    if isinstance(value, dict):
        value = value.get("cases", value.get("profile_cases", value.get("rows", value)))
    if not isinstance(value, list) or len(value) != len(EXPECTED_BATCHES):
        raise RuntimeError(f"profile cases must contain exactly {len(EXPECTED_BATCHES)} batch records")
    batches = []
    for expected, case in zip(EXPECTED_BATCHES, value):
        if not isinstance(case, dict) or case.get("batch") is None:
            raise RuntimeError("each profile case must include an explicit batch")
        batch = int(case["batch"])
        if batch != expected:
            raise RuntimeError(f"profile case order/batch mismatch: expected {expected}, got {batch}")
        batches.append({"batch": batch, "label": case.get("label", f"B{batch}"),
                        "request_ids": case.get("request_ids", []),
                        "padded_tokens": case.get("padded_tokens", case.get("padded", batch))})
    return batches


def _selected_window(events, run, rank):
    """Build the exact 9-replay window expected by step_graph.

    PP and Broadcast events are selected by their actual replay ordinal (3 and
    2 per decode step). Other stream events are retained only around the chosen
    graph window, preventing neighboring batches from being mispaired.
    """
    middle = run["replays"][STABLE_START:STABLE_START + STABLE_REPLAYS]
    graph_events = [e for replay in middle for e in replay]
    graph_ids = {id(e) for e in graph_events}
    first = min(e["ts"] for e in graph_events)
    last = max(e["ts"] + e.get("dur", 0) for e in graph_events)
    margin = max(1000.0, (last - first) * 0.25)
    pp = [e for e in events if "ncclDevKernel_SendRecv" in e.get("name", "") and not _is_graph(e)]
    bc = [e for e in events if "ncclDevKernel_Broadcast" in e.get("name", "") and not _is_graph(e)]
    starts = [min(e["ts"] for e in replay) for replay in run["replays"]]
    ends = [max(e["ts"] + e.get("dur", 0) for e in replay) for replay in run["replays"]]
    selected_pp, selected_bc = [], []
    for i in range(STABLE_START, STABLE_START + STABLE_REPLAYS):
        # PP sends are after stage0 graph completion; PP receives are before
        # stage1 graph entry.  Both are bounded by adjacent graph replays.
        if rank < 4:
            pp_lo, pp_hi = ends[i], starts[i + 1] if i + 1 < len(starts) else ends[i] + 10000
        else:
            pp_lo, pp_hi = ends[i - 1] if i else ends[i] - 10000, starts[i]
        bc_lo, bc_hi = ends[i], starts[i + 1] if i + 1 < len(starts) else ends[i] + 10000
        pp_i = [e for e in pp if pp_lo <= e["ts"] < pp_hi]
        bc_i = [e for e in bc if bc_lo <= e["ts"] < bc_hi]
        if len(pp_i) != 3 or len(bc_i) != 2:
            raise RuntimeError(
                f"rank {rank}, replay {i}: timestamp boundary expected 3 PP/2 BC, "
                f"got {len(pp_i)}/{len(bc_i)}; capture has ambiguous prefill/exterior timing"
            )
        selected_pp.extend(pp_i)
        selected_bc.extend(bc_i)
    selected_ids = {id(e) for e in selected_pp + selected_bc}
    # Identify the runtime-assigned main/control stream from actual anchors.
    # This captures all sampling/logits work on that stream, not just three
    # named kernels. Graph events are excluded from this normalization.
    controls = ("_post_update_num_computed_tokens_kernel", "AllGather",
                "_get_num_sampled_and_rejected_kernel")
    anchor_streams = Counter(
        e.get("args", {}).get("stream") for e in events
        if first - margin <= e["ts"] <= last + margin
        and not _is_graph(e) and any(token in e.get("name", "") for token in controls)
        and e.get("args", {}).get("stream") is not None)
    if not anchor_streams:
        raise RuntimeError(f"rank {rank}: no main-stream anchor in selected graph window")
    main_stream, _ = anchor_streams.most_common(1)[0]
    if len(anchor_streams) != 1:
        raise RuntimeError(f"rank {rank}: control anchors use multiple streams {dict(anchor_streams)}")
    # Keep a generous local window for all GPU outside work (including
    # broadcast side streams and copies), but never retain neighboring batch
    # PP/Broadcast operations. Selected communications are retained by identity
    # even if their duration extends beyond the local timestamp margin.
    margin = max(1000.0, (last - first) * 0.25)
    local = [e for e in events if id(e) in selected_ids
             or first - margin <= e["ts"] <= last + margin]
    local = [e for e in local if (id(e) in graph_ids)
             or id(e) in selected_ids
             or (not _is_graph(e)
                 and "ncclDevKernel_SendRecv" not in e.get("name", "")
                 and "ncclDevKernel_Broadcast" not in e.get("name", ""))]
    # The installed analyzer expects the PP control stream as 23. Collector
    # traces may expose it as 13 (or another runtime-assigned value); normalize
    # the identified main stream while retaining each event's raw value.
    normalized = []
    for event in local:
        if not _is_graph(event) and event.get("args", {}).get("stream") == main_stream:
            event = dict(event)
            args = dict(event.get("args", {}))
            args["original_stream"] = args.get("stream")
            args["stream"] = 23
            event["args"] = args
        normalized.append(event)
    return middle, normalized


def _validate_sidecars(root, cases):
    paths = sorted((root / "rank_timings").glob("*.jsonl"))
    if len(paths) != 8:
        raise RuntimeError(f"expected 8 rank timing sidecars, got {len(paths)}")
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for case in cases:
            batch = case["batch"]
            expected_ids = set(map(str, case.get("request_ids", [])))
            if not expected_ids:
                raise RuntimeError(f"profile case B{batch} has no request_ids for sidecar filtering")
            matched = [r for r in rows
                       if not r.get("prefill")
                       and set(map(str, r.get("request_ids", []))) == expected_ids]
            if len(matched) != REPLAYS_PER_BATCH:
                raise RuntimeError(f"{path.name} batch {batch}: expected {REPLAYS_PER_BATCH} sidecar rows, got {len(matched)}")
            for row in matched:
                if int(row.get("n", -1)) != batch:
                    raise RuntimeError(f"{path.name} batch {batch}: sidecar n mismatch")
                padded = row.get("padded_tokens", row.get("padded"))
                expected = case.get("padded_tokens")
                if padded is None or (expected is not None and int(padded) != int(expected)):
                    raise RuntimeError(f"{path.name} batch {batch}: missing/mismatched padded_tokens")
                mode = str(row.get("graph_mode", ""))
                if not mode.startswith("FULL"):
                    raise RuntimeError(f"{path.name} batch {batch}: graph_mode={mode!r}, expected FULL")


def _timeline(events, origin, end):
    # Shared bounds retain the first stage1 receive and last stage0 send.
    return [dict(name=e.get("name", ""), cat=e.get("cat", ""), ts_us=max(e["ts"], origin) - origin,
                 dur_us=min(e["ts"] + e.get("dur", 0), end) - max(e["ts"], origin), graph_id=_graph_id(e),
                 stream=e.get("args", {}).get("stream"))
            for e in events if e["ts"] < end and e["ts"] + e.get("dur", 0) > origin
            and e.get("cat") in GPU_CATS]


def _purpose(name):
    lower = name.lower()
    if "sparse_mla" in lower or "mla" in lower:
        return "sparse MLA decode"
    if "allreduce" in lower:
        return "TP collective"
    if "allgather" in lower:
        return "TP gather"
    if "sendrecv" in lower:
        return "PP transfer"
    if "broadcast" in lower:
        return "sampled-token broadcast"
    if "gemm" in lower or "cutlass" in lower:
        return "linear/GEMM"
    if "silu" in lower or "swiglu" in lower:
        return "activation"
    if "topk" in lower or "sample" in lower:
        return "sampling"
    return "GPU kernel"


def analyze(root: Path, output: Path):
    graphs = _load_graphs(root)
    cases = _cases(root)
    _validate_sidecars(root, cases)
    traces = {}
    bases = set()
    rank_bases = {}
    cache_dir = root / "analysis_gpu_cache"
    # Process one compressed rank file at a time; retain only compact GPU events
    # and replay groups needed for the nine batch windows.
    for path in _trace_paths(root):
        rank, events, runs, base = _read_rank(path, cache_dir)
        if rank in traces:
            raise RuntimeError(f"duplicate trace for rank {rank}")
        for ordinal, run in enumerate(runs):
            run["ordinal"] = ordinal
        traces[rank] = {"events": events, "runs": runs}
        print(f"Loaded rank {rank}: {len(events)} GPU events", flush=True)
        rank_bases[rank] = base
        bases.add(base)
    if set(traces) != set(range(8)):
        raise RuntimeError(f"expected ranks 0..7, got {sorted(traces)}")
    if set(rank_bases) != set(range(8)) or any(base is None for base in rank_bases.values()):
        raise RuntimeError(f"every rank must provide baseTimeNanoseconds: {rank_bases}")
    if len(bases) != 1:
        raise RuntimeError(f"rank trace clocks differ: {rank_bases}")

    rows = []
    report_data = {"schema": "decode_report_observed_dependency_v1",
                   "mode": "FULL_DECODE_ONLY", "batches": []}
    for batch_case in cases:
        batch = batch_case["batch"]
        print(f"Analyzing batch {batch}", flush=True)
        selected = {}
        for rank in range(8):
            replay, local = _selected_window(traces[rank]["events"],
                                             traces[rank]["runs"][cases.index(batch_case)], rank)
            # The PP scatter optimization requires tensor numel divisible by
            # TP=4. This capture has two eligible tensors at B1/B2, then three.
            selected[rank] = {"events": local, "replays": replay,
                              "pp_allgather_count": 2 if batch < 4 else 3}
        spans = []
        names = Counter()
        alternate = Counter()
        kinds = {}
        kind_totals = Counter()
        gap_total = 0.0
        compact_cases = []
        for step in range(STABLE_REPLAYS):
            nodes, endpoints, sample, audit, comm = step_graph(graphs, selected, step)
            path = backtrace(nodes, endpoints, ordinary_overlap_tolerance_us=.25)
            spans.append(path["span_us"])
            for node in nodes:
                kinds[node["name"]] = node["kind"]
            names.update({n: v for n, v in path["consumer_first"]["amounts"].items()
                          if kinds.get(n) in ("kernel", "communication_tail")})
            alternate.update({n: v for n, v in path["producer_first"]["amounts"].items()
                              if kinds.get(n) in ("kernel", "communication_tail")})
            for name, value in path["consumer_first"]["amounts"].items():
                kind_totals[kinds.get(name, "unknown")] += value
            gap_total += path["consumer_first"]["unattributed_gap_us"]
            compact_cases.append({"step": step + STABLE_START, "span_us": path["span_us"],
                                  "top": path["consumer_first"]["amounts"].most_common(10)
                                  if hasattr(path["consumer_first"]["amounts"], "most_common") else
                                  sorted(path["consumer_first"]["amounts"].items(), key=lambda x: -x[1])[:10],
                                  "overlap_links": len(path["overlapping_dependency_links"]),
                                  "submicro_ordinary_overlaps": [x for x in path["overlapping_dependency_links"] if not x["pdl"]],
                                  "unattributed_gap_us": path["consumer_first"]["unattributed_gap_us"]})
        top10 = [{"kernel": name, "full_kernel_symbol": name,
                  "attributed_us_per_step": value / STABLE_REPLAYS,
                  "alternate_overlap_allocation_us_per_step": alternate[name] / STABLE_REPLAYS,
                  "purpose": _purpose(name)}
                 for name, value in names.most_common(10)]
        origin = min(e["ts"] for r in range(4) for e in traces[r]["runs"][cases.index(batch_case)]["replays"][0])
        display_end = max(e["ts"] + e.get("dur", 0) for r in range(8)
                          for e in traces[r]["runs"][cases.index(batch_case)]["replays"][4])
        timeline_events = []
        phases = []
        for rank in (0, 4):
            compact = _timeline(traces[rank]["events"], origin, display_end)
            for event in compact:
                timeline_events.append({"rank": rank, "start_us": event["ts_us"],
                                        "duration_us": event["dur_us"], "name": event["name"],
                                        "kind": event["cat"],
                                        "stream": event.get("stream"),
                                        "graph_id": event.get("graph_id")})
            for step, replay in enumerate(traces[rank]["runs"][cases.index(batch_case)]["replays"][:5]):
                phases.append({"rank": rank, "step": step,
                               "start_us": min(e["ts"] for e in replay) - origin,
                               "end_us": max(e["ts"] + e.get("dur", 0) for e in replay) - origin})
        timeline_span = max((e["start_us"] + e["duration_us"] for e in timeline_events), default=0.0)
        timeline = {"origin_us": origin, "span_us": timeline_span,
                    "events": timeline_events, "phases": phases, "display_steps": 5}
        row = {"batch": batch, "label": batch_case["label"], "steps": STABLE_REPLAYS,
               "stable_replay_start": STABLE_START, "mean_span_us": sum(spans) / len(spans),
               "min_span_us": min(spans), "max_span_us": max(spans), "rows": top10,
               "accounting": {"kernel_us_per_step": kind_totals["kernel"] / STABLE_REPLAYS,
                              "communication_tail_us_per_step": kind_totals["communication_tail"] / STABLE_REPLAYS,
                              "copy_memset_us_per_step": (kind_totals["gpu_memcpy"] + kind_totals["gpu_memset"]) / STABLE_REPLAYS,
                              "unattributed_gap_us_per_step": gap_total / STABLE_REPLAYS},
               "timeline": timeline,
               "limitations": ["observed dependency-chain attribution, not exact critical path",
                               "PDL trigger/consumer wait timestamps are absent",
                               "collective tails are rendezvous-to-completion proxies",
                               "profiler-instrumented GPU span; host/client time is outside",
                               "ordinary-edge timestamp overlaps up to 0.25us retained and accounted once; not classified as PDL"],
               "step_rows": compact_cases}
        rows.append(row)
        report_data["batches"].append(row)
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(json.dumps({"schema": report_data["schema"],
        "mode": "FULL_DECODE_ONLY", "capture_root": str(root), "batch_rows": rows,
        "validation": {"ranks": 8, "batches": len(rows), "replays_per_batch": REPLAYS_PER_BATCH,
                        "stable_replays": STABLE_REPLAYS, "shared_base_time_nanoseconds": list(bases)},
        "limitations": report_data["batches"][0]["limitations"]}, indent=2) + "\n")
    with gzip.open(output / "report_data.json.gz", "wt") as stream:
        json.dump(report_data, stream, separators=(",", ":"))
    print(json.dumps({"batches": len(rows), "output": str(output),
                      "mean_spans_us": {str(r["batch"]): r["mean_span_us"] for r in rows}}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    analyze(args.root, args.output)


if __name__ == "__main__":
    main()
