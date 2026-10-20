#!/usr/bin/env python3
"""Analyze the B64 Marlin + PCIe-IPC PP2 TP4 EP4 capture.

This keeps the decode-report loader and window selection strict, then uses the
observed dependency path ending at stage-1 sample readiness.  It never treats
the sum of raw durations on eight GPUs as wall time.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from analyze_critical_path import backtrace, step_graph  # noqa: E402
from decode_report_20260929 import analyze as base  # noqa: E402
from decode_report_20260929.extract_detailed_lanes import classify as detailed_classify  # noqa: E402


EXPECTED_BATCHES = [64]
REPLAYS_PER_BATCH = 15
STABLE_REPLAYS = 9
STABLE_START = (REPLAYS_PER_BATCH - STABLE_REPLAYS) // 2
IPC_PREFIXES = ("flashinfer::comm::pcie_ipc::",)


_CATEGORY_NAMES = {
    "attention": "attention", "moe_gemm": "MoE", "dense_gemm": "dense projection",
    "engram": "engram", "mhc_norm": "norm/mhc", "route_quant_act": "routequantactivation",
    "kv_meta": "kvmeta", "allreduce": "AllReduce", "allgather": "AllGather",
    "pp": "PP", "feedback": "broadcast", "memcpy": "memcpy", "memset": "memset",
}


def _category(name: str, source_cat="kernel", in_graph=True) -> str:
    lower = name.lower()
    if "flashinfer::comm::pcie_ipc::" in lower:
        return "AllReduce"
    if any(x in lower for x in ("hash_ids", "write_hash_cache", "writehashcache")):
        return "engram"
    if "marlin_moe" in lower:
        return "MoE"
    event = {"name": name, "cat": source_cat}
    label = detailed_classify(event, in_graph, {})
    if label == "moe_gemm" and "marlin_moe" not in lower:
        label = "dense_gemm" if any(x in lower for x in ("gemm", "gemv", "cublas", "splitkreduce")) else "graph_other"
    if label in ("graph_other", "runtime", "output") and any(
            x in lower for x in ("_dsv4_topk", "_fwd_kernel_ep_", "silu", "swiglu",
                                 "act_and_mul", "quant", "ue8m0", "expert_routing",
                                 "moe_align", "sum")):
        label = "route_quant_act"
    return _CATEGORY_NAMES.get(label, "other")


def _load_traces(root: Path):
    traces, bases, rank_bases = {}, set(), {}
    cache_dir = root / "analysis_gpu_cache"
    for path in base._trace_paths(root):
        rank, events, runs, clock = base._read_rank(path, cache_dir)
        if rank in traces:
            raise RuntimeError(f"duplicate trace for rank {rank}")
        if len(runs) != 1 or len(runs[0]["replays"]) != REPLAYS_PER_BATCH:
            raise RuntimeError(f"rank {rank}: expected one B64 run with 15 replays")
        traces[rank] = {"events": events, "runs": runs}
        rank_bases[rank] = clock
        bases.add(clock)
    if set(traces) != set(range(8)):
        raise RuntimeError(f"expected ranks 0..7, got {sorted(traces)}")
    if any(clock is None for clock in rank_bases.values()) or len(bases) != 1:
        raise RuntimeError(f"rank trace clocks differ: {rank_bases}")
    return traces, rank_bases


def _graph_stats(selected, step, step_stats):
    for stage in range(2):
        ranks = range(stage * 4, stage * 4 + 4)
        spans = []
        raw = Counter()
        counts = Counter()
        for rank in ranks:
            graph_events = selected[rank]["replays"][step]
            if not graph_events:
                raise RuntimeError(f"stage {stage} rank {rank}: no graph nodes")
            spans.append(max(e["ts"] + e.get("dur", 0) for e in graph_events) - min(e["ts"] for e in graph_events))
            for event in graph_events:
                category = _category(event["name"], event.get("cat", ""), True)
                if event.get("cat") == "kernel":
                    raw[category] += event.get("dur", 0)
                    counts[category] += 1
        stage_stats = step_stats[stage]
        stage_stats["graph_span_us"].extend(spans)
        stage_stats["raw_duration_us"] += raw
        stage_stats["kernel_count"] += counts


def analyze(root: Path, output: Path):
    # Override only the imported loader's batch contract for this single-case capture.
    base.EXPECTED_BATCHES = EXPECTED_BATCHES
    base.REPLAYS_PER_BATCH = REPLAYS_PER_BATCH
    base.STABLE_REPLAYS = STABLE_REPLAYS
    base.STABLE_START = STABLE_START
    graphs = base._load_graphs(root)
    cases = base._cases(root)
    if len(cases) != 1 or cases[0]["batch"] != 64:
        raise RuntimeError(f"expected exactly one B64 profile case, got {cases}")
    base._validate_sidecars(root, cases)
    traces, rank_bases = _load_traces(root)

    selected = {}
    for rank in range(8):
        replay, local = base._selected_window(traces[rank]["events"],
                                              traces[rank]["runs"][0], rank)
        selected[rank] = {"events": local, "replays": replay,
                          "pp_allgather_count": 3}

    totals = Counter()
    alternate = Counter()
    raw = Counter()
    kinds = {}
    category_totals = Counter()
    stage_stats = {0: {"graph_span_us": [], "raw_duration_us": Counter(), "kernel_count": Counter()},
                   1: {"graph_span_us": [], "raw_duration_us": Counter(), "kernel_count": Counter()}}
    cases_out = []
    node_cases = []
    for step in range(STABLE_REPLAYS):
        _graph_stats(selected, step, stage_stats)
        nodes, _, sample_ends, audit, comm = step_graph(
            graphs, selected, step, extra_collective_prefixes=IPC_PREFIXES)
        path = backtrace(nodes, sample_ends, ordinary_overlap_tolerance_us=.25)
        for node in nodes:
            kinds[node["name"]] = node["kind"]
            if node["kind"] in ("kernel", "communication_tail"):
                raw[node["name"]] += node["raw_duration"]
        totals.update(path["consumer_first"]["amounts"])
        alternate.update(path["producer_first"]["amounts"])
        for name, value in path["consumer_first"]["amounts"].items():
            source_cat = next((n.get("source_cat", "") for n in nodes if n["name"] == name), "")
            category_totals[_category(name, source_cat)] += value
        compact_case = {"step": step + STABLE_START, "path": path,
                          "graph_audit": audit, "communication_audit": comm,
                          "sample_endpoint_ids": sample_ends}
        cases_out.append(compact_case)
        node_cases.append({"step": step + STABLE_START, "nodes": nodes})

    names = [name for name in totals if kinds.get(name) in ("kernel", "communication_tail")]
    rows = []
    for number, name in enumerate(sorted(names, key=lambda item: totals[item], reverse=True), 1):
        rows.append({"number": number, "kernel": name, "category": _category(name),
                     "attributed_us_per_step": totals[name] / STABLE_REPLAYS,
                     "alternate_overlap_allocation_us_per_step": alternate[name] / STABLE_REPLAYS,
                     "all_rank_raw_us_per_step": raw[name] / STABLE_REPLAYS,
                     "kind": kinds[name]})

    stage_rows = []
    for stage in range(2):
        stats = stage_stats[stage]
        denominator = STABLE_REPLAYS * 4
        for category in sorted(set(stats["raw_duration_us"]) | set(stats["kernel_count"])):
            stage_rows.append({"stage": stage, "category": category,
                "mean_graph_span_us": statistics.mean(stats["graph_span_us"]),
                "raw_kernel_duration_us_per_rank_step": stats["raw_duration_us"][category] / denominator,
                "raw_kernel_duration_us_all_rank_step": stats["raw_duration_us"][category] / STABLE_REPLAYS,
                "mean_kernel_count_per_rank_step": stats["kernel_count"][category] / denominator})

    mean_span = statistics.mean(c["path"]["span_us"] for c in cases_out)
    mean_gap = statistics.mean(c["path"]["consumer_first"]["unattributed_gap_us"] for c in cases_out)
    category_us = {k: v / STABLE_REPLAYS for k, v in category_totals.items()}
    category_us["unattributed gaps"] = mean_gap
    category_percent = {k: v / mean_span * 100 for k, v in category_us.items()}
    selfcheck_us = sum(category_us.values())
    if abs(selfcheck_us - mean_span) > 1e-5:
        raise RuntimeError(f"category accounting does not close: {selfcheck_us} vs {mean_span}")
    selfcheck_percent = sum(category_percent.values())
    if abs(selfcheck_percent - 100.0) > 1e-8:
        raise RuntimeError(f"category percentages do not close: {selfcheck_percent}")
    summary = {
        "schema": "marlin_ipc_b64_observed_dependency_v1", "batch": 64,
        "pp": 2, "tp": 4, "ep": 4, "steps": STABLE_REPLAYS,
        "stable_replay_start": STABLE_START, "replays_per_batch": REPLAYS_PER_BATCH,
        "same_clock_base_time_nanoseconds": next(iter(rank_bases.values())),
        "endpoint": "stage0 graph activity to stage1 sample ready (sample_ends)",
        "mean_span_us": mean_span, "mean_unattributed_gap_us": mean_gap,
        "category_attributed_us_per_step": category_us,
        "category_attribution_percent": category_percent,
        "accounting_selfcheck": {"sum_us": selfcheck_us, "span_us": mean_span,
                                  "sum_percent": selfcheck_percent},
        "top30": rows[:30], "stage_graph_stats": stage_rows,
        "step_summaries": cases_out,
        "validation": {"ranks": 8, "graph_dags": True, "sidecars": True,
                        "same_clock": len(set(rank_bases.values())) == 1},
        "limitations": [
            "This is an observed dependency path proxy; PDL trigger and consumer-wait timestamps are absent.",
            "Profiler overhead is present; this result is not an unprofiled TPOT measurement.",
            "Collective tails are rendezvous-to-completion proxies, not physical wire time.",
            "Raw graph-kernel durations are reported per rank and as an eight-rank sum; raw sums are not wall time and are not divided by TPOT.",
            "Consumer-first and producer-first allocations are sensitivity conventions, not marginal latency bounds.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    with gzip.open(output / "nodes.json.gz", "wt") as stream:
        json.dump({"schema": "marlin_ipc_b64_nodes_v1", "steps": node_cases}, stream, separators=(",", ":"))
    with (output / "top30.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else
                                ["number", "kernel", "category", "attributed_us_per_step",
                                 "alternate_overlap_allocation_us_per_step", "all_rank_raw_us_per_step", "kind"])
        writer.writeheader(); writer.writerows(rows[:30])
    with (output / "stage_graph_stats.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["stage", "category", "mean_graph_span_us",
            "raw_kernel_duration_us_per_rank_step", "raw_kernel_duration_us_all_rank_step",
            "mean_kernel_count_per_rank_step"])
        writer.writeheader(); writer.writerows(stage_rows)
    report = ["# Marlin + PCIe-IPC B64 PP2 TP4 EP4", "",
        "Nine stable replays from the single B64 case were analyzed across all eight ranks. "
        "The endpoint is the stage-1 sample-ready boundary (`sample_ends`), so the span starts "
        "at stage-0 graph activity and ends when stage 1 has sampled.", "",
        f"Mean observed dependency span: {summary['mean_span_us'] / 1000:.3f} ms. "
        f"Mean unattributed gap: {summary['mean_unattributed_gap_us'] / 1000:.3f} ms. "
        f"Category accounting closes at {summary['accounting_selfcheck']['sum_percent']:.6f}%.", "",
        "Raw graph-kernel duration is shown per rank and as an eight-rank sum. It is not wall time, "
        "and the eight-rank sum is not divided by TPOT. Stage graph spans are separate wall-clock "
        "intervals and must not be added to raw duration totals.", "",
        "The result is an observed dependency path proxy with consumer-first and producer-first "
        "overlap allocations. Profiler overhead is present; this is not an unprofiled TPOT "
        "measurement. Collective tails are rendezvous-to-completion proxies, not physical wire time.", "",
        "## Category attribution", "", "| Category | us/step | Percent of mean span |", "|---|---:|---:|"]
    for category in sorted(category_us, key=category_us.get, reverse=True):
        report.append(f"| {category} | {category_us[category]:.3f} | {category_percent[category]:.6f}% |")
    report += [
        "", 
        "## Top 30", "", "| # | Kernel | Category | Consumer us/step | Producer us/step | Raw all-rank us/step |",
        "|---:|---|---|---:|---:|---:|"]
    for row in rows[:30]:
        report.append(f"| {row['number']} | `{row['kernel']}` | {row['category']} | {row['attributed_us_per_step']:.3f} | "
                      f"{row['alternate_overlap_allocation_us_per_step']:.3f} | {row['all_rank_raw_us_per_step']:.3f} |")
    report += ["", "## Stage graph statistics", "", "See `stage_graph_stats.csv`; categories report raw kernel durations and mean per-rank graph spans.", ""]
    (output / "REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"output": str(output), "mean_span_us": summary["mean_span_us"],
                      "top30": len(rows[:30])}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    analyze(args.root, args.output)


if __name__ == "__main__":
    main()
