#!/usr/bin/env python3
"""Build compact prefill swimlanes and dependency-based critical paths.

Input is a directory containing one or more Kineto/Chrome trace files.  A
trace is assigned to a prompt length from its path/name (``2k`` .. ``128k``),
or from a nearby JSON sidecar containing ``prompt_tokens``/``context_tokens``.
The analyzer constructs same-stream dependencies only. Cross-rank, CPU-launch,
event-wait, and CUDA-graph dependencies require explicit collector metadata and
are reported as unavailable here. It never reports a sum of all kernel
durations as a critical path.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import re
from typing import Any


LENGTHS = (2048, 4096, 8192, 16384, 32768, 65536, 131072)
GPU_CATS = {"kernel", "gpu_kernel", "gpu kernels", "cuda_kernel", "gpu_memcpy", "gpu_memset"}
CPU_CATS = {"cpu_op", "cpu_user_annotation", "user_annotation", "runtime", "driver"}
LEN_RE = re.compile(r"(?<!\d)(2|4|8|16|32|64|128)\s*[kK](?!\w)")


def _events(path: Path):
    try:
        import ijson  # type: ignore
    except ImportError:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            value = json.load(f)
        yield from value.get("traceEvents", value if isinstance(value, list) else [])
        return
    with gzip.open(path, "rb") as f:
        yield from ijson.items(f, "traceEvents.item", use_float=True)


def _arg(event: dict[str, Any], *names: str) -> Any:
    args = event.get("args") if isinstance(event.get("args"), dict) else {}
    for name in names:
        if name in args:
            return args[name]
        if name in event:
            return event[name]
    return None


def _link_ids(event: dict[str, Any]) -> set[str]:
    result = set()
    for key in ("correlation", "Correlation", "External id", "external id", "external_id"):
        value = _arg(event, key)
        if value is not None:
            result.add(f"{key}:{value}")
    return result


def _rank(path: Path, data_rank: Any = None) -> int | None:
    if data_rank is not None:
        return int(data_rank)
    match = re.search(r"(?:rank|global_rank)[_-]?(\d+)", path.name, re.I)
    return int(match.group(1)) if match else None


def _length_from_sidecar(path: Path) -> int | None:
    for candidate in (path.with_suffix(".json"), path.parent / "configuration.json",
                      path.parent / "workload.json", path.parent / "requests.json"):
        if not candidate.exists():
            continue
        try:
            value = json.loads(candidate.read_text())
        except (OSError, ValueError):
            continue
        stack = [value]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                for key in ("prompt_tokens", "context_tokens", "context_length", "length"):
                    if isinstance(item.get(key), int) and item[key] in LENGTHS:
                        return int(item[key])
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
    return None


def _length(path: Path) -> int | None:
    match = LEN_RE.search(str(path))
    if match:
        return int(match.group(1)) * 1024
    return _length_from_sidecar(path)


def _kind(name: str, cat: str) -> str:
    low = name.lower()
    if "sendrecv" in low or "allreduce" in low or "allgather" in low or "broadcast" in low or "nccl" in low:
        return "collective"
    if cat in {"gpu_memcpy", "gpu_memset"} or "memcpy" in low or "memset" in low:
        return "memory"
    return "kernel"


def _load_trace(path: Path) -> tuple[int | None, list[dict[str, Any]], dict[str, Any]]:
    events = []
    rank = None
    base = None
    for event in _events(path):
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        ts, dur = event.get("ts"), event.get("dur")
        if not isinstance(ts, (int, float)) or not isinstance(dur, (int, float)) or dur < 0:
            continue
        cat = str(event.get("cat", "")).lower()
        if cat not in GPU_CATS and cat not in CPU_CATS:
            continue
        if rank is None:
            rank = _rank(path, _arg(event, "rank", "global_rank"))
        if cat in GPU_CATS:
            event = dict(event)
            event["_kind"] = "gpu"
            event["_category"] = cat
            event["_links"] = _link_ids(event)
            events.append(event)
    # ijson does not materialize the document header.  Rank is therefore
    # required in the filename unless a producer copies it into an event.
    # Missing base time is reported as unknown rather than triggering a second
    # full-file JSON load for large profiler captures.
    return rank, events, {"base_time_nanoseconds": base, "cpu_records": 0}


def _node(event: dict[str, Any], rank: int, index: int) -> dict[str, Any]:
    start = float(event["ts"])
    return {"id": index, "rank": rank, "name": str(event.get("name", "")),
            "kind": _kind(str(event.get("name", "")), event["_category"]),
            "category": event["_category"], "start": start,
            "end": start + float(event.get("dur", 0)),
            "duration_us": float(event.get("dur", 0)),
            "stream": _arg(event, "stream", "Stream", "stream id"),
            "links": sorted(event.get("_links", set())), "deps": []}


def _path(nodes: list[dict[str, Any]], endpoints: list[int]) -> list[int]:
    if not nodes:
        return []
    indegree = [0] * len(nodes)
    dependents: dict[int, list[int]] = defaultdict(list)
    for node in nodes:
        indegree[node["id"]] = len(node["deps"])
        for dep in node["deps"]:
            dependents[dep].append(node["id"])
    ready = [node["id"] for node in nodes if indegree[node["id"]] == 0]
    order = []
    while ready:
        current = ready.pop()
        order.append(current)
        for dependent in dependents[current]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
    if len(order) != len(nodes):
        return []
    for index in order:
        node = nodes[index]
        node["score"] = node["duration_us"]
        node["prev"] = None
        for dep in node["deps"]:
            parent = nodes[dep]
            gap = max(0.0, node["start"] - parent["end"])
            score = parent["score"] + gap + node["duration_us"]
            if score > node["score"]:
                node["score"], node["prev"] = score, dep
    end = max(endpoints or [n["id"] for n in nodes], key=lambda i: nodes[i]["score"])
    result = []
    while end is not None:
        result.append(end)
        end = nodes[end]["prev"]
    return list(reversed(result))


def _dependency_graph(records: list[tuple[int, list[dict[str, Any]]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    by_rank: dict[int, list[int]] = defaultdict(list)
    for rank, events in records:
        for event in events:
            if event["_kind"] != "gpu":
                continue
            index = len(nodes)
            nodes.append(_node(event, rank, index))
            by_rank[rank].append(index)
    edge_counts = defaultdict(int)
    for rank, ids in by_rank.items():
        streams: dict[Any, list[int]] = defaultdict(list)
        for i in ids:
            streams[nodes[i]["stream"]].append(i)
        for stream_ids in streams.values():
            stream_ids.sort(key=lambda i: (nodes[i]["start"], nodes[i]["end"], i))
            for left, right in zip(stream_ids, stream_ids[1:]):
                nodes[right]["deps"].append(left)
                edge_counts["stream_order"] += 1
    # Do not pair collectives by name/ordinal: P and D have separate TP
    # groups, and equal names can overlap or occur in different order.  A
    # rendezvous edge is accepted only when a future collector supplies an
    # explicit communicator/group and sequence identifier.
    used = {dep for node in nodes for dep in node["deps"]}
    endpoints = [node["id"] for node in nodes if node["id"] not in used]
    return nodes, {"edge_counts": dict(edge_counts), "endpoints": endpoints,
                   "gpu_events": len(nodes), "ranks": sorted(by_rank)}


def _analyze_length(length: int, paths: list[Path]) -> dict[str, Any]:
    records, bases = [], set()
    for path in paths:
        rank, events, meta = _load_trace(path)
        if rank is None:
            continue
        records.append((rank, events))
        if meta["base_time_nanoseconds"] is not None:
            bases.add(meta["base_time_nanoseconds"])
    if not records:
        return {"prompt_tokens": length, "status": "no_rank_traces"}
    ranks = [rank for rank, _ in records]
    if len(set(ranks)) != len(ranks):
        return {"prompt_tokens": length, "status": "ambiguous_duplicate_rank_traces",
                "trace_files": len(records), "ranks": sorted(ranks),
                "critical_path_status": "unavailable",
                "limitations": ["one trace per rank is required; duplicate rank captures were not merged"]}
    nodes, meta = _dependency_graph(records)
    chain = _path(nodes, meta["endpoints"])
    if not chain and nodes:
        return {"prompt_tokens": length, "status": "dependency_cycle",
                "trace_files": len(records), "ranks": meta["ranks"],
                "gpu_events": meta["gpu_events"],
                "critical_path_status": "unavailable",
                "edge_counts": meta["edge_counts"],
                "limitations": ["same-stream dependency graph was cyclic; no partial path was emitted"]}
    # Attribute only the serial increment each path node contributes after its
    # latest dependency. This keeps the top list from becoming a disguised
    # sum of overlapping kernel durations.
    for i in chain:
        predecessor_end = max((nodes[d]["end"] for d in nodes[i]["deps"]), default=nodes[i]["start"])
        nodes[i]["critical_increment_us"] = max(
            0.0, nodes[i]["end"] - max(nodes[i]["start"], predecessor_end)
        )
    top = []
    for i in sorted(chain, key=lambda x: nodes[x]["critical_increment_us"], reverse=True)[:10]:
        node = nodes[i]
        top.append({"name": node["name"], "rank": node["rank"], "kind": node["kind"],
                    "duration_us": node["duration_us"],
                    "critical_increment_us": node["critical_increment_us"],
                    "stream": node["stream"]})
    start = min((nodes[i]["start"] for i in chain), default=0.0)
    end = max((nodes[i]["end"] for i in chain), default=0.0)
    return {"prompt_tokens": length, "status": "ok", "trace_files": len(records),
            "ranks": meta["ranks"], "gpu_events": meta["gpu_events"],
            "critical_path_status": "unavailable",
            "partial_stream_path_us": max(0.0, end - start),
            "partial_stream_path_nodes": len(chain),
            "partial_stream_path_top10": top,
            "edge_counts": {"stream_order": meta["edge_counts"].get("stream_order", 0),
                            "cpu_launch": 0, "event_wait": 0,
                            "collective_rendezvous": 0},
            "coverage": {"rank_traces": len(records), "expected_ranks": 8,
                         "base_time_values": len(bases),
                         "dependency_model": "same-stream order only; cross-rank, CPU-launch, event-wait and graph-DAG edges unavailable"},
            "swimlane": [{"rank": rank, "events": sum(1 for e in events if e["_kind"] == "gpu"),
                          "start_us": min((e["ts"] for e in events if e["_kind"] == "gpu"), default=0),
                          "end_us": max((e["ts"] + e.get("dur", 0) for e in events if e["_kind"] == "gpu"), default=0)}
                         for rank, events in sorted(records)],
            "limitations": ["CPU launch scopes and CUDA graph DAGs are not available to this analyzer input path",
                            "collectives are intentionally not connected without explicit communicator/group and sequence metadata",
                            "partial stream path is not a critical path and must not be used as a latency claim"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    paths = sorted(args.input.rglob("*.trace.json.gz"))
    grouped: dict[int, list[Path]] = defaultdict(list)
    for path in paths:
        length = _length(path)
        if length in LENGTHS:
            grouped[length].append(path)
    rows = [_analyze_length(length, grouped.get(length, [])) for length in LENGTHS]
    report = {"schema": "prefill_report_observed_dependency_v1",
              "input": str(args.input.resolve()), "mode": "CED_P4D4",
              "lengths": rows,
              "collection_contract": {"required": ["traceEvents X records", "rank in filename or an event", "one capture per rank and prompt length"],
                                      "recommended": ["baseTimeNanoseconds", "CPU runtime records with correlation", "CUDA graph DAG sidecars", "explicit communicator/group and collective sequence ids", "event record/wait correlation"]},
              "limitations": ["critical_path is unavailable unless graph, stream, launch, wait and collective dependencies are captured", "partial stream paths are diagnostic only", "missing dependency evidence lowers coverage", "the ijson path is streaming; the fallback parser materializes a trace when ijson is unavailable"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "lengths": [r["prompt_tokens"] for r in rows]}, indent=2))


if __name__ == "__main__":
    main()
