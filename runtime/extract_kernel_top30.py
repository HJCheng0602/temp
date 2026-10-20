#!/usr/bin/env python3
"""Stream graph/eager profiler traces into exact per-rank kernel summaries.

Graph timings and eager shape metadata are deliberately kept separate.  Graph
traces also get a small all-rank event compact so later phase audits can use
the original timestamps without reopening the raw files.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from compact_graph_probe_trace import _device, _iter_trace_events, _stream_id


RESERVOIR_SIZE = 512
RANK_RE = re.compile(r"(?:rank|worker)[_-]?(\d+)", re.I)
GPU_CATEGORIES = {"kernel", "cuda_kernel", "gpu_kernel", "cuda kernels", "gpu kernels"}


class Reservoir:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.minimum: float | None = None
        self.maximum: float | None = None
        self.values: list[float] = []
        self.state = 0x9E3779B9

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        if len(self.values) < RESERVOIR_SIZE:
            self.values.append(value)
            return
        # Deterministic bounded sampling makes output reproducible across runs.
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        slot = self.state % self.count
        if slot < RESERVOIR_SIZE:
            self.values[slot] = value

    def result(self) -> dict[str, float | int | None]:
        ordered = sorted(self.values)
        def percentile(fraction: float) -> float | None:
            if not ordered:
                return None
            index = max(0, min(len(ordered) - 1,
                               int((fraction * len(ordered) + 0.999999) - 1)))
            return ordered[index]
        return {
            "count": self.count,
            "sum_us": self.total,
            "mean_us": self.total / self.count if self.count else None,
            "p50_us": percentile(0.50),
            "p95_us": percentile(0.95),
            "min_us": self.minimum,
            "max_us": self.maximum,
            "quantile_method": f"bounded_reservoir_{RESERVOIR_SIZE}",
        }


def trace_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.trace.json.gz"))


def rank_of(path: Path) -> str:
    match = RANK_RE.search(path.name)
    return match.group(1) if match else path.stem


def metadata_for(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest,
    }


def is_kernel(event: dict[str, Any]) -> bool:
    if event.get("ph") != "X":
        return False
    if str(event.get("cat", "")).lower() not in GPU_CATEGORIES:
        return False
    duration = event.get("dur")
    if not isinstance(duration, (int, float)) or duration <= 0:
        return False
    name = str(event.get("name", "")).lower()
    # CUDA memcpy/memset records are transfers, not kernel work for this table.
    return not any(token in name for token in ("memcpy", "memset"))


def cpu_record(event: dict[str, Any]) -> dict[str, Any] | None:
    args = event.get("args")
    if event.get("cat") != "cpu_op" or event.get("ph") != "X" or not isinstance(args, dict):
        return None
    external = args.get("External id")
    if external is None:
        return None
    keep = {key: args[key] for key in
            ("Input Dims", "Input type", "ConcreteInputs", "Output", "output")
            if key in args}
    return {"external_id": str(external), "name": event.get("name", ""),
            "ts": event.get("ts"), "dur": event.get("dur"), "args": keep}


def compact_event(event: dict[str, Any]) -> dict[str, Any]:
    args = event.get("args", {})
    result = {"ts": event.get("ts"), "dur": float(event["dur"]),
              "name": event.get("name", ""), "device": _device(event),
              "stream": _stream_id(event)}
    if isinstance(args, dict):
        for key in ("channel", "channel_type", "correlation", "graph id", "graph node id"):
            if key in args:
                result[key] = args[key]
    return result


def extract_trace(path: Path, mode: str, compact_dir: Path | None) -> dict[str, Any]:
    stats: dict[str, Reservoir] = {}
    streams: dict[str, Counter[str]] = {}
    graph_ids: dict[str, Counter[str]] = {}
    cpu_ops: dict[str, dict[str, Any]] = {}
    links: Counter[tuple[str, str]] = Counter()
    pending_links: list[tuple[str, str, str]] = []
    compact_handle = None
    compact_path = None
    if mode == "graph" and compact_dir is not None:
        compact_dir.mkdir(parents=True, exist_ok=True)
        compact_path = compact_dir / (path.name.removesuffix(".gz") + ".compact.json.gz")
        compact_handle = gzip.open(compact_path, "wt", encoding="utf-8")
        compact_handle.write('{"source":')
        json.dump(str(path), compact_handle)
        compact_handle.write(',"traceEvents":[\n')
    compact_first = True
    event_count = kernel_count = cpu_count = 0
    device_counts: Counter[str] = Counter()
    for event in _iter_trace_events(path):
        event_count += 1
        record = cpu_record(event)
        if record is not None:
            cpu_count += 1
            cpu_ops[record["external_id"]] = record
        if not is_kernel(event):
            continue
        kernel_count += 1
        name = str(event.get("name", ""))
        duration = float(event["dur"])
        stats.setdefault(name, Reservoir()).add(duration)
        stream = _stream_id(event)
        streams.setdefault(name, Counter())[str(stream) if stream is not None else "unknown"] += 1
        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        graph = args.get("graph id", event.get("graph id"))
        graph_ids.setdefault(name, Counter())[str(graph) if graph is not None else "unknown"] += 1
        device = _device(event)
        device_counts[str(device) if device is not None else "unknown"] += 1
        if mode == "graph":
            if not compact_first:
                compact_handle.write(",\n")
            json.dump(compact_event(event), compact_handle, separators=(",", ":"))
            compact_first = False
        external = args.get("External id")
        correlation = args.get("correlation", event.get("correlation"))
        if external is not None:
            pending_links.append((name, str(external), str(correlation) if correlation is not None else ""))
    for name, external, correlation in pending_links:
        links[(name, external)] += 1
    kernels = {}
    for name, value in sorted(stats.items()):
        row = value.result()
        row.update({"streams": dict(streams[name]), "graph_ids": dict(graph_ids[name])})
        row["cpu_external_ids"] = {
            external: count for (kernel, external), count in links.items() if kernel == name
        }
        kernels[name] = row
    if compact_handle is not None:
        compact_handle.write('\n]}\n')
        compact_handle.close()
    output: dict[str, Any] = {
        "mode": mode, "rank": rank_of(path), "source": metadata_for(path),
        "event_count": event_count, "kernel_event_count": kernel_count,
        "cpu_op_count": cpu_count, "device_event_counts": dict(device_counts),
        "kernel_names": len(kernels), "kernels": kernels,
    }
    output["cpu_ops_by_external_id"] = cpu_ops
    if mode == "graph":
        output["compact_timeline"] = str(compact_path) if compact_path else None
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graphdir", type=Path)
    parser.add_argument("eagerdir", type=Path)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--allow-single-mode", action="store_true")
    args = parser.parse_args()
    graph_files, eager_files = trace_files(args.graphdir), trace_files(args.eagerdir)
    if not graph_files and not args.allow_single_mode:
        parser.error(f"no *.trace.json.gz files below {args.graphdir}")
    if not eager_files and not args.allow_single_mode:
        parser.error(f"no *.trace.json.gz files below {args.eagerdir}")
    graph_out = args.outdir / "graph"
    eager_out = args.outdir / "eager"
    graph_rows = [extract_trace(path, "graph", args.outdir / "graph_compact") for path in graph_files]
    eager_rows = [extract_trace(path, "eager", None) for path in eager_files]
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "manifest.json").write_text(json.dumps({
        "schema": "kernel_top30_extract/v1",
        "graph": {"directory": str(args.graphdir), "ranks": graph_rows,
                  "timing_scope": "summed GPU kernel durations; not wall or critical-path time"},
        "eager": {"directory": str(args.eagerdir), "ranks": eager_rows,
                  "scope": "CPU Input Dims/type metadata only; eager durations are not mixed into graph timing"},
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
