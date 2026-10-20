#!/usr/bin/env python3
"""Stream large PyTorch traces into compact CUDA-kernel summaries.

The input is read one gzip member at a time.  No complete trace or event list
is materialized; quantiles use bounded reservoirs and the compact timeline is
written incrementally.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Iterator


CHUNK_SIZE = 1024 * 1024
RESERVOIR_SIZE = 512
MAX_KERNEL_NAMES = 50_000
MAX_CPU_SCOPES = 2_048
MAX_METADATA = 512
GPU_TIMELINE = {0, 4}
_RANK_RE = re.compile(r"(?:rank|worker)[_-]?(\d+)", re.I)
_CPU_SCOPE_RE = re.compile(r"forward|execute|nvtx", re.I)


def _iter_trace_events(path: Path) -> Iterator[dict[str, Any]]:
    """Yield objects from a top-level traceEvents array incrementally."""
    decoder = json.JSONDecoder()
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as stream:
        buf = ""
        pos = 0
        array_found = False
        eof = False
        need_more = True
        while True:
            if not eof and (not array_found or pos >= len(buf) or need_more):
                piece = stream.read(CHUNK_SIZE)
                if piece:
                    buf = buf[pos:] + piece
                    pos = 0
                    need_more = False
                else:
                    eof = True

            if not array_found:
                stripped = buf.lstrip()
                offset = len(buf) - len(stripped)
                marker = stripped.find('"traceEvents"')
                if marker >= 0:
                    bracket = stripped.find("[", marker + len('"traceEvents"'))
                    if bracket < 0:
                        if eof:
                            raise ValueError(f"traceEvents array not found in {path}")
                        continue
                    pos = offset + bracket + 1
                    array_found = True
                elif stripped.startswith("["):
                    pos = offset + 1
                    array_found = True
                elif eof:
                    raise ValueError(f"traceEvents array not found in {path}")
                else:
                    # Preserve a split key at the end of a chunk.
                    buf = buf[-64:]
                    pos = 0
                    continue

            while True:
                while pos < len(buf) and buf[pos] in " \t\r\n,":
                    pos += 1
                if pos < len(buf) and buf[pos] == "]":
                    return
                if pos >= len(buf):
                    break
                try:
                    value, end = decoder.raw_decode(buf, pos)
                except json.JSONDecodeError:
                    if eof:
                        raise ValueError(f"truncated trace event in {path}")
                    need_more = True
                    break
                pos = end
                if isinstance(value, dict):
                    yield value


class _Reservoir:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.maximum = 0.0
        self.values: list[float] = []

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.maximum = max(self.maximum, value)
        if len(self.values) < RESERVOIR_SIZE:
            self.values.append(value)
        else:
            slot = random.randrange(self.count)
            if slot < RESERVOIR_SIZE:
                self.values[slot] = value

    def result(self) -> dict[str, float | int | None]:
        values = sorted(self.values)
        return {
            "count": self.count,
            "sum_us": self.total,
            "p50_us": _percentile(values, 0.50),
            "p95_us": _percentile(values, 0.95),
            "max_us": self.maximum if self.count else None,
        }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    index = max(0, min(len(values) - 1, int(math.ceil(fraction * len(values))) - 1))
    return values[index]


def _device(event: dict[str, Any]) -> int | str | None:
    args = event.get("args")
    if not isinstance(args, dict):
        args = {}
    value = event.get("device", args.get("device", args.get("device_id", args.get("deviceId"))))
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, str)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    return None


def _stream_id(event: dict[str, Any]) -> int | str | None:
    args = event.get("args")
    if not isinstance(args, dict):
        args = {}
    return event.get("stream", args.get("stream", args.get("stream_id", args.get("streamId"))))


def _category(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("engram", "hash_lookup", "hashlookup", "embedding_lookup")):
        return "Engramlookup"
    if any(token in lowered for token in ("nccl", "allreduce", "all_reduce", "send", "recv", "p2p")):
        return "NCCL"
    if any(token in lowered for token in ("gemm", "mxfp8", "matmul", "linear", "mm_")):
        return "GEMM"
    if any(token in lowered for token in ("attention", "flashinfer", "sparse_mla", "mla_")):
        return "attention"
    return "other"


def _is_gpu_kernel(event: dict[str, Any], device: int | str | None) -> bool:
    category = str(event.get("cat", "")).lower()
    return event.get("ph") == "X" and category in {
        "kernel", "cuda_kernel", "gpu_kernel", "cuda kernels", "gpu kernels"
    }


def _rank(path: Path) -> str:
    match = _RANK_RE.search(path.name)
    return match.group(1) if match else path.stem


def compact_trace(path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = path.name.removesuffix(".gz").removesuffix(".json").removesuffix(".trace")
    timeline_path = output_dir / f"{stem}.compact.json.gz"
    kernel_stats: dict[str, _Reservoir] = {}
    kernel_devices: dict[str, Counter[str]] = {}
    kernel_streams: dict[str, Counter[str]] = {}
    category_stats: dict[str, _Reservoir] = {}
    device_counts: Counter[str] = Counter()
    metadata: list[dict[str, Any]] = []
    cpu_scopes: list[dict[str, Any]] = []
    cpu_scope_dropped = 0
    timeline_count = 0

    with gzip.open(timeline_path, "wt", encoding="utf-8") as compact:
        compact.write('{"source":')
        compact.write(json.dumps(str(path)))
        compact.write(',"traceEvents":[\n')
        first = True
        for event in _iter_trace_events(path):
            category = str(event.get("cat", ""))
            name = str(event.get("name", ""))
            if event.get("ph") == "M":
                if len(metadata) < MAX_METADATA:
                    metadata.append({k: event.get(k) for k in ("name", "ph", "pid", "tid", "args")})
            if category == "user_annotation" and _CPU_SCOPE_RE.search(name):
                if len(cpu_scopes) < MAX_CPU_SCOPES:
                    cpu_scopes.append({k: event.get(k) for k in ("name", "ts", "dur", "pid", "tid", "args")})
                else:
                    cpu_scope_dropped += 1

            device = _device(event)
            if not _is_gpu_kernel(event, device):
                continue
            duration = event.get("dur")
            if not isinstance(duration, (int, float)) or duration < 0:
                continue
            duration = float(duration)
            key = name or "<unnamed>"
            if key not in kernel_stats and len(kernel_stats) >= MAX_KERNEL_NAMES:
                key = "__other_kernel_names__"
            kernel_stats.setdefault(key, _Reservoir()).add(duration)
            kernel_devices.setdefault(key, Counter())[str(device) if device is not None else "unknown"] += 1
            stream = _stream_id(event)
            kernel_streams.setdefault(key, Counter())[str(stream) if stream is not None else "unknown"] += 1
            cat = _category(name)
            category_stats.setdefault(cat, _Reservoir()).add(duration)
            device_counts[str(device) if device is not None else "unknown"] += 1
            if device in GPU_TIMELINE:
                compact_event = {
                    "ts": event.get("ts"), "dur": duration, "name": name,
                    "stream": _stream_id(event), "device": device,
                }
                args = event.get("args", {})
                if isinstance(args, dict):
                    for field in ("channel", "channel_type", "correlation", "graph id", "graph node id"):
                        if field in args:
                            compact_event[field] = args[field]
                if not first:
                    compact.write(",\n")
                compact.write(json.dumps(compact_event, separators=(",", ":")))
                first = False
                timeline_count += 1
        compact.write('\n]}\n')

    summary = {
        "source": str(path),
        "rank": _rank(path),
        "metadata": metadata,
        "metadata_truncated": len(metadata) >= MAX_METADATA,
        "cpu_user_annotation_scopes": cpu_scopes,
        "cpu_scope_dropped": cpu_scope_dropped,
        "gpu_kernel_events": sum(stat.count for stat in kernel_stats.values()),
        "compact_gpu0_gpu4_events": timeline_count,
        "device_event_counts": dict(device_counts),
        "quantile_method": f"reservoir_{RESERVOIR_SIZE}",
        "by_category": {key: value.result() for key, value in sorted(category_stats.items())},
        "by_kernel": {
            key: {
                "category": _category(key),
                "devices": dict(kernel_devices.get(key, {})),
                "streams": dict(kernel_streams.get(key, {})),
                **value.result(),
            }
            for key, value in sorted(kernel_stats.items())
        },
        "compact_timeline": str(timeline_path),
    }
    summary_path = output_dir / f"{stem}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracefolder", type=Path)
    parser.add_argument("outputdirectory", type=Path)
    args = parser.parse_args()
    traces = sorted({*args.tracefolder.rglob("*.trace.json.gz"), *args.tracefolder.rglob("*.json.gz")})
    if not traces:
        parser.error(f"no gzip trace files found under {args.tracefolder}")
    summaries = [compact_trace(path, args.outputdirectory) for path in traces]
    index = {"trace_count": len(summaries), "traces": summaries}
    (args.outputdirectory / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
