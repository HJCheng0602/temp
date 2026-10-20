#!/usr/bin/env python3
"""Build the five-case 2K PP2 compact swimlane payload from sweep artifacts.

This orchestration helper is stdlib-only.  It discovers completed cases and
measured client ITL values, validates the shared trace clock/window, then
delegates event encoding to ``compact_batch_swimlanes.extract``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from compact_batch_swimlanes import _base_time_nanoseconds, _choose_trace, _events, _gpu_class, extract


TARGET_BATCHES = (16, 32, 64, 128, 256)
TRACE_GLOB = "*.pt.trace.json.gz"
ITL_KEYS = ("median_client_itl_ms", "median_itl_ms", "client_itl_ms", "median_client_itl")


def _find_number(obj: object, keys: tuple[str, ...]) -> float | None:
    if isinstance(obj, dict):
        for key in keys:
            value = obj.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return float(value)
        for value in obj.values():
            found = _find_number(value, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_number(value, keys)
            if found is not None:
                return found
    return None


def _batch_from_path(path: Path) -> int | None:
    text = str(path).lower()
    match = re.search(r"(?:batch|b)[_=-]?(16|32|64|128|256)(?:\D|$)", text)
    return int(match.group(1)) if match else None


def _trace_root(root: Path) -> Path:
    candidates = []
    for path in root.rglob(TRACE_GLOB):
        try:
            _choose_trace(path.parent, 0, 0)
            _choose_trace(path.parent, 1, 4)
        except FileNotFoundError:
            continue
        candidates.append(path.parent)
    if not candidates:
        raise FileNotFoundError(f"no PP2 representative trace pair below {root}")
    # Prefer the deepest directory containing the pair; it avoids accidentally
    # selecting another profile's traces from a broad sweep root.
    unique = set(candidates)
    if len(unique) != 1:
        raise RuntimeError(f"expected one profile directory below {root}, found {len(unique)}")
    return next(iter(unique))


def _completed_case(sweep_root: Path, batch: int) -> tuple[Path, float, Path]:
    matches = []
    for complete in sweep_root.rglob("COMPLETE.json"):
        if _batch_from_path(complete.parent) != batch:
            continue
        try:
            payload = json.loads(complete.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"cannot read {complete}: {exc}") from exc
        itl = _find_number(payload, ITL_KEYS)
        if itl is None:
            metric = complete.parent / "metrics" / f"decode_ctx2048_b{batch}_timing.json"
            if metric.is_file():
                itl = _find_number(json.loads(metric.read_text()), ITL_KEYS)
        if itl is None:
            raise RuntimeError(f"no measured client ITL in {complete}")
        matches.append((complete, itl, _trace_root(complete.parent)))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one completed batch {batch} below {sweep_root}, found {len(matches)}")
    return matches[0]


def _largebatch_case(root: Path) -> tuple[float, Path]:
    metric = root / "metrics" / "decode_ctx2048_b256_timing.json"
    itl = _find_number(json.loads(metric.read_text()), ITL_KEYS)
    if itl is None:
        raise RuntimeError(f"no measured client ITL metric in {metric}")
    return itl, _trace_root(root / "timelines" / "decode_ctx2048_b256_profile")


def _gpu_bounds(trace: Path) -> tuple[float, float, int | None]:
    events = _events(trace)
    starts, ends = [], []
    for event in events:
        if _gpu_class(str(event.get("name", "")), str(event.get("cat", ""))) is None:
            continue
        try:
            ts, dur = float(event.get("ts", 0)), float(event.get("dur", 0))
        except (TypeError, ValueError):
            continue
        if dur > 0:
            starts.append(ts)
            ends.append(ts + dur)
    if not starts:
        raise RuntimeError(f"no GPU events in {trace}")
    return min(starts), max(ends), _base_time_nanoseconds(trace)


def build(sweep_root: Path, largebatch_root: Path, output_dir: Path, trim_ms: float = 50.0) -> None:
    cases = []
    for batch in TARGET_BATCHES[:-1]:
        _complete, itl, trace_root = _completed_case(sweep_root, batch)
        cases.append({"batch": batch, "median_itl_ms": itl, "trace_dir": str(trace_root)})
    itl, trace_root = _largebatch_case(largebatch_root)
    cases.append({"batch": 256, "median_itl_ms": itl, "trace_dir": str(trace_root)})

    available_windows = []
    for case in cases:
        stage0 = _choose_trace(Path(case["trace_dir"]), 0, 0)
        stage1 = _choose_trace(Path(case["trace_dir"]), 1, 4)
        bounds = [_gpu_bounds(stage0), _gpu_bounds(stage1)]
        bases = [value[2] for value in bounds]
        if any(value is None for value in bases) and any(value is not None for value in bases):
            raise RuntimeError(f"incomplete baseTimeNanoseconds metadata for batch {case['batch']}: {bases}")
        if len({value for value in bases if value is not None}) > 1:
            raise RuntimeError(f"stage clocks differ for batch {case['batch']}: {bases}")
        case_start = max(value[0] for value in bounds) + trim_ms * 1000.0
        case_end = min(value[1] for value in bounds)
        case["window_start_ts_us"] = case_start
        case["available_window_ms"] = (case_end - case_start) / 1000.0
        available_windows.append(case["available_window_ms"])

    available_ms = min(available_windows)
    window_ms = min(500.0, math.floor(available_ms / 50.0) * 50.0)
    if window_ms < 200.0:
        raise RuntimeError(f"common post-trim window is only {available_ms:.3f} ms; need at least 200 ms")
    manifest = {"cases": cases, "trim_ms": trim_ms, "window_ms": window_ms,
                "alignment": "Each capture has its own origin; both stages retain their original relative timestamps."}
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    compact_path = output_dir / "compact.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    extract(manifest_path, compact_path, window_ms=window_ms, trim_ms=trim_ms)
    payload = json.loads(compact_path.read_text(encoding="utf-8"))
    stats = {
        "settings": {"batches": list(TARGET_BATCHES), "trim_ms": trim_ms, "window_ms": window_ms, "bin_ms": payload["bin_ms"]},
        "cases": [],
    }
    for case in payload["cases"]:
        annotation_durations = [float(annotation["dur_ms"]) for lane in case["lanes"] for annotation in lane["annotations"]]
        histogram = {}
        for duration in annotation_durations:
            bucket = f"{math.floor(duration)}-{math.floor(duration) + 1}"
            histogram[bucket] = histogram.get(bucket, 0) + 1
        stats["cases"].append({
            "batch": case["batch"],
            "median_itl_ms": case["median_itl_ms"],
            "gpu_fractions": [{"stage": lane["stage"], "union_busy": lane["gpu_union_busy_fraction"], "comm_occupied": lane["comm_occupied_fraction"], "occupied_fraction": lane["occupied_fraction"]} for lane in case["lanes"]],
            "two_execution_cycle_stats_ms": case["execution_cycle_interval_stats_ms"],
            "annotation_duration_histogram_ms": histogram,
        })
    (output_dir / "STATS.json").write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    print(manifest_path)
    print(compact_path)
    print(output_dir / "STATS.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--largebatch-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trim-ms", type=float, default=50.0)
    args = parser.parse_args()
    build(args.sweep_root, args.largebatch_root, args.output_dir, args.trim_ms)


if __name__ == "__main__":
    main()
