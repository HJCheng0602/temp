#!/usr/bin/env python3
"""Extract and render compact PP2 batch swimlanes.

Extraction uses only the Python standard library and is suitable for running
next to trace files on a worker.  Rendering imports matplotlib lazily.

The compact representation is intentionally observational: each fixed time
bin contains a bitmask for the four GPU classes seen in that bin.  CPU
``execute_N`` annotations are retained separately and are never treated as
GPU work.
"""

from __future__ import annotations

import argparse
import gzip
import gzip
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Iterable


BIN_MS = 0.1
WINDOW_MS = 500.0
TRIM_MS = 50.0
BITS = {"nccl": 1, "sendrecv": 2, "compute": 4, "memcpy": 8}
CLASS_NAMES = tuple(BITS)


def _events(path: Path) -> list[dict]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        obj = json.load(handle)
    return obj.get("traceEvents", obj) if isinstance(obj, dict) else obj


def _base_time_nanoseconds(path: Path) -> int | None:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        obj = json.load(handle)
    value = obj.get("baseTimeNanoseconds") if isinstance(obj, dict) else None
    return int(value) if value is not None else None


def _trace_files(root: Path) -> list[Path]:
    return sorted(root.glob("**/*.pt.trace.json.gz")) + sorted(root.glob("**/*.pt.trace.json"))


def _choose_trace(root: Path, pp: int, rank: int) -> Path:
    files = _trace_files(root)
    token = re.compile(rf"(?:^|[_-])pp{pp}(?:[_-]|$).*?(?:^|[_-])rank{rank}(?:[._-]|$)", re.I)
    exact = [p for p in files if token.search(p.name)]
    if exact:
        return exact[0]
    # Some exporters put pp/rank tokens in a parent directory instead.
    exact = [p for p in files if token.search(str(p))]
    if exact:
        return exact[0]
    raise FileNotFoundError(f"no representative PP{pp} rank{rank} trace below {root}")


def _gpu_class(name: str, cat: str) -> str | None:
    low = name.lower()
    cat_low = cat.lower()
    if cat_low not in ("kernel", "gpu_memcpy", "gpu_memset"):
        return None
    if "sendrecv" in low or "send_recv" in low:
        return "sendrecv"
    if "nccl" in low or any(k in low for k in ("allreduce", "allgather", "broadcast", "reducescatter")):
        return "nccl"
    if cat_low in ("gpu_memcpy", "gpu_memset") or "memcpy" in low or "memset" in low:
        return "memcpy"
    if cat_low == "kernel":
        # Generic kernels remain compute.  Do not infer attention/MoE labels.
        return "compute"
    return None


def _interval_union(intervals: Iterable[tuple[float, float]]) -> float:
    ordered = sorted((a, b) for a, b in intervals if b > a)
    total = 0.0
    start = stop = None
    for a, b in ordered:
        if start is None:
            start, stop = a, b
        elif a > stop:
            total += stop - start
            start, stop = a, b
        else:
            stop = max(stop, b)
    return total + (0.0 if start is None else stop - start)


def _rle(values: list[int]) -> list[list[int]]:
    if not values:
        return []
    result: list[list[int]] = []
    last = values[0]
    count = 1
    for value in values[1:]:
        if value == last:
            count += 1
        else:
            result.append([count, last])
            last, count = value, 1
    result.append([count, last])
    return result


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _lane(path: Path, start_us: float, window_us: float, bin_us: float) -> dict:
    events = _events(path)
    gpu: list[tuple[float, float, str]] = []
    annotations: list[dict] = []
    all_gpu_times: list[float] = []
    for event in events:
        try:
            ts = float(event.get("ts", 0))
            dur = float(event.get("dur", 0))
        except (TypeError, ValueError):
            continue
        if dur <= 0:
            continue
        name = str(event.get("name", ""))
        cat = str(event.get("cat", ""))
        if cat == "user_annotation" and name.startswith("execute_"):
            end = ts + dur
            if end > start_us and ts < start_us + window_us:
                annotations.append({
                    "offset_ms": round((ts - start_us) / 1000.0, 6),
                    "dur_ms": round(dur / 1000.0, 6),
                    "label": name,
                })
        cls = _gpu_class(name, cat)
        if cls is None:
            continue
        all_gpu_times.extend((ts, ts + dur))
        if ts + dur > start_us and ts < start_us + window_us:
            gpu.append((max(ts, start_us), min(ts + dur, start_us + window_us), cls))

    n_bins = int(math.ceil(window_us / bin_us))
    masks = [0] * n_bins
    by_class: dict[str, list[tuple[float, float]]] = {key: [] for key in CLASS_NAMES}
    for left, right, cls in gpu:
        by_class[cls].append((left, right))
        first = max(0, int(math.floor((left - start_us) / bin_us)))
        last = min(n_bins - 1, int(math.ceil((right - start_us) / bin_us)) - 1)
        for index in range(first, last + 1):
            bin_left = start_us + index * bin_us
            bin_right = min(start_us + window_us, bin_left + bin_us)
            if left < bin_right and right > bin_left:
                masks[index] |= BITS[cls]

    union = _interval_union((a, b) for a, b, _ in gpu)
    comm = _interval_union(by_class["nccl"] + by_class["sendrecv"])
    class_occupied = {key: _interval_union(value) / window_us for key, value in by_class.items()}
    ann_durations = [float(item["dur_ms"]) for item in annotations]
    starts = sorted(float(item["offset_ms"]) for item in annotations)
    cycles = [b - a for a, b in zip(starts, starts[1:])]
    return {
        "trace": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "first_gpu_ts_us": min(all_gpu_times) if all_gpu_times else None,
        "bin_count": n_bins,
        "bins_rle": _rle(masks),
        "occupied_fraction": {key: round(value, 6) for key, value in class_occupied.items()},
        "gpu_union_busy_fraction": round(union / window_us, 6),
        "comm_occupied_fraction": round(comm / window_us, 6),
        "annotations": annotations,
        "annotation_cycle_stats_ms": {
            "count": len(annotations),
            "duration_median": round(statistics.median(ann_durations), 6) if ann_durations else 0.0,
            "duration_p95": round(_percentile(ann_durations, 0.95), 6),
            "interval_median": round(statistics.median(cycles), 6) if cycles else 0.0,
            "interval_p95": round(_percentile(cycles, 0.95), 6),
            "start_delta_median": round(statistics.median(cycles), 6) if cycles else 0.0,
        },
    }


def extract(manifest_path: Path, output: Path, bin_ms: float = BIN_MS,
             window_ms: float = WINDOW_MS, trim_ms: float = TRIM_MS) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest if isinstance(manifest, list) else manifest.get("cases", [])
    if not cases:
        raise ValueError("manifest must contain a non-empty cases list")
    bin_us = bin_ms * 1000.0
    window_us = window_ms * 1000.0
    output_cases = []
    trace_pairs: list[tuple[Path, Path]] = []
    for item in cases:
        root = Path(item["trace_dir"])
        stage0 = _choose_trace(root, 0, 0)
        stage1 = _choose_trace(root, 1, 4)
        paths = [stage0, stage1]
        trace_pairs.append((stage0, stage1))
        base_times = [_base_time_nanoseconds(path) for path in paths]
        known_base_times = {value for value in base_times if value is not None}
        if any(value is None for value in base_times) and known_base_times:
            raise RuntimeError(
                f"representative trace clocks have incomplete baseTimeNanoseconds metadata for batch {item.get('batch')}: {base_times}"
            )
        if len(known_base_times) > 1:
            raise RuntimeError(
                f"representative trace clocks differ for batch {item.get('batch')}: {base_times}; "
                "cannot preserve absolute stage offsets"
            )
        raw_first = []
        raw_last = []
        for path in paths:
            times = []
            path_last = []
            for event in _events(path):
                cls = _gpu_class(str(event.get("name", "")), str(event.get("cat", "")))
                try:
                    ts, dur = float(event.get("ts", 0)), float(event.get("dur", 0))
                except (TypeError, ValueError):
                    continue
                if cls is not None and dur > 0:
                    times.append(ts)
                    path_last.append(ts + dur)
            if times:
                raw_first.append(min(times))
                raw_last.append(max(path_last))
        if not raw_first:
            raise RuntimeError(f"no GPU events found in representative traces for batch {item.get('batch')}")
        # Both stages share this absolute trace clock.  Start after the later
        # first GPU event so the 50 ms trim is valid for both lanes.
        first_gpu = max(raw_first)
        start = first_gpu + trim_ms * 1000.0
        common_end = min(raw_last)
        if start + window_us > common_end:
            available_ms = max(0.0, (common_end - start) / 1000.0)
            raise RuntimeError(
                f"batch {item.get('batch')} has only {available_ms:.3f} ms after the {trim_ms:g} ms trim; "
                f"requested {window_ms:g} ms shared window"
            )
        lanes = [
            {"stage": 0, "rank": 0, **_lane(stage0, start, window_us, bin_us)},
            {"stage": 1, "rank": 4, **_lane(stage1, start, window_us, bin_us)},
        ]
        # A decode cycle spans the corresponding execute annotation on two
        # successive iterations.  Keep each stage's timeline separate so
        # cross-stage scheduling offsets do not become fake latency samples.
        all_deltas = []
        for lane in lanes:
            starts = sorted(float(annotation["offset_ms"]) for annotation in lane["annotations"])
            all_deltas.extend(b - a for a, b in zip(starts, starts[2:]))
        output_cases.append({
            "batch": item.get("batch"),
            "median_itl_ms": item.get("median_itl_ms"),
            "window_start_ts_us": start,
            "first_gpu_ts_us": first_gpu,
            "shared_trace_base_time_nanoseconds": next(iter(known_base_times), None),
            "clock_assumption": "matching baseTimeNanoseconds" if known_base_times else "both traces omit baseTimeNanoseconds; timestamps assumed to share a clock",
            "common_gpu_end_ts_us": common_end,
            "trim_ms": trim_ms,
            "lanes": lanes,
            "execution_cycle_interval_stats_ms": {
                "count": len(all_deltas),
                "median": round(statistics.median(all_deltas), 6) if all_deltas else 0.0,
                "p95": round(_percentile(all_deltas, 0.95), 6),
                "definition": "pooled per-stage execute start[i+2]-start[i] deltas across the two representative stages",
            },
        })
    payload = {
        "schema": "compact_batch_swimlanes/v1",
        "bin_ms": bin_ms,
        "window_ms": window_ms,
        "classes": {key: value for key, value in BITS.items()},
        "time_units": "trace timestamps are microseconds; offsets and durations are milliseconds",
        "cases": output_cases,
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    # Alternating occupancy can make a fine grid larger than the transfer
    # budget.  Preserve the same window and raw-derived stats while coarsening
    # only the occupancy grid in that case.
    if len(gzip.compress(encoded.encode("utf-8"), compresslevel=9)) > 20000 and bin_ms < 0.25:
        bin_ms = 0.25
        bin_us = bin_ms * 1000.0
        for case, (stage0, stage1) in zip(output_cases, trace_pairs):
            start = float(case["window_start_ts_us"])
            case["lanes"] = [
                {"stage": 0, "rank": 0, **_lane(stage0, start, window_us, bin_us)},
                {"stage": 1, "rank": 4, **_lane(stage1, start, window_us, bin_us)},
            ]
        payload["bin_ms"] = bin_ms
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded, encoding="utf-8")


def _expand(rle: list[list[int]]) -> list[int]:
    values = []
    for count, mask in rle:
        values.extend([int(mask)] * int(count))
    return values


def render(compact_path: Path, outdir: Path, title: str | None = None) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    payload = json.loads(compact_path.read_text(encoding="utf-8"))
    bin_ms = float(payload["bin_ms"])
    window_ms = float(payload["window_ms"])
    colors = {"nccl": "#c2410c", "sendrecv": "#7c3aed", "compute": "#2563eb", "memcpy": "#ca8a04"}
    fig, axes = plt.subplots(len(payload["cases"]), 1, figsize=(18, 2.45 * len(payload["cases"])), sharex=True, squeeze=False)
    axes_flat = [row[0] for row in axes]
    for ax, case in zip(axes_flat, payload["cases"]):
        lane_h = 0.82
        y_positions = {0: 1.25, 1: 0.05}
        ax.axhspan(1.18, 2.28, facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=0.6, zorder=0)
        ax.axhspan(-0.12, 1.10, facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=0.6, zorder=0)
        for lane in case["lanes"]:
            y = y_positions[int(lane["stage"])]
            masks = _expand(lane["bins_rle"])
            for bit, cls in enumerate(CLASS_NAMES):
                flag = BITS[cls]
                start = None
                for index in range(len(masks) + 1):
                    present = index < len(masks) and (masks[index] & flag) != 0
                    if present and start is None:
                        start = index
                    if start is not None and not present:
                        ax.broken_barh([(start * bin_ms, (index - start) * bin_ms)],
                                       (y + bit * 0.145, 0.115), facecolors=colors[cls], edgecolors="none", alpha=0.9)
                        start = None
            for ann in lane["annotations"]:
                x = float(ann["offset_ms"])
                w = max(0.03, float(ann["dur_ms"]))
                ax.broken_barh([(x, w)], (y + 0.59, 0.09), facecolors="#64748b", alpha=0.42, edgecolors="none")
                match = re.search(r"execute_(\d+)", ann["label"])
                if match:
                    ax.text(max(0.0, x) + 0.25, y + 0.70, f"N={match.group(1)}", fontsize=5.7, color="#475569", clip_on=True)
            ax.text(-window_ms * 0.012, y + 0.29, f"Stage {int(lane['stage']) + 1} / GPU {lane['rank']}", ha="right", va="center", fontsize=8.5, fontweight="bold", color="#334155")
        ax.set_ylim(-0.12, 2.28)
        ax.set_xlim(0, window_ms)
        ax.set_yticks([])
        ax.grid(axis="x", color="#cbd5e1", linestyle=":", linewidth=0.7)
        ax.set_axisbelow(True)
        stats = [lane for lane in case["lanes"]]
        batch = case.get("batch", "?")
        itl = case.get("median_itl_ms")
        itl_text = f"; unprofiled median client ITL {itl:.2f} ms" if isinstance(itl, (int, float)) else "; unprofiled median client ITL unavailable"
        ax.set_title(f"B={batch}{itl_text}", loc="left", fontsize=10, fontweight="bold")
    axes_flat[-1].set_xlabel(f"Time from common first GPU event + trim (ms), fixed {window_ms:g} ms window; bin resolution {bin_ms:g} ms")
    legend = [Patch(facecolor=colors[key], label=label) for key, label in (("nccl", "NCCL"), ("sendrecv", "PP SendRecv"), ("compute", "GPU compute"), ("memcpy", "Memcpy/ memset"))]
    legend.append(Patch(facecolor="#64748b", alpha=0.42, label="execute_N = CPU annotation label"))
    fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(0.995, 0.995), ncol=5, frameon=False, fontsize=7.2)
    fig.suptitle(title or "GPU actual 2K activity by batch", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.006, "GPU actual 2K activity; NCCL kernel duration can include wait time; CPU execute_N labels are not GPU durations.", fontsize=7, color="#64748b")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / "compact_batch_swimlanes.png"
    pdf = outdir / "compact_batch_swimlanes.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", type=Path, metavar="MANIFEST")
    parser.add_argument("--output", "--output-compact", dest="output", type=Path)
    parser.add_argument("--render-compact", "--rendercompact", dest="render_compact", type=Path)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--bin-ms", type=float, default=BIN_MS)
    parser.add_argument("--window-ms", type=float, default=WINDOW_MS)
    parser.add_argument("--trim-ms", type=float, default=TRIM_MS)
    args = parser.parse_args()
    if args.extract:
        if not args.output:
            parser.error("--extract requires --output")
        extract(args.extract, args.output, args.bin_ms, args.window_ms, args.trim_ms)
        print(args.output)
    elif args.render_compact:
        if not args.outdir:
            parser.error("--render-compact requires --outdir")
        for path in render(args.render_compact, args.outdir):
            print(path)
    else:
        parser.error("choose --extract or --render-compact")


if __name__ == "__main__":
    main()
