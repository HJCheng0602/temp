"""Summarize large-batch request records into JSON and CSV.

The throughput calculation uses streamed token timestamps strictly inside each
case's common decode window. It never derives throughput from batch size or a
median ITL. No external packages are required.
"""

import argparse
import csv
import gzip
import json
from pathlib import Path
import re
import statistics
from collections import Counter


CASE_RE = re.compile(r"^decode_ctx(?P<context>\d+)_b(?P<batch>\d+)(?:_(?P<kind>timing|profile))?$")
PREEMPT_RE = re.compile(r"preempt|preemption", re.IGNORECASE)


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def load_json(path):
    with path.open() as handle:
        return json.load(handle)


def token_count(events, start, end):
    return sum(
        len(event.get("token_ids", []))
        for event in events
        if start < event.get("t", float("inf")) <= end
    )


def summarize_case(request_path, metric_path=None, metric_value=None):
    request = load_json(request_path)
    metric = (metric_value if metric_value is not None else
              load_json(metric_path) if metric_path is not None and metric_path.is_file() else {})
    match = CASE_RE.match(request_path.stem.removesuffix(".json"))
    identifiers = {
        "case": request_path.stem.removesuffix(".json"),
        "context_tokens": int(match.group("context")) if match else None,
        "requested_batch": int(request.get("batch", 0)),
    }
    records = request.get("records", [])
    driver_status = metric.get("status")
    driver_metadata = {
        "driver_status": driver_status,
        "driver_error": metric.get("error"),
        "driver_traceback": metric.get("traceback"),
        "preemptions_before": metric.get("preemptions_before"),
        "preemptions_after": metric.get("preemptions_after"),
        "preemptions_delta": metric.get("preemptions_delta"),
        "preemption_status": metric.get("preemption_status"),
    }
    if not records:
        return {**identifiers, **driver_metadata, "status": "failed",
                "failure": metric.get("error", "request record has no records")}
    record_errors = [r.get("error") for r in records if r.get("error")]
    if record_errors:
        return {**identifiers, **driver_metadata, "status": "failed",
                "failure": metric.get("error", "request record errors"),
                "record_errors": record_errors}
    first = [r.get("events", [])[0]["t"] for r in records if r.get("events")]
    last = [r.get("events", [])[-1]["t"] for r in records if r.get("events")]
    if len(first) != len(records) or len(last) != len(records):
        return {**identifiers, **driver_metadata, "status": "failed",
                "failure": metric.get("error", "request record has empty token events")}
    common_start = max(first)
    common_end = min(last)
    seconds = max(0.0, common_end - common_start)
    count = sum(token_count(r["events"], common_start, common_end) for r in records)
    itls = []
    for record in records:
        events = record["events"]
        # Match parallel_matrix.summarize: discard each request's first eight
        # intervals before applying the common-window and single-token tests.
        for before, after in zip(events[8:], events[9:]):
            if (before["t"] >= common_start and after["t"] <= common_end
                    and len(after.get("token_ids", [])) == 1):
                itls.append(after["t"] - before["t"])
    verified = bool(metric.get("steady_batch_verified")) and min(
        metric.get("steady_intervals_per_request", [0])
    ) >= 32
    preemption_delta = metric.get("preemptions_delta")
    preemption_status = metric.get("preemption_status")
    if preemption_status == "detected" or (preemption_delta is not None and preemption_delta != 0):
        status = "preempted"
        resident_batch = None
    elif preemption_status == "unknown":
        status = "preemption_unknown"
        resident_batch = None
    else:
        status = "complete" if verified else "unverified"
        resident_batch = int(request.get("batch", 0)) if verified else None
    row = {
        **identifiers,
        "resident_verified_batch": resident_batch,
        "status": status,
        "prompt_tokens": request.get("prompt_tokens"),
        "requested_output_tokens": request.get("requested_output_tokens"),
        "common_decode_start": common_start,
        "common_decode_end": common_end,
        "common_decode_seconds": seconds,
        "common_window_output_tokens": count,
        "aggregate_output_tokens_per_second": count / seconds if seconds else None,
        "steady_intervals_per_request_min": min(metric.get("steady_intervals_per_request", [0])),
        "steady_intervals_per_request_max": max(metric.get("steady_intervals_per_request", [0])),
        "itl_count": len(itls),
        "median_itl_ms": statistics.median(itls) * 1000 if itls else None,
        "mean_itl_ms": statistics.mean(itls) * 1000 if itls else None,
        "p90_itl_ms": percentile(itls, 0.90) * 1000 if itls else None,
        "p99_itl_ms": percentile(itls, 0.99) * 1000 if itls else None,
        **driver_metadata,
    }
    if driver_status == "timing_failed":
        row["status"] = "failed"
        row["failure"] = metric.get("error", "driver timing failed")
    elif status == "preempted":
        row["failure"] = "preemption detected during case"
    elif status == "preemption_unknown":
        row["failure"] = "preemption counter unavailable"
    elif not verified:
        row["failure"] = "fewer than 32 common steady ITLs per request"
    return row


def preemption_evidence(root):
    paths = [root / "server.log"] + sorted(root.glob("**/server.log"))
    seen = set()
    lines = []
    for path in paths:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if PREEMPT_RE.search(line):
                lines.append({"file": str(path), "line": line_number, "text": line[:500]})
    return {"count": len(lines), "lines": lines}


def trace_annotation_histogram(root):
    """Return explicit execute/generation annotations grouped by rank trace."""
    output = {}
    pattern = re.compile(r"execute_(\d+).*generation_(\d+)")
    for path in sorted(root.glob("**/*.pt.trace.json.gz")):
        execute = Counter()
        generation = Counter()
        with gzip.open(path, "rt") as handle:
            trace = json.load(handle)
        for event in trace.get("traceEvents", []):
            if event.get("cat") != "user_annotation":
                continue
            match = pattern.search(str(event.get("name", "")))
            if match:
                execute[match.group(1)] += 1
                generation[match.group(2)] += 1
        if execute:
            output[path.name] = {
                "execute_histogram": dict(execute),
                "generation_histogram": dict(generation),
            }
    return {
        "interpretation": (
            "Explicit user_annotation labels from profiler traces; generation_N "
            "is reported verbatim and is not asserted to be a scheduler microbatch size."
        ),
        "per_rank": output,
    }


def summarize(root):
    request_dir = root / "requests"
    metric_dir = root / "metrics"
    rows = []
    for request_path in sorted(request_dir.glob("decode_ctx*_b*_timing.json")):
        stem = request_path.stem
        metric_path = metric_dir / f"{stem}.json"
        metric = load_json(metric_path) if metric_path.is_file() else {}
        sidecar_path = metric_dir / stem.removesuffix("_timing")
        sidecar_path = sidecar_path.with_suffix(".json")
        if sidecar_path.is_file():
            sidecar = load_json(sidecar_path)
            metric = {**metric, **sidecar}
            row = summarize_case(request_path, metric_value=metric)
        else:
            row = summarize_case(request_path, metric_path)
        rows.append(row)
    seen_cases = {row.get("case") for row in rows}
    for metric_path in sorted(metric_dir.glob("decode_ctx*_b*.json")):
        case = metric_path.stem
        if case in seen_cases:
            continue
        value = load_json(metric_path)
        if value.get("status") == "capacity_skip":
            rows.append({
                "case": case,
                "context_tokens": value.get("context_tokens"),
                "requested_batch": value.get("requested_batch"),
                "resident_verified_batch": None,
                "status": "capacity_skip",
                "failure": value.get("reason", "capacity skip"),
                "logged_gpu_kv_capacity_tokens": value.get("logged_gpu_kv_capacity_tokens"),
                "conservative_usable_capacity_tokens": value.get("conservative_usable_capacity_tokens"),
            })
    for path in sorted(root.glob("**/UNSUPPORTED.json")):
        value = load_json(path)
        rows.append({
            "case": path.parent.name,
            "status": "capacity_skip",
            "requested_batch": value.get("requested_batch"),
            "resident_verified_batch": None,
            "failure": value.get("reason") or value.get("status"),
        })
    configuration = load_json(root / "configuration.json") if (root / "configuration.json").is_file() else None
    return {
        "configuration": configuration,
        "cases": rows,
        "global_preemption_evidence": preemption_evidence(root),
        "trace_annotation_histogram": trace_annotation_histogram(root),
        "driver_failure": load_json(root / "FAILED.json")
        if (root / "FAILED.json").is_file() else None,
    }


def write_outputs(summary, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    csv_path = output.with_suffix(".csv")
    fields = sorted({key for row in summary["cases"] for key in row})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary["cases"])
    return csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path,
                        help="case root containing requests/ and metrics/")
    parser.add_argument("--output", required=True, type=Path,
                        help="JSON output path; CSV is written beside it")
    args = parser.parse_args()
    summary = summarize(args.root)
    csv_path = write_outputs(summary, args.output)
    print(json.dumps({"json": str(args.output), "csv": str(csv_path),
                      "cases": len(summary["cases"]),
                      "preemption_evidence_count": summary["global_preemption_evidence"]["count"]}, indent=2))


if __name__ == "__main__":
    main()
