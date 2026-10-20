"""Summarize downloaded Marlin/FlashInfer and service A/B artifacts.

Usage::

    python summarize.py /path/to/results

The input directory is treated as an artifact root.  Missing artifacts are
reported and skipped; this script never invents measurements.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def _read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _files(root, directory, pattern):
    path = root / directory
    return sorted(path.glob(pattern)) if path.is_dir() else []


def _median(values):
    return statistics.median(values) if values else None


def _gain(old, new):
    if old in (None, 0) or new is None:
        return None
    return 100.0 * (old - new) / old


def _sweep_records(root):
    records = []
    paths = _files(root, "sweep", "*marlin.json")
    for path in paths:
        data = _read(path)
        if not isinstance(data, dict):
            continue
        for batch_result in data.get("results", []):
            batch = batch_result.get("batch")
            for row in batch_result.get("rows", []):
                record = dict(row)
                record.update({"path": str(path), "batch": batch,
                               "canonical_order": data.get("canonical_order"),
                               "capture": data.get("source")})
                records.append(record)
    return paths, records


def _fi_records(root):
    records = []
    paths = _files(root, "fi_fixed", "*flashinfer.json")
    for path in paths:
        data = _read(path)
        if isinstance(data, dict):
            records.append({"path": str(path), "data": data})
    return paths, records


def _profile_records(root, sweep_records):
    paths = []
    direct = root / "profile_l20.json"
    if direct.is_file():
        paths.append(direct)
    paths.extend(_files(root, "sweep", "*profile_l20.json"))
    profiles = []
    for path in paths:
        data = _read(path)
        if data is not None:
            profiles.append({"path": str(path), "data": data})
    # New sweep files embed profile blocks.  Keep them as independent records.
    for row in sweep_records:
        if row.get("profile"):
            profiles.append({"path": row["path"], "batch": row.get("batch"),
                             "block_size_m": row.get("block_size_m"),
                             "data": row["profile"]})
    return profiles


def _capture_histograms(root):
    paths = [root / "captures.json", root / "sweep" / "captures.json"]
    path = next((item for item in paths if item.is_file()), None)
    if path is None:
        return {}, []
    data = _read(path)
    chosen = data.get("chosen", []) if isinstance(data, dict) else []
    exact = {}
    basenames = {}
    for item in chosen:
        if not isinstance(item, dict) or not isinstance(item.get("expert_rows"), dict):
            continue
        source = str(item.get("path", ""))
        hist = {int(key): int(value) for key, value in item["expert_rows"].items()}
        exact[source] = hist
        basenames.setdefault(Path(source).name, []).append(hist)
    for name, values in basenames.items():
        if len(values) == 1:
            exact.setdefault(name, values[0])
    return exact, [str(path)]


def _padding_summary(root, records):
    histograms, paths = _capture_histograms(root)
    rows = []
    for record in records:
        source = str(record.get("capture", ""))
        histogram = histograms.get(source) or histograms.get(Path(source).name)
        tile = record.get("block_size_m")
        if histogram is None or tile is None:
            continue
        tile = int(tile)
        useful = sum(histogram.values())
        padded = sum(((count + tile - 1) // tile) * tile
                     for count in histogram.values())
        rows.append({"path": record.get("path"), "capture": source,
                     "batch": record.get("batch"), "block_size_m": tile,
                     "useful_assignments": useful, "padded_rows": padded,
                     "padding_rows": padded - useful,
                     "m_tiles": sum((count + tile - 1) // tile
                                    for count in histogram.values()),
                     "experts_with_assignments": len(histogram)})
    return rows, paths


def _profile_rows(profiles):
    rows = []
    for item in profiles:
        data = item["data"]
        candidates = []
        if isinstance(data, dict) and isinstance(data.get("results"), list):
            for batch_result in data["results"]:
                if not isinstance(batch_result, dict):
                    continue
                for row in batch_result.get("rows", []):
                    if isinstance(row, dict) and row.get("profile"):
                        candidates.append({"batch": batch_result.get("batch"),
                                           "block_size_m": row.get("block_size_m"),
                                           "profile": row["profile"]})
        elif isinstance(data, list):
            candidates.extend(data)
        else:
            candidates.append(data)
        for value in candidates:
            if not isinstance(value, dict):
                continue
            # Accept either the graph_measure fields or a profile_l20 wrapper.
            source = value.get("profile", value)
            if not isinstance(source, dict):
                continue
            local = source.get("local_assignments")
            gate = source.get("gate_up_us", source.get("gate_up"))
            down = source.get("down_us", source.get("down"))
            gmfu = source.get("gate_up_mfu_pct", source.get("gate_up_mfu"))
            dmfu = source.get("down_mfu_pct", source.get("down_mfu"))
            if all(value is None for value in (local, gate, down, gmfu, dmfu)):
                continue
            rows.append({
                "path": item["path"],
                "batch": value.get("batch", item.get("batch", source.get("batch"))),
                "block_size_m": value.get("block_size_m", item.get("block_size_m", source.get("block_size_m"))),
                "local_assignments": local,
                "gate_up_us": gate,
                "down_us": down,
                "gate_up_mfu_pct": gmfu,
                "down_mfu_pct": dmfu,
            })
    return rows


def _cold_summary(records):
    by_tile = {}
    for row in records:
        tile = row.get("block_size_m")
        value = row.get("cold_median_us")
        if tile is None or value is None:
            continue
        by_tile.setdefault(int(tile), []).append(float(value))
    aggregate = {}
    for tile, values in sorted(by_tile.items()):
        aggregate[str(tile)] = {
            "cases": len(values), "min_us": min(values),
            "median_us": _median(values), "values_us": values,
        }
    pairs = []
    for row in records:
        if row.get("block_size_m") != 8 or row.get("cold_median_us") is None:
            continue
        candidates = [other for other in records
                      if other.get("block_size_m") == 32
                      and other.get("batch") == row.get("batch")
                      and other.get("capture") == row.get("capture")]
        if candidates and candidates[0].get("cold_median_us") is not None:
            new = float(candidates[0]["cold_median_us"])
            old = float(row["cold_median_us"])
            pairs.append({"batch": row.get("batch"), "capture": row.get("capture"),
                          "m8_us": old, "m32_us": new,
                          "m32_gain_pct": _gain(old, new)})
    gains = [x["m32_gain_pct"] for x in pairs if x["m32_gain_pct"] is not None]
    return {"by_tile": aggregate, "m8_vs_m32_pairs": pairs,
            "m8_vs_m32": {"cases": len(gains), "min_gain_pct": min(gains) if gains else None,
                          "median_gain_pct": _median(gains), "gains_pct": gains}}


def _fp32_summary(records):
    out = []
    for row in records:
        error = row.get("output")
        if isinstance(error, dict):
            out.append({"path": row["path"], "batch": row.get("batch"),
                        "block_size_m": row.get("block_size_m"),
                        "relative_l2": error.get("rel_l2"),
                        "max_abs": error.get("max_abs"),
                        "finite": error.get("finite")})
    return out


def _fi_summary(records):
    rows = []
    for item in records:
        data = item["data"]
        timing = data.get("timing", {})
        ref = data.get("flashinfer_vs_fp32_reference", {})
        rows.append({"path": item["path"], "batch": data.get("batch"),
                     "batch_provenance": data.get("batch_provenance"),
                     "cold_median_ms": timing.get("cold_median_ms"),
                     "hot_median_ms": timing.get("median_ms"),
                     "cold_samples_ms": timing.get("cold_samples_ms", []),
                     "relative_l2": ref.get("rel_l2"), "max_abs": ref.get("max_abs"),
                     "finite": data.get("output_finite"),
                     "repeat_exact": timing.get("graph_repeat_exact"),
                     "local_experts": data.get("local_experts")})
    return rows


def _find_service(root, name):
    candidates = [root / name / "summary.json", root / f"{name}.json",
                  root / "service" / name / "summary.json"]
    return next(((p, _read(p)) for p in candidates if p.is_file()), (None, None))


def _service_summary(root):
    result = {}
    loaded = {}
    for name in ("service_baseline", "service_m32"):
        path, data = _find_service(root, name)
        if path is None:
            continue
        loaded[name] = (path, data)
        cases = data if isinstance(data, list) else data.get("cases", data.get("summary", []))
        if isinstance(cases, dict):
            cases = [cases]
        rows = []
        for case in cases if isinstance(cases, list) else []:
            path_stats = case.get("path", {})
            step = path_stats.get("stage0_submit_to_tokens_ready_ms", {})
            if not step:
                step = case.get("whole_batch_step", {})
            rows.append({"label": case.get("label"), "batch": case.get("batch"),
                         "p50_ms": step.get("p50_ms"), "p95_ms": step.get("p95_ms"),
                         "path": str(path)})
        result[name] = rows
    if len(loaded) == 2:
        base = {x.get("label"): x for x in result["service_baseline"]}
        tuned = {x.get("label"): x for x in result["service_m32"]}
        result["paired_tpot"] = [
            {"label": label, "batch": tuned[label].get("batch"),
             "baseline_p50_ms": base[label].get("p50_ms"),
             "m32_p50_ms": tuned[label].get("p50_ms"),
             "gain_pct": _gain(base[label].get("p50_ms"), tuned[label].get("p50_ms"))}
            for label in sorted(base.keys() & tuned.keys())
        ]
    return result


def _output_identity(root):
    found = {}
    for name in ("service_baseline", "service_m32"):
        candidates = [root / name / "output_ids.json", root / f"{name}_output_ids.json",
                      root / "service" / name / "output_ids.json"]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            continue
        data = _read(path)
        if isinstance(data, dict):
            found[name] = {str(k): v for k, v in data.items()}
    if len(found) != 2:
        return {"available": False, "reason": "both service output_ids.json files are required"}
    labels = sorted(found["service_baseline"].keys() & found["service_m32"].keys())
    rows = []
    for label in labels:
        left, right = found["service_baseline"][label], found["service_m32"][label]
        rows.append({"label": label, "requests": min(len(left), len(right)),
                     "exact": left == right,
                     "matching_requests": sum(a == b for a, b in zip(left, right))})
    return {"available": True, "labels": rows,
            "exact_labels": sum(row["exact"] for row in rows),
            "labels_compared": len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=None,
                        help="summary JSON path; defaults to ROOT/summary.json")
    args = parser.parse_args()
    root = args.root
    sweep_paths, sweep = _sweep_records(root)
    fi_paths, fi = _fi_records(root)
    profiles = _profile_records(root, sweep)
    padding, capture_paths = _padding_summary(root, sweep)
    summary = {
        "schema": "moe_tuning_summary_v1",
        "root": str(root),
        "artifacts": {"sweep": [str(p) for p in sweep_paths],
                       "fi_fixed": [str(p) for p in fi_paths],
                       "profiles": [item["path"] for item in profiles],
                       "captures": capture_paths},
        "marlin": {"cold_timing": _cold_summary(sweep),
                   "fp32_errors": _fp32_summary(sweep),
                   "padding": padding,
                   "records": len(sweep)},
        "flashinfer": {"records": _fi_summary(fi)},
        "profile_l20": _profile_rows(profiles),
        "service": _service_summary(root),
        "output_identity": _output_identity(root),
        "skipped": [],
    }
    expected = {"sweep": sweep_paths, "fi_fixed": fi_paths}
    for name, paths in expected.items():
        if not paths:
            summary["skipped"].append(f"no {name}/*.json artifacts found")
    if not profiles:
        summary["skipped"].append("no profile_l20.json or embedded profiles found")
    if not summary["service"]:
        summary["skipped"].append("no service_baseline/service_m32 summaries found")
    if not summary["output_identity"].get("available"):
        summary["skipped"].append("service output identity comparison unavailable")

    output = args.output or root / "summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report = ["# MoE Tuning Summary", "", f"Artifact root: `{root}`", ""]
    report.append(f"Marlin sweep records: {len(sweep)}; FlashInfer records: {len(fi)}; profiles: {len(summary['profile_l20'])}.")
    cold = summary["marlin"]["cold_timing"]
    report.append("")
    report.append("## Cold Marlin Timing")
    report.append("")
    for tile, values in cold["by_tile"].items():
        report.append(f"- M{tile}: {values['cases']} cases, min {values['min_us']:.3f} us, median {values['median_us']:.3f} us")
    gain = cold["m8_vs_m32"]
    report.append(f"- M8 to M32 paired gain: {gain['cases']} cases; min {gain['min_gain_pct']!s}%, median {gain['median_gain_pct']!s}%")
    report.extend(["", "## Correctness", ""])
    errors = summary["marlin"]["fp32_errors"]
    report.append(f"Marlin FP32-reference rows: {len(errors)}; finite rows: {sum(x.get('finite') is True for x in errors)}.")
    report.append(f"Profile rows with local-assignment-aware MFU: {len(summary['profile_l20'])}.")
    report.extend(["", "## Service", ""])
    report.append(json.dumps(summary["service"], sort_keys=True))
    identity = summary["output_identity"]
    report.append(f"Output identity: {identity.get('exact_labels', 0)}/{identity.get('labels_compared', 0)} labels exact.")
    if summary["skipped"]:
        report.extend(["", "## Skipped", ""])
        report.extend(f"- {item}" for item in summary["skipped"])
    report_path = output.with_name("REPORT.md")
    report_path.write_text("\n".join(report) + "\n")
    print(json.dumps({"summary": str(output), "report": str(report_path),
                      "marlin_records": len(sweep), "flashinfer_records": len(fi)}))


if __name__ == "__main__":
    main()
