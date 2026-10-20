"""Summarize baseline/candidate fixed-cohort fast-GEMM trial artifacts."""

import argparse
import csv
import json
from pathlib import Path
import statistics


BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
DEFAULT_CASES = 27
METRICS = (
    ("stage0_submit_to_tokens_ready_ms", "Stage 0 submit to tokens ready (ms)"),
    ("scheduler_step_ms", "Scheduler step (ms)"),
    ("stage0_gpu_max_ms", "Stage 0 GPU max (ms)"),
    ("stage1_gpu_max_ms", "Stage 1 GPU max (ms)"),
    ("pooled_itl_p50_ms", "Client pooled ITL p50 (ms)"),
    ("pooled_itl_p95_ms", "Client pooled ITL p95 (ms)"),
)


def _load_trial(path: Path, expected_cases: int, allow_incomplete: bool) -> list[dict]:
    complete = path / "COMPLETE.json"
    summary = path / "summary.json"
    if not complete.exists() and not allow_incomplete:
        raise RuntimeError(f"Missing COMPLETE.json: {path}")
    if not summary.exists():
        raise RuntimeError(f"Missing summary.json: {path}")
    cases = json.loads(summary.read_text())
    if not isinstance(cases, list):
        raise RuntimeError(f"summary.json is not a case list: {path}")
    if len(cases) != expected_cases and not allow_incomplete:
        raise RuntimeError(f"Expected {expected_cases} cases in {path}, got {len(cases)}")
    return cases


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _aggregate(cases: list[dict]) -> list[dict]:
    grouped: dict[int, list[dict]] = {}
    for case in cases:
        grouped.setdefault(int(case["batch"]), []).append(case)
    rows = []
    for batch in sorted(grouped):
        entries = grouped[batch]
        row = {"batch": batch, "cases": len(entries)}
        for key, _ in METRICS:
            values = []
            for entry in entries:
                value = entry.get(key)
                if isinstance(value, (int, float)):
                    values.append(float(value))
                elif key in entry.get("path", {}):
                    value = entry["path"][key].get("p50_ms")
                    if isinstance(value, (int, float)):
                        values.append(float(value))
            row[f"{key}_median"] = statistics.median(values) if values else None
            row[f"{key}_min"] = min(values) if values else None
            row[f"{key}_max"] = max(values) if values else None
        rows.append(row)
    return rows


def _raw_rows(label: str, cases: list[dict]) -> list[dict]:
    rows = []
    for case in cases:
        row = {"trial": label, "label": case.get("label"),
               "batch": case.get("batch"), "repeat": case.get("repeat")}
        for key, _ in METRICS:
            value = case.get(key)
            if value is None:
                value = case.get("path", {}).get(key, {}).get("p50_ms")
            row[key] = value
        row["whole_batch_step_p50_ms"] = case.get("whole_batch_step", {}).get("p50_ms")
        row["whole_batch_step_p95_ms"] = case.get("whole_batch_step", {}).get("p95_ms")
        row["whole_batch_output_period_p50_ms"] = case.get(
            "whole_batch_output_period", {}).get("p50_ms")
        row["whole_batch_output_period_p95_ms"] = case.get(
            "whole_batch_output_period", {}).get("p95_ms")
        rows.append(row)
    return rows


def _token_ids(path: Path, expected_batch: int | None = None) -> list[list[int]]:
    payload = json.loads(path.read_text())
    output = {}
    records = payload.get("records", [])
    if expected_batch is not None and len(records) != expected_batch:
        raise RuntimeError(f"Expected {expected_batch} records in {path}, got {len(records)}")
    for index, record in enumerate(records):
        output[index] = [
            token for event in record.get("events", [])
            for token in event.get("token_ids", [])
        ]
        if len(output[index]) != 96:
            raise RuntimeError(
                f"Expected 96 output tokens in record {index} of {path}, "
                f"got {len(output[index])}"
            )
    return [output[index] for index in range(len(records))]


def _repeat_diagnostics(path: Path) -> dict:
    request_dir = path / "requests"
    by_batch: dict[int, list[list[list[int]]]] = {}
    for request_path in sorted(request_dir.glob("b*_r*.json")):
        try:
            batch = int(request_path.name.split("_", 1)[0][1:])
        except (ValueError, IndexError):
            continue
        by_batch.setdefault(batch, []).append(_token_ids(request_path, batch))
    exact = prefix = total = requests = 0
    batches = 0
    for batch, repeats in by_batch.items():
        if len(repeats) < 2:
            continue
        batches += 1
        first = repeats[0]
        for other in repeats[1:]:
            for left, right in zip(first, other):
                total += len(right)
                common = 0
                for a, b in zip(left, right):
                    if a != b:
                        break
                    common += 1
                prefix += common
                exact += int(left == right)
                requests += 1
    return dict(batches_compared=batches, exact_requests=exact,
                comparable_requests=requests, compared_tokens=total, prefix_tokens=prefix,
                prefix_agreement=(prefix / total if total else None))


def _cross_trial_diagnostics(baseline: Path, candidate: Path) -> dict:
    exact = prefix = total = requests = 0
    for batch in BATCHES:
        base_path = baseline / "requests" / f"b{batch}_r1.json"
        cand_path = candidate / "requests" / f"b{batch}_r1.json"
        if not base_path.exists() or not cand_path.exists():
            continue
        left, right = _token_ids(base_path, batch), _token_ids(cand_path, batch)
        for first, second in zip(left, right):
            requests += 1
            total += len(second)
            common = 0
            for a, b in zip(first, second):
                if a != b:
                    break
                common += 1
            prefix += common
            exact += int(first == second)
    return dict(exact_requests=exact, comparable_requests=requests,
                prefix_tokens=prefix, total_tokens=total,
                prefix_agreement=(prefix / total if total else None))


def _whole_batch_checks(path: Path, cases: list[dict]) -> dict:
    valid = 0
    graph_full = 0
    ranks = {str(rank): [] for rank in range(8)}
    for case in cases:
        verification = case.get("dispatch_verification", {})
        if (verification.get("graph_mode") == "FULL"
                and verification.get("actual_tokens") == case.get("batch")
                and verification.get("padded_tokens") == case.get("batch")):
            valid += 1
        if case.get("path"):
            graph_full += 1
    timing_dir = path / "rank_timings"
    for rank in ranks:
        files = sorted(timing_dir.glob(f"rank_{rank}_*.jsonl"))
        values = []
        for timing_path in files:
            for line in timing_path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not row.get("prefill") and row.get("graph_mode") == "FULL":
                    value = row.get("gpu_ms")
                    if isinstance(value, (int, float)):
                        values.append(float(value))
        ranks[rank] = dict(samples=len(values), gpu_ms_p95=_percentile(values, .95))
    return dict(scope="all non-prefill FULL-Graph rank records; mixed batches and warmups",
                cases_with_full_graph_dispatch=valid,
                cases_with_path_summary=graph_full, expected_cases=len(cases),
                ranks=ranks)


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _markdown(baseline_rows: list[dict], candidate_rows: list[dict],
              baseline_name: str, candidate_name: str, diagnostics: dict, mode: str) -> str:
    lines = ["# Fast GEMM Trial Summary", "",
             f"Baseline: `{baseline_name}`; candidate: `{candidate_name}`.", "",
             ("Candidate uses BF16 activation for Marlin MXFP4 W4A16; baseline uses FP8 activation. "
              if mode == 'all' else "Candidate retains the original MoE backend and activation precision. ") +
             "Token differences are diagnostics on one repeated prompt, not a model quality evaluation.", ""]
    lines += ["## Aggregate latency", "",
              "Each value is the median of the three repeat p50 values; the range is shown in parentheses.", "",
              "| Batch | Stage0 submit->ready baseline | Stage0 submit->ready candidate | Reduction | Speedup | Candidate client ITL p50 |",
              "|---:|---:|---:|---:|---:|---:|"]
    base_by_batch = {row["batch"]: row for row in baseline_rows}
    cand_by_batch = {row["batch"]: row for row in candidate_rows}
    for batch in sorted(set(base_by_batch) | set(cand_by_batch)):
        base = base_by_batch.get(batch, {})
        cand = cand_by_batch.get(batch, {})
        b = base.get("stage0_submit_to_tokens_ready_ms_median")
        c = cand.get("stage0_submit_to_tokens_ready_ms_median")
        reduction = ((b - c) / b * 100) if b and c is not None else None
        speedup = (b / c) if b and c else None
        brange = f"({_fmt(base.get('stage0_submit_to_tokens_ready_ms_min'))}-{_fmt(base.get('stage0_submit_to_tokens_ready_ms_max'))})"
        crange = f"({_fmt(cand.get('stage0_submit_to_tokens_ready_ms_min'))}-{_fmt(cand.get('stage0_submit_to_tokens_ready_ms_max'))})"
        lines.append(f"| {batch} | {_fmt(b)} {brange} | {_fmt(c)} {crange} | "
                     f"{_fmt(reduction)}% | {_fmt(speedup)}x | "
                     f"{_fmt(cand.get('pooled_itl_p50_ms_median'))} |")
    lines += ["", "## Diagnostics", "",
              f"Baseline repeat determinism: `{diagnostics['baseline_repeat']}`.",
              f"Candidate repeat determinism: `{diagnostics['candidate_repeat']}`.",
              f"Cross-trial request comparison: `{diagnostics['cross_trial']}`.", "",
              "## Whole-batch checks", "",
              f"Baseline: `{diagnostics['baseline_checks']['cases_with_full_graph_dispatch']}` / "
              f"`{diagnostics['baseline_checks']['expected_cases']}` cases with FULL Graph dispatch.",
              f"Candidate: `{diagnostics['candidate_checks']['cases_with_full_graph_dispatch']}` / "
              f"`{diagnostics['candidate_checks']['expected_cases']}` cases with FULL Graph dispatch.", "",
              "| Rank | Baseline GPU p95 (ms) | Candidate GPU p95 (ms) |",
              "|---:|---:|---:|"]
    for rank in map(str, range(8)):
        base_rank = diagnostics["baseline_checks"]["ranks"].get(rank, {})
        cand_rank = diagnostics["candidate_checks"]["ranks"].get(rank, {})
        lines.append(f"| {rank} | {_fmt(base_rank.get('gpu_ms_p95'))} | "
                     f"{_fmt(cand_rank.get('gpu_ms_p95'))} |")
    lines += ["",
              "Rank p95 is an aggregate mixed-batch diagnostic over all non-prefill FULL-Graph records, including warmups; it is not a per-case latency result.",
              "Raw repeat data and the complete diagnostic scope are in `summary.json` and `raw_data.csv`."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--outputdir", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=DEFAULT_CASES)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    baseline_cases = _load_trial(args.baseline, args.expected_cases, args.allow_incomplete)
    candidate_cases = _load_trial(args.candidate, args.expected_cases, args.allow_incomplete)
    mode = json.loads((args.candidate / 'configuration.json').read_text())['mode']
    args.outputdir.mkdir(parents=True, exist_ok=True)
    baseline_rows, candidate_rows = _aggregate(baseline_cases), _aggregate(candidate_cases)
    diagnostics = dict(
        baseline_repeat=_repeat_diagnostics(args.baseline),
        candidate_repeat=_repeat_diagnostics(args.candidate),
        cross_trial=_cross_trial_diagnostics(args.baseline, args.candidate),
        baseline_checks=_whole_batch_checks(args.baseline, baseline_cases),
        candidate_checks=_whole_batch_checks(args.candidate, candidate_cases),
    )
    raw = _raw_rows("baseline", baseline_cases) + _raw_rows("candidate", candidate_cases)
    fields = sorted({key for row in raw for key in row})
    with (args.outputdir / "raw_data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(raw)
    report = dict(mode=mode, baseline=_aggregate(baseline_cases), candidate=_aggregate(candidate_cases),
                  diagnostics=diagnostics, expected_cases=args.expected_cases)
    (args.outputdir / "summary.json").write_text(json.dumps(report, indent=2))
    (args.outputdir / "REPORT.md").write_text(_markdown(
        baseline_rows, candidate_rows, args.baseline.name, args.candidate.name, diagnostics, mode))


if __name__ == "__main__":
    main()
