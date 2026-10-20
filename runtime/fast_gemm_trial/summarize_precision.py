"""Summarize bounded decode precision artifacts without declaring quality pass/fail."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _read(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def _first(root: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        path = root / name
        if path.exists():
            return path
    return None


def _completed_dir(root: Path, mode: str, required: bool = False) -> Path | None:
    """Find exactly one completed artifact whose recorded mode matches."""
    candidates = []
    directories = [root] + sorted(path for path in root.iterdir() if path.is_dir())
    for directory in directories:
        configuration = _read(directory / "configuration.json")
        complete = _read(directory / "COMPLETE.json")
        if configuration and configuration.get("mode") == mode and complete and complete.get("completed") is True:
            candidates.append(directory)
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous completed {mode} artifacts: {[str(path) for path in candidates]}")
    if not candidates and required:
        raise FileNotFoundError(f"No completed {mode} artifact below {root}")
    return candidates[0] if candidates else None


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"samples": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    ordered = sorted(values)

    def quantile(q: float) -> float:
        index = min(len(ordered) - 1, int(q * (len(ordered) - 1)))
        return float(ordered[index])

    return {
        "samples": len(values),
        "mean": float(sum(values) / len(values)),
        "p50": quantile(0.50),
        "p95": quantile(0.95),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
    }


def _workload_index(workload: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    requests: dict[str, dict[str, Any]] = {}
    cases: dict[str, dict[str, Any]] = {}
    for case in workload.get("cases", []):
        label = str(case["label"])
        if label in cases or not case.get("requests") or int(case["batch"]) != len(case["requests"]):
            raise ValueError(f"Invalid or duplicate workload case metadata: {label}")
        cases[label] = {
            "label": label,
            "batch": int(case["batch"]),
            "prompt_tokens": int(case["requests"][0]["target_prompt_tokens"]),
            "output_tokens": int(case["output_tokens"]),
            "request_keys": [str(request["key"]) for request in case.get("requests", [])],
        }
        for request in case.get("requests", []):
            key = str(request["key"])
            if key in requests:
                raise ValueError(f"Duplicate workload request key: {key}")
            requests[key] = {"case": label, **cases[label]}
    return requests, cases


def _case_rows(cases: dict[str, dict[str, Any]], rows: list[dict[str, Any]], key_field: str = "key") -> dict[str, list[dict[str, Any]]]:
    output = {label: [] for label in cases}
    known_keys = {key for case in cases.values() for key in case["request_keys"]}
    for row in rows:
        key = row.get(key_field)
        if key not in known_keys:
            raise ValueError(f"Unknown workload request key in artifact: {key!r}")
        case = next((label for label, data in cases.items() if key in data["request_keys"]), None)
        if case is not None:
            output[case].append(row)
    return output


def _free_summary(data: dict[str, Any], requests: dict[str, dict[str, Any]], cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = data.get("rows", [])
    per_case = []
    grouped = _case_rows(cases, rows)
    for label, case_rows in grouped.items():
        per_case.append({
            "label": label,
            "batch": cases[label]["batch"],
            "prompt_tokens": cases[label]["prompt_tokens"],
            "output_tokens": cases[label]["output_tokens"],
            "requests": len(case_rows),
            "exact_requests": sum(bool(row.get("exact")) for row in case_rows),
            "tokens": sum(int(row.get("tokens", 0)) for row in case_rows),
            "positional_equal": sum(int(row.get("positional_equal", 0)) for row in case_rows),
            "common_prefix_tokens": sum(int(row.get("common_prefix", 0)) for row in case_rows),
        })
    return {
        "requests": int(data.get("requests", len(rows))),
        "exact_requests": int(data.get("exact_requests", 0)),
        "tokens": int(data.get("tokens", 0)),
        "positional_equal": int(data.get("positional_equal", 0)),
        "common_prefix_tokens": int(data.get("common_prefix_tokens", 0)),
        "rows": per_case,
    }


def _logit_summary(data: dict[str, Any], cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    samples = data.get("samples", [])
    grouped = _case_rows(cases, samples)
    differences = data.get("raw_argmax_differences", [])
    differences_by_key: dict[str, int] = {}
    for difference in differences:
        key = str(difference.get("key", ""))
        differences_by_key[key] = differences_by_key.get(key, 0) + 1
    fields = ("max_abs", "relative_l2", "centered_rmse", "cosine", "kl", "top10_overlap")
    per_case = []
    for label, case_samples in grouped.items():
        per_case.append({
            "label": label,
            "batch": cases[label]["batch"],
            "prompt_tokens": cases[label]["prompt_tokens"],
            "output_tokens": cases[label]["output_tokens"],
            "full_vocab_samples": len(case_samples),
            "all_decode_rows": sum(cases[label]["output_tokens"] - 1 for _ in cases[label]["request_keys"]),
            "all_row_argmax_differences": sum(differences_by_key.get(key, 0) for key in cases[label]["request_keys"]),
            "all_row_raw_argmax_agreement": (
                1.0 - sum(differences_by_key.get(key, 0) for key in cases[label]["request_keys"])
                / sum(cases[label]["output_tokens"] - 1 for _ in cases[label]["request_keys"])
                if cases[label]["request_keys"] and cases[label]["output_tokens"] > 1 else None
            ),
            "distributions": {field: _distribution([float(row[field]) for row in case_samples if field in row]) for field in fields},
        })
    return {
        "full_vocab_samples": int(data.get("full_vocab_samples", len(samples))),
        "all_decode_rows": int(data.get("all_decode_rows", 0)),
        "raw_argmax_agreement": (
            float((data.get("raw_argmax_equal", data.get("all_decode_rows", 0) - len(differences))) / data["all_decode_rows"])
            if data.get("all_decode_rows") else None
        ),
        "raw_argmax_differences": len(data.get("raw_argmax_differences", [])),
        "selected_argmax_agreement": data.get("selected_argmax_agreement"),
        "target_logprob_delta": {
            "mean": data.get("mean_baseline_target_logprob_delta"),
            "max_abs": data.get("max_abs_baseline_target_logprob_delta"),
        },
        "distributions": {
            field: _distribution([float(row[field]) for row in samples if field in row])
            for field in fields
        },
        "rows": per_case,
    }


def _artifact(directory: Path, filename: str) -> dict[str, Any] | None:
    path = directory / filename
    return _read(path) if path.exists() else None


def _validate_free(data: dict[str, Any], requests: dict[str, dict[str, Any]]) -> None:
    keys = [str(row.get("key")) for row in data.get("rows", [])]
    expected = set(requests)
    actual = set(keys)
    if actual != expected or len(keys) != len(actual):
        raise ValueError(
            f"free comparison request coverage differs: missing={sorted(expected - actual)[:5]}, "
            f"extra={sorted(actual - expected)[:5]}, duplicate_rows={len(keys) - len(actual)}"
        )


def _validate_logit_keys(data: dict[str, Any], requests: dict[str, dict[str, Any]]) -> None:
    known = set(requests)
    keys = [str(row.get("key")) for row in data.get("samples", [])]
    keys.extend(str(row.get("key")) for row in data.get("raw_argmax_differences", []))
    unknown = sorted(set(keys) - known)
    if unknown:
        raise ValueError(f"Unknown workload request keys in logits artifact: {unknown[:5]}")


def summarize(root: Path) -> dict[str, Any]:
    baseline_dir = _completed_dir(root, "baseline", required=True)
    assert baseline_dir is not None
    workload_path = _first(baseline_dir, ("workload.json",)) or _first(root, ("workload.json",))
    if workload_path is None:
        raise FileNotFoundError("workload.json not found below artifact root")
    requests, cases = _workload_index(_read(workload_path) or {})
    result: dict[str, Any] = {
        "artifact_root": str(root),
        "workload_cases": len(cases),
        "workload_requests": len(requests),
        "modes": {},
        "notes": [
            "Observed distributions only; no pass thresholds or model-quality guarantees are manufactured.",
            "Forced trace logits measure numerical agreement under identical decode prefixes.",
        ],
    }
    baseline = _artifact(baseline_dir, "repeat_logits.json")
    if baseline is None:
        raise FileNotFoundError(f"Completed baseline artifact has no repeat_logits.json: {baseline_dir}")
    _validate_logit_keys(baseline, requests)
    result["baseline_repeat"] = _logit_summary(baseline, cases)
    for mode in ("dense", "all"):
        mode_dir = _completed_dir(root, mode, required=True)
        assert mode_dir is not None
        free = _artifact(mode_dir, "free_comparison.json")
        logits = _artifact(mode_dir, "logits_comparison.json")
        if free is None or logits is None:
            raise FileNotFoundError(f"Completed {mode} artifact must contain both comparison outputs: {mode_dir}")
        entry: dict[str, Any] = {}
        if free is not None:
            _validate_free(free, requests)
            entry["free_output_equivalence"] = _free_summary(free, requests, cases)
        if logits is not None:
            _validate_logit_keys(logits, requests)
            entry["forced_numeric_comparison"] = _logit_summary(logits, cases)
        result["modes"][mode] = entry
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.artifact_root / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summarize(args.artifact_root), indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
