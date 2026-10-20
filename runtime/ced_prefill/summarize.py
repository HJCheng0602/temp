"""Summarize a CED coordinator result directory without importing torch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any


BASELINE_TTFT_MS = {
    2048: 248.120,
    4096: 453.526,
    8192: 860.412,
    16384: 1303.555,
    32768: 2207.657,
    65536: 4151.031,
    131072: 8665.498,
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def values(rows: list[dict], key: str) -> list[float]:
    return [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]


def stats(rows: list[dict], key: str) -> dict[str, float] | None:
    data = values(rows, key)
    if not data:
        return None
    return {"median_ms": statistics.median(data), "min_ms": min(data), "max_ms": max(data)}


def validation(warmup: dict) -> dict:
    references = warmup.get("reference") or []
    reference_rows = [reply.get("result", {}) for reply in references
                      if isinstance(reply, dict) and reply.get("status") == "ok"]
    allclose = [row.get("allclose") for row in reference_rows]
    argmax = [row.get("argmax_equal") for row in reference_rows]
    continuation_equal = [row.get("continuation_equal") for row in reference_rows]
    sampled = []
    for row in warmup.get("decode_workers") or []:
        sampled.append(row.get("result", {}).get("token_ids", []))
    head_checks = []
    for row in warmup.get("decode_workers") or []:
        if row.get("head_check") is not None:
            head_checks.append(row["head_check"])
    return dict(
        reference_ranks=len(reference_rows),
        reference_allclose=allclose,
        reference_argmax_equal=argmax,
        reference_continuation_equal=continuation_equal,
        sampled_token_ids_by_rank=sampled,
        sampled_tokens_nonempty=len(sampled) == 4 and all(valid_sample(ids) for ids in sampled),
        head_checks=head_checks,
        approximation_to_full=[row.get("approximation_to_full") for row in reference_rows],
    )


def valid_sample(ids):
    return (isinstance(ids, list) and len(ids) == 1
            and isinstance(ids[0], list) and len(ids[0]) == 1
            and isinstance(ids[0][0], int) and ids[0][0] >= 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    required = ["configuration.json", "manifests.json", "warmup.json", "records.json", "COMPLETE.json"]
    missing = [name for name in required if not (root / name).exists()]
    failed = (root / "FAILED.json").exists()
    config = read_json(root / "configuration.json") if (root / "configuration.json").exists() else {}
    manifests = read_json(root / "manifests.json") if (root / "manifests.json").exists() else {}
    warmup = read_json(root / "warmup.json") if (root / "warmup.json").exists() else {}
    records = read_json(root / "records.json") if (root / "records.json").exists() else []
    if not isinstance(records, list):
        records = []

    by_length: dict[str, dict] = {}
    for length in sorted({int(row["length"]) for row in records if "length" in row}):
        rows = [row for row in records if int(row.get("length", -1)) == length]
        entry = {"repeats": len(rows)}
        for key in ("ttft_ms", "prefill_ms", "export_ms", "restore_ms", "replay_and_sample_ms"):
            entry[key] = stats(rows, key)
        export_bytes = [sum(item.get("bytes", 0) for item in row.get("exports", [])
                            if isinstance(item, dict) and isinstance(item.get("bytes"), (int, float)))
                        for row in rows]
        entry["aggregate_export_bytes"] = (dict(median=statistics.median(export_bytes), min=min(export_bytes),
                                                max=max(export_bytes)) if export_bytes else None)
        entry["baseline_http_ttft_ms"] = BASELINE_TTFT_MS.get(length)
        by_length[str(length)] = entry

    validation_result = validation(warmup)
    configured_sizes = [int(size) for size in config.get("sizes", [])]
    configured_repeats = int(config.get("repeats", 0) or 0)
    observed_counts = {size: sum(1 for row in records if int(row.get("length", -1)) == size)
                       for size in configured_sizes}
    shape_complete = bool(configured_sizes and configured_repeats and
                          all(observed_counts[size] == configured_repeats
                              for size in configured_sizes))
    reference_complete = (
        validation_result["reference_ranks"] == 4
        and all(validation_result["reference_allclose"])
        and all(validation_result["reference_argmax_equal"])
        and all(validation_result["reference_continuation_equal"])
        and validation_result["sampled_tokens_nonempty"]
    )
    samples_complete = bool(records) and all(
        len(row.get("token_ids_by_rank", [])) == 4
        and all(valid_sample(ids) for ids in row["token_ids_by_rank"])
        for row in records
    )
    reasons = missing + (["FAILED.json present"] if failed else [])
    if not shape_complete:
        reasons.append("records do not contain every configured length/repeat")
    if not reference_complete:
        reasons.append("four-rank reference or sampled-token validation is incomplete")
    if not samples_complete:
        reasons.append("measured requests have incomplete sampled tokens")
    complete = not reasons
    summary = dict(
        passed=complete,
        incomplete_reasons=reasons,
        configuration=config,
        manifests=manifests,
        validation=validation_result,
        lengths=by_length,
        caveats=[
            "CED TTFT starts at the coordinator P/D case boundary and includes export, restore, and replay/sample.",
            "The supplied baseline is HTTP request-start to first streamed token; it is not an apples-to-apples comparison.",
            "Decode CPU offload is configurable and must be read from configuration.json.",
        ],
    )
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = ["# CED Prefill Summary", "", f"Status: {'complete' if complete else 'incomplete'}", ""]
    if reasons:
        lines.append("Incomplete evidence: " + ", ".join(reasons) + ".")
        lines.append("")
    lines += ["| Prompt | Repeats | TTFT median/min/max ms | P median ms | Export median ms | Restore median ms | Replay/sample median ms | Aggregate export bytes median | Baseline HTTP TTFT ms |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for length, entry in by_length.items():
        def fmt(item):
            return "n/a" if not item else f"{item['median_ms']:.3f}/{item['min_ms']:.3f}/{item['max_ms']:.3f}"
        def med(item):
            return "n/a" if not item else f"{item['median_ms']:.3f}"
        lines.append(f"| {length} | {entry['repeats']} | {fmt(entry['ttft_ms'])} | {med(entry['prefill_ms'])} | {med(entry['export_ms'])} | {med(entry['restore_ms'])} | {med(entry['replay_and_sample_ms'])} | {(entry['aggregate_export_bytes'] or {}).get('median', 'n/a')} | {entry['baseline_http_ttft_ms'] or 'n/a'} |")
    lines += ["", "Validation:", "", f"- Reference allclose: `{validation_result['reference_allclose']}`; argmax equal: `{validation_result['reference_argmax_equal']}`; continuation equal: `{validation_result['reference_continuation_equal']}`.", f"- Sampled token IDs nonempty on all recorded ranks: `{validation_result['sampled_tokens_nonempty']}`.", f"- Bounded versus full diagnostic: `{validation_result['approximation_to_full']}` (diagnostic only; not a pass criterion).", f"- Source-head checks recorded separately: `{len(validation_result['head_checks'])}`.", "", "CED TTFT includes the P/D handoff and bounded replay. The HTTP baseline uses a different measurement boundary, and CPU decode offload is configurable; no speedup claim is made from this table."]
    (root / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
