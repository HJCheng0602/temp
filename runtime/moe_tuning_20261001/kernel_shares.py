"""Build stage kernel-share and dependency-path reports for one profile run."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path


def _load_gemm(root: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "marlin_ipc_b64_20261001" / "gemm_mfu.py"
    spec = importlib.util.spec_from_file_location("dsv41_gemm_mfu", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load GEMM analyzer: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RESULT = root
    module.main()


def _stage_rows(path: Path) -> tuple[list[dict], dict[int, float]]:
    rows = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            row["stage"] = int(row["stage"])
            row["raw_kernel_duration_us_per_rank_step"] = float(
                row["raw_kernel_duration_us_per_rank_step"]
            )
            rows.append(row)
    totals: dict[int, float] = {}
    for row in rows:
        stage = row["stage"]
        totals[stage] = totals.get(stage, 0.0) + row["raw_kernel_duration_us_per_rank_step"]
    for row in rows:
        total = totals[row["stage"]]
        value = row["raw_kernel_duration_us_per_rank_step"]
        row["single_gpu_stage_share_percent"] = 100.0 * value / total if total else None
    return rows, totals


def _load_analysis(root: Path, existing: Path | None) -> dict:
    path = existing or (root / "analysis" / "analysis.json")
    if not path.is_file():
        raise FileNotFoundError(f"missing analysis JSON: {path}")
    return json.loads(path.read_text())


def _write(root: Path, analysis: dict, stage_rows: list[dict], totals: dict[int, float],
           skipped_gemm: bool) -> None:
    output = root / "analysis"
    path_data = analysis.get("category_attributed_us_per_step", {})
    path_percent = analysis.get("category_attribution_percent", {})
    top10 = analysis.get("top30", [])[:10]
    payload = {
        "schema": "dsv41_kernel_shares_v1",
        "root": str(root),
        "batch": analysis.get("batch"),
        "stages": [
            {"stage": stage, "single_gpu_graph_kernel_us_per_step": total,
             "scope": "mean over four ranks; sum of graph kernel lifetimes; may overlap; not TPOT"}
            for stage, total in sorted(totals.items())
        ],
        "stage_kernel_rows": stage_rows,
        "dependency_path_categories": [
            {"category": key, "attributed_us_per_step": value,
             "percent_of_dependency_span": path_percent.get(key)}
            for key, value in sorted(path_data.items(), key=lambda item: item[1], reverse=True)
        ],
        "dependency_path_top10": top10,
        "gemm_analysis": None if skipped_gemm else str(output / "gemm_mfu.json"),
        "limitations": [
            "Stage kernel totals are summed GPU kernel lifetimes on one average GPU and can overlap.",
            "They are not wall-clock TPOT and must not be derived by dividing eight-GPU sums by TPOT.",
            "Dependency-path percentages use the observed sample-ready backtrace and are a separate denominator.",
        ],
    }
    (output / "kernel_shares.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Kernel Shares",
        "",
        f"Profile root: `{root}`. Batch: `{analysis.get('batch', 'unknown')}`.",
        "",
        "Stage values are mean raw graph-kernel lifetimes per GPU per step, averaged over the four ranks in that PP stage. "
        "The category sum is a lifetime denominator; kernels may overlap, so it is not wall time, TPOT, or an eight-GPU sum.",
        "",
        "## Stage Graph Kernel Shares",
        "",
        "| Stage | Category | us / GPU / step | Share of stage lifetime |",
        "|---:|---|---:|---:|",
    ]
    for row in sorted(stage_rows, key=lambda value: (value["stage"], -value["raw_kernel_duration_us_per_rank_step"])):
        lines.append(
            f"| {row['stage']} | {row['category']} | "
            f"{row['raw_kernel_duration_us_per_rank_step']:.3f} | "
            f"{row['single_gpu_stage_share_percent']:.3f}% |"
        )
    lines += ["", "## Observed Dependency Path", "",
              "These percentages use the observed dependency path from stage-0 graph activity to stage-1 sample readiness. "
              "They are not the same denominator as stage kernel lifetime shares.", "",
              "| Category | Attributed us / step | Percent of dependency span |", "|---|---:|---:|"]
    for row in payload["dependency_path_categories"]:
        lines.append(f"| {row['category']} | {row['attributed_us_per_step']:.3f} | {row['percent_of_dependency_span']:.3f}% |")
    lines += ["", "## Dependency Top 10", "",
              "| # | Kernel | Category | Consumer us / step | Producer us / step |",
              "|---:|---|---|---:|---:|"]
    for index, row in enumerate(top10, 1):
        lines.append(
            f"| {index} | `{row.get('kernel', '')}` | {row.get('category', '')} | "
            f"{float(row.get('attributed_us_per_step', 0)):.3f} | "
            f"{float(row.get('alternate_overlap_allocation_us_per_step', 0)):.3f} |"
        )
    lines += ["", "## Scope", "",
              "The report does not make an end-to-end latency or hardware-counter claim. "
              "GEMM MFU, when present, is written separately by the existing GEMM analyzer with its own per-stage single-GPU denominator."]
    (output / "KERNEL_SHARES.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--skip-gemm", action="store_true")
    parser.add_argument("--existing-analysis", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    analysis = _load_analysis(root, args.existing_analysis)
    if not args.skip_gemm:
        _load_gemm(root)
    stage_rows, totals = _stage_rows(output / "stage_graph_stats.csv")
    _write(root, analysis, stage_rows, totals, args.skip_gemm)
    print(json.dumps({"root": str(root), "output": str(output / "KERNEL_SHARES.md"),
                      "skip_gemm": args.skip_gemm}, indent=2))


if __name__ == "__main__":
    main()
