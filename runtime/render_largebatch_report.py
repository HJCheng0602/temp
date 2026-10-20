"""Combine large-batch summaries and render measured throughput plots."""

import argparse
import csv
import json
from pathlib import Path


def load(path):
    with Path(path).open() as handle:
        return json.load(handle)


def status_score(row, summary):
    status = row.get("status")
    observed = summary.get("configuration", {}).get("capacity_mode") == "observed"
    if status != "complete" or row.get("resident_verified_batch") is None:
        # Preserve observed runtime evidence over a conservative skip when
        # both are candidates for the same context/batch.
        return 2 if observed and status in (
            "preempted", "preemption_unknown", "unverified", "failed"
        ) else 1 if observed else 0
    clean = row.get("preemption_status") in (None, "clean")
    return 4 if observed and clean else 3 if clean else 1


def combine(paths):
    grouped = {}
    failures = []
    for source in paths:
        summary = load(source)
        for row in summary.get("cases", []):
            item = dict(row, source=str(source))
            key = (row.get("context_tokens"), row.get("requested_batch"))
            grouped.setdefault(key, []).append((status_score(row, summary), item))
    selected = []
    alternates = []
    for key, candidates in sorted(grouped.items(), key=lambda pair: (
            pair[0][0] is None, pair[0][0], pair[0][1] is None, pair[0][1])):
        candidates.sort(key=lambda pair: (-pair[0], pair[1].get("source", "")))
        selected.append(candidates[0][1])
        alternates.extend(item for _, item in candidates[1:])
    for row in selected + alternates:
        if row.get("status") != "complete" or row.get("resident_verified_batch") is None:
            failures.append(row)
    return {
        "sources": [str(path) for path in paths],
        "cases": selected,
        "alternates": alternates,
        "failures_and_skips": failures,
    }


def write_csv(rows, path):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def render(combined, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {2048: "#1769aa", 8192: "#c45508", 32768: "#2f855a"}
    labels = {2048: "2K", 8192: "8K", 32768: "32K"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for context in (2048, 8192, 32768):
        rows = [row for row in combined["cases"]
                if row.get("context_tokens") == context
                and row.get("status") == "complete"
                and row.get("resident_verified_batch") is not None]
        rows.sort(key=lambda row: row.get("requested_batch", 0))
        batches = [row["requested_batch"] for row in rows]
        itl = [row.get("median_itl_ms") for row in rows]
        tps = [row.get("aggregate_output_tokens_per_second") for row in rows]
        measured_itl = [(b, y) for b, y in zip(batches, itl) if y is not None]
        measured_tps = [(b, y) for b, y in zip(batches, tps) if y is not None]
        if measured_itl:
            axes[0].plot(*zip(*measured_itl), marker="o", color=colors[context], label=labels[context])
        if measured_tps:
            axes[1].plot(*zip(*measured_tps), marker="o", color=colors[context], label=labels[context])
    axes[0].set(xlabel="Requested batch", ylabel="Median ITL (ms)", title="Measured median ITL")
    axes[1].set(xlabel="Requested batch", ylabel="Aggregate output (tokens/s)", title="Measured aggregate throughput")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(title="Context")
    fig.savefig(output / "largebatch_report.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", nargs="+", type=Path, required=True,
                        help="summarize_largebatch JSON files")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined = combine(args.summaries)
    (args.output_dir / "combined.json").write_text(json.dumps(combined, indent=2))
    write_csv(combined["cases"] + combined["alternates"], args.output_dir / "combined.csv")
    render(combined, args.output_dir)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "selected_cases": len(combined["cases"]),
        "failures_and_skips": len(combined["failures_and_skips"]),
    }, indent=2))


if __name__ == "__main__":
    main()
