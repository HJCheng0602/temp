"""Compare measured chunked and unchunked TTFT, retaining failed-run evidence."""

import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib.pyplot as plt

from analyze_sweep import matches, read_rows, request_stages


def collect(directory):
    requests = json.loads((directory / "requests.json").read_text()) if (directory / "requests.json").exists() else []
    rows = read_rows(directory)
    output = {}
    for length in sorted({r["prompt_tokens"] for r in requests}):
        selected = [r for r in requests if r["prompt_tokens"] == length and r["label"].startswith("measure_")]
        if not selected:
            continue
        values = [r["ttft_ms"] for r in selected]
        stages = [request_stages(rows, r["request_id"]) for r in selected]
        expected = 1 if "no_chunk" in directory.name else (length + 8191) // 8192
        for r in selected:
            for rank in range(8):
                actual = [row for row in rows if row.get("global_rank") == rank and row.get("prefill") and matches(row, r["request_id"])]
                if len(actual) != expected or sum(row["n"] for row in actual) != length:
                    raise RuntimeError(f"Incomplete rank audit: {directory.name}/{r['label']}/rank{rank}")
        output[length] = dict(median_ms=statistics.median(values), min_ms=min(values),
                              max_ms=max(values), repeats=len(values), chunks=expected,
                              stage0_ms=statistics.median(sum(s[0]) for s in stages),
                              stage1_ms=statistics.median(sum(s[1]) for s in stages))
    return output


def plot_trace(path, output):
    trace = json.loads(path.read_text())
    rows = []
    selectors = {
        "mqa_ms": "mqa_logits", "topk_ms": "topkperrow",
        "candidate_mask_ms": "_mask_candidates_kernel",
        "candidate_blocks_ms": "_block_scores_kernel",
        "sparse_mla_ms": "sparse_mla_prefill", "nccl_ms": "nccl",
    }
    for rank in trace["ranks"]:
        for chunk in rank["chunks"]:
            row = dict(rank=int(rank["rank"]), stage=int(rank["rank"]) // 4,
                       chunk=chunk["chunk_label"] + 1,
                       context_k=(chunk["chunk_label"] + 1) * 8,
                       kernel_sum_ms=chunk["sum_gpu_us"] / 1000,
                       kernel_union_ms=chunk["union_gpu_us"] / 1000)
            for field, pattern in selectors.items():
                row[field] = sum(k["sum_gpu_us"] for k in chunk["top_kernels_by_sum_gpu_us"]
                                 if pattern in k["name"].lower()) / 1000
            rows.append(row)
    with (output / "trace_components.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    series = [("mqa_ms", "MQA index logits", "#b64d36"),
              ("topk_ms", "Indexer top-k", "#98722b"),
              ("candidate_mask_ms", "Candidate mask", "#68734c"),
              ("sparse_mla_ms", "Sparse MLA", "#17778d"),
              ("nccl_ms", "In-forward NCCL", "#75648a")]
    for stage, ax in enumerate(axes):
        selected = [r for r in rows if r["stage"] == stage]
        for field, label, color in series:
            ax.plot([r["context_k"] for r in selected], [r[field] for r in selected],
                    label=label, color=color, linewidth=1.8)
        ax.set(xlabel="Cumulative context (K tokens)", title=f"Stage {stage}, rank {stage * 4}")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Summed kernel duration per 8K chunk (ms)")
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle("128K prefill: index selection drives the growing stage-1 cost", fontsize=12)
    fig.tight_layout()
    fig.savefig(output / "stage_growth.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--no-chunk", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = collect(args.baseline)
    trials = [(directory, collect(directory)) for directory in args.no_chunk]
    rows = []
    for length, previous in baseline.items():
        available = [(directory, data[length]) for directory, data in trials if length in data]
        row = dict(context_k=length // 1024, baseline_ms=previous["median_ms"])
        if available:
            directory, current = available[0]
            row.update(no_chunk_ms=current["median_ms"], ratio=current["median_ms"] / previous["median_ms"],
                       min_ms=current["min_ms"], max_ms=current["max_ms"], repeats=current["repeats"],
                       stage0_ms=current["stage0_ms"], stage1_ms=current["stage1_ms"], run=directory.name)
        else:
            row.update(no_chunk_ms=None, ratio=None, min_ms=None, max_ms=None, repeats=0,
                       stage0_ms=None, stage1_ms=None, run="not measured")
        rows.append(row)
    with (args.output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "comparison.json").write_text(json.dumps(rows, indent=2))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([r["context_k"] for r in rows], [r["baseline_ms"] / 1000 for r in rows], "o-", color="#17778d", label="8K scheduler chunks")
    valid = [r for r in rows if r["no_chunk_ms"] is not None]
    ax.plot([r["context_k"] for r in valid], [r["no_chunk_ms"] / 1000 for r in valid], "s-", color="#c65332", label="No scheduler chunking")
    ax.set(xlabel="Prompt length (K tokens, K = 1024)", ylabel="Median TTFT (seconds)", title="B=1 | PP2 (20+20), TP4, EP4 | Eager")
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output / "comparison.png", dpi=180)
    plt.close(fig)
    if args.trace:
        plot_trace(args.trace, args.output)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
