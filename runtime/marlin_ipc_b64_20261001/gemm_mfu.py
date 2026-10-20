"""Account for useful GEMM FLOPs against the existing B64 raw GPU trace."""

import collections
import gzip
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "results/marlin_ipc_b64_20261001"
B = 64
PEAKS = {"BF16": 258e12, "FP8": 516e12, "TF32": 129e12}


def gemm(calls, k, n, groups=1):
    return calls * groups * 2 * B * k * n


# Counts include TP replicas. Routed experts instead count each selection once
# across the complete EP group. Padding receives no useful FLOP credit.
LEDGER = {
    "Marlin gate/up": (160, "BF16", 2 * B * 6 * 40 * 5120 * 4608),
    "Marlin down": (160, "BF16", 2 * B * 6 * 40 * 2304 * 5120),
    "FP8 attention, indexer and Engram projections": (
        520, "FP8",
        gemm(160, 5120, 1792) + gemm(160, 1280, 8192)
        + gemm(160, 2048, 5120) + gemm(32, 1280, 4096)
        + gemm(8, 6144, 25600),
    ),
    "FP8 shared expert gate/up": (160, "FP8", gemm(160, 5120, 1152)),
    "FP8 shared expert down": (160, "FP8", gemm(160, 576, 5120)),
    "BF16 grouped wo_a": (160, "BF16", gemm(160, 4096, 1024, groups=2)),
    "BF16 indexer key": (16, "BF16", gemm(16, 512, 128)),
    "BF16 router": (160, "BF16", gemm(160, 5120, 384)),
    "BF16 LM head": (4, "BF16", gemm(4, 5120, 32320)),
    "BF16 indexer head weights": (32, "BF16", gemm(32, 5120, 32)),
    "BF16 ratio-2 compressor": (12, "BF16", gemm(12, 5120, 1024)),
    "BF16 ratio-1 compressor": (4, "BF16", gemm(4, 5120, 512)),
    "TF32 mHC full stream": (316, "TF32", gemm(316, 20480, 24)),
    "TF32 mHC first-layer collapsed stream": (4, "TF32", gemm(4, 5120, 24)),
}

IO = {
    "Marlin gate/up": ("BF16", "MXFP4 (E2M1 + block scales)", "BF16", "[m_e,5120] @ [5120,4608] -> [m_e,4608]", "BF16 MMA; FP32 accumulation/reduction"),
    "Marlin down": ("BF16", "MXFP4 (E2M1 + block scales)", "BF16", "[m_e,2304] @ [2304,5120] -> [m_e,5120]", "BF16 MMA; FP32 accumulation/reduction"),
    "FP8 attention, indexer and Engram projections": ("MXFP8 (E4M3 + UE8M0 scales)", "MXFP8 (E4M3 + UE8M0 scales)", "BF16", "Five shapes in the table below", "FP8 MMA; FP32 accumulation"),
    "FP8 shared expert gate/up": ("MXFP8", "MXFP8", "BF16", "[64,5120] @ [5120,1152] -> [64,1152]", "FP8 MMA; FP32 accumulation"),
    "FP8 shared expert down": ("MXFP8", "MXFP8", "BF16", "[64,576] @ [576,5120] -> [64,5120]", "FP8 MMA; FP32 accumulation"),
    "BF16 grouped wo_a": ("BF16", "BF16", "BF16", "[2,64,4096] @ [2,4096,1024] -> [2,64,1024]", "BF16 MMA; FP32 accumulation"),
    "BF16 indexer key": ("BF16", "BF16", "BF16", "[64,512] @ [512,128] -> [64,128]", "BF16 MMA; FP32 accumulation"),
    "BF16 router": ("BF16", "BF16", "FP32", "[64,5120] @ [5120,384] -> [64,384]", "BF16 MMA; FP32 accumulation; separate split-K reduction"),
    "BF16 LM head": ("BF16", "BF16", "BF16", "[64,5120] @ [5120,32320] -> [64,32320]", "BF16 MMA; FP32 accumulation; later logits processing excluded"),
    "BF16 indexer head weights": ("BF16", "BF16", "BF16", "[64,5120] @ [5120,32] -> [64,32]", "BF16 MMA; FP32 accumulation"),
    "BF16 ratio-2 compressor": ("BF16", "BF16", "FP32", "[64,5120] @ [5120,1024] -> [64,1024]", "BF16 MMA; FP32 accumulation; separate split-K reduction"),
    "BF16 ratio-1 compressor": ("BF16", "BF16", "FP32", "[64,5120] @ [5120,512] -> [64,512]", "BF16 MMA; FP32 accumulation"),
    "TF32 mHC full stream": ("BF16", "FP32", "FP32", "[64,20480] @ [20480,24] -> [64,24]", "TF32 path; FP32 partials [s,64,24] and square sums [s,64]"),
    "TF32 mHC first-layer collapsed stream": ("BF16", "FP32", "FP32", "[64,5120] @ [5120,24] -> [64,24]", "TF32 path; FP32 partials [s,64,24] and square sums [s,64]"),
}


def classify(node, previous, marlin_index):
    name = node["name"]
    if "marlin_moe_wna16::Marlin<" in name:
        return "Marlin gate/up" if marlin_index % 2 == 0 else "Marlin down"
    if "b12xDenseGemmKernel" in name:
        if "TVLayoutSrc1409601_" in name:
            return "FP8 attention, indexer and Engram projections"
        assert "TVLayoutSrc1819201_" in name, name
        return "FP8 shared expert gate/up"
    if name.startswith("cutlass3x_sm120_bstensorop_s16832gemm_block_scaled_"):
        return "FP8 shared expert down"
    if "sm120_tf32_hc_prenorm_gemm_impl<24u, 20480u," in name:
        return "TF32 mHC full stream"
    if "sm120_tf32_hc_prenorm_gemm_impl<24u, 5120u," in name:
        return "TF32 mHC first-layer collapsed stream"
    if "cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2" in name:
        if previous == "_fused_save_compress_norm_kernel":
            return "BF16 indexer key"
        assert previous == "Kernel", (name, previous)
        return "BF16 grouped wo_a"
    if "cutlass_80_wmma_tensorop_s161616gemm_bf16_32x32_128x2" in name:
        return "BF16 router"
    if "cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3" in name:
        return "BF16 LM head"
    if "cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x2" in name:
        return "BF16 indexer head weights"
    if "cutlass_80_wmma_tensorop_s161616gemm_bf16_32x32_64x2" in name:
        return "BF16 ratio-2 compressor"
    if "cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6" in name:
        return "BF16 ratio-1 compressor"
    assert "gemm" not in name.lower() or "paged_mqa_logits" in name, name
    return None


def stage_per_gpu_flops(group, stage, calls):
    total_calls, _, total_flops = LEDGER[group]
    if group.startswith("Marlin"):
        # Both stages have 20 layers and all top-6 routes stay within EP4.
        # Mean useful work is exact even when the four expert owners differ.
        return total_flops / 2 / 4
    if group == "FP8 attention, indexer and Engram projections":
        return (
            gemm(20, 5120, 1792) + gemm(20, 1280, 8192)
            + gemm(20, 2048, 5120)
            + gemm(3 if stage == 0 else 5, 1280, 4096)
            + gemm(2 if stage == 0 else 0, 6144, 25600)
        )
    return total_flops / total_calls * calls


def main():
    with gzip.open(RESULT / "analysis/nodes.json.gz", "rt") as f:
        steps = json.load(f)["steps"]
    totals = collections.defaultdict(float)
    rank_totals = collections.defaultdict(lambda: [0.0] * 8)
    rank_counts = collections.defaultdict(lambda: [0] * 8)
    symbols = collections.defaultdict(set)
    for step in steps:
        counts = collections.Counter()
        for rank in range(8):
            nodes = [n for n in step["nodes"] if n["rank"] == rank and n["kind"] == "kernel"]
            marlin_index = 0
            for index, node in enumerate(nodes):
                previous = nodes[index - 1]["name"] if index else ""
                group = classify(node, previous, marlin_index)
                if group is None:
                    continue
                if group.startswith("Marlin"):
                    marlin_index += 1
                counts[group] += 1
                totals[group] += node["raw_duration"]
                rank_totals[group][rank] += node["raw_duration"]
                rank_counts[group][rank] += 1
                symbols[group].add(node["name"])
            assert marlin_index == 40, (step["step"], rank, marlin_index)
        assert dict(counts) == {k: v[0] for k, v in LEDGER.items()}, counts

    rows = []
    all_gemm_us = sum(totals.values()) / len(steps)
    per_rank_gemm_us = [
        sum(value[rank] for value in rank_totals.values()) / len(steps)
        for rank in range(8)
    ]
    per_stage_per_gpu_total_us = [
        sum(per_rank_gemm_us[start:start + 4]) / 4 for start in (0, 4)
    ]
    for group, (count, precision, flops) in LEDGER.items():
        us = totals[group] / len(steps)
        a, b, c, shape, arithmetic = IO[group]
        rank_us = [value / len(steps) for value in rank_totals[group]]
        stage_per_gpu = []
        for stage, start in enumerate((0, 4)):
            mean_us = sum(rank_us[start:start + 4]) / 4
            mean_calls = sum(rank_counts[group][start:start + 4]) / len(steps) / 4
            mean_flops = stage_per_gpu_flops(group, stage, mean_calls)
            stage_per_gpu.append(dict(
                stage=stage,
                mean_us_per_gpu_per_step=mean_us,
                share_of_stage_gemm_time_percent=100 * mean_us / per_stage_per_gpu_total_us[stage],
                min_rank_us_per_step=min(rank_us[start:start + 4]),
                max_rank_us_per_step=max(rank_us[start:start + 4]),
                calls_per_gpu_per_step=mean_calls,
                mean_useful_flops_per_gpu_per_step=mean_flops,
                mfu_percent=100 * mean_flops / (PEAKS[precision] * mean_us * 1e-6) if mean_us else None,
            ))
        reconstructed_flops = sum(s["mean_useful_flops_per_gpu_per_step"] * 4 for s in stage_per_gpu)
        assert abs(reconstructed_flops - flops) < 1, (group, reconstructed_flops, flops)
        rows.append(dict(group=group, calls_per_step=count, precision=precision,
                         flops_per_step=flops, all_rank_raw_us_per_step=us,
                         share_of_listed_gemm_time_percent=100 * us / all_gemm_us,
                         all_rank_raw_us_captured=totals[group],
                         inA=a, inB=b, outC=c, logical_shape=shape,
                         per_rank_us_per_step=rank_us,
                         per_rank_share_of_gemm_time_percent=[100 * t / denominator for t, denominator in zip(rank_us, per_rank_gemm_us)],
                         per_stage_per_gpu=stage_per_gpu,
                         arithmetic=arithmetic,
                         mean_call_us=us / count, effective_tflops=flops / us / 1e6,
                         mfu_percent=100 * flops / (PEAKS[precision] * us * 1e-6),
                         symbols=sorted(symbols[group])))
    aggregates = {}
    for group, selected in {
        "Marlin combined": [r for r in rows if r["group"].startswith("Marlin")],
        "FP8 combined": [r for r in rows if r["precision"] == "FP8"],
        "All listed GEMMs": rows,
    }.items():
        raw_s = sum(r["all_rank_raw_us_per_step"] for r in selected) * 1e-6
        ideal_s = sum(r["flops_per_step"] / PEAKS[r["precision"]] for r in selected)
        aggregates[group] = dict(all_rank_raw_us_per_step=raw_s * 1e6,
                                 precision_weighted_mfu_percent=100 * ideal_s / raw_s)
        aggregates[group]["per_stage_per_gpu"] = []
        for stage in range(2):
            stage_us = sum(r["per_stage_per_gpu"][stage]["mean_us_per_gpu_per_step"] for r in selected)
            stage_ideal_s = sum(r["per_stage_per_gpu"][stage]["mean_useful_flops_per_gpu_per_step"] / PEAKS[r["precision"]] for r in selected)
            aggregates[group]["per_stage_per_gpu"].append(dict(
                stage=stage, mean_us_per_gpu_per_step=stage_us,
                precision_weighted_mfu_percent=100 * stage_ideal_s / (stage_us * 1e-6),
            ))

    payload = dict(batch=B, steps=len(steps), peaks_flops_per_s=PEAKS, rows=rows,
                   aggregates=aggregates,
                   per_rank_total_gemm_us_per_step=per_rank_gemm_us,
                   per_stage_per_gpu_total_gemm_us_per_step=per_stage_per_gpu_total_us,
                   accounting="Useful 2*M*N*K, raw summed GPU kernel seconds; TP replicas included, padding excluded.",
                   evidence="Existing nine-step B64 trace, actual model config/source and historical eager operand ledger; all per-step counts validated.",
                   limitations=["Source/order-inferred shapes, not fresh per-launch operand capture.",
                                "Kernel time includes internal waits and fused overhead; this is not an SM hardware counter or end-to-end model MFU.",
                                "mHC credits only learned GEMM FLOPs, excluding fused square sums.",
                                "Separate quantizers, split-K reductions, attention contractions and communication are excluded."])
    (RESULT / "analysis/gemm_mfu.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# B64 GEMM Time, Shape And Tensor Types", "",
        "Marlin + IPC, PP2 TP4 EP4, 2K context, nine stable steps. No new GPU execution.", "",
        f"GEMM time per GPU per decode step: stage0 **{per_stage_per_gpu_total_us[0] / 1000:.6f} ms**, "
        f"stage1 **{per_stage_per_gpu_total_us[1] / 1000:.6f} ms**.", "",
        "Each stage column is the mean of its four individual GPU times, averaged over nine steps. "
        "Shares use the listed GEMM time on one average GPU of that same stage as denominator. "
        "These are sums of kernel lifetimes on one GPU, including overlap and internal waits, not wall-clock TPOT. "
        "A dash means the stage does not execute that GEMM. The per-rank table below exposes load imbalance.", "",
        "| GEMM | Logical shape | inA x inB -> outC | Stage0 ms / GPU / step (share) | Stage0 MFU | Stage1 ms / GPU / step (share) | Stage1 MFU |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        cells = []
        for stage in r["per_stage_per_gpu"]:
            cells += [f"{stage['mean_us_per_gpu_per_step']/1000:.6f} ({stage['share_of_stage_gemm_time_percent']:.2f}%)", f"{stage['mfu_percent']:.2f}%"] if stage["calls_per_gpu_per_step"] else ["-", "-"]
        shape = r["logical_shape"]
        if r["group"] == "FP8 attention, indexer and Engram projections":
            shape = "<br>".join([
                "q_a + KV: [64,5120] @ [5120,1792] -> [64,1792]",
                "q_b: [64,1280] @ [1280,8192] -> [64,8192]",
                "wo_b: [64,2048] @ [2048,5120] -> [64,5120]",
                "Indexer query: [64,1280] @ [1280,4096] -> [64,4096]",
                "Engram: [64,6144] @ [6144,25600] -> [64,25600]",
            ])
        lines.append(f"| {r['group']} | {shape} | {r['inA']} x {r['inB']} -> {r['outC']} | {' | '.join(cells)} |")
    lines.append(f"| Total | - | Mixed | {per_stage_per_gpu_total_us[0]/1000:.6f} (100%) | - | {per_stage_per_gpu_total_us[1]/1000:.6f} (100%) | - |")
    lines += ["",
              "The five shared-symbol FP8 shapes have combined time and MFU; per-shape durations have not been independently resolved. "
              "Per-GPU calls in stage0/stage1: q_a+KV 20/20, q_b 20/20, wo_b 20/20, indexer query 3/5, Engram 2/0.",
              "", "## Precision And Shape Qualifications", "",
              "Shapes show logical multiplication orientation, not packed weight storage or physical strides. "
              "MXFP8 means E4M3 values with UE8M0 block scales. The GEMM consumes quantized A; the preceding BF16-to-FP8 quantizer is separate.", "",
              "For Marlin, m_e is the selected token count for one expert. In each layer sum(m_e)=64*6=384 across all 384 experts. "
              "Each EP rank owns 96 experts, with unequal local token counts. It is not M=64 for every expert. "
              "The physical gate/up input is [64,5120], and routed output slots cover [384,4608]; down uses [384,2304] "
              "and produces [384,5120] route slots before expert summation. Packed weights and padded tiles differ from these logical shapes.", "",
              "The Marlin kernel dequantizes the four-bit weights for BF16 MMA and uses FP32 accumulation/reduction, then writes BF16. "
              "The other BF16/FP8 GEMMs accumulate in FP32; output dtype depends on the epilogue. "
              "Router and compressors request FP32 results, while LM head here writes BF16; later logits processing is outside this table. "
              "outC denotes the logical GEMM result. A split-K main kernel may write FP32 partials before a separate reduction produces C.", "",
              "mHC stores A as BF16 and B as FP32. TF32 describes the internal compute path, not either tensor's storage dtype. "
              "Its split-K kernel writes FP32 partial matrices [s,64,24] and square sums [s,64]. The table gives the logical final matrix shape; "
              "the split factor s is not inferred from the symbol. No claim about the number of TF32 instructions per logical multiply is made.", "",
              "Separate quantization, activation, split-K reduction and attention QK/AV kernels are excluded from these times. "
              "All nine per-step invocation counts were checked. Shapes are source/order reconstructions checked against historical eager operands, "
              "not newly captured B64 operand metadata.", "",
              "Full-precision timing, shares, symbols and dtype metadata: `analysis/gemm_mfu.json`. "
              "Original evidence and FLOP derivations: `GEMM_MFU.md`.", "",
              "## Tensor-Type Evidence", "",
              "- Actual compressor FP32 output request: `source_snapshot/models/deepseek_v41/attention.py`, lines 846-855.",
              "- Router FP32 output request: `../fast_gemm_trial_20260927/run_bundle/model_executor/layers/fused_moe/router/gate_linear.py`, GateLinear.forward tier 4. "
              "Historical router split-K reduction has float output pointers, distinct from the BF16 indexer reduction.",
              "- mHC BF16 A, FP32 B/C assertions and allocations: `../../deployment/ced_prefill/evidence/pd_g/vllm/model_executor/kernels/mhc/tilelang.py`, lines 127-173.",
              "- Marlin BF16 buffers and FP32 reduction: `source_snapshot/model_executor/layers/fused_moe/experts/marlin_moe.py`, lines 100-218.",
              "- Upstream [Marlin BF16 MMA implementation](https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/quantization/marlin/marlin_mma.h) "
              "uses BF16 operands and FP32 accumulators.",
              "- Upstream [FlashInfer b12x implementation](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py) "
              "declares acc_dtype=Float32. Upstream source corroborates the implementation; it is not a disassembly of the installed binary."]
    (RESULT / "GEMM_DETAIL.md").write_text("\n".join(lines) + "\n")
    with (RESULT / "GEMM_DETAIL.md").open("a") as f:
        f.write("\n## Individual GPU Times\n\nAll values are ms per decode step, averaged over the same nine steps.\n\n")
        f.write("| GEMM | " + " | ".join(f"rank{r}" for r in range(8)) + " |\n")
        f.write("|---|" + "---:|" * 8 + "\n")
        for row in rows:
            f.write("| " + row["group"] + " | " + " | ".join(f"{t/1000:.6f}" for t in row["per_rank_us_per_step"]) + " |\n")
        f.write("| Total | " + " | ".join(f"{t/1000:.6f}" for t in per_rank_gemm_us) + " |\n")
        f.write("\n## MFU For The Per-GPU Stage Means\n\n")
        f.write("MFU = mean useful FLOPs per GPU / (single-GPU reference peak * mean kernel seconds per GPU). "
                "This ratio is not the arithmetic average of individual-rank MFUs. Marlin uses the known total "
                "top-6 expert work in each 20-layer stage divided by four, without assuming equal routing. "
                "The mixed FP8 symbol uses stage-specific shape counts, including Engram only in stage0.\n\n")
        f.write("Single-GPU dense reference peaks: BF16 258 TFLOP/s, FP8 516 TFLOP/s, TF32 129 TFLOP/s.\n")
        f.write("\nThe input/output types and shape qualifications above still apply. Reference precision means "
                "BF16 for W4A16 Marlin, FP8 for block-scaled dense GEMM, and TF32 for mHC's internal compute path. "
                "Fused square sums, dequantization, padding and internal waits receive no GEMM FLOP credit. "
                "These are useful kernel-local ratios, not Tensor Core hardware counters or end-to-end model MFU.\n")
    for row in rows:
        print(f"{row['group']}: {row['flops_per_step']/1e9:.6f} GFLOP, "
              f"{row['all_rank_raw_us_per_step']/1000:.6f} GPU-ms, "
              f"{row['mean_call_us']:.3f} us/call, {row['mfu_percent']:.3f}%")
    print(json.dumps(aggregates, indent=2))


if __name__ == "__main__":
    main()
