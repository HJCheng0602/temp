# B64 GEMM Time, Shape And Tensor Types

Marlin + IPC, PP2 TP4 EP4, 2K context, nine stable steps. No new GPU execution.

GEMM time per GPU per decode step: stage0 **4.651882 ms**, stage1 **4.933365 ms**.

Each stage column is the mean of its four individual GPU times, averaged over nine steps. Shares use the listed GEMM time on one average GPU of that same stage as denominator. These are sums of kernel lifetimes on one GPU, including overlap and internal waits, not wall-clock TPOT. A dash means the stage does not execute that GEMM. The per-rank table below exposes load imbalance.

| GEMM | Logical shape | inA x inB -> outC | Stage0 ms / GPU / step (share) | Stage0 MFU | Stage1 ms / GPU / step (share) | Stage1 MFU |
|---|---|---|---:|---:|---:|---:|
| Marlin gate/up | [m_e,5120] @ [5120,4608] -> [m_e,4608] | BF16 x MXFP4 (E2M1 + block scales) -> BF16 | 0.866322 (18.62%) | 40.53% | 1.158652 (23.49%) | 30.31% |
| Marlin down | [m_e,2304] @ [2304,5120] -> [m_e,5120] | BF16 x MXFP4 (E2M1 + block scales) -> BF16 | 0.507069 (10.90%) | 34.63% | 0.652556 (13.23%) | 26.91% |
| FP8 attention, indexer and Engram projections | q_a + KV: [64,5120] @ [5120,1792] -> [64,1792]<br>q_b: [64,1280] @ [1280,8192] -> [64,8192]<br>wo_b: [64,2048] @ [2048,5120] -> [64,5120]<br>Indexer query: [64,1280] @ [1280,4096] -> [64,4096]<br>Engram: [64,6144] @ [6144,25600] -> [64,25600] | MXFP8 (E4M3 + UE8M0 scales) x MXFP8 (E4M3 + UE8M0 scales) -> BF16 | 1.119606 (24.07%) | 20.68% | 0.833546 (16.90%) | 18.72% |
| FP8 shared expert gate/up | [64,5120] @ [5120,1152] -> [64,1152] | MXFP8 x MXFP8 -> BF16 | 0.328169 (7.05%) | 8.92% | 0.336047 (6.81%) | 8.71% |
| FP8 shared expert down | [64,576] @ [576,5120] -> [64,5120] | MXFP8 x MXFP8 -> BF16 | 0.390805 (8.40%) | 3.74% | 0.357354 (7.24%) | 4.09% |
| BF16 grouped wo_a | [2,64,4096] @ [2,4096,1024] -> [2,64,1024] | BF16 x BF16 -> BF16 | 0.588352 (12.65%) | 14.15% | 0.597582 (12.11%) | 13.93% |
| BF16 indexer key | [64,512] @ [512,128] -> [64,128] | BF16 x BF16 -> BF16 | 0.012943 (0.28%) | 0.75% | 0.004096 (0.08%) | 0.79% |
| BF16 router | [64,5120] @ [5120,384] -> [64,384] | BF16 x BF16 -> FP32 | 0.145536 (3.13%) | 13.40% | 0.145976 (2.96%) | 13.36% |
| BF16 LM head | [64,5120] @ [5120,32320] -> [64,32320] | BF16 x BF16 -> BF16 | - | - | 0.299934 (6.08%) | 27.37% |
| BF16 indexer head weights | [64,5120] @ [5120,32] -> [64,32] | BF16 x BF16 -> BF16 | 0.028822 (0.62%) | 0.85% | 0.022434 (0.45%) | 1.81% |
| BF16 ratio-2 compressor | [64,5120] @ [5120,1024] -> [64,1024] | BF16 x BF16 -> FP32 | 0.051216 (1.10%) | 15.24% | - | - |
| BF16 ratio-1 compressor | [64,5120] @ [5120,512] -> [64,512] | BF16 x BF16 -> FP32 | - | - | 0.007244 (0.15%) | 17.95% |
| TF32 mHC full stream | [64,20480] @ [20480,24] -> [64,24] | BF16 x FP32 -> FP32 | 0.502854 (10.81%) | 3.78% | 0.517945 (10.50%) | 3.77% |
| TF32 mHC first-layer collapsed stream | [64,5120] @ [5120,24] -> [64,24] | BF16 x FP32 -> FP32 | 0.110188 (2.37%) | 0.11% | - | - |
| Total | - | Mixed | 4.651882 (100%) | - | 4.933365 (100%) | - |

The five shared-symbol FP8 shapes have combined time and MFU; per-shape durations have not been independently resolved. Per-GPU calls in stage0/stage1: q_a+KV 20/20, q_b 20/20, wo_b 20/20, indexer query 3/5, Engram 2/0.

## Precision And Shape Qualifications

Shapes show logical multiplication orientation, not packed weight storage or physical strides. MXFP8 means E4M3 values with UE8M0 block scales. The GEMM consumes quantized A; the preceding BF16-to-FP8 quantizer is separate.

For Marlin, m_e is the selected token count for one expert. In each layer sum(m_e)=64*6=384 across all 384 experts. Each EP rank owns 96 experts, with unequal local token counts. It is not M=64 for every expert. The physical gate/up input is [64,5120], and routed output slots cover [384,4608]; down uses [384,2304] and produces [384,5120] route slots before expert summation. Packed weights and padded tiles differ from these logical shapes.

The Marlin kernel dequantizes the four-bit weights for BF16 MMA and uses FP32 accumulation/reduction, then writes BF16. The other BF16/FP8 GEMMs accumulate in FP32; output dtype depends on the epilogue. Router and compressors request FP32 results, while LM head here writes BF16; later logits processing is outside this table. outC denotes the logical GEMM result. A split-K main kernel may write FP32 partials before a separate reduction produces C.

mHC stores A as BF16 and B as FP32. TF32 describes the internal compute path, not either tensor's storage dtype. Its split-K kernel writes FP32 partial matrices [s,64,24] and square sums [s,64]. The table gives the logical final matrix shape; the split factor s is not inferred from the symbol. No claim about the number of TF32 instructions per logical multiply is made.

Separate quantization, activation, split-K reduction and attention QK/AV kernels are excluded from these times. All nine per-step invocation counts were checked. Shapes are source/order reconstructions checked against historical eager operands, not newly captured B64 operand metadata.

Full-precision timing, shares, symbols and dtype metadata: `analysis/gemm_mfu.json`. Original evidence and FLOP derivations: `GEMM_MFU.md`.

## Tensor-Type Evidence

- Actual compressor FP32 output request: `source_snapshot/models/deepseek_v41/attention.py`, lines 846-855.
- Router FP32 output request: `../fast_gemm_trial_20260927/run_bundle/model_executor/layers/fused_moe/router/gate_linear.py`, GateLinear.forward tier 4. Historical router split-K reduction has float output pointers, distinct from the BF16 indexer reduction.
- mHC BF16 A, FP32 B/C assertions and allocations: `../../deployment/ced_prefill/evidence/pd_g/vllm/model_executor/kernels/mhc/tilelang.py`, lines 127-173.
- Marlin BF16 buffers and FP32 reduction: `source_snapshot/model_executor/layers/fused_moe/experts/marlin_moe.py`, lines 100-218.
- Upstream [Marlin BF16 MMA implementation](https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/quantization/marlin/marlin_mma.h) uses BF16 operands and FP32 accumulators.
- Upstream [FlashInfer b12x implementation](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py) declares acc_dtype=Float32. Upstream source corroborates the implementation; it is not a disassembly of the installed binary.

## Individual GPU Times

All values are ms per decode step, averaged over the same nine steps.

| GEMM | rank0 | rank1 | rank2 | rank3 | rank4 | rank5 | rank6 | rank7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Marlin gate/up | 0.832692 | 0.776105 | 0.926436 | 0.930056 | 1.166091 | 1.165205 | 1.232515 | 1.070798 |
| Marlin down | 0.487392 | 0.462276 | 0.540534 | 0.538074 | 0.661983 | 0.655207 | 0.680617 | 0.612417 |
| FP8 attention, indexer and Engram projections | 1.111143 | 1.123163 | 1.117980 | 1.126140 | 0.835581 | 0.834254 | 0.827419 | 0.836928 |
| FP8 shared expert gate/up | 0.324236 | 0.328285 | 0.326781 | 0.333373 | 0.336884 | 0.335946 | 0.331252 | 0.340106 |
| FP8 shared expert down | 0.412697 | 0.380537 | 0.370582 | 0.399403 | 0.382819 | 0.352189 | 0.324989 | 0.369418 |
| BF16 grouped wo_a | 0.578311 | 0.588698 | 0.589447 | 0.596950 | 0.600873 | 0.598421 | 0.591242 | 0.599792 |
| BF16 indexer key | 0.012917 | 0.012978 | 0.012786 | 0.013092 | 0.004046 | 0.004114 | 0.004128 | 0.004096 |
| BF16 router | 0.143435 | 0.145917 | 0.146069 | 0.146724 | 0.146168 | 0.145973 | 0.144463 | 0.147299 |
| BF16 LM head | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.300280 | 0.300103 | 0.299495 | 0.299857 |
| BF16 indexer head weights | 0.028768 | 0.028743 | 0.028586 | 0.029191 | 0.022589 | 0.022450 | 0.022766 | 0.021931 |
| BF16 ratio-2 compressor | 0.050841 | 0.051815 | 0.051154 | 0.051055 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| BF16 ratio-1 compressor | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.007271 | 0.007303 | 0.007275 | 0.007125 |
| TF32 mHC full stream | 0.500680 | 0.500200 | 0.500189 | 0.510348 | 0.522355 | 0.518867 | 0.509877 | 0.520682 |
| TF32 mHC first-layer collapsed stream | 0.105013 | 0.107549 | 0.116167 | 0.112021 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Total | 4.588125 | 4.506266 | 4.726711 | 4.786427 | 4.986940 | 4.940033 | 4.976039 | 4.830449 |

## MFU For The Per-GPU Stage Means

MFU = mean useful FLOPs per GPU / (single-GPU reference peak * mean kernel seconds per GPU). This ratio is not the arithmetic average of individual-rank MFUs. Marlin uses the known total top-6 expert work in each 20-layer stage divided by four, without assuming equal routing. The mixed FP8 symbol uses stage-specific shape counts, including Engram only in stage0.

Single-GPU dense reference peaks: BF16 258 TFLOP/s, FP8 516 TFLOP/s, TF32 129 TFLOP/s.

The input/output types and shape qualifications above still apply. Reference precision means BF16 for W4A16 Marlin, FP8 for block-scaled dense GEMM, and TF32 for mHC's internal compute path. Fused square sums, dequantization, padding and internal waits receive no GEMM FLOP credit. These are useful kernel-local ratios, not Tensor Core hardware counters or end-to-end model MFU.
