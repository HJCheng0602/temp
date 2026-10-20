# B128 Full Decode Path and Kernel Shares

2026-09-30 Marlin + FlashInfer PCIe IPC baseline; old profile, offline reanalysis

PP2 TP4 EP4, layers 20+20, 2048-token prompt, nine stable decode replays.

Path: stage0 embedding and layers -> PP tensor sends/receives -> stage1 TP reconstruction AllGather -> stage1 layers -> final normalization, LM head, logits AllGather and sampling.

Observed path: 28.606376 ms. Endpoint: Earliest stage0 graph activity to latest stage1 sample-ready event.

A single dependency chain is attributed once, with overlapping intervals allocated consumer-first. Values are not sums across GPUs. Collectives report observed completion tails, not pure transfer time.

## Entire Path

| Category | ms/step | Path share |
|---|---:|---:|
| MoE expert GEMM | 11.203648 | 39.1649% |
| Other GEMM/projections | 5.123561 | 17.9106% |
| Attention/indexer | 3.531497 | 12.3451% |
| IPC / Attention output | 1.991264 | 6.9609% |
| IPC / MoE output combine | 1.935231 | 6.7650% |
| AllGather | 1.040387 | 3.6369% |
| Other kernels | 0.918200 | 3.2098% |
| Unattributed gaps | 0.844734 | 2.9530% |
| Routing/quant/activation | 0.841229 | 2.9407% |
| mHC/normalization | 0.784248 | 2.7415% |
| KV/RoPE/metadata | 0.138781 | 0.4851% |
| Sampling/output | 0.084099 | 0.2940% |
| PP SendRecv | 0.082239 | 0.2875% |
| IPC / Embedding | 0.048440 | 0.1693% |
| Engram hash/lookup/post | 0.020416 | 0.0714% |
| Memory copy | 0.010574 | 0.0370% |
| Memory set | 0.007826 | 0.0274% |

Accounting total: 100.000000%.

## Top 30 Kernels on the Path

| # | Full kernel symbol | ms/step | Path share |
|---:|---|---:|---:|
| 1 | `void marlin_moe_wna16::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 2814749767106568l, 128, 1, 8, 4, true, 4, 2, false>(int4 const*, int4 const*, int4*, int4*, int4 const*, float const*, int4 const*, float const*, int4 const*, int const*, int const*, int const*, float const*, int, bool, int, int, int, int*, bool, bool, bool)` | 11.203648 | 39.1649% |
| 2 | `void flashinfer::comm::pcie_ipc::ipc_rsag_push_param_kernel<__nv_bfloat16, 4, false>(flashinfer::comm::pcie_ipc::PushOneshotParamData<__nv_bfloat16>)` | 3.974936 | 13.8953% |
| 3 | `void sparse_mla_prefill_mg_dual_kernel<(ModelType)1, (ComputeMode)1, 16, 64, 64, 1>(__nv_bfloat16 const*, unsigned char const*, int const*, unsigned char const*, int const*, __nv_bfloat16*, float*, float const*, PrefillColdParams)` | 3.478942 | 12.1614% |
| 4 | `kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm120_b12xDenseGemmKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1409601_TVLayoutDst1409601_Valuetypef8E4M3FN_tensor00o11_0` | 1.286262 | 4.4964% |
| 5 | `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1.040387 | 3.6369% |
| 6 | `void deep_gemm::sm120_tf32_hc_prenorm_gemm_impl<24u, 20480u, 128u, 32u, 64u, 16u, 4u, 256u, 128u>(unsigned int, CUtensorMap_st, CUtensorMap_st, float*, float*, long, long)` | 1.029980 | 3.6005% |
| 7 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8::Params)` | 0.972178 | 3.3985% |
| 8 | `cutlass3x_sm120_bstensorop_s16832gemm_block_scaled_ue8m0xe4m3_ue8m0xe4m3_f32_bf16_bf16_128x128x128_1x1x1_0_tnn_align16_q_bias_bf16_relu` | 0.883033 | 3.0868% |
| 9 | `mhc_pre_big_fuse_with_norm_tilelang_kernel` | 0.395095 | 1.3811% |
| 10 | `mhc_post_tilelang_kernel` | 0.386405 | 1.3508% |
| 11 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8>(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8::Params)` | 0.325600 | 1.1382% |
| 12 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1})` | 0.313001 | 1.0942% |
| 13 | `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8::Params)` | 0.267349 | 0.9346% |
| 14 | `void deep_gemm::sm120_fp8_paged_mqa_logits<1u, 32u, 128u, 64u, true, false, 2u, 3u, 128u, 128u, 256u, float>(unsigned int, unsigned int, unsigned int, unsigned int const*, float*, unsigned int const*, unsigned int const*, unsigned int const*, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)` | 0.241131 | 0.8429% |
| 15 | `_swiglu_limit_pad_aware_kernel` | 0.168435 | 0.5888% |
| 16 | `_dsv4_topk_kernel` | 0.163225 | 0.5706% |
| 17 | `Kernel` | 0.147106 | 0.5142% |
| 18 | `void vllm::moe::moe_align_block_size_kernel<int>(int const*, int*, int*, int*, int*, int, int, int, int, unsigned long, int*, int, int, bool)` | 0.145992 | 0.5103% |
| 19 | `void vllm::deepseek_v4_fused_ops::fusedDeepseekV4QNormRopeKVRopeQuantInsertKernel<c10::BFloat16, 16, false, false>(c10::BFloat16 const*, c10::BFloat16*, c10::BFloat16 const*, unsigned char*, long const*, long const*, float const*, float, int, int, int, int, int, bool, bool)` | 0.138781 | 0.4851% |
| 20 | `void vllm::moe::moe_sum_vec_kernel<c10::BFloat16, int, 6, true>(c10::BFloat16*, c10::BFloat16 const*, long, int, long, long, int const*, int const*, long, long)` | 0.126255 | 0.4414% |
| 21 | `void vllm::persistent::persistent_topk_kernel<512, 4u>(vllm::persistent::PersistentTopKParams)` | 0.102329 | 0.3577% |
| 22 | `kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_at__tensorptrbf16gmemalign16o128012801_tensorptri8gmemalign16o128012801_tensorptri8gmem_0` | 0.087052 | 0.3043% |
| 23 | `ncclDevKernel_SendRecv(ncclDevKernelArgsStorage<4096ul>)` | 0.082239 | 0.2875% |
| 24 | `_gumbel_sample_kernel` | 0.078457 | 0.2743% |
| 25 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, float, float, float, false, float, float, float, true, false, false, false>(cublasLt::cublasSplitKParams<float>, float const*, float const*, float*, float*, float const*, float const*, float const*, float const*, float*, void*, long, float*, int*, float*, float*, float const*, float const*, float const*, float const*, float const*)` | 0.073383 | 0.2565% |
| 26 | `kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_at__tensorptrbf16gmemalign16o512051201_tensorptri8gmemalign16o512051201_tensorptri8gmem_0` | 0.071995 | 0.2517% |
| 27 | `_compute_global_topk_indices_and_lens_kernel` | 0.066635 | 0.2329% |
| 28 | `kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_at__tensorptrbf16gmemalign16o204820481_tensorptri8gmemalign16o204820481_tensorptri8gmem_0` | 0.065508 | 0.2290% |
| 29 | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>)` | 0.063029 | 0.2203% |
| 30 | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<unsigned char>, std::array<char*, 1ul> >(int, at::native::FillFunctor<unsigned char>, std::array<char*, 1ul>)` | 0.059627 | 0.2084% |

## Single GPU Captured Graph

Each stage uses one actual GPU (rank0/rank4). Raw kernel lifetimes may overlap; their sum is not stage wall time. These denominators differ from the full-path table. External PP, LM head and sampling are omitted only in this graph-only table.

| Stage | Rank | Category | ms/GPU/step | Graph kernel share |
|---:|---:|---|---:|---:|
| 0 | 0 | IPC / MoE output combine | 3.438169 | 22.9701% |
| 0 | 0 | Other GEMM/projections | 3.264496 | 21.8098% |
| 0 | 0 | MoE expert GEMM | 3.107935 | 20.7639% |
| 0 | 0 | Attention/indexer | 1.684642 | 11.2550% |
| 0 | 0 | IPC / Attention output | 1.058275 | 7.0702% |
| 0 | 0 | Routing/quant/activation | 0.615521 | 4.1122% |
| 0 | 0 | Other kernels | 0.526984 | 3.5207% |
| 0 | 0 | Engram hash/lookup/post | 0.505360 | 3.3763% |
| 0 | 0 | mHC/normalization | 0.394732 | 2.6372% |
| 0 | 0 | IPC / Embedding | 0.156978 | 1.0488% |
| 0 | 0 | AllGather | 0.145373 | 0.9712% |
| 0 | 0 | KV/RoPE/metadata | 0.069540 | 0.4646% |
| 1 | 4 | MoE expert GEMM | 3.556917 | 25.4192% |
| 1 | 4 | Other GEMM/projections | 3.125036 | 22.3328% |
| 1 | 4 | IPC / MoE output combine | 3.053045 | 21.8183% |
| 1 | 4 | Attention/indexer | 1.833367 | 13.1020% |
| 1 | 4 | IPC / Attention output | 1.002422 | 7.1637% |
| 1 | 4 | Routing/quant/activation | 0.573539 | 4.0987% |
| 1 | 4 | mHC/normalization | 0.421829 | 3.0146% |
| 1 | 4 | Other kernels | 0.355437 | 2.5401% |
| 1 | 4 | KV/RoPE/metadata | 0.071446 | 0.5106% |

## Scope

- Old Sep30 baseline profile; no new GPU experiment.
- Collective tails include rendezvous and waiting and are not physical wire time.
- Raw stage rows are per-GPU averages within each stage, not wall time.
- Dependency attribution is an observed path proxy, not a marginal speedup bound.
- This report contains no B64 M32 claim and no extrapolation from B64.
- Graph-only raw statistics omit PP, logits and sampling outside capture; full-path statistics include them.
- Ranks 0 and 4 are actual representative GPUs; they are not stage averages.
- Other GEMM/projections includes attention, Engram, mHC and output projections with shared symbols.
- Input preparation before graph entry and feedback after sample readiness are outside the span.
