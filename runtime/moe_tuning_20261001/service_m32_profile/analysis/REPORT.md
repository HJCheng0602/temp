# Marlin + PCIe-IPC B64 PP2 TP4 EP4

Nine stable replays from the single B64 case were analyzed across all eight ranks. The endpoint is the stage-1 sample-ready boundary (`sample_ends`), so the span starts at stage-0 graph activity and ends when stage 1 has sampled.

Mean observed dependency span: 17.600 ms. Mean unattributed gap: 0.676 ms. Category accounting closes at 100.000000%.

Raw graph-kernel duration is shown per rank and as an eight-rank sum. It is not wall time, and the eight-rank sum is not divided by TPOT. Stage graph spans are separate wall-clock intervals and must not be added to raw duration totals.

The result is an observed dependency path proxy with consumer-first and producer-first overlap allocations. Profiler overhead is present; this is not an unprofiled TPOT measurement. Collective tails are rendezvous-to-completion proxies, not physical wire time.

## Category attribution

| Category | us/step | Percent of mean span |
|---|---:|---:|
| dense projection | 4995.063 | 28.380672% |
| MoE | 4674.804 | 26.561045% |
| attention | 2488.580 | 14.139475% |
| AllReduce | 1989.145 | 11.301815% |
| routequantactivation | 693.786 | 3.941918% |
| unattributed gaps | 676.344 | 3.842815% |
| norm/mhc | 661.802 | 3.760189% |
| other | 618.999 | 3.516994% |
| AllGather | 602.249 | 3.421824% |
| kvmeta | 116.849 | 0.663909% |
| PP | 51.388 | 0.291974% |
| engram | 14.322 | 0.081371% |
| memset | 8.551 | 0.048583% |
| memcpy | 8.345 | 0.047415% |

## Top 30

| # | Kernel | Category | Consumer us/step | Producer us/step | Raw all-rank us/step |
|---:|---|---|---:|---:|---:|
| 1 | `void marlin_moe_wna16::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 2814749767106568l, 128, 2, 8, 4, false, 4, 2, false>(int4 const*, int4 const*, int4*, int4*, int4 const*, float const*, int4 const*, float const*, int4 const*, int const*, int const*, int const*, float const*, int, bool, int, int, int, int*, bool, bool, bool)` | MoE | 4674.804 | 4674.804 | 12738.398 |
| 2 | `void flashinfer::sparse_mla_sm120::sparse_mla_decode_dsv4_kernel<(ModelType)1, 16, 64>(__nv_bfloat16 const*, unsigned char const*, int const*, __nv_bfloat16*, float*, int const*, unsigned char const*, int const*, int const*, int, int, unsigned long, int, int, int, int, int, float, unsigned long, unsigned long, unsigned long)` | attention | 2210.245 | 2210.245 | 7697.431 |
| 3 | `void flashinfer::comm::pcie_ipc::ipc_rsag_push_param_kernel<__nv_bfloat16, 4, false>(flashinfer::comm::pcie_ipc::PushOneshotParamData<__nv_bfloat16>)` | AllReduce | 1989.145 | 1989.483 | 14867.457 |
| 4 | `kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm120_b12xDenseGemmKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1409601_TVLayoutDst1409601_Valuetypef8E4M3FN_tensor00o11_0` | dense projection | 1950.007 | 1939.033 | 7812.608 |
| 5 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8>(cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8::Params)` | dense projection | 1213.371 | 1213.371 | 4811.891 |
| 6 | `void deep_gemm::sm120_tf32_hc_prenorm_gemm_impl<24u, 20480u, 128u, 32u, 64u, 16u, 4u, 256u, 128u>(unsigned int, CUtensorMap_st, CUtensorMap_st, float*, float*, long, long)` | dense projection | 1023.988 | 1015.851 | 4083.198 |
| 7 | `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | AllGather | 602.249 | 602.249 | 2447.759 |
| 8 | `mhc_pre_big_fuse_with_norm_tilelang_kernel` | norm/mhc | 359.261 | 362.595 | 1448.745 |
| 9 | `mhc_post_tilelang_kernel` | norm/mhc | 300.255 | 308.055 | 1260.390 |
| 10 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8>(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8::Params)` | dense projection | 300.191 | 300.191 | 1199.735 |
| 11 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_s161616gemm_bf16_32x32_128x2_tn_align8>(cutlass_80_wmma_tensorop_s161616gemm_bf16_32x32_128x2_tn_align8::Params)` | dense projection | 291.719 | 291.719 | 1166.049 |
| 12 | `void flashinfer::sparse_mla_sm120::sparse_mla_decode_dsv4_merge_kernel<16, 512, 64, 8>(__nv_bfloat16 const*, float const*, __nv_bfloat16*, float*, float const*, int, int, int, int, unsigned long)` | attention | 142.299 | 142.299 | 497.193 |
| 13 | `void deep_gemm::sm120_fp8_paged_mqa_logits<1u, 32u, 128u, 64u, true, false, 2u, 3u, 128u, 128u, 256u, float>(unsigned int, unsigned int, unsigned int, unsigned int const*, float*, unsigned int const*, unsigned int const*, unsigned int const*, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)` | attention | 134.524 | 134.524 | 534.574 |
| 14 | `void vllm::moe::moe_align_block_size_kernel<int>(int const*, int*, int*, int*, int*, int, int, int, int, unsigned long, int*, int, int, bool)` | routequantactivation | 129.831 | 129.831 | 515.563 |
| 15 | `void deep_gemm::sm120_tf32_hc_prenorm_gemm_impl<24u, 5120u, 128u, 32u, 64u, 16u, 4u, 256u, 128u>(unsigned int, CUtensorMap_st, CUtensorMap_st, float*, float*, long, long)` | dense projection | 127.236 | 4.544 | 440.751 |
| 16 | `_swiglu_limit_pad_aware_kernel` | routequantactivation | 121.757 | 121.757 | 435.530 |
| 17 | `_dsv4_topk_kernel` | routequantactivation | 117.500 | 96.599 | 470.250 |
| 18 | `void vllm::deepseek_v4_fused_ops::fusedDeepseekV4QNormRopeKVRopeQuantInsertKernel<c10::BFloat16, 16, false, false>(c10::BFloat16 const*, c10::BFloat16*, c10::BFloat16 const*, unsigned char*, long const*, long const*, float const*, float, int, int, int, int, int, bool, bool)` | kvmeta | 116.849 | 118.442 | 476.032 |
| 19 | `void vllm::moe::moe_sum_vec_kernel<c10::BFloat16, int, 6, true>(c10::BFloat16*, c10::BFloat16 const*, long, int, long, long, int const*, int const*, long, long)` | routequantactivation | 114.838 | 114.838 | 432.030 |
| 20 | `Kernel` | other | 114.037 | 135.391 | 546.843 |
| 21 | `_gumbel_sample_kernel` | other | 86.112 | 86.112 | 315.918 |
| 22 | `kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_at__tensorptrbf16gmemalign16o128012801_tensorptri8gmemalign16o128012801_tensorptri8gmem_0` | routequantactivation | 74.879 | 56.291 | 330.249 |
| 23 | `kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_at__tensorptrbf16gmemalign16o204820481_tensorptri8gmemalign16o204820481_tensorptri8gmem_0` | routequantactivation | 65.574 | 70.773 | 282.077 |
| 24 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, float, float, float, false, float, float, float, true, false, false, false>(cublasLt::cublasSplitKParams<float>, float const*, float const*, float*, float*, float const*, float const*, float const*, float const*, float*, void*, long, float*, int*, float*, float*, float const*, float const*, float const*, float const*, float const*)` | dense projection | 64.365 | 85.905 | 373.402 |
| 25 | `void vllm::persistent::persistent_topk_kernel<512, 4u>(vllm::persistent::PersistentTopKParams)` | other | 62.585 | 62.585 | 248.981 |
| 26 | `_compute_global_topk_indices_and_lens_kernel` | other | 61.887 | 61.887 | 245.848 |
| 27 | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<unsigned char>, std::array<char*, 1ul> >(int, at::native::FillFunctor<unsigned char>, std::array<char*, 1ul>)` | other | 59.460 | 59.460 | 236.954 |
| 28 | `kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_at__tensorptrbf16gmemalign16o512051201_tensorptri8gmemalign16o512051201_tensorptri8gmem_0` | routequantactivation | 57.863 | 55.172 | 517.571 |
| 29 | `ncclDevKernel_SendRecv(ncclDevKernelArgsStorage<4096ul>)` | PP | 51.388 | 51.388 | 30357.199 |
| 30 | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>)` | other | 50.183 | 50.183 | 199.306 |

## Stage graph statistics

See `stage_graph_stats.csv`; categories report raw kernel durations and mean per-rank graph spans.

