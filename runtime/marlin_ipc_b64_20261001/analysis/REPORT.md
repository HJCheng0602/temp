# Marlin + PCIe-IPC B64 PP2 TP4 EP4

Nine stable replays from the single B64 case were analyzed across all eight ranks. The endpoint is the stage-1 sample-ready boundary (`sample_ends`), so the span starts at stage-0 graph activity and ends when stage 1 has sampled.

Mean observed dependency span: 19.098 ms. Mean unattributed gap: 0.707 ms. Category accounting closes at 100.000000%.

Raw graph-kernel duration is shown per rank and as an eight-rank sum. It is not wall time, and the eight-rank sum is not divided by TPOT. Stage graph spans are separate wall-clock intervals and must not be added to raw duration totals.

The result is an observed dependency path proxy with consumer-first and producer-first overlap allocations. Profiler overhead is present; this is not an unprofiled 19.180 ms measurement. Collective tails are rendezvous-to-completion proxies, not physical wire time.

## Category attribution

| Category | us/step | Percent of mean span |
|---|---:|---:|
| MoE | 6144.823 | 32.175900% |
| dense projection | 4987.857 | 26.117723% |
| attention | 2485.940 | 13.017033% |
| AllReduce | 1985.320 | 10.395657% |
| routequantactivation | 710.990 | 3.722930% |
| unattributed gaps | 706.527 | 3.699558% |
| norm/mhc | 658.821 | 3.449757% |
| other | 618.517 | 3.238717% |
| AllGather | 602.796 | 3.156395% |
| kvmeta | 116.633 | 0.610723% |
| PP | 48.275 | 0.252780% |
| engram | 14.233 | 0.074527% |
| memset | 8.572 | 0.044886% |
| memcpy | 8.291 | 0.043416% |

## Top 30

| # | Kernel | Category | Consumer us/step | Producer us/step | Raw all-rank us/step |
|---:|---|---|---:|---:|---:|
| 1 | `void marlin_moe_wna16::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 2814749767106568l, 128, 1, 8, 4, true, 4, 2, false>(int4 const*, int4 const*, int4*, int4*, int4 const*, float const*, int4 const*, float const*, int4 const*, int const*, int const*, int const*, float const*, int, bool, int, int, int, int*, bool, bool, bool)` | MoE | 6144.823 | 6144.823 | 15780.036 |
| 2 | `void flashinfer::sparse_mla_sm120::sparse_mla_decode_dsv4_kernel<(ModelType)1, 16, 64>(__nv_bfloat16 const*, unsigned char const*, int const*, __nv_bfloat16*, float*, int const*, unsigned char const*, int const*, int const*, int, int, unsigned long, int, int, int, int, int, float, unsigned long, unsigned long, unsigned long)` | attention | 2206.271 | 2206.271 | 7696.677 |
| 3 | `void flashinfer::comm::pcie_ipc::ipc_rsag_push_param_kernel<__nv_bfloat16, 4, false>(flashinfer::comm::pcie_ipc::PushOneshotParamData<__nv_bfloat16>)` | AllReduce | 1985.320 | 1985.637 | 17786.367 |
| 4 | `kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm120_b12xDenseGemmKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1409601_TVLayoutDst1409601_Valuetypef8E4M3FN_tensor00o11_0` | dense projection | 1953.606 | 1942.145 | 7814.332 |
| 5 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8>(cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8::Params)` | dense projection | 1212.722 | 1212.722 | 4806.070 |
| 6 | `void deep_gemm::sm120_tf32_hc_prenorm_gemm_impl<24u, 20480u, 128u, 32u, 64u, 16u, 4u, 256u, 128u>(unsigned int, CUtensorMap_st, CUtensorMap_st, float*, float*, long, long)` | dense projection | 1023.232 | 1015.438 | 4080.631 |
| 7 | `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | AllGather | 602.796 | 602.796 | 2427.146 |
| 8 | `mhc_pre_big_fuse_with_norm_tilelang_kernel` | norm/mhc | 359.614 | 363.140 | 1448.365 |
| 9 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8>(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8::Params)` | dense projection | 300.398 | 300.398 | 1199.716 |
| 10 | `mhc_post_tilelang_kernel` | norm/mhc | 296.892 | 304.369 | 1253.749 |
| 11 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_s161616gemm_bf16_32x32_128x2_tn_align8>(cutlass_80_wmma_tensorop_s161616gemm_bf16_32x32_128x2_tn_align8::Params)` | dense projection | 289.052 | 289.052 | 1157.395 |
| 12 | `void flashinfer::sparse_mla_sm120::sparse_mla_decode_dsv4_merge_kernel<16, 512, 64, 8>(__nv_bfloat16 const*, float const*, __nv_bfloat16*, float*, float const*, int, int, int, int, unsigned long)` | attention | 142.038 | 142.038 | 496.327 |
| 13 | `void vllm::moe::moe_align_block_size_kernel<int>(int const*, int*, int*, int*, int*, int, int, int, int, unsigned long, int*, int, int, bool)` | routequantactivation | 134.806 | 134.806 | 532.253 |
| 14 | `void deep_gemm::sm120_fp8_paged_mqa_logits<1u, 32u, 128u, 64u, true, false, 2u, 3u, 128u, 128u, 256u, float>(unsigned int, unsigned int, unsigned int, unsigned int const*, float*, unsigned int const*, unsigned int const*, unsigned int const*, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)` | attention | 134.645 | 134.645 | 535.103 |
| 15 | `_swiglu_limit_pad_aware_kernel` | routequantactivation | 128.559 | 128.559 | 473.963 |
| 16 | `void vllm::moe::moe_sum_vec_kernel<c10::BFloat16, int, 6, true>(c10::BFloat16*, c10::BFloat16 const*, long, int, long, long, int const*, int const*, long, long)` | routequantactivation | 120.946 | 120.946 | 440.712 |
| 17 | `void deep_gemm::sm120_tf32_hc_prenorm_gemm_impl<24u, 5120u, 128u, 32u, 64u, 16u, 4u, 256u, 128u>(unsigned int, CUtensorMap_st, CUtensorMap_st, float*, float*, long, long)` | dense projection | 120.629 | 4.512 | 431.566 |
| 18 | `_dsv4_topk_kernel` | routequantactivation | 116.746 | 96.210 | 468.541 |
| 19 | `void vllm::deepseek_v4_fused_ops::fusedDeepseekV4QNormRopeKVRopeQuantInsertKernel<c10::BFloat16, 16, false, false>(c10::BFloat16 const*, c10::BFloat16*, c10::BFloat16 const*, unsigned char*, long const*, long const*, float const*, float, int, int, int, int, int, bool, bool)` | kvmeta | 116.633 | 118.574 | 477.966 |
| 20 | `Kernel` | other | 114.150 | 135.627 | 547.597 |
| 21 | `_gumbel_sample_kernel` | other | 83.911 | 83.911 | 309.617 |
| 22 | `kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_at__tensorptrbf16gmemalign16o128012801_tensorptri8gmemalign16o128012801_tensorptri8gmem_0` | routequantactivation | 74.965 | 56.459 | 329.916 |
| 23 | `kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_at__tensorptrbf16gmemalign16o204820481_tensorptri8gmemalign16o204820481_tensorptri8gmem_0` | routequantactivation | 65.647 | 70.777 | 281.650 |
| 24 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, float, float, float, false, float, float, float, true, false, false, false>(cublasLt::cublasSplitKParams<float>, float const*, float const*, float*, float*, float const*, float const*, float const*, float const*, float*, void*, long, float*, int*, float*, float*, float const*, float const*, float const*, float const*, float const*)` | dense projection | 63.952 | 85.195 | 373.095 |
| 25 | `void vllm::persistent::persistent_topk_kernel<512, 4u>(vllm::persistent::PersistentTopKParams)` | other | 62.475 | 62.475 | 248.445 |
| 26 | `_compute_global_topk_indices_and_lens_kernel` | other | 62.095 | 62.095 | 246.794 |
| 27 | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<unsigned char>, std::array<char*, 1ul> >(int, at::native::FillFunctor<unsigned char>, std::array<char*, 1ul>)` | other | 59.480 | 59.480 | 237.066 |
| 28 | `kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_at__tensorptrbf16gmemalign16o512051201_tensorptri8gmemalign16o512051201_tensorptri8gmem_0` | routequantactivation | 57.816 | 54.823 | 517.300 |
| 29 | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>)` | other | 51.950 | 51.950 | 203.019 |
| 30 | `ncclDevKernel_SendRecv(ncclDevKernelArgsStorage<4096ul>)` | PP | 48.275 | 48.275 | 34218.025 |

## Stage graph statistics

See `stage_graph_stats.csv`; categories report raw kernel durations and mean per-rank graph spans.

