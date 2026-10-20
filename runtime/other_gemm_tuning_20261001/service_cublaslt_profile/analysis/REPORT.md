# Marlin + PCIe-IPC B64 PP2 TP4 EP4

Nine stable replays from the single B64 case were analyzed across all eight ranks. The endpoint is the stage-1 sample-ready boundary (`sample_ends`), so the span starts at stage-0 graph activity and ends when stage 1 has sampled.

Mean observed dependency span: 17.343 ms. Mean unattributed gap: 0.672 ms. Category accounting closes at 100.000000%.

Raw graph-kernel duration is shown per rank and as an eight-rank sum. It is not wall time, and the eight-rank sum is not divided by TPOT. Stage graph spans are separate wall-clock intervals and must not be added to raw duration totals.

The result is an observed dependency path proxy with consumer-first and producer-first overlap allocations. Profiler overhead is present; this is not an unprofiled TPOT measurement. Collective tails are rendezvous-to-completion proxies, not physical wire time.

## Category attribution

| Category | us/step | Percent of mean span |
|---|---:|---:|
| dense projection | 4772.357 | 27.517779% |
| MoE | 4650.245 | 26.813675% |
| attention | 2486.302 | 14.336208% |
| AllReduce | 1985.754 | 11.450011% |
| routequantactivation | 693.520 | 3.998892% |
| unattributed gaps | 671.846 | 3.873918% |
| norm/mhc | 662.039 | 3.817371% |
| other | 622.907 | 3.591731% |
| AllGather | 602.508 | 3.474108% |
| kvmeta | 116.203 | 0.670034% |
| PP | 48.180 | 0.277809% |
| engram | 14.044 | 0.080981% |
| memset | 8.526 | 0.049161% |
| memcpy | 8.381 | 0.048323% |

## Top 30

| # | Kernel | Category | Consumer us/step | Producer us/step | Raw all-rank us/step |
|---:|---|---|---:|---:|---:|
| 1 | `void marlin_moe_wna16::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 2814749767106568l, 128, 2, 8, 4, false, 4, 2, false>(int4 const*, int4 const*, int4*, int4*, int4 const*, float const*, int4 const*, float const*, int4 const*, int const*, int const*, int const*, float const*, int, bool, int, int, int, int*, bool, bool, bool)` | MoE | 4650.245 | 4650.245 | 12687.098 |
| 2 | `void flashinfer::sparse_mla_sm120::sparse_mla_decode_dsv4_kernel<(ModelType)1, 16, 64>(__nv_bfloat16 const*, unsigned char const*, int const*, __nv_bfloat16*, float*, int const*, unsigned char const*, int const*, int const*, int, int, unsigned long, int, int, int, int, int, float, unsigned long, unsigned long, unsigned long)` | attention | 2209.299 | 2209.299 | 7698.033 |
| 3 | `void flashinfer::comm::pcie_ipc::ipc_rsag_push_param_kernel<__nv_bfloat16, 4, false>(flashinfer::comm::pcie_ipc::PushOneshotParamData<__nv_bfloat16>)` | AllReduce | 1985.754 | 1985.893 | 14841.402 |
| 4 | `kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm120_b12xDenseGemmKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1409601_TVLayoutDst1409601_Valuetypef8E4M3FN_tensor00o11_0` | dense projection | 1948.788 | 1938.100 | 7809.958 |
| 5 | `void deep_gemm::sm120_tf32_hc_prenorm_gemm_impl<24u, 20480u, 128u, 32u, 64u, 16u, 4u, 256u, 128u>(unsigned int, CUtensorMap_st, CUtensorMap_st, float*, float*, long, long)` | dense projection | 1024.564 | 1017.167 | 4082.823 |
| 6 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8::Params)` | dense projection | 981.007 | 981.007 | 3880.128 |
| 7 | `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | AllGather | 602.508 | 602.508 | 2427.595 |
| 8 | `mhc_pre_big_fuse_with_norm_tilelang_kernel` | norm/mhc | 359.779 | 363.210 | 1447.943 |
| 9 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8>(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8::Params)` | dense projection | 300.256 | 300.256 | 1200.423 |
| 10 | `mhc_post_tilelang_kernel` | norm/mhc | 299.953 | 307.211 | 1258.425 |
| 11 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_s161616gemm_bf16_32x32_128x2_tn_align8>(cutlass_80_wmma_tensorop_s161616gemm_bf16_32x32_128x2_tn_align8::Params)` | dense projection | 290.997 | 290.997 | 1164.391 |
| 12 | `void flashinfer::sparse_mla_sm120::sparse_mla_decode_dsv4_merge_kernel<16, 512, 64, 8>(__nv_bfloat16 const*, float const*, __nv_bfloat16*, float*, float const*, int, int, int, int, unsigned long)` | attention | 143.025 | 143.025 | 498.300 |
| 13 | `void deep_gemm::sm120_fp8_paged_mqa_logits<1u, 32u, 128u, 64u, true, false, 2u, 3u, 128u, 128u, 256u, float>(unsigned int, unsigned int, unsigned int, unsigned int const*, float*, unsigned int const*, unsigned int const*, unsigned int const*, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)` | attention | 133.977 | 133.977 | 533.910 |
| 14 | `void vllm::moe::moe_align_block_size_kernel<int>(int const*, int*, int*, int*, int*, int, int, int, int, unsigned long, int*, int, int, bool)` | routequantactivation | 129.763 | 129.763 | 515.469 |
| 15 | `_swiglu_limit_pad_aware_kernel` | routequantactivation | 121.568 | 121.568 | 434.179 |
| 16 | `void deep_gemm::sm120_tf32_hc_prenorm_gemm_impl<24u, 5120u, 128u, 32u, 64u, 16u, 4u, 256u, 128u>(unsigned int, CUtensorMap_st, CUtensorMap_st, float*, float*, long, long)` | dense projection | 121.227 | 4.551 | 420.097 |
| 17 | `_dsv4_topk_kernel` | routequantactivation | 117.933 | 97.421 | 469.845 |
| 18 | `void vllm::deepseek_v4_fused_ops::fusedDeepseekV4QNormRopeKVRopeQuantInsertKernel<c10::BFloat16, 16, false, false>(c10::BFloat16 const*, c10::BFloat16*, c10::BFloat16 const*, unsigned char*, long const*, long const*, float const*, float, int, int, int, int, int, bool, bool)` | kvmeta | 116.203 | 117.874 | 475.020 |
| 19 | `void vllm::moe::moe_sum_vec_kernel<c10::BFloat16, int, 6, true>(c10::BFloat16*, c10::BFloat16 const*, long, int, long, long, int const*, int const*, long, long)` | routequantactivation | 114.536 | 114.536 | 431.501 |
| 20 | `Kernel` | other | 114.025 | 135.168 | 546.295 |
| 21 | `_gumbel_sample_kernel` | other | 88.722 | 88.722 | 335.043 |
| 22 | `kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_at__tensorptrbf16gmemalign16o128012801_tensorptri8gmemalign16o128012801_tensorptri8gmem_0` | routequantactivation | 74.859 | 56.321 | 330.041 |
| 23 | `kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_at__tensorptrbf16gmemalign16o204820481_tensorptri8gmemalign16o204820481_tensorptri8gmem_0` | routequantactivation | 65.063 | 70.081 | 279.849 |
| 24 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, float, float, float, false, float, float, float, true, false, false, false>(cublasLt::cublasSplitKParams<float>, float const*, float const*, float*, float*, float const*, float const*, float const*, float const*, float*, void*, long, float*, int*, float*, float*, float const*, float const*, float const*, float const*, float const*)` | dense projection | 64.121 | 85.252 | 372.597 |
| 25 | `_compute_global_topk_indices_and_lens_kernel` | other | 62.993 | 62.993 | 249.896 |
| 26 | `void vllm::persistent::persistent_topk_kernel<512, 4u>(vllm::persistent::PersistentTopKParams)` | other | 62.666 | 62.666 | 250.109 |
| 27 | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<unsigned char>, std::array<char*, 1ul> >(int, at::native::FillFunctor<unsigned char>, std::array<char*, 1ul>)` | other | 59.531 | 59.531 | 236.832 |
| 28 | `kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_at__tensorptrbf16gmemalign16o512051201_tensorptri8gmemalign16o512051201_tensorptri8gmem_0` | routequantactivation | 58.117 | 55.343 | 517.749 |
| 29 | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>)` | other | 50.283 | 50.283 | 199.535 |
| 30 | `ncclDevKernel_SendRecv(ncclDevKernelArgsStorage<4096ul>)` | PP | 48.180 | 48.180 | 30139.241 |

## Stage graph statistics

See `stage_graph_stats.csv`; categories report raw kernel durations and mean per-rank graph spans.

