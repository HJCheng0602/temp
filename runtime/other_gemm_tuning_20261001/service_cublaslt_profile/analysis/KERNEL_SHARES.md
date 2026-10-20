# Kernel Shares

Profile root: `/mnt/fuse/dsv41-tools-20260917/vllm/other_gemm_tuning_20261001/service_cublaslt_profile`. Batch: `64`.

Stage values are mean raw graph-kernel lifetimes per GPU per step, averaged over the four ranks in that PP stage. The category sum is a lifetime denominator; kernels may overlap, so it is not wall time, TPOT, or an eight-GPU sum.

## Stage Graph Kernel Shares

| Stage | Category | us / GPU / step | Share of stage lifetime |
|---:|---|---:|---:|
| 0 | dense projection | 3247.940 | 34.921% |
| 0 | AllReduce | 1952.167 | 20.989% |
| 0 | MoE | 1378.856 | 14.825% |
| 0 | attention | 1086.502 | 11.682% |
| 0 | routequantactivation | 563.827 | 6.062% |
| 0 | other | 341.146 | 3.668% |
| 0 | norm/mhc | 330.809 | 3.557% |
| 0 | engram | 224.010 | 2.408% |
| 0 | AllGather | 117.313 | 1.261% |
| 0 | kvmeta | 58.354 | 0.627% |
| 1 | dense projection | 2791.697 | 31.391% |
| 1 | MoE | 1792.918 | 20.161% |
| 1 | AllReduce | 1758.183 | 19.770% |
| 1 | attention | 1305.148 | 14.676% |
| 1 | routequantactivation | 547.681 | 6.158% |
| 1 | norm/mhc | 348.076 | 3.914% |
| 1 | other | 289.089 | 3.251% |
| 1 | kvmeta | 60.401 | 0.679% |

## Observed Dependency Path

These percentages use the observed dependency path from stage-0 graph activity to stage-1 sample readiness. They are not the same denominator as stage kernel lifetime shares.

| Category | Attributed us / step | Percent of dependency span |
|---|---:|---:|
| dense projection | 4772.357 | 27.518% |
| MoE | 4650.245 | 26.814% |
| attention | 2486.302 | 14.336% |
| AllReduce | 1985.754 | 11.450% |
| routequantactivation | 693.520 | 3.999% |
| unattributed gaps | 671.846 | 3.874% |
| norm/mhc | 662.039 | 3.817% |
| other | 622.907 | 3.592% |
| AllGather | 602.508 | 3.474% |
| kvmeta | 116.203 | 0.670% |
| PP | 48.180 | 0.278% |
| engram | 14.044 | 0.081% |
| memset | 8.526 | 0.049% |
| memcpy | 8.381 | 0.048% |

## Dependency Top 10

| # | Kernel | Category | Consumer us / step | Producer us / step |
|---:|---|---|---:|---:|
| 1 | `void marlin_moe_wna16::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 2814749767106568l, 128, 2, 8, 4, false, 4, 2, false>(int4 const*, int4 const*, int4*, int4*, int4 const*, float const*, int4 const*, float const*, int4 const*, int const*, int const*, int const*, float const*, int, bool, int, int, int, int*, bool, bool, bool)` | MoE | 4650.245 | 4650.245 |
| 2 | `void flashinfer::sparse_mla_sm120::sparse_mla_decode_dsv4_kernel<(ModelType)1, 16, 64>(__nv_bfloat16 const*, unsigned char const*, int const*, __nv_bfloat16*, float*, int const*, unsigned char const*, int const*, int const*, int, int, unsigned long, int, int, int, int, int, float, unsigned long, unsigned long, unsigned long)` | attention | 2209.299 | 2209.299 |
| 3 | `void flashinfer::comm::pcie_ipc::ipc_rsag_push_param_kernel<__nv_bfloat16, 4, false>(flashinfer::comm::pcie_ipc::PushOneshotParamData<__nv_bfloat16>)` | AllReduce | 1985.754 | 1985.893 |
| 4 | `kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm120_b12xDenseGemmKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1409601_TVLayoutDst1409601_Valuetypef8E4M3FN_tensor00o11_0` | dense projection | 1948.788 | 1938.100 |
| 5 | `void deep_gemm::sm120_tf32_hc_prenorm_gemm_impl<24u, 20480u, 128u, 32u, 64u, 16u, 4u, 256u, 128u>(unsigned int, CUtensorMap_st, CUtensorMap_st, float*, float*, long, long)` | dense projection | 1024.564 | 1017.167 |
| 6 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8::Params)` | dense projection | 981.007 | 981.007 |
| 7 | `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | AllGather | 602.508 | 602.508 |
| 8 | `mhc_pre_big_fuse_with_norm_tilelang_kernel` | norm/mhc | 359.779 | 363.210 |
| 9 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8>(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8::Params)` | dense projection | 300.256 | 300.256 |
| 10 | `mhc_post_tilelang_kernel` | norm/mhc | 299.953 | 307.211 |

## Scope

The report does not make an end-to-end latency or hardware-counter claim. GEMM MFU, when present, is written separately by the existing GEMM analyzer with its own per-stage single-GPU denominator.
