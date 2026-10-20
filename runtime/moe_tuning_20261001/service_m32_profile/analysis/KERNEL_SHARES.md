# Kernel Shares

Profile root: `/mnt/fuse/dsv41-tools-20260917/vllm/moe_tuning_20261001/service_m32_profile`. Batch: `64`.

Stage values are mean raw graph-kernel lifetimes per GPU per step, averaged over the four ranks in that PP stage. The category sum is a lifetime denominator; kernels may overlap, so it is not wall time, TPOT, or an eight-GPU sum.

## Stage Graph Kernel Shares

| Stage | Category | us / GPU / step | Share of stage lifetime |
|---:|---|---:|---:|
| 0 | dense projection | 3334.797 | 35.486% |
| 0 | AllReduce | 1950.493 | 20.755% |
| 0 | MoE | 1373.391 | 14.614% |
| 0 | attention | 1086.829 | 11.565% |
| 0 | routequantactivation | 559.088 | 5.949% |
| 0 | other | 345.513 | 3.677% |
| 0 | norm/mhc | 330.276 | 3.514% |
| 0 | engram | 237.249 | 2.525% |
| 0 | AllGather | 121.535 | 1.293% |
| 0 | kvmeta | 58.378 | 0.621% |
| 1 | dense projection | 2878.192 | 31.961% |
| 1 | MoE | 1811.208 | 20.113% |
| 1 | AllReduce | 1766.372 | 19.615% |
| 1 | attention | 1304.354 | 14.484% |
| 1 | routequantactivation | 546.632 | 6.070% |
| 1 | norm/mhc | 349.293 | 3.879% |
| 1 | other | 288.660 | 3.205% |
| 1 | kvmeta | 60.630 | 0.673% |

## Observed Dependency Path

These percentages use the observed dependency path from stage-0 graph activity to stage-1 sample readiness. They are not the same denominator as stage kernel lifetime shares.

| Category | Attributed us / step | Percent of dependency span |
|---|---:|---:|
| dense projection | 4995.063 | 28.381% |
| MoE | 4674.804 | 26.561% |
| attention | 2488.580 | 14.139% |
| AllReduce | 1989.145 | 11.302% |
| routequantactivation | 693.786 | 3.942% |
| unattributed gaps | 676.344 | 3.843% |
| norm/mhc | 661.802 | 3.760% |
| other | 618.999 | 3.517% |
| AllGather | 602.249 | 3.422% |
| kvmeta | 116.849 | 0.664% |
| PP | 51.388 | 0.292% |
| engram | 14.322 | 0.081% |
| memset | 8.551 | 0.049% |
| memcpy | 8.345 | 0.047% |

## Dependency Top 10

| # | Kernel | Category | Consumer us / step | Producer us / step |
|---:|---|---|---:|---:|
| 1 | `void marlin_moe_wna16::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 2814749767106568l, 128, 2, 8, 4, false, 4, 2, false>(int4 const*, int4 const*, int4*, int4*, int4 const*, float const*, int4 const*, float const*, int4 const*, int const*, int const*, int const*, float const*, int, bool, int, int, int, int*, bool, bool, bool)` | MoE | 4674.804 | 4674.804 |
| 2 | `void flashinfer::sparse_mla_sm120::sparse_mla_decode_dsv4_kernel<(ModelType)1, 16, 64>(__nv_bfloat16 const*, unsigned char const*, int const*, __nv_bfloat16*, float*, int const*, unsigned char const*, int const*, int const*, int, int, unsigned long, int, int, int, int, int, float, unsigned long, unsigned long, unsigned long)` | attention | 2210.245 | 2210.245 |
| 3 | `void flashinfer::comm::pcie_ipc::ipc_rsag_push_param_kernel<__nv_bfloat16, 4, false>(flashinfer::comm::pcie_ipc::PushOneshotParamData<__nv_bfloat16>)` | AllReduce | 1989.145 | 1989.483 |
| 4 | `kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm120_b12xDenseGemmKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1409601_TVLayoutDst1409601_Valuetypef8E4M3FN_tensor00o11_0` | dense projection | 1950.007 | 1939.033 |
| 5 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8>(cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8::Params)` | dense projection | 1213.371 | 1213.371 |
| 6 | `void deep_gemm::sm120_tf32_hc_prenorm_gemm_impl<24u, 20480u, 128u, 32u, 64u, 16u, 4u, 256u, 128u>(unsigned int, CUtensorMap_st, CUtensorMap_st, float*, float*, long, long)` | dense projection | 1023.988 | 1015.851 |
| 7 | `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | AllGather | 602.249 | 602.249 |
| 8 | `mhc_pre_big_fuse_with_norm_tilelang_kernel` | norm/mhc | 359.261 | 362.595 |
| 9 | `mhc_post_tilelang_kernel` | norm/mhc | 300.255 | 308.055 |
| 10 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8>(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8::Params)` | dense projection | 300.191 | 300.191 |

## Scope

The report does not make an end-to-end latency or hardware-counter claim. GEMM MFU, when present, is written separately by the existing GEMM analyzer with its own per-stage single-GPU denominator.
