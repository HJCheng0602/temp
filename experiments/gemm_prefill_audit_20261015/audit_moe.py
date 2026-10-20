"""Controlled FP8xFP4 grouped GEMMs: distinguish useful work from M padding.

Uses installed vLLM scale packing and DeepGEMM wrappers. The 96 experts,
W13/W2 dimensions, recipes, and 61440-row workspace match the traced layer.
Routing is synthetic, explicitly recorded, and is not a claim about that trace.
Launch with the deployment LD_LIBRARY_PATH, as for audit_dense.py.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys

ROOT = Path('/home/admin/dsv41-vllm-20260917')
sys.path.insert(0, str(ROOT))
import parallel_matrix
os.environ.update(parallel_matrix.environment())
os.environ.update(MAX_JOBS='2', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
sys.path.insert(0, '/home/admin/prefill_allreduce_20261013/deps')
from audit_dense import time_graph, smi
import torch
from vllm.utils import deep_gemm as dg
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import _pack_deepgemm_mxfp4_scales


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='/home/admin/gemm_prefill_audit_20261015/moe.json')
    args = p.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    torch.manual_seed(20261015)
    torch.backends.cuda.matmul.allow_tf32 = False
    mod = dg._import_deep_gemm()
    print('DEEPGEMM', mod.__file__, flush=True)
    w1 = torch.randint(0, 256, (96, 4608, 2560), dtype=torch.uint8, device='cuda')
    w2 = torch.randint(0, 256, (96, 5120, 1152), dtype=torch.uint8, device='cuda')
    s1 = torch.full((96, 4608, 160), 120, dtype=torch.uint8, device='cuda')
    s2 = torch.full((96, 5120, 72), 120, dtype=torch.uint8, device='cuda')
    s1, s2 = _pack_deepgemm_mxfp4_scales(w1, w2, s1, s2)
    data = {'metadata': {'deepgemm': mod.__file__, 'smi_before': smi(),
        'workspace_rows': 61440, 'experts': 96, 'useful_rows': 12288,
        'scope': 'synthetic grouped GEMM only; same useful rows, controlled per-expert counts'},
        'cases': []}
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6],
                       device='cuda', dtype=torch.float32) / 128
    for name, n, k, w, sw in [('W13', 4608, 5120, w1, s1), ('W2', 5120, 2304, w2, s2)]:
        useful = torch.randn(12288, k, device='cuda', dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        for distribution, counts in [('balanced128', [128] * 96),
                                      ('jitter112144', [112, 144] * 48)]:
            for alignment in [128, 64]:
                padded = [((v + alignment - 1) // alignment) * alignment for v in counts]
                aq = torch.zeros(61440, k, device='cuda', dtype=torch.float8_e4m3fn)
                ids = torch.full((61440,), -1, device='cuda', dtype=torch.int32)
                starts = []
                dest = src = 0
                for expert, (count, extent) in enumerate(zip(counts, padded)):
                    aq[dest:dest+count].copy_(useful[src:src+count])
                    ids[dest:dest+count] = expert
                    starts.append((dest, src))
                    dest += extent
                    src += count
                sa = dg.get_mn_major_tma_aligned_packed_ue8m0_tensor(
                    torch.ones((61440, k // 128), device='cuda', dtype=torch.float32))
                out = torch.empty(61440, n, device='cuda', dtype=torch.bfloat16)

                def run():
                    return dg.m_grouped_fp8_fp4_gemm_nt_contiguous(
                        (aq, sa), (w.view(torch.int8), sw), out, ids,
                        recipe_a=(1, 128), recipe_b=(1, 32))

                case = {'name': name, 'distribution': distribution, 'alignment': alignment,
                    'counts': counts, 'useful_rows': sum(counts), 'padded_rows': sum(padded),
                    'workspace_rows': 61440, 'm_padding_factor': sum(padded) / sum(counts)}
                try:
                    with dg.mk_alignment_scope(alignment):
                        run()
                        torch.cuda.synchronize()
                        errs = []
                        for expert in [0, 31, 95]:
                            dest, src = starts[expert]
                            raw = w[expert]
                            wb = torch.stack((lut[(raw & 15).long()], lut[(raw >> 4).long()]), -1).flatten(1)
                            ref = useful[src:src+2].float() @ wb.T
                            got = out[dest:dest+2].float()
                            errs.append(float((got-ref).norm()/ref.norm()))
                            del wb, ref, got
                        case['relative_l2_checked_experts'] = errs
                        if max(errs) >= .006:
                            raise ValueError('independent reference relative L2 >= .006')
                        case.update(time_graph(run, repeats=8, trials=5))
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
                            run()
                            torch.cuda.synchronize()
                        prof.export_chrome_trace(str(output.parent / f'moe_{name}_{distribution}_{alignment}.json'))
                    case['useful_tflops'] = 2 * sum(counts) * n * k / (case['median_us'] * 1e6)
                    case['tile_padded_tflops'] = 2 * sum(padded) * n * k / (case['median_us'] * 1e6)
                    case['status'] = 'ok'
                except Exception as exc:
                    case.update(status='error', error=f'{type(exc).__name__}: {exc}')
                data['cases'].append(case)
                output.write_text(json.dumps(data, indent=2))
                print(name, distribution, alignment, case.get('status'),
                      round(case.get('median_us', 0), 2), round(case.get('useful_tflops', 0), 1),
                      'padding', case['m_padding_factor'], case.get('error', ''), flush=True)
                del aq, sa, ids, out
                gc.collect()
                torch.cuda.empty_cache()
        del useful
    data['metadata']['smi_after'] = smi()
    output.write_text(json.dumps(data, indent=2))
    print('COMPLETE', str(output), flush=True)


if __name__ == '__main__':
    main()
