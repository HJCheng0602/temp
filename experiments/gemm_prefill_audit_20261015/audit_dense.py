"""Isolate the five observed 8K MXFP8 GEMMs and sweep installed CUTLASS tactics.

Run only on the authorized SM120 developer machine. No model weights are loaded.
The same quantized operands and BF16 output are used by all tactics; timing
excludes quantization, allocation, correctness checks, and communication.
"""
import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path('/home/admin/dsv41-vllm-20260917')
DEPS = '/home/admin/prefill_allreduce_20261013/deps'
sys.path.insert(0, str(ROOT))
import parallel_matrix
os.environ.update(parallel_matrix.environment())
os.environ.update(MAX_JOBS='2', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  MKL_NUM_THREADS='1', PYTHONPATH=DEPS)
sys.path.insert(0, DEPS)

import torch
import flashinfer
from flashinfer.gemm import gemm_base as gb

SHAPES = [('WqaWkv', 1792, 5120), ('Wqb', 8192, 1280),
          ('Wo_b', 5120, 2048), ('SharedUp', 1152, 5120),
          ('SharedDown', 5120, 576)]


def smi():
    return subprocess.run(['nvidia-smi', '-i', '0',
        '--query-gpu=name,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu',
        '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()


def time_graph(fn, repeats=16, trials=7):
    for _ in range(4):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(trials):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000 / repeats)
    return {'median_us': statistics.median(samples), 'samples_us': samples,
            'repeats': repeats, 'trials': trials}


def dequant(codes, scales):
    scales = scales.reshape(codes.shape[0], codes.shape[1] // 32)
    if scales.dtype == torch.uint8:
        scales = torch.ldexp(torch.ones_like(scales, dtype=torch.float32), scales.int() - 127)
    return codes.float() * scales.float().repeat_interleave(32, dim=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='/home/admin/gemm_prefill_audit_20261015/dense.json')
    p.add_argument('--shape', type=int, nargs='+', default=list(range(5)))
    p.add_argument('--m', type=int, default=8192)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--tactic', type=int, default=-1)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    torch.manual_seed(20261015)
    torch.backends.cuda.matmul.allow_tf32 = False
    mod = gb._load_gemm_sm120_mxfp8_module()
    nt = int(mod.mxfp8_gemm_tactic_num())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source = inspect.getsource(gb.gen_gemm_sm120_module_cutlass_mxfp8)
    (output.parent / 'jit_generator.py.txt').write_text(source)
    (output.parent / 'runner.py.txt').write_text(inspect.getsource(gb._create_cutlass_mxfp8_gemm_module))
    props = torch.cuda.get_device_properties(0)
    data = {'metadata': {'gpu': props.name, 'sm_count': props.multi_processor_count,
        'torch': torch.__version__, 'flashinfer': flashinfer.__version__,
        'flashinfer_path': flashinfer.__file__, 'cuda_home': os.environ.get('CUDA_HOME'),
        'smi_before': smi(), 'tactics': nt, 'm': args.m,
        'scope': 'synthetic same-shape GEMM-only; quantization excluded; one idle GPU',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, 'cases': []}
    for idx in args.shape:
        name, n, k = SHAPES[idx]
        m = args.m
        a0 = torch.randn(m, k, device='cuda', dtype=torch.bfloat16) * 0.05
        w0 = torch.randn(n, k, device='cuda', dtype=torch.bfloat16) * 0.05
        a, sa = flashinfer.mxfp8_quantize(a0, is_sf_swizzled_layout=True)
        w, sw = flashinfer.mxfp8_quantize(w0, is_sf_swizzled_layout=True)
        out = torch.empty(m, n, device='cuda', dtype=torch.bfloat16)
        ws = torch.empty(32 * 1024 * 1024, device='cuda', dtype=torch.uint8)

        def run(tactic):
            mod.mxfp8_gemm(a, w, sa, sw, out, ws, tactic)
            return out

        if args.profile:
            for _ in range(5):
                run(args.tactic)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            run(args.tactic)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            print('PROFILE_DONE', name, args.tactic, flush=True)
            continue

        ac, asc = flashinfer.mxfp8_quantize(a0, is_sf_swizzled_layout=False)
        wc, wsc = flashinfer.mxfp8_quantize(w0, is_sf_swizzled_layout=False)
        rows = torch.arange(0, m, max(1, m // 32), device='cuda')[:32]
        ref = dequant(ac[rows], asc.reshape(m, k // 32)[rows]) @ dequant(wc, wsc).T
        baseline = run(-1).clone()
        torch.cuda.synchronize()
        case = {'name': name, 'm': m, 'n': n, 'k': k,
                'quantized_input_bytes': a.numel() + w.numel() + sa.numel() + sw.numel(),
                'rows_checked': rows.cpu().tolist(), 'tactics': []}
        for tactic in [-1] + list(range(nt)):
            row = {'tactic': tactic}
            try:
                run(tactic)
                torch.cuda.synchronize()
                delta = out[rows].float() - ref
                row['relative_l2_vs_dequant_fp32'] = float(delta.norm() / ref.norm())
                row['max_abs_vs_dequant_fp32'] = float(delta.abs().max())
                row['relative_l2_vs_default'] = float((out.float() - baseline.float()).norm() /
                                                     baseline.float().norm())
                if row['relative_l2_vs_dequant_fp32'] >= 0.006:
                    raise ValueError('independent-reference relative L2 >= 0.006')
                row.update(time_graph(lambda: run(tactic)))
                row['tflops'] = 2 * m * n * k / (row['median_us'] * 1e6)
                row['status'] = 'ok'
                row['gpu_state'] = smi()
            except Exception as exc:
                row.update(status='error', error=f'{type(exc).__name__}: {exc}')
            case['tactics'].append(row)
            print(name, tactic, row.get('status'), round(row.get('median_us', 0), 3),
                  round(row.get('tflops', 0), 2), flush=True)
        data['cases'].append(case)
        output.write_text(json.dumps(data, indent=2))
        del a0, w0, a, sa, w, sw, out, ws, ac, asc, wc, wsc, ref, baseline, rows
        gc.collect()
        torch.cuda.empty_cache()
    data['metadata']['smi_after'] = smi()
    output.write_text(json.dumps(data, indent=2))
    print('COMPLETE', str(output), flush=True)


if __name__ == '__main__':
    main()
