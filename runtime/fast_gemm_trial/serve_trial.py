"""Launch isolated fast GEMM trials with the verified PP receive fix."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import parallel_matrix as matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('baseline', 'dense', 'all'), required=True)
    parser.add_argument('--timing-dir', type=Path, required=True)
    parser.add_argument('--fixed-batch-control', type=Path, required=True)
    parser.add_argument('--enable-trace-replay', action='store_true')
    args = parser.parse_args()
    runtime = Path(importlib.metadata.distribution('vllm').locate_file('vllm'))
    manifest = json.loads((matrix.ROOT / 'pp_deferred_fix/manifest.json').read_text())
    for relative, hashes in manifest['files'].items():
        if hashlib.sha256((runtime / relative).read_bytes()).hexdigest() != hashes['after']:
            raise RuntimeError(f'Runtime differs from verified fix: {relative}')
    strategy = dict(name='pp2_tp4_ep4', pp=2, tp=4, dp=1, ep=True, pp_layers=[20, 20])
    command = matrix.command(strategy, matrix.ROOT)
    command.remove('--enforce-eager')
    index = command.index('--profiler-config')
    del command[index:index + 2]
    for key, value in {'--port': 8024, '--max-model-len': 4096,
                       '--max-num-seqs': 256, '--max-num-batched-tokens': 8192,
                       '--gpu-memory-utilization': .85}.items():
        command[command.index(key) + 1] = str(value)
    command += ['--engram-config', '{"cpu_offload":true}', '--async-scheduling',
                '--compilation-config', json.dumps(dict(
                    mode=0, cudagraph_mode='FULL_DECODE_ONLY',
                    cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32, 64, 96,
                                             120, 128, 136, 144, 160, 192, 224, 256])),
                '--scheduler-cls',
                'precision_prefill_scheduler.PrecisionPrefillScheduler'
                if os.environ.get('DSV41_PRECISION_PREFILL_BARRIER') == '1'
                else 'pp2_fixed_batch_scheduler.FixedBatchScheduler']
    if args.mode == 'all':
        command += ['--kernel-config', '{"moe_backend":"marlin"}']
    if args.enable_trace_replay:
        command += ['--enable-trace-replay']
    env = matrix.environment(strategy)
    for key in list(env):
        if key.startswith(('DSV41_WAVE_', 'DSV41_PP_', 'DSV41_GRAPH_PROBE_', 'DSV41_FAST_')):
            env.pop(key)
    for key in ('CUDA_DEVICE_MAX_CONNECTIONS', 'CUDA_DEVICE_MAX_COPY_CONNECTIONS',
                'NCCL_GRAPH_MIXING_SUPPORT', 'NCCL_GRAPH_STREAM_ORDERING', 'VLLM_MARLIN_INPUT_DTYPE'):
        env.pop(key, None)
    trial = Path(__file__).resolve().parent
    env.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', MAX_JOBS='2',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               DSV41_PP_TIMING_DIR=str(args.timing_dir), DSV41_PP_TIMING_PER_RANK='1',
               DSV41_WAVE_BATCH='1', DSV41_WAVE_SIZE='1',
               DSV41_WAVE_CONTROL=str(args.fixed_batch_control),
               PYTHONPATH=os.pathsep.join((str(trial), str(matrix.ROOT), env.get('PYTHONPATH', ''))))
    if args.mode != 'baseline':
        env['DSV41_FAST_ENABLED'] = '1'
        env['DSV41_FAST_EXTENSION'] = str(trial / 'dsv4_wo_a_cublaslt.so')
    print(json.dumps(dict(mode=args.mode, strategy=strategy, command=command)), flush=True)
    os.execve(command[0], command, env)


if __name__ == '__main__':
    main()
