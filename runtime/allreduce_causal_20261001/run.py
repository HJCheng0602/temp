"""Bounded sequential experiments; only terminates child groups it creates."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/home/admin/dsv41-vllm-20260917')
OUT = ROOT / 'allreduce_causal_20261001'
OUT.mkdir(exist_ok=True)
env = {k: v for k, v in os.environ.items() if not k.startswith('NCCL_')}
env.update(CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           MAX_JOBS='2', CUDA_HOME='/usr/local/cuda', TORCH_CUDA_ARCH_LIST='12.0',
           PYTHONPATH=f'{ROOT}/allreduce_trial_20260930/deps:{ROOT}/vllm',
           FLASHINFER_AUTOTUNE_DIR=f'{ROOT}/allreduce_range_20261001/autotune4096',
           NCCL_P2P_LEVEL='PHB', NCCL_ALGO='allreduce:Ring', NCCL_PROTO='allreduce:LL',
           NCCL_DEBUG='INFO', NCCL_DEBUG_SUBSYS='INIT,GRAPH,TUNING,P2P,SHM')
env['PATH'] = '/usr/local/cuda/bin:' + env.get('PATH', '')
jobs = [('nccl_lengths', ['--backend', 'nccl']), ('ipc_lengths', ['--backend', 'ipc']),
        ('ipc_variants', ['--backend', 'ipc', '--variants']),
        ('nccl_profile', ['--backend', 'nccl', '--profile']),
        ('ipc_profile', ['--backend', 'ipc', '--profile']),
        ('nccl_domain0', ['--backend', 'nccl', '--profile']), ('probe', []), ('probe_batched', []),
        ('nccl_dump', ['--backend', 'nccl', '--dump']), ('ipc_dump', ['--backend', 'ipc', '--dump']),
        ('nccl_nomixing', ['--backend', 'nccl', '--profile']),
        ('nccl_nomixing_dump', ['--backend', 'nccl', '--dump'])]
if len(sys.argv) > 1:
    jobs = [j for j in jobs if j[0] in sys.argv[1:]]
status = []
for name, extra in jobs:
    job_env = env.copy()
    if name == 'nccl_domain0':
        job_env['NCCL_MEM_SYNC_DOMAIN'] = '0'
    if name.startswith('nccl_nomixing'):
        job_env['NCCL_GRAPH_MIXING_SUPPORT'] = '0'
    cmd = [str(ROOT / 'env/bin/python3.12'), '-m', 'torch.distributed.run', '--standalone',
           '--nproc-per-node=4', str(OUT / 'steady.py'), '--output', str(OUT / (name + '.json'))] + extra
    if name in ('probe', 'probe_batched'):
        base = '/home/admin/dsv41-existing-ops-20260917/env'
        job_env.update(CXX=f'{base}/bin/x86_64-conda-linux-gnu-c++',
                       CC=f'{base}/bin/x86_64-conda-linux-gnu-cc',
                       NVCC_PREPEND_FLAGS=f'-ccbin={base}/bin/x86_64-conda-linux-gnu-c++',
                       LD_LIBRARY_PATH=f'{base}/lib:' + env.get('LD_LIBRARY_PATH', ''))
        cmd = [str(ROOT / 'env/bin/python3.12'), '-m', 'torch.distributed.run', '--standalone',
               '--nproc-per-node=4', str(OUT / 'probe_driver.py'), '--output', str(OUT / (name + '_results.jsonl'))]
        if name == 'probe':
            cmd += ['--modes', '0', '1', '--protocols', '0', '1']
    with (OUT / (name + '.log')).open('w') as log:
        child = subprocess.Popen(cmd, env=job_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print('START', name, child.pid, flush=True)
        try:
            code = child.wait(timeout=480)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            code = -999
        status.append({'name': name, 'returncode': code, 'time': time.time()})
        (OUT / 'status.json').write_text(json.dumps(status, indent=2))
        print('END', name, code, flush=True)
        if code != 0:
            break
