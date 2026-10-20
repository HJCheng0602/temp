"""Run isolated four-GPU comparisons sequentially, retaining every command."""

import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path('/home/admin/dsv41-vllm-20260917')
OUT = ROOT / 'allreduce_range_20261001'
ALL = '1,2,4,8,16,24,32,48,64,96,128,192,256,384,512,768,1024,1536,2048,3072,4096'
SELECT = '1,8,32,64,128,256,512,1024,4096'
SMALL = '1,2,4,8,16,32,64,128,256'


def run(name, batches, overrides=None, backends=('nccl',), devices='0,1,2,3',
        profile=False, capacity=4096, tune=False):
    env = os.environ.copy()
    for key in list(env):
        if key.startswith('NCCL_'):
            del env[key]
    env.update(CUDA_VISIBLE_DEVICES=devices, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONPATH=str(ROOT / 'allreduce_trial_20260930/deps') + ':' + str(ROOT / 'vllm'),
               FLASHINFER_AUTOTUNE_DIR=str(OUT / ('autotune' + str(capacity))))
    env.update(overrides or {})
    if profile:
        env.update(NCCL_DEBUG='INFO', NCCL_DEBUG_SUBSYS='INIT,ENV,GRAPH,TUNING,P2P,SHM',
                   NCCL_DEBUG_FILE=str(OUT / (name + '.%h.%p.nccl.log')),
                   NCCL_TOPO_DUMP_FILE=str(OUT / (name + '.topology.xml')))
    cmd = [str(ROOT / 'env/bin/python3.12'), '-m', 'torch.distributed.run',
           '--standalone', '--nproc-per-node=4', str(OUT / 'microbench.py'),
           '--output', str(OUT / (name + '.json')), '--batches', batches,
           '--workspace-max-batch', str(capacity), '--repeats', '7',
           '--backends', *backends]
    if profile:
        cmd += ['--profile-batches', batches]
    if tune:
        cmd += ['--tune']
    record = {'name': name, 'command': cmd, 'env': {k: v for k, v in env.items()
              if k.startswith(('NCCL_', 'CUDA_', 'FLASHINFER_', 'OMP_', 'MKL_'))},
              'started': time.time()}
    (OUT / (name + '.launch.json')).write_text(json.dumps(record, indent=2))
    with (OUT / (name + '.log')).open('w') as log:
        child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        record['pid'] = child.pid
        (OUT / 'active.json').write_text(json.dumps(record, indent=2))
        try:
            record['exit_code'] = child.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            record['exit_code'] = 'timeout'
    record['finished'] = time.time()
    (OUT / (name + '.launch.json')).write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)
    if record['exit_code'] != 0:
        raise RuntimeError(f'{name} failed: inspect its log before continuing')


def main():
    run('default_profile', SELECT, backends=('nccl', 'ipc'), profile=True)
    run('phb_profile', SELECT, {'NCCL_P2P_LEVEL': 'PHB'}, profile=True)
    run('p2p_profile', SELECT, {'NCCL_P2P_LEVEL': 'SYS'}, profile=True)
    run('p2p_range', ALL, {'NCCL_P2P_LEVEL': 'SYS'})
    run('capacity256', SMALL, backends=('nccl', 'ipc'), capacity=256, tune=True)
    run('numa1_range', ALL, backends=('nccl', 'ipc'), devices='4,5,6,7', tune=True)
    run('numa1_p2p', SELECT, {'NCCL_P2P_LEVEL': 'SYS'}, devices='4,5,6,7')
    for algo in ('Ring', 'Tree'):
        for proto in ('Simple', 'LL'):
            run(f'p2p_{algo}_{proto}', SELECT, {'NCCL_P2P_LEVEL': 'SYS',
                'NCCL_ALGO': algo, 'NCCL_PROTO': proto})
    for ctas in (1, 2, 8):
        run(f'p2p_ring_simple_ctas{ctas}', SELECT, {'NCCL_P2P_LEVEL': 'SYS',
            'NCCL_ALGO': 'Ring', 'NCCL_PROTO': 'Simple',
            'NCCL_MIN_CTAS': str(ctas), 'NCCL_MAX_CTAS': str(ctas)})
    (OUT / 'diagnostics.completed').write_text(str(time.time()))
    (OUT / 'active.json').write_text(json.dumps({'state': 'completed'}))


if __name__ == '__main__':
    main()
