"""Run isolated GPU probes serially, retaining each failure as evidence."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import parallel_matrix as matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--families', nargs='+', default=['fp8', 'mhc', 'woa'])
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    env = matrix.environment()
    env.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MAX_JOBS='2',
               PYTHONPATH=os.pathsep.join((str(ROOT / 'allreduce_trial_20260930/deps'),
                                          str(ROOT), str(HERE))))
    python = str(ROOT / 'env/bin/python3.12')
    jobs = []
    for path in sorted(args.capture_dir.glob('*rank0*.json')):
        metadata = json.loads(path.read_text())
        kind = metadata['kind']
        if kind not in args.families:
            continue
        shape = metadata['args'][0]['shape']
        name = path.stem
        base = [python, str(HERE / f'{kind}_probe.py')]
        if kind == 'fp8':
            base += ['--capture-dir', str(args.capture_dir), '--shape',
                     str(shape[1]), str(metadata['args'][1]['shape'][1]),
                     '--backends', 'b12x', 'cutlass', 'cudnn']
            if shape[1] == 576:
                base.append('b12x_padded_k576')
        else:
            base += ['--input-capture', str(path.with_suffix('.pt'))]
        if kind == 'woa':
            base += ['--extension', str(ROOT / 'fast_gemm_trial/dsv4_wo_a_cublaslt.so')]
        if kind == 'mhc':
            base += ['--candidate', 'all']
        jobs.append((name, base))
    results = []
    for name, command in jobs:
        output = args.directory / (name + '.json')
        command += ['--output', str(output)]
        (args.directory / 'status.json').write_text(json.dumps(
            dict(current=name, completed=len(results), total=len(jobs))))
        start = time.time()
        with (args.directory / (name + '.log')).open('w') as log:
            try:
                proc = subprocess.run(command, env=env, stdout=log,
                                      stderr=subprocess.STDOUT, timeout=600)
                status = dict(returncode=proc.returncode)
            except subprocess.TimeoutExpired:
                status = dict(error='timeout')
        results.append(dict(name=name, command=command, seconds=time.time()-start,
                            output_exists=output.exists(), **status))
        (args.directory / 'jobs.json').write_text(json.dumps(results, indent=2))
        print(results[-1], flush=True)
    (args.directory / 'status.json').write_text(json.dumps(
        dict(phase='complete', jobs=len(results))))


if __name__ == '__main__':
    main()
