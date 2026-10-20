"""Run serial bounded controls for sparse-order and GEMM correctness."""

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--stage', choices=('smoke', 'full'), default='smoke')
    args = parser.parse_args()
    root = Path('/home/admin/dsv41-vllm-20260917')
    work = args.directory
    work.mkdir(parents=True, exist_ok=False)
    workload = json.loads((root / 'precision_validation_20260928/workload.json').read_text())
    if args.stage == 'smoke':
        workload = copy.deepcopy(workload)
        workload['cases'] = [c for c in workload['cases']
                             if (c['batch'] == 1 and len(c['requests'][0]['prompt_token_ids']) in (64, 511, 2048))
                             or c['batch'] == 8]
        for case in workload['cases']:
            case['output_tokens'] = 32
        assert len(workload['cases']) == 4
    path = work / 'workload.json'
    path.write_text(json.dumps(workload, indent=2))
    sequence = [('original', 'baseline'), ('canonical', 'baseline')] if args.stage == 'smoke' else [
        ('canonical', 'baseline'), ('canonical', 'dense'), ('canonical', 'all')]
    for order, mode in sequence:
        directory = work / f'{order}_{mode}'
        env = os.environ.copy()
        env.update(DSV41_SPARSE_ORDER=order, DSV41_SPARSE_AUDIT='1',
                   DSV41_MOE_CAPTURE='1' if args.stage == 'full' else '0',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   PYTHONPATH=str(root / 'fast_gemm_trial') + ':' + str(root))
        command = [sys.executable, str(root / 'fast_gemm_trial/run_precision.py'),
                   '--mode', mode, '--directory', str(directory), '--workload', str(path),
                   '--free-repeat']
        if mode != 'baseline':
            command += ['--reference', str(work / 'canonical_baseline')]
        with (work / f'{order}_{mode}.log').open('w') as stream:
            subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                           check=True, timeout=7500)
        result = json.loads((directory / ('repeat_logits.json' if mode == 'baseline'
                                         else 'logits_comparison.json')).read_text())
        print(json.dumps(dict(order=order, mode=mode, rows=result['all_decode_rows'],
                              argmax_equal=result['raw_argmax_equal'],
                              kl=result['metrics']['kl'], max_abs=result['metrics']['max_abs'])), flush=True)
        deadline = time.monotonic() + 120
        while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                       '--format=csv,noheader'], text=True).strip():
            if time.monotonic() > deadline:
                raise RuntimeError('GPU workers did not exit after the completed run')
            time.sleep(2)
    (work / 'COMPLETE.json').write_text(json.dumps(dict(completed=True)))


if __name__ == '__main__':
    main()
