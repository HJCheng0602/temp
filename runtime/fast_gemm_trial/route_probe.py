"""Bounded B256 MoE routing-concentration check after serving tests finish."""

import json
import os
from pathlib import Path
import subprocess

import parallel_matrix as matrix

root = Path(__file__).resolve().parent
target = root / 'route_probe'
target.mkdir(exist_ok=False)
active = subprocess.check_output(
    ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
if active:
    raise RuntimeError(f'GPU process active: {active}')
env = matrix.environment()
env.update(CUDA_VISIBLE_DEVICES='0', MAX_JOBS='2', OMP_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
env.pop('DSV41_FAST_ENABLED', None)
rows = []
for name, experts in {'two_local': [0, 1, 96, 97, 192, 288],
                      'six_local': [0, 1, 2, 3, 4, 5]}.items():
    route = target / (name + '_routes.json')
    route.write_text(json.dumps(dict(topk_ids=[experts] * 256)))
    for backend in ('deepgemm', 'marlin'):
        output = target / f'{name}_{backend}.json'
        script = matrix.ROOT / 'gemm_candidates_20260927' / f'compare_{backend}_candidate.py'
        with output.with_suffix('.log').open('w') as log:
            subprocess.run([str(matrix.ROOT / 'env/bin/python'), str(script),
                            '--run', '--batches', '256', '--route-file', str(route),
                            '--output', str(output)], env=env, check=True, timeout=300,
                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        payload = json.loads(output.read_text())
        if len(payload['rows']) != 1 or payload['rows'][0]['status'] != 'ok':
            raise RuntimeError(f'Invalid route probe: {output}: {payload}')
        rows.append(dict(pattern=name, backend=backend, result=payload))
        print(json.dumps(rows[-1]), flush=True)
(target / 'summary.json').write_text(json.dumps(rows, indent=2))
print('ROUTE_PROBE_COMPLETE', flush=True)
