"""Install the measured policy while preserving the original CED worker."""
import hashlib
import json
from pathlib import Path
import shutil

trial = Path(__file__).resolve().parent
ced = Path('/home/admin/dsv41-vllm-20260917/ced_prefill')
worker = ced / 'worker.py'
original_sha = '5824364f31092f9077151d38c52258be3cc64e08cf0c1ac8932215c7215d6283'
anchor = ('            worker.initialize_from_config(cache_config)\n'
          '            torch.cuda.synchronize()\n')
hook = ('            if role == "prefill":\n'
        '                from prefill_dense import install as install_prefill_dense\n'
        '                install_prefill_dense()\n')
before = worker.read_bytes()
digest = hashlib.sha256(before).hexdigest()
backup = trial / 'backup'
backup.mkdir(exist_ok=True)
if hook not in before.decode():
    assert digest == original_sha, f'CED worker changed: {digest}; inspect before patching'
    assert before.decode().count(anchor) == 1
    shutil.copy2(worker, backup / 'worker.py')
    worker.write_text(before.decode().replace(anchor, anchor + hook))
else:
    assert hashlib.sha256((backup / 'worker.py').read_bytes()).hexdigest() == original_sha
for name in ('prefill_dense.py', 'prefill_dense_policy.json'):
    shutil.copy2(trial / name, ced / name)
evidence = {'before_sha256': digest,
            'after_sha256': hashlib.sha256(worker.read_bytes()).hexdigest(),
            'backup': str(backup / 'worker.py'),
            'policy': json.loads((ced / 'prefill_dense_policy.json').read_text())}
(trial / 'installation.json').write_text(json.dumps(evidence, indent=2))
print('INSTALLED', evidence['after_sha256'], flush=True)
