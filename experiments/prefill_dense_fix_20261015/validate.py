"""Exercise the real public FlashInfer dispatch before loading model weights."""
import gc
import json
import os
from pathlib import Path
import sys

root = Path('/home/admin/dsv41-vllm-20260917')
sys.path.insert(0, str(root))
import parallel_matrix
os.environ.update(parallel_matrix.environment())
os.environ.update(MAX_JOBS='2', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
sys.path.insert(0, '/home/admin/prefill_allreduce_20261013/deps')
sys.path.insert(0, str(root / 'ced_prefill'))
import torch
import flashinfer
from flashinfer.gemm.gemm_base import mm_mxfp8
from prefill_dense import install

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.cuda.set_device(0)
torch.manual_seed(20261015)
state = install()
rows = []
for entry in json.loads((root / 'ced_prefill/prefill_dense_policy.json').read_text())['entries']:
    m, n, k = entry['shape']
    af = torch.randn(m, k, device='cuda', dtype=torch.bfloat16) * .05
    wf = torch.randn(n, k, device='cuda', dtype=torch.bfloat16) * .05
    a, sa = flashinfer.mxfp8_quantize(af, is_sf_swizzled_layout=True)
    w, sw = flashinfer.mxfp8_quantize(wf, is_sf_swizzled_layout=True)
    state.configure(False)
    default = mm_mxfp8(a, w.T, sa, sw, out_dtype=torch.bfloat16, backend='cutlass').clone()
    old = state.snapshot()
    state.configure(True)
    repaired = mm_mxfp8(a, w.T, sa, sw, out_dtype=torch.bfloat16, backend='cutlass')
    new = state.snapshot()
    equal = torch.equal(default, repaired)
    row = {'name': entry['name'], 'equal': equal, 'default': old, 'repaired': new}
    assert equal and torch.isfinite(repaired).all(), row
    assert old['calls'] > 0 and old['selected_calls'] == 0, row
    assert new['selected_calls'] == new['calls'] > 0, row
    rows.append(row)
    print('VALIDATED', entry['name'], equal, new['shapes'], flush=True)
    del af, wf, a, sa, w, sw, default, repaired
    gc.collect()
    torch.cuda.empty_cache()
Path(__file__).with_name('validation.json').write_text(json.dumps(rows, indent=2))
print('VALIDATION_COMPLETE', flush=True)
