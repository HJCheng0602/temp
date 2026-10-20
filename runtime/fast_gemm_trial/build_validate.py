"""Compile once, then validate the exact strided wo_a path before deployment."""

import json
import os
from pathlib import Path
import shutil
import statistics
import sys

import parallel_matrix as matrix

os.environ.update(matrix.environment())
os.environ.update(MAX_JOBS='2', TORCH_CUDA_ARCH_LIST='12.0', OMP_NUM_THREADS='1')
os.environ.pop('DSV41_FAST_ENABLED', None)

import torch
from torch.utils.cpp_extension import load

root = Path(__file__).resolve().parent
cuda = Path(os.environ['CUDA_HOME'])
library = next(path for directory in ('lib', 'lib64')
               for path in sorted((cuda / directory).glob('libcublasLt.so*')) if path.is_file())
extension = load(name='dsv4_wo_a_cublaslt', sources=[str(root / 'cublaslt_fast.cu')],
                 extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr'],
                 extra_ldflags=[str(library)], verbose=True)
shutil.copy2(extension.__file__, root / 'dsv4_wo_a_cublaslt.so')
if '--compile-only' in sys.argv:
    print('BUILD_COMPLETE', flush=True)
    raise SystemExit(0)
torch.set_num_threads(1)
torch.manual_seed(1729)
rows = []

def timing(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(20):
            fn()
    values = []
    for _ in range(7):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000 / 20)
    return statistics.median(values)

for batch in (128, 136, 144, 160, 192, 224, 256):
    a = (torch.randn((batch, 2, 4096), device='cuda', dtype=torch.bfloat16) * .05).transpose(0, 1)
    b = (torch.randn((2, 1024, 4096), device='cuda', dtype=torch.bfloat16) * .05).transpose(1, 2)
    out = torch.empty((batch, 2, 1024), device='cuda', dtype=torch.bfloat16).transpose(0, 1)
    assert a.stride() == (4096, 8192, 1)
    assert b.stride() == (4194304, 1, 4096)
    assert out.stride() == (1024, 2048, 1)
    ref = torch.bmm(a.float(), b.float())
    extension.bmm_out(a, b, out, 1)
    torch.testing.assert_close(out.float(), ref, rtol=.02, atol=.02)
    relative_l2 = float((out.float() - ref).norm() / ref.norm())
    rows.append(dict(batch=batch, relative_l2=relative_l2,
                     baseline_us=timing(lambda: torch.bmm(a, b, out=out)),
                     candidate_us=timing(lambda: extension.bmm_out(a, b, out, 1)),
                     strides=[list(x.stride()) for x in (a, b, out)]))
    print(json.dumps(rows[-1]), flush=True)
(root / 'wo_a_validation.json').write_text(json.dumps(rows, indent=2))
print('BUILD_VALIDATION_COMPLETE', flush=True)
