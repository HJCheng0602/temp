"""Capture real B64 operands at the three non-Marlin kernel boundaries."""

from __future__ import annotations

import functools
import importlib.machinery
import json
import os
from pathlib import Path
import sys

COUNTS = set()
TARGETS = {
    "torch": ("bmm", "woa"),
    "vllm.utils.flashinfer": ("mm_mxfp8", "fp8"),
    "vllm.utils.deep_gemm": ("tf32_hc_prenorm_gemm", "mhc"),
}


def _patch(module):
    name, kind = TARGETS[module.__name__]
    original = getattr(module, name, None)
    if original is None or getattr(original, "_other_capture", False):
        return

    @functools.wraps(original)
    def call(*args, **kwargs):
        result = original(*args, **kwargs)
        arm = os.environ.get("DSV41_OTHER_CAPTURE_ARM")
        if not arm or not Path(arm).is_file():
            return result
        import torch
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_initialized() else -1
        if rank not in (0, 4) or torch.cuda.is_current_stream_capturing():
            return result
        a = args[0] if args else None
        if not isinstance(a, torch.Tensor):
            return result
        if kind == "woa":
            if tuple(a.shape) != (2, 64, 4096) or tuple(args[1].shape) != (2, 4096, 1024):
                return result
        elif a.ndim != 2 or a.shape[0] != 64:
            return result
        shape_key = "_".join("x".join(map(str, x.shape)) for x in args[:2])
        key = (kind, rank, shape_key)
        if key in COUNTS:
            return result
        COUNTS.add(key)

        def cpu(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu()
            if isinstance(value, tuple):
                return tuple(cpu(x) for x in value)
            if isinstance(value, list):
                return [cpu(x) for x in value]
            if isinstance(value, dict):
                return {k: cpu(v) for k, v in value.items()}
            return value

        def description(value):
            if isinstance(value, torch.Tensor):
                return dict(shape=list(value.shape), stride=list(value.stride()),
                            dtype=str(value.dtype), device=str(value.device))
            return repr(value)

        metadata = dict(kind=kind, rank=rank, callable=f"{module.__name__}.{name}",
                        args=[description(x) for x in args],
                        kwargs={k: description(v) for k, v in kwargs.items()})
        out = Path(os.environ["DSV41_OTHER_CAPTURE_DIR"])
        out.mkdir(parents=True, exist_ok=True)
        stem = f"{kind}_rank{rank}_{shape_key}"
        torch.save(dict(metadata=metadata, args=cpu(args), kwargs=cpu(kwargs),
                        output=cpu(result)), out / f"{stem}.pt")
        (out / f"{stem}.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print("DSV41_OTHER_INPUT_CAPTURED", stem, flush=True)
        return result

    call._other_capture = True
    setattr(module, name, call)
    print("DSV41_OTHER_CAPTURE_PATCH", module.__name__, name, flush=True)


def install():
    class Finder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in TARGETS:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return None
            original = spec.loader

            class Loader:
                def create_module(self, module_spec):
                    return original.create_module(module_spec) if hasattr(original, "create_module") else None

                def exec_module(self, module):
                    original.exec_module(module)
                    _patch(module)

            spec.loader = Loader()
            return spec

    sys.meta_path.insert(0, Finder())
    for name in TARGETS:
        if name in sys.modules:
            _patch(sys.modules[name])
