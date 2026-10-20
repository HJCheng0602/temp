"""Opt-in B64 wo_a cuBLASLt hook with lazy loading and counters."""

from __future__ import annotations

import atexit
import importlib.util
import os
from pathlib import Path
import sys
import types
from typing import Any


_EXTENSION = None
_WARMED = False
_MODE = os.environ.get("DSV41_WOA_MODE", "mm_full_accum")
if _MODE not in ("cublaslt", "mm_full_accum"):
    raise RuntimeError(f"DSV41_WOA_MODE must be cublaslt or mm_full_accum, got {_MODE!r}")
_COUNTS = {"mode": _MODE, "patched": 0, "warmup": 0, "hit": 0,
           "fallback": 0, "error": 0}
_PATCHED = False


def _extension_path() -> Path:
    value = os.environ.get("DSV41_WOA_EXTENSION")
    if value:
        return Path(value)
    return Path(__file__).resolve().parents[1] / "fast_gemm_trial" / "dsv4_wo_a_cublaslt.so"


def _load_extension() -> Any:
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    path = _extension_path()
    spec = importlib.util.spec_from_file_location("dsv4_wo_a_cublaslt", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load wo_a extension: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "bmm_out"):
        raise RuntimeError(f"wo_a extension has no bmm_out: {path}")
    _EXTENSION = module
    return module


def _warm_b64(extension: Any) -> None:
    import torch

    device = torch.device("cuda", torch.cuda.current_device())
    a = torch.zeros((2, 64, 4096), device=device, dtype=torch.bfloat16)
    b = torch.empty_strided((2, 4096, 1024), (4194304, 1, 4096),
                            device=device, dtype=torch.bfloat16)
    b.zero_()
    out = torch.empty_strided((2, 64, 1024), (1024, 2048, 1),
                              device=device, dtype=torch.bfloat16)
    extension.bmm_out(a, b, out, 1)
    torch.cuda.synchronize(device)
    _COUNTS["warmup"] += 1
    print("DSV41_WOA_WARM", {"batch": 64, "a_stride": list(a.stride()),
                              "b_stride": list(b.stride()),
                              "out_stride": list(out.stride())}, file=sys.stderr,
          flush=True)


def _ensure_warmed() -> None:
    global _WARMED
    if _WARMED:
        return
    import torch
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("wo_a cuBLASLt plan was not warmed before CUDA graph capture")
    _warm_b64(_load_extension())
    _WARMED = True


def _eligible(a: Any, b: Any, out: Any) -> bool:
    return (getattr(a, "ndim", 0) == 3 and tuple(a.shape) == (2, 64, 4096)
            and tuple(b.shape) == (2, 4096, 1024)
            and a.dtype == b.dtype and str(a.dtype) == "torch.bfloat16"
            and tuple(a.stride()) == (262144, 4096, 1)
            and tuple(b.stride()) == (4194304, 1, 4096)
            and out is not None and tuple(out.shape) == (2, 64, 1024)
            and out.dtype == a.dtype and out.stride(2) == 1
            and out.stride(1) >= 1024)


def _wrap_bmm(original):
    def bmm(a, b, *, out=None):
        if not _eligible(a, b, out):
            _COUNTS["fallback"] += 1
            return original(a, b, out=out)
        try:
            if not _COUNTS["hit"]:
                import torch
                print("DSV41_WOA_FIRST_HIT", dict(
                    mode=_MODE, a_shape=list(a.shape), a_stride=list(a.stride()),
                    b_stride=list(b.stride()), out_stride=list(out.stride()),
                    capturing=torch.cuda.is_current_stream_capturing()),
                    file=sys.stderr, flush=True)
            if _MODE == "mm_full_accum":
                import torch
                previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
                try:
                    for group in range(2):
                        torch.mm(a[group], b[group], out=out[group])
                finally:
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous
                result = out
            else:
                _ensure_warmed()
                result = _EXTENSION.bmm_out(a, b, out, 1)
            _COUNTS["hit"] += 1
            return result
        except Exception:
            _COUNTS["error"] += 1
            raise
    bmm.original = original
    return bmm


def _patch_module(module: types.ModuleType) -> None:
    global _PATCHED
    if _PATCHED or getattr(module, "_dsv41_woa_hook", False):
        return
    fn = getattr(module, "deep_gemm_fp8_o_proj", None)
    if fn is None or not hasattr(fn, "__globals__"):
        return
    original_torch = fn.__globals__.get("torch")
    if original_torch is None or not hasattr(original_torch, "bmm"):
        return
    class TorchProxy:
        def __getattr__(self, name):
            return getattr(original_torch, name)

        def bmm(self, a, b, *, out=None):
            return _wrap_bmm(original_torch.bmm)(a, b, out=out)

    globals_copy = dict(fn.__globals__)
    globals_copy["torch"] = TorchProxy()
    wrapped = types.FunctionType(fn.__code__, globals_copy, fn.__name__,
                                 fn.__defaults__, fn.__closure__)
    wrapped.__kwdefaults__ = fn.__kwdefaults__
    setattr(module, "deep_gemm_fp8_o_proj", wrapped)
    module._dsv41_woa_hook = True
    _COUNTS["patched"] += 1
    _PATCHED = True
    print("DSV41_WOA_PATCH", {"mode": _MODE}, file=sys.stderr, flush=True)


class _Finder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "vllm.models.deepseek_v4.nvidia.ops.o_proj":
            return None
        import importlib.machinery
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader

        class Loader:
            def create_module(self, module_spec):
                return loader.create_module(module_spec) if hasattr(loader, "create_module") else None

            def exec_module(self, module):
                loader.exec_module(module)
                _patch_module(module)

        spec.loader = Loader()
        return spec


def install() -> None:
    finder = _Finder()
    sys.meta_path.insert(0, finder)
    existing = sys.modules.get("vllm.models.deepseek_v4.nvidia.ops.o_proj")
    if existing is not None:
        _patch_module(existing)
    def report():
        if _COUNTS["patched"] or _COUNTS["hit"]:
            print("DSV41_WOA_COUNTS", _COUNTS, file=sys.stderr, flush=True)
    atexit.register(report)
