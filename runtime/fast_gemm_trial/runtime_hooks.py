"""Lazy, process-local hooks for the installed DeepSeek V4 runtime.

This module is safe to import from ``sitecustomize``: it imports no torch,
vLLM, FlashInfer, or CUDA extension. The target module is patched only after
Python has loaded it.
"""

from __future__ import annotations

import sys
import types
from types import ModuleType
from typing import Any


TARGETS = {
    "vllm.models.deepseek_v4.nvidia.ops.o_proj",
    "vllm.model_executor.kernels.linear.mxfp8.flashinfer",
}
_PATCHED = False


def _wo_a_bmm(a: Any, b: Any, *, out: Any = None, **kwargs: Any) -> Any:
    # Import lazily because this function is reached only in a CUDA worker.
    try:
        from .dense_fast import wo_a_bmm
    except ImportError:
        from dense_fast import wo_a_bmm

    if getattr(a, "ndim", 0) != 3 or getattr(b, "ndim", 0) != 3:
        return _wo_a_bmm.original(a, b, out=out, **kwargs)
    if tuple(a.shape[::2]) != (2, 4096) or tuple(b.shape) != (2, 4096, 1024):
        return _wo_a_bmm.original(a, b, out=out, **kwargs)
    if a.shape[1] > 256:
        return _wo_a_bmm.original(a, b, out=out, **kwargs)
    return wo_a_bmm(a, b, out=out)


def _patch_o_proj(module: ModuleType) -> None:
    global _PATCHED
    if _PATCHED or getattr(module, "_dsv4_fast_gemm_hook", False):
        return
    fn = getattr(module, "deep_gemm_fp8_o_proj", None)
    if fn is None or not hasattr(fn, "__globals__"):
        return
    original_torch = fn.__globals__.get("torch")
    if original_torch is None or not hasattr(original_torch, "bmm"):
        return
    class TorchProxy:
        def __getattr__(self, name: str) -> Any:
            return getattr(original_torch, name)

        def bmm(self, a: Any, b: Any, *, out: Any = None) -> Any:
            _wo_a_bmm.original = original_torch.bmm
            return _wo_a_bmm(a, b, out=out)

    # Clone the function with a module-local torch proxy. No global torch symbol
    # is replaced, so unrelated operators and concurrent workers are unaffected.
    cloned_globals = dict(fn.__globals__)
    cloned_globals["torch"] = TorchProxy()
    wrapped = types.FunctionType(fn.__code__, cloned_globals, fn.__name__, fn.__defaults__, fn.__closure__)
    wrapped.__kwdefaults__ = fn.__kwdefaults__
    setattr(module, "deep_gemm_fp8_o_proj", wrapped)
    module._dsv4_fast_gemm_hook = True
    _PATCHED = True


def patch_loaded_module(module: ModuleType) -> None:
    if module.__name__ == "vllm.models.deepseek_v4.nvidia.ops.o_proj":
        _patch_o_proj(module)
    # Dense dispatch is patched only when FlashInfer is already loaded. No
    # import here prevents sitecustomize from triggering CUDA initialization.
    try:
        from .dense_fast import install_dense_mm_mxfp8
    except ImportError:
        from dense_fast import install_dense_mm_mxfp8
    for name in ("vllm.utils.flashinfer",):
        candidate = sys.modules.get(name)
        if candidate is not None and hasattr(candidate, "mm_mxfp8"):
            install_dense_mm_mxfp8(candidate)


class _Finder:
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname not in TARGETS:
            return None
        import importlib.machinery
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader

        class Loader:
            def create_module(self, spec: Any) -> Any:
                return loader.create_module(spec) if hasattr(loader, "create_module") else None

            def exec_module(self, module: ModuleType) -> None:
                loader.exec_module(module)
                patch_loaded_module(module)

        spec.loader = Loader()
        return spec


def install_import_hook() -> None:
    if not any(type(f).__name__ == "_Finder" and getattr(f, "_dsv4", False) for f in sys.meta_path):
        finder = _Finder()
        finder._dsv4 = True
        sys.meta_path.insert(0, finder)
    for target in TARGETS:
        existing = sys.modules.get(target)
        if existing is not None:
            patch_loaded_module(existing)
