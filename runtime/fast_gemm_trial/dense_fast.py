"""Opt-in dispatch for measured dense MXFP8 and wo_a paths.

The module has no import-time vLLM or CUDA side effects.  Call
``install_dense_mm_mxfp8`` at model setup, and call ``install_wo_a`` with the
specific attention class used by the deployment.  ``warmup_wo_a`` must run
before CUDA graph capture; it creates the cuBLASLt plans for the actual
strides, device, and algorithm index.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable


# Primary streamed measurements, keyed by (N, K). Values are the best measured
# backend for each M bucket (1, 8, 32, 128, 256). Unknown M uses the nearest
# measured bucket. K=576 deliberately has no b12x entry.
_DENSE_POLICY: dict[tuple[int, int], tuple[str, ...]] = {
    (1792, 5120): ("b12x", "b12x", "b12x", "b12x", "b12x"),
    (8192, 1280): ("b12x", "b12x", "b12x", "cudnn", "b12x"),
    (5120, 2048): ("b12x", "b12x", "b12x", "b12x", "b12x"),
    (1152, 5120): ("b12x", "b12x", "b12x", "b12x", "cutlass"),
    (25600, 6144): ("b12x", "b12x", "b12x", "cudnn", "cudnn"),
    (5120, 576): ("cudnn", "cudnn", "cudnn", "cutlass", "cutlass"),
    (4096, 1280): ("b12x", "b12x", "b12x", "b12x", "b12x"),
}
_M_BUCKETS = (1, 8, 32, 128, 256)
_log = logging.getLogger("dsv4_fast_gemm")
_logged_dense: set[tuple[int, int, int, str]] = set()
_warmed_devices: set[tuple[int, int]] = set()


def dense_backend(m: int, n: int, k: int) -> str | None:
    """Return the measured backend, or ``None`` for an unknown shape."""
    # The table stops at M=256. Do not extrapolate a decode policy into
    # prefill or large prompt batches where the measured winner can change.
    if int(m) < 1 or int(m) > _M_BUCKETS[-1]:
        return None
    choices = _DENSE_POLICY.get((int(n), int(k)))
    if choices is None:
        return None
    index = min(range(len(_M_BUCKETS)), key=lambda i: abs(_M_BUCKETS[i] - int(m)))
    return choices[index]


@dataclass
class DenseDispatch:
    """Callable wrapper around FlashInfer's existing ``mm_mxfp8`` API."""

    mm_mxfp8: Callable[..., Any]

    def __call__(self, a: Any, b: Any, a_scale: Any, b_scale: Any, **kwargs: Any) -> Any:
        backend = dense_backend(a.shape[0], b.shape[1], a.shape[1])
        if backend is None:
            return self.mm_mxfp8(a, b, a_scale, b_scale, **kwargs)
        # b12x requires K divisible by 128. Keep this guard beside dispatch so
        # a future policy edit cannot route the known K=576 shape incorrectly.
        if backend == "b12x" and (a.shape[1] % 128 or b.shape[1] % 128):
            backend = "cutlass"
        log_key = (int(a.shape[0]), int(b.shape[1]), int(a.shape[1]), backend)
        if log_key not in _logged_dense:
            _logged_dense.add(log_key)
            print('DSV41_FAST_DENSE', log_key, flush=True)
        # Replace a caller-provided backend rather than creating duplicate
        # keywords. This also makes the wrapper safe around existing explicit
        # CUTLASS calls.
        call_kwargs = dict(kwargs)
        call_kwargs["backend"] = backend
        return self.mm_mxfp8(a, b, a_scale, b_scale, **call_kwargs)


def install_dense_mm_mxfp8(module: Any) -> DenseDispatch:
    """Install a narrow wrapper on the supplied module's ``mm_mxfp8`` symbol.

    Pass the module that the model actually imported (usually ``flashinfer`` or
    ``flashinfer.gemm.gemm_base``). This avoids changing installed packages and
    avoids patching unrelated GEMM operators.
    """
    original = getattr(module, "mm_mxfp8")
    if isinstance(original, DenseDispatch):
        return original
    wrapper = DenseDispatch(original)
    setattr(module, "mm_mxfp8", wrapper)
    return wrapper


_WO_A = None


def build_wo_a_extension() -> Any:
    """Load the bridge compiled separately before starting model workers."""
    import importlib.util
    import os

    prebuilt = os.environ.get("DSV41_FAST_EXTENSION")
    if prebuilt:
        spec = importlib.util.spec_from_file_location("dsv4_wo_a_cublaslt", prebuilt)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load DSV41_FAST_EXTENSION={prebuilt}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    raise RuntimeError("set DSV41_FAST_EXTENSION to a prebuilt dsv4_wo_a_cublaslt .so")


def warmup_wo_a(extension: Any, device: Any, batches: tuple[int, ...] = (128, 256), algorithm: int = 1) -> None:
    """Create cuBLASLt plans before graph capture for the production layout."""
    import torch

    device_index = torch.device(device).index
    key = (device_index if device_index is not None else torch.cuda.current_device(), algorithm)
    if key in _warmed_devices:
        return
    batches = tuple(sorted(set(int(batch) for batch in batches if int(batch) >= 128)))
    if not batches:
        return
    for batch in batches:
        # Match the deployment tensors exactly: A is a [B, G, K] buffer viewed
        # as [G, B, K], and B is stored as [G, N, K] then viewed as [G, K, N].
        a = torch.zeros((batch, 2, 4096), device=device, dtype=torch.bfloat16).transpose(0, 1)
        b = torch.zeros((2, 1024, 4096), device=device, dtype=torch.bfloat16).transpose(1, 2)
        if tuple(a.stride()) != (4096, 8192, 1) or tuple(b.stride()) != (4194304, 1, 4096):
            raise RuntimeError(f"wo_a warmup layout mismatch: A={a.stride()} B={b.stride()}")
        if hasattr(extension, "bmm_out"):
            z = torch.empty((batch, 2, 1024), device=device, dtype=torch.bfloat16).transpose(0, 1)
            extension.bmm_out(a, b, z, algorithm)
        else:
            extension.bmm(a, b, algorithm)
    torch.cuda.synchronize(device)
    _warmed_devices.add(key)
    print('DSV41_FAST_WO_A', dict(device=key[0], batches=batches, algorithm=algorithm), flush=True)


def wo_a_bmm(a: Any, b: Any, out: Any | None = None) -> Any:
    """Dispatch only the production wo_a layout; small B remains aten.bmm."""
    if a.ndim != 3 or b.ndim != 3 or tuple(a.shape[::2]) != (2, 4096):
        raise ValueError(f"unexpected wo_a A shape: {tuple(a.shape)}")
    if tuple(b.shape) != (2, 4096, 1024):
        raise ValueError(f"unexpected wo_a B shape: {tuple(b.shape)}")
    if a.dtype != b.dtype:
        raise TypeError("wo_a operands must have the same dtype")
    import torch
    if a.shape[1] < 128 or a.shape[1] > 256:
        return torch.bmm(a, b, out=out)
    global _WO_A
    if _WO_A is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('wo_a extension was not warmed before graph capture')
        _WO_A = build_wo_a_extension()
        warmup_wo_a(_WO_A, a.device, (128, 136, 144, 160, 192, 224, 256))
    # Runtime plan creation is forbidden here: install_wo_a/warmup_wo_a must
    # have populated the extension cache before this function is captured.
    if out is not None and hasattr(_WO_A, "bmm_out"):
        return _WO_A.bmm_out(a, b, out, 1)
    result = _WO_A.bmm(a, b, 1)
    if out is not None:
        out.copy_(result)
        return out
    return result


def install_wo_a(attention_cls: Any, extension: Any | None = None) -> Any:
    """Patch only ``attention_cls._o_proj``'s supplied wo_a call helper.

    The class's existing method should call ``wo_a_bmm`` at the exact wo_a
    site. This function stores the prebuilt extension and returns it; it does
    not replace the full fused projection, whose surrounding rotary and wo_b
    semantics are model-specific.
    """
    global _WO_A
    if extension is None:
        raise RuntimeError(
            "install_wo_a requires a prebuilt extension; call "
            "build_wo_a_extension() outside worker startup"
        )
    _WO_A = extension
    setattr(attention_cls, "_dsv4_wo_a_bmm", staticmethod(wo_a_bmm))
    return extension
