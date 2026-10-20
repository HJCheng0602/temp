"""Small opt-in source-20 diagnostics for the independent accuracy report."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any


def _torch():
    import torch

    return torch


def _tensor(value: Any):
    torch = _torch()
    if isinstance(value, (tuple, list)):
        value = next((item for item in value if isinstance(item, torch.Tensor)), None)
    return value if isinstance(value, torch.Tensor) else None


class DiagnosticHandles:
    def __init__(self, cleanup):
        self._cleanup = cleanup
        self._closed = False

    def remove(self):
        if not self._closed:
            self._cleanup()
            self._closed = True

    close = remove

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.remove()


def install(runner: Any) -> DiagnosticHandles:
    """Install opt-in hooks and return a removable handle bundle.

    Hooks are inactive unless the runner is rank zero and has capture_logits
    enabled.  P source-head output is intercepted at the class method because
    the runtime calls ``Source20CacheHead.forward`` explicitly.
    """
    if getattr(runner, "rank", 1) != 0:
        return DiagnosticHandles(lambda: None)
    torch = _torch()
    output = Path(runner.output)
    counter = {"value": 0}
    hooks = []
    original_head = None

    def save(kind: str, value: Any, positions: Any = None):
        if not getattr(runner, "capture_logits", False):
            return
        value = _tensor(value)
        if value is None or value.shape[0] <= 128:
            return
        mode = str(getattr(runner, "diagnostic_mode", "ced"))
        case = str(getattr(runner, "case_name", "case"))
        cpu = value.detach().float().cpu().clone()
        index = counter["value"]
        counter["value"] += 1
        stem = f"{case}_{mode}_{kind}_{index:04d}"
        torch.save(cpu, output / f"{stem}.pt")
        metadata = {"case": case, "mode": mode, "kind": kind,
                    "index": index, "shape": list(cpu.shape),
                    "dtype": str(cpu.dtype)}
        if positions is not None and hasattr(positions, "numel") and positions.numel():
            metadata["position_min"] = int(positions.min().item())
            metadata["position_max"] = int(positions.max().item())
        (output / f"{stem}.json").write_text(json.dumps(metadata, sort_keys=True))

    backbone = runner.backbone
    layer20 = backbone.layers[20]
    role = str(getattr(runner, "role", ""))

    if role == "prefill":
        def source_norm(_module, _args, out):
            save("normalized_source20_input", out,
                 getattr(backbone, "_ced_positions", None))

        hooks.append(layer20.attn_norm.register_forward_hook(source_norm))
    else:
        def attention_input(_module, args):
            positions = args[0] if args and hasattr(args[0], "dtype") else None
            save("normalized_source20_input", args[1] if len(args) > 1 else None,
                 positions)

        hooks.append(layer20.attn.register_forward_pre_hook(attention_input))

    compressor = getattr(layer20.attn, "compressor", None)
    if compressor is not None and role != "prefill":
        hooks.append(compressor.register_forward_hook(
            lambda _m, _args, out: save("compressor_latent", out)
        ))

    indexer = getattr(layer20.attn, "indexer", None)
    wk = getattr(indexer, "wk", None) if indexer is not None else None
    if wk is not None:
        hooks.append(wk.register_forward_hook(
            lambda _m, _args, out: save("index_pre", out)
        ))

    if role == "prefill":
        from cache_head import Source20CacheHead

        original_head = Source20CacheHead.forward

        def wrapped_head(self, *args, **kwargs):
            result = original_head(self, *args, **kwargs)
            save("compressor_latent", result,
                 args[1] if len(args) > 1 else kwargs.get("positions"))
            return result

        Source20CacheHead.forward = wrapped_head

    def cleanup():
        nonlocal original_head
        for hook in reversed(hooks):
            hook.remove()
        if original_head is not None:
            from cache_head import Source20CacheHead
            Source20CacheHead.forward = original_head
            original_head = None

    return DiagnosticHandles(cleanup)


def tensor_metrics(lhs: Any, rhs: Any) -> dict[str, Any]:
    """JSON-safe comparison with explicit missing/non-finite states."""
    torch = _torch()
    if lhs is None or rhs is None:
        return {"status": "missing"}
    if tuple(lhs.shape) != tuple(rhs.shape):
        return {"status": "shape_mismatch", "lhs_shape": list(lhs.shape),
                "rhs_shape": list(rhs.shape)}
    if lhs.numel() == 0:
        return {"status": "empty"}
    finite = bool(torch.isfinite(lhs).all() and torch.isfinite(rhs).all())
    delta = lhs.float() - rhs.float()
    return {"status": "ok" if finite else "nonfinite", "finite": finite,
            "shape": list(lhs.shape), "max_abs": float(delta.abs().max()),
            "rmse": float(delta.square().mean().sqrt()),
            "relative_l2": float(delta.norm() / rhs.float().norm().clamp_min(1e-12)),
            "torch_equal": bool(torch.equal(lhs, rhs))}


def page_byte_metrics(lhs: Any, rhs: Any) -> dict[str, Any]:
    """Compare copied whole-page views; stride padding is intentionally excluded."""
    torch = _torch()
    if lhs is None or rhs is None or tuple(lhs.shape) != tuple(rhs.shape):
        return {"status": "missing_or_shape_mismatch"}
    left = lhs.detach().cpu().contiguous().view(torch.uint8)
    right = rhs.detach().cpu().contiguous().view(torch.uint8)
    if left.numel() != right.numel():
        return {"status": "byte_size_mismatch", "lhs": left.numel(), "rhs": right.numel()}
    changed = left.ne(right)
    return {"status": "ok", "bytes": left.numel(),
            "byte_equal": bool(torch.equal(left, right)),
            "different_byte_fraction": float(changed.float().mean())}


def _load_kind(folder: Path, case: str, mode: str, kind: str):
    torch = _torch()
    paths = sorted(folder.glob(f"{case}_{mode}_{kind}_*.pt"))
    if not paths:
        return None
    values = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    return torch.cat(values, dim=0) if len(values) > 1 else values[0]


def compare_folder(folder: str | Path, case: str, left_mode: str,
                   right_mode: str) -> dict[str, Any]:
    """Compare accumulated P/native tensors, excluding decoder replay labels."""
    folder = Path(folder)
    result = {"case": case, "left_mode": left_mode, "right_mode": right_mode,
              "comparisons": {}}
    for kind in ("normalized_source20_input", "compressor_latent", "index_pre"):
        left = _load_kind(folder, case, left_mode, kind)
        right = _load_kind(folder, case, right_mode, kind)
        result["comparisons"][kind] = tensor_metrics(left, right)
    return result
