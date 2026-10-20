"""Offline accuracy checks for the CED prefill experiment.

The live runner supplies model-specific callbacks.  This module deliberately
does not construct a model or infer physical cache slots from token positions.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from typing import Any, Callable, Iterator, Mapping


def _torch():
    import torch

    return torch


def tensor_metrics(lhs: Any, rhs: Any, *, atol: float = 0.02,
                   rtol: float = 0.01) -> dict[str, Any]:
    """Compare tensors without treating a shape/type mismatch as a pass."""
    torch = _torch()
    if lhs is None or rhs is None:
        return {"status": "invalid", "reason": "missing_tensor"}
    if tuple(lhs.shape) != tuple(rhs.shape):
        return {"status": "invalid", "reason": "shape_mismatch",
                "lhs_shape": list(lhs.shape), "rhs_shape": list(rhs.shape)}
    if lhs.numel() == 0:
        return {"status": "invalid", "reason": "empty_tensor", "numel": 0}
    a = lhs.detach().float()
    b = rhs.detach().float()
    delta = a - b
    scale = b.norm().clamp_min(1e-12)
    return {
        "status": "ok",
        "dtype_equal": str(lhs.dtype) == str(rhs.dtype),
        "torch_equal": bool(torch.equal(lhs, rhs)),
        "allclose": bool(torch.allclose(lhs, rhs, atol=atol, rtol=rtol)),
        "shape": list(lhs.shape),
        "numel": lhs.numel(),
        "max_abs": float(delta.abs().max().item()),
        "rmse": float(delta.square().mean().sqrt().item()),
        "relative_l2": float((delta.norm() / scale).item()),
        "argmax_equal": bool(torch.equal(lhs.argmax(-1), rhs.argmax(-1))),
    }


def sampled_metrics(lhs: Any, rhs: Any) -> dict[str, Any]:
    """Compare greedy token IDs, accepting nested worker result records."""
    left = lhs.get("token_ids", lhs) if isinstance(lhs, Mapping) else lhs
    right = rhs.get("token_ids", rhs) if isinstance(rhs, Mapping) else rhs
    return {"equal": left == right, "lhs": left, "rhs": right,
            "nonempty": bool(left) and bool(right)}


def cache_snapshot(cache: Any, slots: Any, *, label: str = "cache") -> dict[str, Any]:
    """Digest selected physical pages without decoding packed cache records.

    ``slots`` must contain physical *page* IDs indexing dimension zero.  The
    whole page, including padding and scale bytes, is hashed as raw uint8.
    Token-to-page conversion belongs to the live adapter; this function never
    interprets 528/584-byte SM120 records as logical rows.
    """
    torch = _torch()
    if cache is None or slots is None:
        return {"status": "invalid", "reason": "missing_cache_or_slots",
                "label": label}
    slot_tensor = slots.detach().to(device=cache.device, dtype=torch.long).reshape(-1)
    if bool((slot_tensor < 0).any()) or bool((slot_tensor >= cache.shape[0]).any()):
        return {"status": "invalid", "reason": "slot_out_of_range",
                "label": label}
    pages = cache.index_select(0, slot_tensor).detach().cpu().contiguous()
    raw = pages.view(torch.uint8)
    digest = hashlib.sha256(raw.numpy().tobytes()).hexdigest()
    return {
        "status": "ok", "label": label, "shape": list(cache.shape),
        "dtype": str(cache.dtype), "stride": list(cache.stride()),
        "page_shape": list(pages.shape), "page_stride": list(pages.stride()),
        "page_ids": slot_tensor.cpu().tolist(),
        "bytes_sha256": digest, "bytes_nbytes": raw.numel(),
    }


def compare_cache_snapshots(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> dict[str, Any]:
    """Compare snapshots by bytes and metadata, never by dequantized values only."""
    if lhs.get("status") != "ok" or rhs.get("status") != "ok":
        return {"status": "invalid", "lhs": lhs, "rhs": rhs}
    left = lhs.get("bytes_sha256")
    right = rhs.get("bytes_sha256")
    return {
        "status": "ok" if left == right else "different",
        "byte_equal": left == right,
        "shape_equal": lhs.get("shape") == rhs.get("shape"),
        "dtype_equal": lhs.get("dtype") == rhs.get("dtype"),
        "stride_equal": lhs.get("stride") == rhs.get("stride"),
        "page_equal": lhs.get("page_ids") == rhs.get("page_ids"),
        "lhs_bytes": lhs.get("bytes_nbytes"), "rhs_bytes": rhs.get("bytes_nbytes"),
    }


def kl_divergence(lhs_logits: Any, rhs_logits: Any) -> float | None:
    """Mean KL(lhs || rhs) after float32 log-softmax conversion."""
    torch = _torch()
    if lhs_logits is None or rhs_logits is None or tuple(lhs_logits.shape) != tuple(rhs_logits.shape):
        return None
    if lhs_logits.numel() == 0:
        return None
    p = torch.log_softmax(lhs_logits.float(), dim=-1)
    q = torch.log_softmax(rhs_logits.float(), dim=-1)
    return float(torch.nn.functional.kl_div(q, p.exp(), reduction="batchmean").item())


def target_nll(logits: Any, targets: Any) -> float | None:
    """Mean target-token NLL; returns None when targets are absent/mismatched."""
    torch = _torch()
    if logits is None or targets is None or logits.numel() == 0:
        return None
    target = targets.detach().to(device=logits.device, dtype=torch.long).reshape(-1)
    flat = logits.reshape(-1, logits.shape[-1])
    if flat.shape[0] != target.numel():
        return None
    return float(torch.nn.functional.cross_entropy(flat.float(), target,
                                                   reduction="mean").item())


def compare_paths(paths: Mapping[str, Mapping[str, Any]], *, atol: float = 0.02,
                  rtol: float = 0.01) -> dict[str, Any]:
    """Build explicit comparisons for original/canonical/native-CED paths."""
    names = tuple(paths)
    result: dict[str, Any] = {"paths": list(names), "comparisons": {}}
    for left_name in names:
        for right_name in names:
            if names.index(left_name) >= names.index(right_name):
                continue
            left, right = paths[left_name], paths[right_name]
            row: dict[str, Any] = {
                "logits": tensor_metrics(left.get("logits"), right.get("logits"),
                                          atol=atol, rtol=rtol),
                "sample": sampled_metrics(left.get("sample"), right.get("sample")),
            }
            row["kl"] = kl_divergence(left.get("logits"), right.get("logits"))
            left_caches = left.get("cache_snapshots", {})
            right_caches = right.get("cache_snapshots", {})
            common_caches = sorted(set(left_caches) & set(right_caches))
            if common_caches:
                row["cache"] = {
                    name: compare_cache_snapshots(left_caches[name], right_caches[name])
                    for name in common_caches
                }
            if "boundary" in left and "boundary" in right:
                row["boundary"] = tensor_metrics(left["boundary"], right["boundary"],
                                                   atol=atol, rtol=rtol)
            result["comparisons"][f"{left_name}_vs_{right_name}"] = row
    return result


def _metadata_objects(metadata: Any) -> Iterator[Any]:
    if isinstance(metadata, Mapping):
        yield from metadata.values()
    elif isinstance(metadata, (list, tuple)):
        yield from metadata
    else:
        yield metadata


@contextmanager
def native_bounded_swa(metadata: Any, positions: Any, *, window_start: int,
                       slot_to_position: Callable[[Any], Any] | None = None
                       ) -> Iterator[dict[str, Any]]:
    """Temporarily clip only tail-query SWA rows for a native full model.

    ``slot_to_position`` is mandatory because sparse metadata stores physical
    slots, not absolute positions.  It receives a tensor of slot IDs and must
    return positions with the same shape.  Older query rows remain untouched;
    only rows whose query position is at least ``window_start`` are compacted.
    """
    torch = _torch()
    if slot_to_position is None:
        raise ValueError("native_bounded_swa requires explicit slot_to_position")
    saved: list[tuple[Any, Any, Any, int]] = []
    seen: set[int] = set()
    changed = 0
    try:
        for obj in _metadata_objects(metadata):
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            indices = getattr(obj, "prefill_swa_indices", None)
            lengths = getattr(obj, "prefill_swa_lens", None)
            count = int(getattr(obj, "num_prefill_tokens", 0))
            if indices is None or lengths is None or count <= 0:
                continue
            saved.append((obj, indices.clone(), lengths.clone(), count))
            index_view = indices[:count]
            length_view = lengths[:count]
            query_positions = getattr(obj, "query_positions", None)
            if query_positions is None:
                query_positions = positions
            if query_positions is None or query_positions.numel() < count:
                raise ValueError("metadata object lacks enough query positions")
            query_positions = query_positions[-count:]
            flat_indices = index_view.reshape(count, -1)
            width = flat_indices.shape[1]
            cols = torch.arange(width, device=flat_indices.device)[None, :]
            valid = cols < length_view.reshape(-1, 1)
            valid &= flat_indices >= 0
            positions_for_slots = slot_to_position(flat_indices.clamp_min(0))
            target = query_positions >= window_start
            keep = valid & (positions_for_slots >= window_start) & (
                positions_for_slots <= query_positions.reshape(-1, 1))
            keep &= target.reshape(-1, 1)
            # Stable argsort puts selected entries first while retaining their
            # original candidate order; this is the vectorized equivalent of
            # front-compaction without per-row GPU synchronizations.
            key = torch.where(keep, cols, cols + width)
            order = torch.argsort(key, dim=1, stable=True)
            packed = flat_indices.gather(1, order)
            counts = keep.sum(dim=1).to(length_view.dtype)
            updated = flat_indices.clone()
            updated[target] = torch.where(
                torch.arange(width, device=flat_indices.device)[None, :] < counts[target, None],
                packed[target], torch.full_like(packed[target], -1),
            )
            flat_indices.copy_(torch.where(target[:, None], updated, flat_indices))
            length_view.copy_(torch.where(target, counts, length_view))
            changed += int(target.sum().item())
        yield {"status": "ok", "rows_changed": changed,
               "window_start": int(window_start)}
    finally:
        for obj, old_indices, old_lengths, count in saved:
            obj.prefill_swa_indices[:count].copy_(old_indices[:count])
            obj.prefill_swa_lens[:count].copy_(old_lengths[:count])


def build_accuracy_report(paths: Mapping[str, Mapping[str, Any]], *,
                          prompt_tokens: int, continuation: Mapping[str, Any] | None = None,
                          atol: float = 0.02, rtol: float = 0.01) -> dict[str, Any]:
    """Build a conservative report; no missing comparison is marked passed."""
    torch = _torch()
    report = {"prompt_tokens": int(prompt_tokens),
              "paths": list(paths),
              "comparisons": compare_paths(paths, atol=atol, rtol=rtol),
              "same_semantics_pass": False,
              "original_equivalence_pass": False,
              "coverage": {"finite_logits": {}, "logit_metrics": {},
                            "cache_bytes": {}, "continuation": {}}}
    for name, path in paths.items():
        logits = path.get("logits")
        report["coverage"]["finite_logits"][name] = bool(
            logits is not None and logits.numel() > 0 and bool(torch.isfinite(logits).all()))
        report["coverage"]["cache_bytes"][name] = {
            key: value.get("status") == "ok"
            for key, value in path.get("cache_snapshots", {}).items()
        }
    if continuation is not None:
        report["continuation"] = continuation
        report["coverage"]["continuation"] = {"present": True,
                                                "equal": continuation.get("equal")}
    same = report["comparisons"]["comparisons"].get(
        "native_bounded_vs_transferred_ced")
    if same:
        log = same["logits"]
        report["same_semantics_pass"] = bool(
            log.get("status") == "ok" and log.get("allclose") and
            log.get("argmax_equal") and same["sample"].get("equal") and
            bool(continuation and continuation.get("equal")) and
            bool(same.get("cache")) and all(
                item.get("byte_equal") for item in same["cache"].values()))
        report["coverage"]["logit_metrics"]["native_bounded_vs_transferred_ced"] = log
        if "cache" in same:
            report["coverage"]["cache_bytes"][
                "native_bounded_vs_transferred_ced"] = same["cache"]
    full = report["comparisons"]["comparisons"].get(
        "full_original_vs_transferred_ced")
    if full:
        report["original_equivalence_pass"] = bool(
            full["logits"].get("status") == "ok" and
            full["logits"].get("torch_equal"))
    return report
