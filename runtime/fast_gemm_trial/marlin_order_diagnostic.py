"""Opt-in canonical ordering for Marlin's expert dispatch list.

Marlin can reduce split-K blocks in the order presented by
``sorted_token_ids``.  This diagnostic keeps the alignment result and its
counts unchanged, while sorting the valid dispatch region by expert and token
id on the GPU.  It is disabled unless ``DSV41_MARLIN_CANONICAL_ORDER=1``.
"""

from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
from types import ModuleType


def _enabled() -> bool:
    return os.environ.get("DSV41_MARLIN_CANONICAL_ORDER") == "1"


def _canonical_sorted_token_ids(
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    block_size_m: int,
):
    """Stable-sort valid dispatch slots, leaving the capacity tail untouched."""
    import torch

    if sorted_token_ids.ndim != 1 or expert_ids.ndim != 1:
        raise RuntimeError("Marlin canonical order requires flat dispatch tensors")
    if block_size_m <= 0:
        raise RuntimeError("Marlin canonical order requires positive block size")
    if sorted_token_ids.device != expert_ids.device:
        raise RuntimeError("Marlin dispatch tensors must share a device")

    nslots = sorted_token_ids.numel()
    if nslots == 0:
        return sorted_token_ids
    nblocks = (nslots + block_size_m - 1) // block_size_m
    if expert_ids.numel() < nblocks:
        raise RuntimeError("Marlin expert_ids is shorter than dispatch capacity")

    # The scalar count stays on the device: this path is safe during graph
    # replay and works when the final expert has a partial capacity block.
    count = num_tokens_post_padded.reshape(())
    positions = torch.arange(nslots, device=sorted_token_ids.device,
                             dtype=torch.int64)
    valid = positions < count.to(dtype=torch.int64)
    slot_experts = expert_ids.to(dtype=torch.int64).repeat_interleave(block_size_m)
    slot_experts = slot_experts[:nslots]
    token_ids = sorted_token_ids.to(dtype=torch.int64)

    # Give each contiguous expert run its own segment id. This preserves the
    # alignment layout even if a producer emits repeated, non-contiguous runs
    # for the same expert. Invalid capacity slots receive a final segment id.
    boundaries = torch.ones_like(slot_experts, dtype=torch.bool)
    boundaries[1:] = slot_experts[1:] != slot_experts[:-1]
    segments = torch.cumsum(boundaries.to(dtype=torch.int64), dim=0)
    segment_key = torch.where(valid, segments, segments[-1] + 1)

    # Sort by token first, then by contiguous segment with stable argsort. This
    # avoids a composite integer key overflow for large global expert ids.
    token_key = torch.where(valid, token_ids, nslots + positions)
    token_order = torch.argsort(token_key, stable=True)
    order = token_order[torch.argsort(segment_key[token_order], stable=True)]
    reordered = token_ids[order].to(dtype=sorted_token_ids.dtype)
    return torch.where(valid, reordered, sorted_token_ids)


def install(module: ModuleType) -> None:
    """Patch the Marlin internal dispatch entry point once."""
    marker = "_dsv41_marlin_canonical_order"
    if getattr(module, marker, False):
        return
    original = getattr(module, "_fused_marlin_moe", None)
    if original is None:
        raise RuntimeError("Marlin module has no _fused_marlin_moe entry point")

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        if not _enabled():
            return original(*args, **kwargs)

        names = ("sorted_token_ids", "expert_ids", "num_tokens_post_padded",
                 "block_size_m")
        positional_names = (
            "hidden_states", "w1", "w2", "bias1", "bias2", "w1_scale",
            "w2_scale", "topk_weights", "num_topk", "quant_type",
            "apply_router_weight_on_input", "expert_map", "block_size_m",
            "sorted_token_ids", "expert_ids", "num_tokens_post_padded",
        )
        values = dict(zip(positional_names, args))
        values.update({name: kwargs[name] for name in names if name in kwargs})
        missing = [name for name in names if name not in values]
        if missing:
            raise RuntimeError(f"Marlin canonical order missing arguments: {missing}")
        canonical = _canonical_sorted_token_ids(
            values["sorted_token_ids"], values["expert_ids"],
            values["num_tokens_post_padded"], int(values["block_size_m"]),
        )

        if "sorted_token_ids" in kwargs:
            kwargs = dict(kwargs)
            kwargs["sorted_token_ids"] = canonical
            return original(*args, **kwargs)
        if len(args) < 1:
            raise RuntimeError("Marlin canonical order cannot rewrite positional args")
        rewritten = list(args)
        rewritten[positional_names.index("sorted_token_ids")] = canonical
        return original(*rewritten, **kwargs)

    module._fused_marlin_moe = wrapped
    setattr(module, marker, True)
    print("DSV41_MARLIN_CANONICAL_ORDER_INSTALLED", flush=True)


class _MarlinFinder(importlib.abc.MetaPathFinder):
    target = "vllm.model_executor.layers.fused_moe.experts.marlin_moe"

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.target:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _MarlinLoader(spec.loader)
        return spec


class _MarlinLoader(importlib.abc.Loader):
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def create_module(self, spec):
        creator = getattr(self.wrapped, "create_module", None)
        return creator(spec) if creator else None

    def exec_module(self, module):
        self.wrapped.exec_module(module)
        install(module)


def install_import_hook() -> None:
    if not _enabled():
        return
    if not any(isinstance(finder, _MarlinFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _MarlinFinder())
    if _MarlinFinder.target in sys.modules:
        install(sys.modules[_MarlinFinder.target])


def self_test() -> None:
    """CPU-only check for expert grouping, stable ties, and tail preservation."""
    import torch

    old = os.environ.get("DSV41_MARLIN_CANONICAL_ORDER")
    os.environ["DSV41_MARLIN_CANONICAL_ORDER"] = "1"
    try:
        ids = torch.tensor([4, 1, 7, 3, 99, 99, 5, 0, 2, 88, 88, 88])
        experts = torch.tensor([0, 0, 1])
        count = torch.tensor(9)
        out = _canonical_sorted_token_ids(ids, experts, count, 4)
        assert out.tolist() == [0, 1, 3, 4, 5, 7, 99, 99, 2, 88, 88, 88]

        repeated = torch.tensor([4, 1, 7, 3, 9, 2, 8, 6, 5])
        repeated_experts = torch.tensor([0, 0, 1, 1, 0])
        repeated_out = _canonical_sorted_token_ids(
            repeated, repeated_experts, torch.tensor(9), 2)
        assert repeated_out.tolist() == [1, 3, 4, 7, 2, 6, 8, 9, 5]
    finally:
        if old is None:
            os.environ.pop("DSV41_MARLIN_CANONICAL_ORDER", None)
        else:
            os.environ["DSV41_MARLIN_CANONICAL_ORDER"] = old
