"""Opt-in Marlin block-size override for production A/B runs."""

from __future__ import annotations

import functools
import importlib.machinery
import json
import os
import sys
import threading
from types import ModuleType
from typing import Any


_STATE = threading.local()
_PATCHED = "_dsv41_tile_override_patched"
_LOGGED: set[tuple[int, int, int | None, int | None]] = set()
_NAMES = (
    "hidden_states", "w1", "w2", "bias1", "bias2", "w1_scale",
    "w2_scale", "topk_weights", "num_topk", "quant_type",
    "apply_router_weight_on_input", "expert_map", "block_size_m",
    "sorted_token_ids", "expert_ids", "num_tokens_post_padded",
)


def _policy() -> dict[str, int]:
    raw = os.environ.get("DSV41_MARLIN_TILE_CONFIG", "")
    if raw:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("DSV41_MARLIN_TILE_CONFIG must be a JSON object")
        return {str(k): int(v) for k, v in value.items()}
    raw = os.environ.get("DSV41_MARLIN_TILE", "")
    return {"default": int(raw)} if raw else {}


def _tile(layer: int | None, batch: int | None) -> int | None:
    values = _policy()
    if not values:
        return None
    keys = []
    if batch is not None and layer is not None:
        keys.append(f"batch{batch}.layer{layer}")
    if layer is not None:
        keys.append(f"layer{layer}")
    if batch is not None:
        keys.append(f"batch{batch}")
    keys.append("default")
    result = next((values[key] for key in keys if key in values), None)
    if result not in (8, 16, 32, 48, 64):
        if result is not None:
            raise ValueError(f"unsupported Marlin tile size: {result}")
        return None
    return result


def _rewrite_block(args: tuple[Any, ...], kwargs: dict[str, Any], block: int):
    if "block_size_m" in kwargs:
        out = dict(kwargs)
        out["block_size_m"] = block
        return args, out
    index = _NAMES.index("block_size_m")
    if len(args) <= index:
        raise RuntimeError("unexpected Marlin _fused_marlin_moe signature")
    out = list(args)
    out[index] = block
    return tuple(out), kwargs


def _patch_marlin(module: ModuleType) -> None:
    if getattr(module, _PATCHED, False):
        return
    align = getattr(module, "moe_align_block_size", None)
    fused = getattr(module, "_fused_marlin_moe", None)
    if align is None or fused is None:
        return

    @functools.wraps(align)
    def align_wrapper(topk_ids, block_size_m, *args, **kwargs):
        block = getattr(_STATE, "block", None)
        return align(topk_ids, block if block is not None else block_size_m,
                     *args, **kwargs)

    @functools.wraps(fused)
    def fused_wrapper(*args, **kwargs):
        block = getattr(_STATE, "block", None)
        if block is None:
            return fused(*args, **kwargs)
        rewritten, rewritten_kwargs = _rewrite_block(args, kwargs, block)
        return fused(*rewritten, **rewritten_kwargs)

    module.moe_align_block_size = align_wrapper
    module._fused_marlin_moe = fused_wrapper
    setattr(module, _PATCHED, True)
    print("DSV41_MARLIN_TILE_OVERRIDE_INSTALLED", flush=True)


def _patch_quant(module: ModuleType) -> None:
    cls = getattr(module, "Mxfp4MoEMethod", None)
    if cls is None or getattr(cls, _PATCHED, False):
        return
    original = cls.apply

    @functools.wraps(original)
    def apply(self, layer, x, topk_weights, topk_ids, shared_experts,
              shared_experts_input):
        import torch.distributed as dist

        prefix = str(getattr(layer, "layer_name", getattr(layer, "prefix", "")))
        try:
            layer_no = int(prefix.split(".layers.", 1)[1].split(".", 1)[0])
        except (IndexError, ValueError):
            layer_no = None
        previous = getattr(_STATE, "block", None)
        _STATE.block = _tile(layer_no, int(x.shape[0]) if x.ndim else None)
        rank = (dist.get_rank() if dist.is_initialized()
                else int(os.environ.get("LOCAL_RANK", "-1")))
        marker = (rank, int(x.shape[0]) if x.ndim else -1, layer_no, _STATE.block)
        if _STATE.block is not None and marker not in _LOGGED and marker[0] in (0, 4):
            _LOGGED.add(marker)
            print("DSV41_MARLIN_TILE_SELECTED " + json.dumps({
                "rank": marker[0], "batch": marker[1], "layer": layer_no,
                "tile": _STATE.block,
            }, sort_keys=True), flush=True)
        try:
            return original(self, layer, x, topk_weights, topk_ids,
                            shared_experts, shared_experts_input)
        finally:
            _STATE.block = previous

    cls.apply = apply
    setattr(cls, _PATCHED, True)


def install_import_hook() -> None:
    targets = {
        "vllm.model_executor.layers.quantization.mxfp4": _patch_quant,
        "vllm.model_executor.layers.fused_moe.experts.marlin_moe": _patch_marlin,
    }

    class Finder:
        def find_spec(self, fullname, path=None, target=None):
            patch = targets.get(fullname)
            if patch is None:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return None
            loader = spec.loader

            class Loader:
                def create_module(self, module_spec):
                    return loader.create_module(module_spec) if hasattr(loader, "create_module") else None

                def exec_module(self, module):
                    loader.exec_module(module)
                    patch(module)

            spec.loader = Loader()
            return spec

    sys.meta_path.insert(0, Finder())
    for name, patch in targets.items():
        existing = sys.modules.get(name)
        if existing is not None:
            patch(existing)
