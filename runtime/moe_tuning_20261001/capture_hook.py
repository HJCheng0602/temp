"""Capture selected Marlin MoE inputs for one real B64 decode request."""

from __future__ import annotations

import functools
import importlib.machinery
import json
import os
from pathlib import Path
import sys
from dataclasses import asdict


LAYERS = frozenset((0, 5, 10, 19, 20, 25, 30, 39))


def _armed() -> bool:
    value = os.environ.get("DSV41_MOE_CAPTURE_ARM")
    return bool(value) and Path(value).is_file()


def _layer_number(prefix: str) -> int | None:
    marker = ".layers."
    if marker not in prefix:
        return None
    try:
        return int(prefix.split(marker, 1)[1].split(".", 1)[0])
    except (IndexError, ValueError):
        return None


def _patch(module: object) -> None:
    cls = getattr(module, "Mxfp4MoEMethod", None)
    if cls is None or getattr(cls, "_dsv41_capture_patched", False):
        return
    original = cls.apply
    counts: dict[tuple[int, int], int] = {}

    @functools.wraps(original)
    def apply(self, layer, x, topk_weights, topk_ids, shared_experts,
              shared_experts_input):
        import torch
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_initialized() else 0
        prefix = str(getattr(layer, "layer_name", getattr(layer, "prefix", "")))
        layer_number = _layer_number(prefix)
        key = (rank, layer_number) if layer_number is not None else None
        if (_armed() and key is not None and layer_number in LAYERS
                and counts.get(key, 0) < 2 and x.ndim == 2
                and x.shape[0] == 64 and topk_ids.shape == (64, 6)
                and not torch.cuda.is_current_stream_capturing()):
            if x.ndim != 2:
                raise RuntimeError(
                    f"capture expected a 2D MoE activation, got x={tuple(x.shape)} "
                    f"at {prefix} rank {rank}"
                )
            ordinal = counts.get(key, 0)
            counts[key] = ordinal + 1
            root = Path(os.environ["DSV41_MOE_CAPTURE_DIR"])
            root.mkdir(parents=True, exist_ok=True)
            experts = getattr(self.moe_kernel, "fused_experts", None)
            expert_map = getattr(layer, "expert_map", None)
            activation_config = getattr(experts, "activation_config", None)
            metadata = {
                "schema": "dsv41_moe_input_capture_v1",
                "rank": rank,
                "layer": layer_number,
                "prefix": prefix,
                "ordinal": ordinal,
                "x_shape": list(x.shape),
                "x_dtype": str(x.dtype),
                "topk_ids_shape": list(topk_ids.shape),
                "topk_weights_shape": list(topk_weights.shape),
                "expert_map": (expert_map.detach().cpu().tolist()
                                if expert_map is not None else None),
                "apply_router_weight_on_input": getattr(
                    layer, "apply_router_weight_on_input", None),
                "activation": str(getattr(layer, "activation", None)),
                "swiglu_limit": getattr(layer, "swiglu_limit", None),
                "gemm1_clamp_limit": getattr(experts, "gemm1_clamp_limit", None),
                "expert_backend": type(experts).__name__ if experts is not None else None,
                "activation_config": (asdict(activation_config)
                                       if activation_config is not None else None),
            }
            payload = {
                **metadata,
                "metadata": metadata,
                "x": x.detach().cpu(),
                "topk_ids": topk_ids.detach().cpu(),
                "topk_weights": topk_weights.detach().cpu(),
                "expert_map": (expert_map.detach().cpu()
                                if expert_map is not None else None),
            }
            torch.save(payload, root / f"rank{rank:02d}_layer{layer_number:02d}_call{ordinal}.pt")
            (root / f"rank{rank:02d}_layer{layer_number:02d}_call{ordinal}.json").write_text(
                json.dumps(metadata, indent=2) + "\n"
            )
        return original(self, layer, x, topk_weights, topk_ids,
                        shared_experts, shared_experts_input)

    cls.apply = apply
    cls._dsv41_capture_patched = True
    print("DSV41_MOE_CAPTURE_INSTALLED", flush=True)


def install_import_hook() -> None:
    target = "vllm.model_executor.layers.quantization.mxfp4"

    class Finder:
        def find_spec(self, fullname, path=None, target_module=None):
            if fullname != target:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return None
            loader = spec.loader

            class Loader:
                def create_module(self, module_spec):
                    return (loader.create_module(module_spec)
                            if hasattr(loader, "create_module") else None)

                def exec_module(self, module):
                    loader.exec_module(module)
                    _patch(module)

            spec.loader = Loader()
            return spec

    sys.meta_path.insert(0, Finder())
    if target in sys.modules:
        _patch(sys.modules[target])
