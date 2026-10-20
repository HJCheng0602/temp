"""Bounded single-GPU Marlin repeat probe.

This probe isolates repeated-output instability from the full model.  It uses
the captured BF16 MoE input/routes and loads only the global experts selected
by the capture.  Three reduction/allocation modes are reported:

* ``normal``: let ``fused_marlin_moe`` allocate its work buffers;
* ``modular``: reuse the same workspace/cache layout used by
  ``MarlinExperts.workspace_shapes`` and ``MarlinExperts.apply``;
* ``det_reduce``: use the same modular buffers but replace ``_moe_C.moe_sum``
  with a masked, ordered FP32 sum before casting to the output dtype.

The script is intentionally diagnostic: it does not apply a tolerance or call
the full model.  Run it on the GPU host with the same vLLM environment as the
service.  It accepts the capture/checkpoint format used by
``marlin_real_oracle.py`` and writes JSON with cloned outputs from every eager
call and graph replay.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path

from marlin_real_oracle import (
    _individual_key_map,
    _index_keys,
    _load_capture,
    _load_individual_rows,
    _load_rows,
    _resolve_rows,
)


def _report(values):
    import torch
    base = values[0].float()
    rows = []
    for value in values:
        delta = value.float() - base
        rows.append({
            "finite": bool(torch.isfinite(value).all().item()),
            "max_abs_from_first": float(delta.abs().max().item()),
            "rel_l2_from_first": float(delta.norm().item() /
                                         base.norm().clamp_min(1e-12).item()),
        })
    return {
        "count": len(values),
        "shape": list(base.shape),
        "max_abs_repeat": max(row["max_abs_from_first"] for row in rows),
        "rows": rows,
    }


def _load_selected(capture, checkpoint: Path, layer: int, device):
    import torch
    expert_map = capture.get("expert_map")
    gids = sorted({int(gid) for gid in capture["topk_ids"].reshape(-1).tolist()
                   if expert_map is None or
                   (0 <= int(gid) < expert_map.numel() and
                    int(expert_map[int(gid)].item()) >= 0)})
    if not gids:
        raise ValueError("capture contains no local expert assignments")
    individual = _individual_key_map(checkpoint, layer, gids)
    if individual:
        if len(individual) != len(gids):
            raise KeyError(f"missing selected per-expert keys: {sorted(set(gids) - set(individual))}")
        weights = _load_individual_rows(checkpoint, individual, gids)
        local = {gid: gid for gid in gids}
        rows = gids
    else:
        keys = _index_keys(checkpoint, layer)
        w13_key = next(key for key in keys if key.endswith("w13_weight"))
        from safetensors import safe_open
        with safe_open(str(checkpoint / keys[w13_key]), framework="pt", device="cpu") as handle:
            shape = tuple(handle.get_slice(w13_key).get_shape())
        rows, local = _resolve_rows(capture["topk_ids"], expert_map, shape[0])
        weights = _load_rows(checkpoint, keys, rows)
    weights = {name: tensor.to(device=device) for name, tensor in weights.items()}
    for name, tensor in weights.items():
        if tensor.dtype != torch.uint8:
            raise TypeError(f"{name} must be raw uint8, got {tensor.dtype}")
    compact = {row: index for index, row in enumerate(rows)}
    local = {gid: compact[row] for gid, row in local.items()}
    emap_size = int(expert_map.numel()) if expert_map is not None else max(local) + 1
    emap = torch.full((emap_size,), -1, dtype=torch.int32, device=device)
    for gid, row in local.items():
        emap[gid] = row
    return weights, emap


def _tile_capture(capture, batch: int, device):
    import torch
    source_tokens = capture["x"].shape[0]
    index = torch.arange(batch, device=device, dtype=torch.long) % source_tokens
    return (capture["x"].index_select(0, index).contiguous(),
            capture["topk_ids"].index_select(0, index).to(torch.int32).contiguous(),
            capture["topk_weights"].index_select(0, index).contiguous())


def _deterministic_moe_sum(moe_output, output, topk_ids, expert_map):
    """Ordered FP32 reduction for already router-weighted FC2 rows.

    ``moe_output`` is laid out as [tokens, topk, hidden] by
    ``fused_marlin_moe``.  The route weights have already been applied by
    the FC2 Marlin call in this probe, so this function only masks invalid or
    remote routes and reduces the valid rows in their original top-k order.
    """
    import torch
    values = moe_output.float()
    ids = topk_ids.to(dtype=torch.long)
    if expert_map is not None:
        if expert_map.numel() == 0:
            valid = torch.zeros_like(ids, dtype=torch.bool)
        else:
            in_range = (ids >= 0) & (ids < expert_map.numel())
            safe_ids = ids.clamp(0, max(0, expert_map.numel() - 1))
            valid = in_range & (expert_map.index_select(0, safe_ids.reshape(-1))
                                .reshape_as(ids) >= 0)
        values = torch.where(valid.unsqueeze(-1), values,
                             torch.zeros_like(values))
    output.copy_(values.sum(dim=1, dtype=torch.float32).to(output.dtype))
    return output


def _run_mode(fused, kwargs, mode: str, repeats: int):
    import torch
    base = dict(kwargs)
    if mode == "normal":
        base.pop("workspace", None)
        base.pop("intermediate_cache13", None)
        base.pop("intermediate_cache2", None)
    if mode == "det_reduce":
        base["moe_sum"] = _deterministic_moe_sum
    values = [fused(**base).clone() for _ in range(repeats)]
    torch.cuda.synchronize()
    eager = _report(values)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_value = fused(**base)
    graph_values = []
    for _ in range(repeats):
        graph.replay()
        torch.cuda.synchronize()
        graph_values.append(graph_value.clone())
    return ({"eager": eager, "graph": _report(graph_values),
             "eager_vs_graph_max_abs": float((values[-1].float() - graph_values[-1].float()).abs().max().item())},
            values[-1], graph_values[-1])


def _input_variant(capture, batch: int, device, variant: str):
    """Return real capture rows or a route-order probe with the same routes."""
    import torch
    x, ids, weights = _tile_capture(capture, batch, device)
    if variant == "real":
        return x, ids, weights
    if variant == "reverse":
        order = torch.arange(batch - 1, -1, -1, device=device)
        return (x.index_select(0, order).contiguous(),
                ids.index_select(0, order).contiguous(),
                weights.index_select(0, order).contiguous())
    if variant == "derived_bf16":
        # A deterministic, tiny BF16 perturbation prevents identical tiled
        # rows from hiding physical alignment/reduction-order sensitivity.
        delta = torch.arange(x.numel(), device=device, dtype=torch.float32)
        delta = (delta.remainder(17) - 8).reshape_as(x) * (2.0 ** -10)
        return (torch.add(x, delta.to(torch.bfloat16)).to(torch.bfloat16).contiguous(),
                ids, weights)
    raise ValueError(f"unknown input variant: {variant}")


def _alignment_report(module, ids, expert_map, local_experts, repeats):
    from marlin_order_diagnostic import _canonical_sorted_token_ids
    estimated_m = math.ceil(ids.shape[0] * local_experts / expert_map.numel())
    for block in (8, 16, 32, 48, 64):
        if estimated_m * ids.shape[1] / local_experts / block < .9:
            break
    raw_hashes, canonical_hashes, expert_hashes, set_hashes, counts = [], [], [], [], []

    def digest(tensor):
        return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()

    for _ in range(repeats):
        tokens, experts, count = module.moe_align_block_size(
            ids, block, expert_map.numel(), expert_map, ignore_invalid_experts=True)
        canonical = _canonical_sorted_token_ids(tokens, experts, count, block)
        size = int(count.item())
        counts.append(size)
        raw_hashes.append(digest(tokens[:size]))
        canonical_hashes.append(digest(canonical[:size]))
        expert_hashes.append(digest(experts[:(size + block - 1) // block]))
        set_hashes.append(digest(tokens[:size].sort().values))
    return dict(block_size=block, raw_orders=len(set(raw_hashes)),
                canonical_orders=len(set(canonical_hashes)),
                expert_layouts=len(set(expert_hashes)),
                token_multisets=len(set(set_hashes)), padded_counts=sorted(set(counts)))


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 8, 64, 256, 512])
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("this probe must run on a CUDA device")
    torch.cuda.set_device(device)
    capture = _load_capture(args.capture, False, device)
    layer = args.layer if args.layer is not None else int(capture.get("layer", 0))
    if capture["x"].dtype != torch.bfloat16:
        raise ValueError("capture x must be exact BF16")
    weights, expert_map = _load_selected(capture, Path(args.checkpoint), layer, device)
    from vllm.model_executor.layers.fused_moe.activation import ApplyMoEActivationConfig, MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
    import vllm.model_executor.layers.fused_moe.experts.marlin_moe as marlin_module
    from marlin_order_diagnostic import install as install_canonical_order
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_moe_mxfp4_layer_for_marlin
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
    from vllm.scalar_type import scalar_types
    class LayerStub:
        params_dtype = torch.bfloat16
    packed = prepare_moe_mxfp4_layer_for_marlin(
        LayerStub(), weights["w13_weight"], weights["w2_weight"],
        weights["w13_weight_scale"], weights["w2_weight_scale"], None, None)
    config_values = dict(capture.get("activation_config", {}))
    config_values.setdefault("clamp_limit", capture.get("swiglu_limit", 10.0))
    config_values.setdefault("alpha", 1.0)
    config_values.setdefault("beta", 0.0)
    allowed = set(inspect.signature(ApplyMoEActivationConfig).parameters)
    unknown = sorted(set(config_values) - allowed)
    if unknown:
        raise ValueError(f"capture activation_config has unknown fields: {unknown}")
    activation_config = ApplyMoEActivationConfig(**config_values)
    # Install the wrapper once; its environment flag is checked per call, so
    # raw and canonical dispatch can be measured in one process.
    install_canonical_order(marlin_module)
    os.environ.pop("DSV41_MARLIN_CANONICAL_ORDER", None)
    variants = ["real", "reverse", "derived_bf16"]
    result = {"layer": layer, "batches": args.batches, "repeats": args.repeats,
              "capture_metadata": capture.get("metadata", capture.get("capture_metadata", {})),
              "modes": {}}
    for batch in args.batches:
      result["modes"][str(batch)] = {}
      variant_outputs = {}
      _, align_ids, _ = _tile_capture(capture, batch, device)
      result["modes"][str(batch)]["alignment"] = _alignment_report(
          marlin_module, align_ids, expert_map, packed[0].shape[0], args.repeats)
      for variant in variants:
        x, ids, topk_weights = _input_variant(capture, batch, device, variant)
        n = weights["w13_weight"].shape[1] // 2
        k = x.shape[1]
        topk = ids.shape[1]
        # Same swapped workspace allocation as MarlinExperts.apply/workspace_shapes.
        workspace13 = torch.empty((batch * topk, max(n, k)), device=device, dtype=torch.bfloat16)
        workspace2 = torch.empty((batch * topk * max(2 * n, k),), device=device, dtype=torch.bfloat16)
        workspace = marlin_make_workspace_new(device, 4)
        common = dict(hidden_states=x, w1=packed[0], w2=packed[1], bias1=None,
                      bias2=None, w1_scale=packed[2], w2_scale=packed[3],
                      topk_weights=topk_weights, topk_ids=ids,
                      quant_type_id=scalar_types.float4_e2m1f.id,
                      global_num_experts=int(expert_map.numel()), expert_map=expert_map,
                      activation=MoEActivation.SILU,
                      activation_config=activation_config, input_dtype=None)
        normal_kwargs = dict(common)
        modular_kwargs = dict(common, workspace=workspace,
                              intermediate_cache13=workspace2,
                              intermediate_cache2=workspace13,
                              output=workspace13.reshape(-1)[:batch * k].view(batch, k))
        normal_report, normal_eager, normal_graph = _run_mode(
            fused_marlin_moe, normal_kwargs, "normal", args.repeats)
        modular_report, modular_eager, modular_graph = _run_mode(
            fused_marlin_moe, modular_kwargs, "modular", args.repeats)
        det_report, det_eager, det_graph = _run_mode(
            fused_marlin_moe, modular_kwargs, "det_reduce", args.repeats)
        result["modes"][str(batch)][variant] = {
            "normal": normal_report,
            "modular": modular_report,
            "det_reduce": det_report,
            "normal_vs_modular_max_abs": float((normal_eager.float() - modular_eager.float()).abs().max().item()),
            "modular_vs_det_reduce_max_abs": float((modular_eager.float() - det_eager.float()).abs().max().item()),
            "graph_modular_vs_det_reduce_max_abs": float((modular_graph.float() - det_graph.float()).abs().max().item()),
        }
        os.environ["DSV41_MARLIN_CANONICAL_ORDER"] = "1"
        try:
            canonical_report, canonical_eager, _ = _run_mode(
                fused_marlin_moe, modular_kwargs, "modular", args.repeats)
        finally:
            os.environ.pop("DSV41_MARLIN_CANONICAL_ORDER", None)
        result["modes"][str(batch)][variant]["canonical_order"] = {
            "modular": canonical_report,
            "raw_vs_canonical_max_abs": float(
                (modular_eager.float() - canonical_eager.float()).abs().max().item()),
        }
        variant_outputs[variant] = normal_eager.detach().clone()
      if "real" in variant_outputs and "reverse" in variant_outputs:
        order = torch.arange(batch - 1, -1, -1, device=device)
        reverse_restored = variant_outputs["reverse"].index_select(0, order)
        result["modes"][str(batch)]["alignment_probe"] = {
            "kind": "reverse_input_order_proxy",
            "normal_eager_max_abs_after_restore": float(
                (variant_outputs["real"].float() - reverse_restored.float())
                .abs().max().item()),
        }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    Path(args.output).write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
