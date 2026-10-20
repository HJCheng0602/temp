"""Small real-weight MXFP4/Marlin oracle for one MoE layer.

This is deliberately a standalone diagnostic.  It does not start vLLM and it
does not load a whole checkpoint into a model.  The input capture is a torch
``.pt``/``.pth`` dictionary containing at least::

    x:             [tokens, hidden] BF16/FP16/FP32 hidden states
    topk_ids:      [tokens, topk] global expert ids (int32/int64)
    topk_weights:  [tokens, topk] float32 router weights
    expert_map:    [global_experts] global -> local row, with -1 for remote

Optional keys are ``activation_config`` (dict with ``clamp_limit``, ``alpha``,
and ``beta``), ``layer`` (integer), ``swiglu_limit`` (number),
``input_dtype`` (``bf16``/``fp8``), and already
captured ``fp8_output``/``baseline_output``.  ``expert_map`` may be omitted
only when the checkpoint stores global expert rows and the ids are in range.

The checkpoint path is expected to contain ``model.safetensors.index.json``.
Only rows for experts referenced by ``topk_ids`` are requested from each
safetensors file.  The script refuses a backend that cannot provide row
slicing instead of silently materializing all 384 experts.

Examples (on the GPU host)::

    python marlin_real_oracle.py --capture /tmp/moe.pt --layer 0 \
      --checkpoint /mnt/fuse/models/DeepSeek-V4.1-Flash-local-engram \
      --print-reference

    python marlin_real_oracle.py --capture /tmp/moe.pt --layer 0 \
      --checkpoint /mnt/fuse/models/DeepSeek-V4.1-Flash-local-engram \
      --run-marlin --device cuda:0

The independent reference decodes OCP E2M1 nibbles and UE8M0 block scales in
FP32.  It uses the verified vLLM clamp convention
``silu(clamp(gate, max=limit)) * clamp(up, -limit, limit)``;
``--gate-first`` selects the alternate tensor order for captures from a
different implementation.  A synthetic fallback is available with
``--synthetic`` and is labeled in the JSON output; it is not a real-weight
accuracy result.
"""

from __future__ import annotations

import argparse
import json
import inspect
from pathlib import Path
from typing import Any


LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
       0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _torch():
    import torch
    return torch


def _load_capture(path: str | None, synthetic: bool, device: Any):
    torch = _torch()
    if synthetic:
        gen = torch.Generator(device="cpu").manual_seed(1729)
        x = (torch.randn((2, 64), generator=gen, dtype=torch.bfloat16) * 0.1).contiguous()
        topk_ids = torch.tensor([[3, 1, 7], [1, 3, 5]], dtype=torch.int64)
        topk_weights = torch.tensor([[.5, .3, .2], [.4, .35, .25]], dtype=torch.float32)
        expert_map = torch.full((8,), -1, dtype=torch.int64)
        expert_map[3], expert_map[7], expert_map[1], expert_map[5] = 2, 0, 5, 1
        return dict(x=x, topk_ids=topk_ids, topk_weights=topk_weights,
                    expert_map=expert_map, layer=0, swiglu_limit=10.0,
                    source="synthetic")
    if path is None:
        raise ValueError("--capture is required unless --synthetic is supplied")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("capture must be a dictionary")
    required = ("x", "topk_ids", "topk_weights")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"capture missing keys: {missing}")
    out = dict(payload)
    out["source"] = str(path)
    if out["x"].dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"capture x must preserve BF16/FP16 runtime input, got {out['x'].dtype}")
    out["x"] = out["x"].to(device=device).contiguous()
    out["topk_ids"] = out["topk_ids"].to(device=device, dtype=torch.int64)
    out["topk_weights"] = out["topk_weights"].to(device=device, dtype=torch.float32)
    if "expert_map" in out and out["expert_map"] is not None:
        out["expert_map"] = out["expert_map"].to(device=device, dtype=torch.int64)
    return out


def _index_keys(checkpoint: Path, layer: int) -> dict[str, str]:
    index_path = checkpoint / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    weight_map = json.loads(index_path.read_text())["weight_map"]
    prefix = f"model.layers.{layer}.ffn.experts.routed_experts."
    wanted = {name: file for name, file in weight_map.items()
              if name.startswith(prefix) and name[len(prefix):] in {
                  "w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"}}
    # Some checkpoint variants use a language_model prefix.  Resolve this
    # explicitly while retaining the exact suffix contract above.
    if len(wanted) != 4:
        wanted = {name: file for name, file in weight_map.items()
                  if name.endswith(f".layers.{layer}.ffn.experts.routed_experts.w13_weight")
                  or name.endswith(f".layers.{layer}.ffn.experts.routed_experts.w13_weight_scale")
                  or name.endswith(f".layers.{layer}.ffn.experts.routed_experts.w2_weight")
                  or name.endswith(f".layers.{layer}.ffn.experts.routed_experts.w2_weight_scale")}
    if len(wanted) != 4:
        matches = [name for name in weight_map if "routed_experts" in name and f"layers.{layer}." in name]
        raise KeyError(f"could not resolve four routed expert tensors; candidates={matches[:20]}")
    return wanted


def _individual_key_map(checkpoint: Path, layer: int, gids: list[int]) -> dict[int, dict[str, tuple[str, str]]]:
    """Resolve the actual checkpoint's per-expert w1/w3/w2 tensors."""
    weight_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    found: dict[int, dict[str, tuple[str, str]]] = {}
    for gid in gids:
        base = f".layers.{layer}.ffn.experts.{gid}."
        per: dict[str, tuple[str, str]] = {}
        for name, filename in weight_map.items():
            if not ('.' + name).endswith((f"{base}w1.weight", f"{base}w3.weight",
                                  f"{base}w2.weight", f"{base}w1.scale",
                                  f"{base}w3.scale", f"{base}w2.scale")):
                continue
            suffix = name.rsplit(".", 2)[-2:]
            per[".".join(suffix)] = (filename, name)
        if len(per) != 6:
            continue
        found[gid] = per
    return found


def _load_individual_rows(checkpoint: Path, keys: dict[int, dict[str, tuple[str, str]]], gids: list[int]):
    """Load only selected per-expert tensors and form the fused w13 layout."""
    torch = _torch()
    from safetensors import safe_open

    w13_rows, w13_scales, w2_rows, w2_scales = [], [], [], []
    for gid in gids:
        per = keys[gid]
        values: dict[str, Any] = {}
        for suffix, (filename, tensor_name) in per.items():
            with safe_open(str(checkpoint / filename), framework="pt", device="cpu") as handle:
                tensor = handle.get_tensor(tensor_name)
            if suffix.endswith('.weight') and tensor.dtype == torch.int8:
                tensor = tensor.view(torch.uint8)
            if suffix.endswith(".scale") and tensor.dtype != torch.uint8:
                if tensor.dtype != torch.float8_e8m0fnu:
                    raise TypeError(f"expert {gid} {suffix} must be raw E8M0 bytes")
                tensor = tensor.view(torch.uint8)
            values[suffix] = tensor
        w13_rows.append(torch.cat((values["w1.weight"], values["w3.weight"]), dim=0))
        w13_scales.append(torch.cat((values["w1.scale"], values["w3.scale"]), dim=0))
        w2_rows.append(values["w2.weight"])
        w2_scales.append(values["w2.scale"])
    return {
        "w13_weight": torch.stack(w13_rows),
        "w13_weight_scale": torch.stack(w13_scales),
        "w2_weight": torch.stack(w2_rows),
        "w2_weight_scale": torch.stack(w2_scales),
    }


def _load_rows(checkpoint: Path, keys: dict[str, str], rows: list[int]):
    torch = _torch()
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("safetensors is required on the GPU host") from exc
    result: dict[str, Any] = {}
    logical_suffixes = (
        "w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"
    )
    for logical in logical_suffixes:
        key = next(name for name in keys if name.endswith(logical))
        filename = keys[key]
        with safe_open(str(checkpoint / filename), framework="pt", device="cpu") as handle:
            view = handle.get_slice(key)
            # safe_open slices are lazy in current safetensors.  Refuse a
            # backend that lacks __getitem__, since full expert tensors are huge.
            if not hasattr(view, "__getitem__"):
                raise RuntimeError(f"safetensors backend cannot row-slice {key}")
            # Use one-row slices because PySafeSlice implementations differ
            # on list indexing; this still transfers only selected experts.
            result[logical] = torch.cat([view[row:row + 1] for row in rows], dim=0)
    return result


def _resolve_rows(ids, expert_map, nrows: int, row_space: str = "auto") -> tuple[list[int], dict[int, int]]:
    gids = sorted({int(value) for value in ids.reshape(-1).tolist()})
    if row_space not in {"auto", "global", "local"}:
        raise ValueError(f"invalid row space: {row_space}")
    if row_space == "global" or (row_space == "auto" and expert_map is not None
                                  and nrows >= expert_map.numel()):
        gids = [gid for gid in gids if expert_map is None or
                (0 <= gid < expert_map.numel() and int(expert_map[gid].item()) >= 0)]
        if not gids:
            raise ValueError("capture routes no local experts on this rank")
        if any(gid < 0 or gid >= nrows for gid in gids):
            raise ValueError("global expert id exceeds global checkpoint rows")
        return gids, {gid: gid for gid in gids}
    if expert_map is None:
        if any(gid < 0 or gid >= nrows for gid in gids):
            raise ValueError("expert_map is required when checkpoint rows are not global")
        return gids, {gid: gid for gid in gids}
    mapped = {}
    for gid in gids:
        if gid < 0 or gid >= expert_map.numel():
            raise ValueError(f"global expert id {gid} outside expert_map")
        lid = int(expert_map[gid].item())
        if lid >= 0:
            mapped[gid] = lid
    rows = sorted(set(mapped.values()))
    if any(row >= nrows for row in rows):
        raise ValueError(f"expert_map local row exceeds checkpoint rows ({nrows})")
    if not rows:
        raise ValueError("capture routes no local experts on this rank")
    return rows, mapped


def _decode_weight(packed, scales, rows: int, k: int):
    torch = _torch()
    raw = packed.to(torch.int64)
    low, high = raw & 15, raw >> 4
    lut = torch.tensor(LUT, dtype=torch.float32, device=packed.device)
    values = torch.stack((lut[low], lut[high]), dim=-1).reshape(*packed.shape[:-1], k)
    scale = torch.exp2(scales.to(torch.float32) - 127.0).repeat_interleave(32, dim=-1)
    if scale.shape[-1] != k:
        raise ValueError(f"scale K mismatch: {scale.shape[-1]} vs {k}")
    return values * scale


def _silu_clamped(gate, up, limit: float | None, gate_first: bool):
    torch = _torch()
    if limit is not None:
        gate = gate.clamp(max=float(limit))
        up = up.clamp(min=-float(limit), max=float(limit))
    return (torch.nn.functional.silu(gate) * up
            if not gate_first else torch.nn.functional.silu(up) * gate)


def _reference(capture, weights, limit: float | None, gate_first: bool, stage_dtype=None):
    torch = _torch()
    x = capture["x"] if stage_dtype is not None else capture["x"].to(torch.float32)
    if stage_dtype is not None:
        x = x.to(stage_dtype)
    ids = capture["topk_ids"]
    weights_router = capture["topk_weights"]
    expert_map = capture.get("expert_map")
    w13, s13, w2, s2 = (weights[name] for name in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"))
    k = x.shape[-1]
    dw13 = _decode_weight(w13, s13, w13.shape[1], k)
    dw2 = _decode_weight(w2, s2, w2.shape[1], w2.shape[2] * 2)
    if stage_dtype is not None:
        dw13, dw2 = dw13.to(stage_dtype), dw2.to(stage_dtype)
    output = torch.zeros((x.shape[0], k), dtype=x.dtype, device=x.device)
    local_by_global = capture["_local_by_global"]
    stage1 = torch.full((x.shape[0], ids.shape[1], dw13.shape[1]), float("nan"), device=x.device)
    stage2 = torch.full((x.shape[0], ids.shape[1], k), float("nan"), device=x.device)
    for token in range(x.shape[0]):
        for slot in range(ids.shape[1]):
            gid = int(ids[token, slot].item())
            if gid not in local_by_global:
                continue
            row = local_by_global[gid]
            gu = x[token] @ dw13[row].T
            stage1[token, slot] = gu.float()
            n = gu.numel() // 2
            act = _silu_clamped(gu[:n], gu[n:], limit, gate_first)
            y = act @ dw2[row].T
            stage2[token, slot] = y.float()
            output[token] += y * weights_router[token, slot]
    return output.float(), {"fc1": stage1, "fc2": stage2}


def _report(name, value, reference):
    torch = _torch()
    delta = (value.float() - reference.float())
    return {"name": name, "shape": list(value.shape),
            "finite": bool(torch.isfinite(value).all().item()),
            "max_abs": float(delta.abs().max().item()),
            "rel_l2": float(delta.norm().item() / reference.float().norm().clamp_min(1e-12).item())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture", help="torch dict with x/topk_ids/topk_weights/expert_map")
    parser.add_argument("--checkpoint", default="/mnt/fuse/models/DeepSeek-V4.1-Flash-local-engram")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-marlin", action="store_true", help="also call vLLM fused_marlin_moe on CUDA")
    parser.add_argument("--run-deepgemm", action="store_true", help="also call vLLM DeepGemm FP8xFP4 on CUDA")
    parser.add_argument("--print-reference", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--swiglu-limit", type=float, default=None)
    parser.add_argument("--gate-first", action="store_true", help="use clamp(gate)*silu(up)")
    parser.add_argument("--eager-repeats", type=int, default=3)
    parser.add_argument("--graph-repeats", type=int, default=3)
    parser.add_argument("--output", help="write JSON report to this path")
    args = parser.parse_args()
    torch = _torch()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    device = torch.device(args.device)
    capture = _load_capture(args.capture, args.synthetic, device)
    layer = args.layer if args.layer is not None else int(capture.get("layer", 0))
    limit = args.swiglu_limit if args.swiglu_limit is not None else capture.get("swiglu_limit")
    if limit is not None:
        limit = float(limit)
    checkpoint = Path(args.checkpoint)
    checkpoint_weight_format = "synthetic"
    if args.synthetic:
        torch.manual_seed(1746)
        e = 8
        n, k = 32, capture["x"].shape[-1]
        w13 = torch.randint(0, 256, (e, 2 * n, k // 2), dtype=torch.uint8, device=device)
        w2 = torch.randint(0, 256, (e, k, n // 2), dtype=torch.uint8, device=device)
        s13 = torch.full((e, 2 * n, k // 32), 120, dtype=torch.uint8, device=device)
        s2 = torch.full((e, k, n // 32), 120, dtype=torch.uint8, device=device)
        weights = {"w13_weight": w13, "w13_weight_scale": s13, "w2_weight": w2, "w2_weight_scale": s2}
        rows, local = _resolve_rows(capture["topk_ids"], capture.get("expert_map"), e)
        weights = {name: value[rows] for name, value in weights.items()}
    else:
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError("safetensors is required") from exc
        expert_map = capture.get("expert_map")
        routed = sorted({int(gid) for gid in capture["topk_ids"].reshape(-1).tolist()
                         if expert_map is None or
                         (0 <= int(gid) < expert_map.numel() and int(expert_map[int(gid)].item()) >= 0)})
        if not routed:
            raise ValueError("capture routes no local experts on this rank")
        individual = _individual_key_map(checkpoint, layer, routed)
        if individual:
            if len(individual) != len(routed):
                missing = sorted(set(routed) - set(individual))
                raise KeyError(f"missing per-expert checkpoint keys for global experts {missing}")
            rows, local = routed, {gid: gid for gid in routed}
            weights = _load_individual_rows(checkpoint, individual, routed)
            checkpoint_weight_format = "per_expert_w1_w3_w2"
        else:
            # Compatibility path for fused routed_experts checkpoints.
            keys = _index_keys(checkpoint, layer)
            w13_key = next(k for k in keys if k.endswith("w13_weight"))
            with safe_open(str(checkpoint / keys[w13_key]), framework="pt", device="cpu") as handle:
                view = handle.get_slice(w13_key)
                if not hasattr(view, "get_shape"):
                    raise RuntimeError("safetensors backend lacks lazy shape metadata")
                shape = tuple(view.get_shape())
            rows, local = _resolve_rows(capture["topk_ids"], expert_map, shape[0])
            weights = _load_rows(checkpoint, keys, rows)
            checkpoint_weight_format = "fused_routed_experts"
        weights = {name: value.to(device=device) for name, value in weights.items()}
    for raw_name in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"):
        if weights[raw_name].dtype != torch.uint8:
            raise TypeError(
                f"{raw_name} must remain uint8 raw MXFP4/UE8M0 bytes; "
                f"got {weights[raw_name].dtype}"
            )
    # Row loads are compact. Convert the original local row ids (or global
    # row ids for a full checkpoint) to the selected tensor's dense row ids.
    row_to_compact = {row: index for index, row in enumerate(rows)}
    local = {gid: row_to_compact[row] for gid, row in local.items()}
    capture["_local_by_global"] = local
    reference, stages = _reference(capture, weights, limit, args.gate_first)
    bf16_reference, bf16_stages = _reference(
        capture, weights, limit, args.gate_first, stage_dtype=torch.bfloat16
    )
    result = {"source": capture["source"], "layer": layer, "mode": "synthetic" if args.synthetic else "real",
              "selected_local_rows": rows, "global_to_local": local,
              "swiglu_limit": limit, "gate_first": args.gate_first,
              "checkpoint_weight_format": checkpoint_weight_format,
              "capture_metadata": capture.get("metadata", capture.get("capture_metadata", {})),
              "reference": {"finite": bool(torch.isfinite(reference).all().item()),
                            "max_abs": float(reference.abs().max().item()),
                            "bf16_stage_reference": {
                                "finite": bool(torch.isfinite(bf16_reference).all().item()),
                                "max_abs": float(bf16_reference.abs().max().item()),
                                "fc1_shape": list(bf16_stages["fc1"].shape),
                                "fc2_shape": list(bf16_stages["fc2"].shape)},
                            "clamp_semantics": "gate=max-only; up=symmetric[-limit,+limit]"}}
    for metadata_key in ("prefix", "expert_backend", "gemm1_clamp_limit",
                         "activation_config", "apply_router_weight_on_input"):
        if metadata_key in capture:
            result[metadata_key] = capture[metadata_key]
    if capture.get("apply_router_weight_on_input", False):
        raise ValueError("apply_router_weight_on_input=true is not implemented by this oracle")
    ids = capture["topk_ids"].to(torch.int32).contiguous()
    global_experts = capture['expert_map'].numel() if capture.get('expert_map') is not None else int(ids.max().item()) + 1
    emap = torch.full((global_experts,), -1, dtype=torch.int32, device=device)
    for gid, row in local.items():
        emap[gid] = row
    if args.print_reference:
        result["reference_output"] = reference.detach().cpu().tolist()
        result["fc1_shape"] = list(stages["fc1"].shape)
        result["fc2_shape"] = list(stages["fc2"].shape)
    if args.run_marlin:
        if capture["x"].dtype != torch.bfloat16:
            raise ValueError("--run-marlin requires exact BF16 capture x")
        if device.type != "cuda":
            raise ValueError("--run-marlin requires --device cuda:N")
        import inspect
        from vllm.model_executor.layers.fused_moe.activation import ApplyMoEActivationConfig
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_moe_mxfp4_layer_for_marlin
        from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
        class LayerStub:
            params_dtype = torch.bfloat16
        config_signature = inspect.signature(ApplyMoEActivationConfig)
        config_values = dict(capture.get("activation_config", {}))
        if not isinstance(config_values, dict):
            raise ValueError("capture['activation_config'] must be a JSON-like dict")
        if "clamp_limit" not in config_values and limit is not None:
            config_values["clamp_limit"] = limit
        config_values.setdefault("alpha", 1.0)
        config_values.setdefault("beta", 0.0)
        if config_values.get("alpha") != 1.0 or config_values.get("beta") != 0.0:
            raise ValueError("reference currently supports only alpha=1, beta=0")
        config_values = {key: value for key, value in config_values.items()
                         if key in config_signature.parameters}
        activation_config = ApplyMoEActivationConfig(**config_values)
        packed = prepare_moe_mxfp4_layer_for_marlin(
            LayerStub(), weights["w13_weight"], weights["w2_weight"],
            weights["w13_weight_scale"], weights["w2_weight_scale"], None, None)
        scalar_types = __import__("vllm.scalar_type", fromlist=["scalar_types"]).scalar_types
        def invoke():
            return fused_marlin_moe(hidden_states=capture["x"],
                                    w1=packed[0], w2=packed[1], bias1=None, bias2=None,
                                    w1_scale=packed[2], w2_scale=packed[3],
                                    topk_weights=capture["topk_weights"], topk_ids=ids,
                                    quant_type_id=scalar_types.float4_e2m1f.id,
                                    global_num_experts=int(emap.numel()), expert_map=emap,
                                    activation_config=activation_config, input_dtype=None)
        eager_values = [invoke().clone() for _ in range(max(1, args.eager_repeats))]
        y = eager_values[-1]
        result["marlin_eager_fp32"] = _report("marlin_eager_fp32", y, reference)
        result["marlin_eager_bf16"] = _report("marlin_eager_bf16", y, bf16_reference)
        result["eager_repeat"] = [_report("eager_repeat", value, y) for value in eager_values]
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y_graph = fused_marlin_moe(hidden_states=capture["x"],
                                       w1=packed[0], w2=packed[1], bias1=None, bias2=None,
                                       w1_scale=packed[2], w2_scale=packed[3],
                                       topk_weights=capture["topk_weights"], topk_ids=ids,
                                       quant_type_id=scalar_types.float4_e2m1f.id,
                                       global_num_experts=int(emap.numel()), expert_map=emap,
                                       activation_config=activation_config, input_dtype=None)
        graph_values = []
        for _ in range(max(1, args.graph_repeats)):
            graph.replay()
            torch.cuda.synchronize(device)
            graph_values.append(y_graph.clone())
        result["marlin_graph_fp32"] = _report("marlin_graph_fp32", graph_values[-1], reference)
        result["marlin_graph_bf16"] = _report("marlin_graph_bf16", graph_values[-1], bf16_reference)
        result["eager_vs_graph"] = _report("eager_vs_graph", y, graph_values[-1])
        result["graph_repeat"] = [_report("graph_repeat", value, graph_values[-1]) for value in graph_values]
    if args.run_deepgemm:
        if capture["x"].dtype != torch.bfloat16:
            raise ValueError("--run-deepgemm requires exact BF16 capture x")
        if device.type != "cuda":
            raise ValueError("--run-deepgemm requires --device cuda:N")
        from types import SimpleNamespace
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.deep_gemm_utils import compute_aligned_M_and_alignment
        from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import DeepGemmFP4Experts
        from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
        from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import _pack_deepgemm_mxfp4_scales
        from vllm.utils.deep_gemm import get_mk_alignment_for_contiguous_layout, DeepGemmQuantScaleFMT

        # Both backends receive the same compact selected rows and map. DeepGEMM
        # performs the production FP8 activation quantization in dg_chain().
        w1_scale, w2_scale = _pack_deepgemm_mxfp4_scales(
            weights["w13_weight"], weights["w2_weight"],
            weights["w13_weight_scale"], weights["w2_weight_scale"])
        dg = SimpleNamespace(w1_scale=w1_scale, w2_scale=w2_scale,
                             gemm1_clamp_limit=limit, _ACT_BLOCK_K=128,
                             _WEIGHT_BLOCK_K=32)
        dg.adjust_N_for_activation = lambda n, activation: n // 2
        dg._act_mul_quant = lambda input, output, activation: DeepGemmFP4Experts._act_mul_quant(
            dg, input, output, activation)
        local_experts = weights["w13_weight"].shape[0]
        align = get_mk_alignment_for_contiguous_layout()[0]
        m_sum, align_used = compute_aligned_M_and_alignment(
            capture["x"].shape[0], ids.shape[1], local_experts, align, None)
        n = weights["w13_weight"].shape[1]
        k = capture["x"].shape[1]
        workspace13 = torch.empty((m_sum, max(n, k)), device=device, dtype=torch.bfloat16)
        workspace2 = torch.empty((m_sum, max(n, k)), device=device, dtype=torch.bfloat16)
        output = torch.empty_like(capture["x"])
        global_num_experts = int(emap.numel())
        use_ue8m0 = DeepGemmQuantScaleFMT.from_oracle() == DeepGemmQuantScaleFMT.UE8M0

        def dg_chain():
            aq, aq_scale = per_token_group_quant_fp8(
                capture["x"], 128, column_major_scales=False, use_ue8m0=use_ue8m0)
            DeepGemmFP4Experts.apply(
                dg, output, aq, weights["w13_weight"], weights["w2_weight"],
                capture["topk_weights"], ids, MoEActivation.SILU,
                global_num_experts, emap, aq_scale, None, workspace13, workspace2,
                None, False)
            return output

        dg_values = [dg_chain().clone() for _ in range(max(1, args.eager_repeats))]
        torch.cuda.synchronize(device)
        dg_y = dg_values[-1].float().clone()
        result["deepgemm_eager_fp32"] = _report("deepgemm_eager_fp32", dg_y, reference)
        result["deepgemm_eager_bf16"] = _report("deepgemm_eager_bf16", dg_y, bf16_reference)
        result["deepgemm_eager_repeat"] = [_report("deepgemm_eager_repeat", value.float(), dg_y)
                                             for value in dg_values]
        dg_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(dg_graph):
            dg_graph_value = dg_chain()
        dg_graph_values = []
        for _ in range(max(1, args.graph_repeats)):
            dg_graph.replay()
            torch.cuda.synchronize(device)
            dg_graph_values.append(dg_graph_value.float().clone())
        dg_last = dg_graph_values[-1]
        result["deepgemm_graph_fp32"] = _report("deepgemm_graph_fp32", dg_last, reference)
        result["deepgemm_graph_bf16"] = _report("deepgemm_graph_bf16", dg_last, bf16_reference)
        result["deepgemm_eager_vs_graph"] = _report("deepgemm_eager_vs_graph", dg_y, dg_last)
        result["deepgemm_graph_repeat"] = [_report("deepgemm_graph_repeat", value, dg_last)
                                             for value in dg_graph_values]
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n")


if __name__ == "__main__":
    main()
