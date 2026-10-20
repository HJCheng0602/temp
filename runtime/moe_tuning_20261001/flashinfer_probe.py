"""Real-capture FlashInfer SM120 W4A16/MXFP4 probe.

This is intentionally a host-side probe rather than a vLLM integration.  It
loads the selected expert rows using the existing Marlin capture loader,
prepares the upstream b12x ``fp4_e8m0_k32`` W4A16 layout, and calls the
low-level FlashInfer runner with the DeepSeek clamp parameters.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
HELPERS = HERE / "fast_gemm_trial"
if str(HELPERS) not in sys.path:
    sys.path.insert(0, str(HELPERS))

from marlin_repeat_probe import _load_selected  # noqa: E402
from marlin_real_oracle import _load_capture, _reference, _report  # noqa: E402


def _tile(capture, batch, device):
    import torch

    rows = int(capture["x"].shape[0])
    idx = torch.arange(batch, device=device) % rows
    return (
        capture["x"].to(device).index_select(0, idx).contiguous(),
        capture["topk_ids"].to(device).index_select(0, idx).to(torch.int32).contiguous(),
        capture["topk_weights"].to(device).index_select(0, idx).contiguous(),
    )


def _install_force_tile(force_tile):
    """Optionally inject force_tile_config into upstream compilation."""
    if force_tile is None:
        return None, None
    import flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_kernel as kernel

    original = kernel.compile_w4a16_fused_moe
    seen = []

    def forced(**kwargs):
        kwargs["force_tile_config"] = tuple(force_tile)
        seen.append({
            "size_m": kwargs.get("size_m"),
            "hidden_size": kwargs.get("hidden_size"),
            "intermediate_size": kwargs.get("intermediate_size"),
            "num_experts": kwargs.get("num_experts"),
            "top_k": kwargs.get("top_k"),
            "moe_block_size": kwargs.get("moe_block_size"),
            "scale_format": kwargs.get("scale_format"),
            "force_tile_config": list(force_tile),
        })
        return original(**kwargs)

    kernel.compile_w4a16_fused_moe = forced
    return original, seen


def _prepare(weights, device, w13_layout):
    import torch
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_prepare import (
        prepare_w4a16_fp4_e8m0_k32_weights,
    )

    tensors = {name: value.to(device=device) for name, value in weights.items()}
    for name in ("w13_weight_scale", "w2_weight_scale"):
        over = int((tensors[name].to(torch.uint8) > 247).sum().item())
        if over:
            raise ValueError(
                f"{name} contains {over} E8M0 bytes above 247; refusing "
                "FlashInfer's saturating BF16 conversion"
            )
    # Native MXFP4 E2M1/UE8M0-K32 weights have no additional per-expert global
    # scale in the capture contract.  Keep this explicit in the report.
    experts = tensors["w13_weight"].shape[0]
    one = torch.ones(experts, dtype=torch.float32, device=device)
    prepared = prepare_w4a16_fp4_e8m0_k32_weights(
        tensors["w13_weight"], tensors["w13_weight_scale"], one,
        tensors["w2_weight"], tensors["w2_weight_scale"], one,
        activation="silu", params_dtype=torch.bfloat16, w13_layout=w13_layout,
    )
    return tensors, prepared


def _fi_call(x, ids, weights, prepared, emap, buffers):
    import flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_kernel as kernel

    return kernel.run_w4a16_moe(
        x, prepared, weights, ids, activation="silu",
        intermediate_cache13=buffers.intermediate_cache13,
        intermediate_cache2=buffers.intermediate_cache2,
        output=buffers.output, fc1_c_tmp=buffers.fc1_c_tmp,
        fc2_c_tmp=buffers.fc2_c_tmp,
        packed_route_indices=buffers.packed_route_indices,
        block_expert_ids=buffers.block_expert_ids,
        packed_route_count=buffers.packed_route_count,
        expert_offsets=buffers.expert_offsets, expert_map=emap,
        fast_math=True, swiglu_limit=10.0, swiglu_alpha=1.0,
        swiglu_beta=0.0,
    )


def _timed(fn, repeats):
    import torch
    from graph_measure import measure_graph
    output, hot, cold, replay_outputs = measure_graph(fn, 5, repeats, 10)
    samples = [x / 1000 for x in hot]
    return output, {
        "count": len(samples),
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
        "cold_median_ms": statistics.median(cold) / 1000,
        "cold_samples_ms": [x / 1000 for x in cold],
        "cache_displacement_bytes": 256 * 1024 * 1024,
        "replays_per_sample": repeats,
        "graph_repeat_exact": all(torch.equal(replay_outputs[0], x) for x in replay_outputs[1:]),
    }


def _repeat_report(values):
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
    return rows


def _reference_for_batch(capture, x, ids, route_weights, raw, emap):
    ref_capture = dict(capture)
    ref_capture["x"] = x
    ref_capture["topk_ids"] = ids
    ref_capture["topk_weights"] = route_weights
    ref_capture["expert_map"] = None
    ref_capture["_local_by_global"] = {
        int(gid): int(row) for gid, row in enumerate(emap.tolist()) if int(row) >= 0
    }
    return _reference(ref_capture, raw, 10.0, False)[0]


def _marlin_baseline(x, ids, route_weights, raw, emap, repeats):
    import inspect
    from vllm.model_executor.layers.fused_moe.activation import ApplyMoEActivationConfig, MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        prepare_moe_mxfp4_layer_for_marlin,
    )
    from vllm.scalar_type import scalar_types

    class LayerStub:
        params_dtype = x.dtype

    packed = prepare_moe_mxfp4_layer_for_marlin(
        LayerStub(), raw["w13_weight"], raw["w2_weight"],
        raw["w13_weight_scale"], raw["w2_weight_scale"], None, None,
    )
    activation_values = {"clamp_limit": 10.0, "alpha": 1.0, "beta": 0.0}
    activation_values = {
        key: value for key, value in activation_values.items()
        if key in inspect.signature(ApplyMoEActivationConfig).parameters
    }
    values = dict(
        hidden_states=x, w1=packed[0], w2=packed[1], bias1=None, bias2=None,
        w1_scale=packed[2], w2_scale=packed[3], topk_weights=route_weights,
        topk_ids=ids, quant_type_id=scalar_types.float4_e2m1f.id,
        global_num_experts=int(emap.numel()), expert_map=emap,
        activation=MoEActivation.SILU,
        activation_config=ApplyMoEActivationConfig(**activation_values),
        input_dtype=None,
    )
    values = {key: value for key, value in values.items()
              if key in inspect.signature(fused_marlin_moe).parameters}
    eager = [fused_marlin_moe(**values).clone() for _ in range(repeats)]
    return eager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--derived", action="store_true",
                        help="permit deterministic modulo tiling when capture has fewer rows")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--force-tile", nargs=4, type=int, metavar=("K1", "N1", "K2", "N2"),
        help="optional FC1/FC2 tile override; upstream validates it",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--w13-layout", choices=("w31", "w13"), default="w31",
        help="logical FC1 row order: w31=gate/up (loader output), w13=up/gate",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import torch

    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(device)
    capture = _load_capture(args.capture, False, device)
    layer = args.layer if args.layer is not None else int(capture.get("layer", 0))
    if capture["x"].dtype != torch.bfloat16:
        raise ValueError("capture x must be BF16")
    if args.batch > int(capture["x"].shape[0]) and not args.derived:
        raise ValueError(
            f"capture has {capture['x'].shape[0]} rows but --batch={args.batch}; "
            "pass --derived to permit modulo tiling"
        )
    x, ids, route_weights = _tile(capture, args.batch, device)
    weights, emap = _load_selected(capture, Path(args.checkpoint), layer, device)
    raw, prepared = _prepare(weights, device, args.w13_layout)
    if int(prepared.hidden_size) != int(x.shape[1]):
        raise ValueError(f"hidden mismatch: weights={prepared.hidden_size}, x={x.shape[1]}")
    if int(prepared.intermediate_size) != int(raw["w13_weight"].shape[1] // 2):
        raise ValueError("invalid gated W13 shape")
    if int(ids.shape[1]) != 6:
        raise ValueError(f"expected top-k=6, got {ids.shape[1]}")
    if int(raw["w13_weight"].shape[0]) > 96:
        raise ValueError(
            f"selected capture has {raw['w13_weight'].shape[0]} local experts; "
            "the loader compacted only routed local rows (expected <=96)"
        )

    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_host import (
        make_w4a16_packed_buffers,
    )
    buffers = make_w4a16_packed_buffers(
        prepared, m=args.batch, topk=int(ids.shape[1]), dtype=torch.bfloat16,
        device=device, route_num_experts=int(emap.numel()),
    )
    original_compile, compiled_tiles = _install_force_tile(args.force_tile)
    def invoke():
        return _fi_call(x, ids, route_weights, prepared, emap, buffers)
    try:
        first, timing = _timed(invoke, args.repeats)
        eager = [invoke().clone() for _ in range(args.repeats)]
        torch.cuda.synchronize()
    finally:
        if original_compile is not None:
            import flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_kernel as kernel
            kernel.compile_w4a16_fused_moe = original_compile
    if args.force_tile is not None and not compiled_tiles:
        raise RuntimeError("force-tile was requested but no fused compile was observed")

    reference = _reference_for_batch(capture, x, ids, route_weights, raw, emap)
    fi_reference = _report("flashinfer_fp32", first, reference)
    try:
        marlin_values = _marlin_baseline(
            x, ids, route_weights, raw, emap, min(args.repeats, 5)
        )
        marlin_report = _report("marlin_fp32", marlin_values[-1], reference)
        marlin_repeat = [_report("marlin_repeat", value, marlin_values[-1])
                         for value in marlin_values]
        marlin_error = _report("flashinfer_vs_marlin", first, marlin_values[-1])
    except Exception as exc:  # Environment may omit vLLM's Marlin extension.
        marlin_report = {"status": "unavailable", "error": repr(exc)}
        marlin_repeat = []
        marlin_error = {"status": "unavailable"}

    result = {
        "schema": "flashinfer_sm120_w4a16_probe_v1",
        "layer": layer,
        "batch": args.batch,
        "batch_provenance": "captured" if args.batch <= int(capture["x"].shape[0]) else "derived_tile",
        "capture_rows": int(capture["x"].shape[0]),
        "hidden_size": int(x.shape[1]),
        "intermediate_size": int(prepared.intermediate_size),
        "global_experts": int(emap.numel()),
        "local_experts": int(raw["w13_weight"].shape[0]),
        "local_expert_provenance": "compact selected routed rows from the rank's 96-expert shard",
        "top_k": int(ids.shape[1]),
        "weight_contract": (
            "fp4_e2m1 + ue8m0_k32, source_format=fp4_e8m0_k32, "
            f"w13_layout={args.w13_layout}"
        ),
        "activation_contract": "silu with gate max clamp 10 and up +/- clamp 10",
        "clamp_forwarding": "direct low-level run_w4a16_moe arguments",
        "force_tile": list(args.force_tile) if args.force_tile is not None else None,
        "compiled_tile_kwargs": compiled_tiles,
        "global_scale_contract": "unit per-expert global scales; verify against checkpoint metadata",
        "timing": timing,
        "repeat_self_report": _repeat_report(eager),
        "flashinfer_vs_fp32_reference": fi_reference,
        "marlin": marlin_report,
        "marlin_repeat_vs_last": marlin_repeat,
        "flashinfer_vs_marlin": marlin_error,
        "output_finite": bool(torch.isfinite(first).all().item()),
        "output_shape": list(first.shape),
        "note": "Single-GPU compact-expert replay of real captured inputs; excludes EP communication and is not service TPOT.",
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
