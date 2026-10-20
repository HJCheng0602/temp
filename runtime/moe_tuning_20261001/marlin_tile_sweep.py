"""Real-capture, single-GPU Marlin block-M sweep.

This is a local diagnostic for the installed vLLM environment.  It loads only
the selected checkpoint expert rows, runs the fused Marlin MoE chain on real
captured inputs/routes, and forces the host alignment block size to each value
requested.  No synthetic fallback is provided: a capture is required.

Example (on the GPU host, from the repository root)::

    python deployment/moe_tuning_20261001/marlin_tile_sweep.py \
      --capture /path/to/moe_capture.pt \
      --checkpoint /mnt/fuse/models/DeepSeek-V4.1-Flash-local-engram \
      --layer 20 --output /tmp/marlin_tile_sweep.json
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import statistics
import sys
from contextlib import contextmanager
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
HELPERS = HERE / "fast_gemm_trial"
if str(HELPERS) not in sys.path:
    sys.path.insert(0, str(HELPERS))

from marlin_real_oracle import (  # noqa: E402
    _load_capture,
    _reference,
    _report,
)
from marlin_repeat_probe import _load_selected  # noqa: E402


BLOCKS = (8, 16, 32, 48, 64)


def _digest(tensor):
    import torch

    return hashlib.sha256(
        tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _subset(capture, batch, device):
    import torch

    if batch <= 0 or batch > int(capture["x"].shape[0]):
        raise ValueError(
            f"requested batch {batch} exceeds capture rows {capture['x'].shape[0]}"
        )
    out = dict(capture)
    out["x"] = capture["x"][:batch].contiguous().to(device)
    out["topk_ids"] = capture["topk_ids"][:batch].contiguous().to(device, dtype=torch.int64)
    out["topk_weights"] = capture["topk_weights"][:batch].contiguous().to(device, dtype=torch.float32)
    if capture.get("expert_map") is not None:
        out["expert_map"] = capture["expert_map"].to(device, dtype=torch.int64)
    return out


@contextmanager
def _forced_block_size(module, block_size):
    """Force alignment and the identical block size passed to the GEMM op."""
    original_align = module.moe_align_block_size
    original_fused = module._fused_marlin_moe

    def forced(topk_ids, _block_size, *args, **kwargs):
        return original_align(topk_ids, int(block_size), *args, **kwargs)

    def fused(*args, **kwargs):
        names = (
            "hidden_states", "w1", "w2", "bias1", "bias2", "w1_scale",
            "w2_scale", "topk_weights", "num_topk", "quant_type",
            "apply_router_weight_on_input", "expert_map", "block_size_m",
            "sorted_token_ids", "expert_ids", "num_tokens_post_padded",
        )
        if "block_size_m" in kwargs:
            kwargs = dict(kwargs)
            kwargs["block_size_m"] = int(block_size)
            return original_fused(*args, **kwargs)
        if len(args) <= names.index("block_size_m"):
            raise RuntimeError("unexpected Marlin _fused_marlin_moe signature")
        rewritten = list(args)
        rewritten[names.index("block_size_m")] = int(block_size)
        return original_fused(*rewritten, **kwargs)

    module.moe_align_block_size = forced
    module._fused_marlin_moe = fused
    try:
        yield
    finally:
        module.moe_align_block_size = original_align
        module._fused_marlin_moe = original_fused


def _activation_config(capture, limit):
    from vllm.model_executor.layers.fused_moe.activation import ApplyMoEActivationConfig

    values = dict(capture.get("activation_config", {}))
    values.setdefault("clamp_limit", float(limit))
    values.setdefault("alpha", 1.0)
    values.setdefault("beta", 0.0)
    if values["clamp_limit"] != float(limit):
        raise ValueError(
            f"capture clamp_limit={values['clamp_limit']} does not match requested {limit}"
        )
    if values.get("alpha") != 1.0 or values.get("beta") != 0.0:
        raise ValueError("this sweep requires alpha=1 and beta=0")
    signature = inspect.signature(ApplyMoEActivationConfig)
    return ApplyMoEActivationConfig(
        **{key: value for key, value in values.items() if key in signature.parameters}
    )


def _prepare(capture, checkpoint, layer, device):
    import torch
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        prepare_moe_mxfp4_layer_for_marlin,
    )

    weights, emap = _load_selected(capture, checkpoint, layer, device)
    class LayerStub:
        params_dtype = torch.bfloat16

    packed = prepare_moe_mxfp4_layer_for_marlin(
        LayerStub(), weights["w13_weight"], weights["w2_weight"],
        weights["w13_weight_scale"], weights["w2_weight_scale"], None, None,
    )
    capture = dict(capture)
    capture["_local_by_global"] = {
        int(gid): int(row) for gid, row in enumerate(emap.tolist()) if int(row) >= 0
    }
    return weights, emap, packed


def _invoke_factory(module, capture, weights, packed, emap, activation_config):
    import torch
    from vllm.scalar_type import scalar_types
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe

    ids = capture["topk_ids"].to(torch.int32).contiguous()
    x = capture["x"]
    if x.dtype != torch.bfloat16:
        raise ValueError(f"real Marlin sweep requires BF16 capture x, got {x.dtype}")
    topk = capture["topk_weights"]
    # Raw fused W13 has [gate;up] rows, so N is half its logical row count.
    # Deriving this from raw weights matches MarlinExperts.workspace_shapes;
    # packed storage layouts are intentionally not used for this calculation.
    n = weights["w13_weight"].shape[1] // 2
    k = x.shape[1]
    workspace13 = torch.empty((x.shape[0] * ids.shape[1], max(n, k)), device=x.device, dtype=torch.bfloat16)
    workspace2 = torch.empty((x.shape[0] * ids.shape[1] * max(2 * n, k),), device=x.device, dtype=torch.bfloat16)
    workspace = module.marlin_make_workspace_new(x.device, 4)
    output = torch.empty((x.shape[0], k), device=x.device, dtype=x.dtype)
    values = dict(hidden_states=x, w1=packed[0], w2=packed[1], bias1=None, bias2=None,
                  w1_scale=packed[2], w2_scale=packed[3], topk_weights=topk,
                  topk_ids=ids, quant_type_id=scalar_types.float4_e2m1f.id,
                  global_num_experts=int(emap.numel()), expert_map=emap,
                  activation=MoEActivation.SILU,
                  activation_config=activation_config, input_dtype=None,
                  workspace=workspace, intermediate_cache13=workspace2,
                  intermediate_cache2=workspace13, output=output)
    signature = inspect.signature(fused_marlin_moe)
    values = {key: value for key, value in values.items() if key in signature.parameters}

    def invoke():
        return fused_marlin_moe(**values)

    return invoke


def _timed_graph(invoke, repeats, warmup_replays=10, replays_per_sample=10):
    from graph_measure import measure_graph
    return measure_graph(invoke, repeats, replays_per_sample, warmup_replays)


def _run_batch(capture, weights, emap, packed, blocks, repeats, limit,
               warmup_replays, replays_per_sample, canonical_order, profile_directory=None):
    import torch
    import vllm.model_executor.layers.fused_moe.experts.marlin_moe as marlin_module

    # Install the existing canonical-order diagnostic before any graph capture.
    from marlin_order_diagnostic import install as install_canonical
    install_canonical(marlin_module)
    previous_order = os.environ.get("DSV41_MARLIN_CANONICAL_ORDER")
    if canonical_order:
        os.environ["DSV41_MARLIN_CANONICAL_ORDER"] = "1"
    else:
        os.environ.pop("DSV41_MARLIN_CANONICAL_ORDER", None)
    try:
        activation_config = _activation_config(capture, limit)
        # The independent FP32 oracle is evaluated once per batch and is kept
        # separate from timing; it uses the same selected real checkpoint rows.
        fp32_reference, _ = _reference(capture, weights, limit, False)
        rows = []
        reference_output = None
        # Always establish the comparison reference with the production tile
        # first, even when the caller requests a subset of candidate blocks.
        sweep_blocks = tuple(dict.fromkeys((8,) + tuple(blocks)))
        for block in sweep_blocks:
            with _forced_block_size(marlin_module, block):
                invoke = _invoke_factory(marlin_module, capture, weights, packed, emap, activation_config)
                value, samples, cold_samples, repeated_outputs = _timed_graph(
                    invoke, repeats, warmup_replays, replays_per_sample)
                profile = None
                if profile_directory:
                    from graph_measure import profile_graph
                    profile = profile_graph(invoke, Path(profile_directory) / f"b{capture['x'].shape[0]}_m{block}.json")
                    ids = capture['topk_ids'].long()
                    assignments = int(((ids >= 0) & (emap[ids.clamp_min(0)] >= 0)).sum().item())
                    profile['local_assignments'] = assignments
                    profile['gate_up_mfu_pct'] = 100 * (2 * assignments * 5120 * 4608) / (profile['gate_up_us'] * 1e-6 * 258e12)
                    profile['down_mfu_pct'] = 100 * (2 * assignments * 2304 * 5120) / (profile['down_us'] * 1e-6 * 258e12)
            if reference_output is None:
                reference_output = value.clone()
            if block == 8:
                reference_output = value.clone()
            row = {
                "block_size_m": block,
                "median_us": statistics.median(samples),
                "samples_us": samples,
                "cold_median_us": statistics.median(cold_samples),
                "cold_samples_us": cold_samples,
                "cache_displacement_bytes": 256 * 1024 * 1024,
                "repeat_hashes": [_digest(output) for output in repeated_outputs],
                "repeat_exact": len({_digest(output) for output in repeated_outputs}) == 1,
                "output": _report(f"marlin_block_{block}", value, fp32_reference),
                "vs_block8": _report(f"block_{block}_vs_block8", value, reference_output),
                "finite": bool(torch.isfinite(value).all().item()),
                "profile": profile,
            }
            rows.append(row)
            del invoke, value
            torch.cuda.empty_cache()
        return rows
    finally:
        if previous_order is None:
            os.environ.pop("DSV41_MARLIN_CANONICAL_ORDER", None)
        else:
            os.environ["DSV41_MARLIN_CANONICAL_ORDER"] = previous_order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, help="real torch capture dictionary")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batches", nargs="+", type=int, default=[8, 32, 64])
    parser.add_argument("--blocks", nargs="+", type=int, default=list(BLOCKS), choices=BLOCKS)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-replays", type=int, default=10)
    parser.add_argument("--replays-per-sample", type=int, default=10)
    parser.add_argument("--swiglu-limit", type=float, default=10.0)
    parser.add_argument("--canonical-order", action="store_true",
                        help="opt into deterministic canonical dispatch ordering")
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile-directory")
    args = parser.parse_args()

    if args.repeats < 1 or args.warmup_replays < 1 or args.replays_per_sample < 1:
        raise ValueError("repeats and graph replay counts must be positive")
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("this sweep requires a CUDA device")
    torch.cuda.set_device(device)
    capture = _load_capture(args.capture, False, device)
    layer = args.layer if args.layer is not None else int(capture.get("layer", 0))
    if capture.get("swiglu_limit", args.swiglu_limit) != args.swiglu_limit:
        raise ValueError("capture swiglu_limit does not match --swiglu-limit")
    if capture.get("apply_router_weight_on_input", False):
        raise ValueError("apply_router_weight_on_input=true is unsupported")
    if capture["x"].shape[1] != 5120 or capture["topk_ids"].shape[1] != 6:
        raise ValueError("expected real DeepSeek V4 H=5120 and top-k=6 capture")
    checkpoint = Path(args.checkpoint)
    weights, emap, packed = _prepare(capture, checkpoint, layer, device)
    capture["_local_by_global"] = {
        int(gid): int(row) for gid, row in enumerate(emap.tolist()) if int(row) >= 0
    }
    results = []
    for batch in args.batches:
        current = _subset(capture, batch, device)
        current["_local_by_global"] = capture["_local_by_global"]
        results.append({"batch": batch, "rows": _run_batch(
            current, weights, emap, packed,
            tuple(args.blocks), args.repeats, args.swiglu_limit,
            args.warmup_replays, args.replays_per_sample,
            args.canonical_order,
            args.profile_directory,
        )})
    payload = {
        "schema": "real_marlin_block_m_sweep_v1",
        "source": str(args.capture),
        "checkpoint": str(checkpoint),
        "mode": "real_capture_only",
        "device": str(device),
        "layer": layer,
        "batches": args.batches,
        "blocks": args.blocks,
        "repeats": args.repeats,
        "warmup_replays": args.warmup_replays,
        "replays_per_sample": args.replays_per_sample,
        "hidden": 5120,
        "topk": 6,
        "swiglu_limit": args.swiglu_limit,
        "canonical_order": args.canonical_order,
        "weights_selected_local_rows": [int(x) for x in sorted(set(emap[emap >= 0].tolist()))],
        "route_rows": int(capture["x"].shape[0]),
        "results": results,
        "limitations": [
            "Timing is single-GPU fused-chain timing, not PP/TP/EP service latency.",
            "Captured routes are real, but only the supplied layer/input rows are covered.",
            "The custom Marlin op's thread-K and nblocks are not exposed by this wrapper.",
        ],
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(path), "schema": payload["schema"], "batches": args.batches}))


if __name__ == "__main__":
    main()
