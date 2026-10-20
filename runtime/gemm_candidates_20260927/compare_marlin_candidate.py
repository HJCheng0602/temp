"""Bounded genuine vLLM Marlin MXFP4 W4A16 MoE candidate.

The operands are synthetic but deliberately shared with the FP8xFP4 candidates:
OCP E2M1 packed weights and constant UE8M0 (2^-7) scales.  This measures the
Marlin weight-only path with BF16 activations; it is not an end-to-end model
benchmark.  The script does not perform any work until ``--run`` is supplied.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path

# Marlin's host-side setup should not add an uncontrolled CPU thread pool.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")

import torch


HIDDEN = 5120
INTERMEDIATE = 2304
GLOBAL_EXPERTS = 384
LOCAL_EXPERTS = 96
TOPK = 6
SCALE_BYTE = 0x78  # UE8M0 exponent 120 = 2**-7
LUT = (0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6)


def _check_contract():
    """Resolve APIs and report their signatures without importing guessed names."""
    from vllm.model_executor.layers.quantization.utils import marlin_utils_fp4
    from vllm.model_executor.layers.fused_moe.experts import marlin_moe

    pack = getattr(marlin_utils_fp4, "prepare_moe_mxfp4_layer_for_marlin", None)
    fused = getattr(marlin_moe, "fused_marlin_moe", None)
    if pack is None or fused is None:
        raise RuntimeError(
            "installed vLLM lacks the current MXFP4 Marlin API: "
            "need prepare_moe_mxfp4_layer_for_marlin and fused_marlin_moe"
        )
    return pack, fused


class _LayerStub:
    # The upstream pure packing helper only reads params_dtype.  Keeping this a
    # tiny object avoids constructing a full FusedMoE module for the microbench.
    params_dtype = torch.bfloat16


def _packed_weights(seed: int, device: torch.device):
    gen = torch.Generator(device=device).manual_seed(seed)
    w13 = torch.randint(
        0, 256, (LOCAL_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2),
        dtype=torch.uint8, device=device, generator=gen,
    ).contiguous()
    w2 = torch.randint(
        0, 256, (LOCAL_EXPERTS, HIDDEN, INTERMEDIATE // 2),
        dtype=torch.uint8, device=device, generator=gen,
    ).contiguous()
    s13 = torch.full(
        (LOCAL_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 32),
        SCALE_BYTE, dtype=torch.uint8, device=device,
    )
    s2 = torch.full(
        (LOCAL_EXPERTS, HIDDEN, INTERMEDIATE // 32),
        SCALE_BYTE, dtype=torch.uint8, device=device,
    )
    return w13, w2, s13, s2


def _routes(path: str | None, batch: int, seed: int, device: torch.device):
    if path is None:
        gen = torch.Generator(device=device).manual_seed(seed + 101)
        return torch.rand((batch, GLOBAL_EXPERTS), device=device, generator=gen).topk(
            TOPK, dim=-1
        ).indices.to(torch.int32)
    payload = json.loads(Path(path).read_text())
    values = (payload.get("topk_ids", payload.get("routes", payload))
              if isinstance(payload, dict) else payload)
    ids = torch.tensor(values, dtype=torch.int32, device=device)
    if ids.shape != (batch, TOPK):
        raise ValueError(f"route file must contain shape {(batch, TOPK)}, got {tuple(ids.shape)}")
    if ids.min().item() < 0 or ids.max().item() >= GLOBAL_EXPERTS:
        raise ValueError("route expert IDs must be in [0, 384)")
    if any(len(set(row)) != TOPK for row in ids.tolist()):
        raise ValueError("each route must contain six distinct experts")
    return ids


def _reference(x, routes, w13, w2):
    raw_lut = torch.tensor(LUT, dtype=torch.float32, device=x.device)

    def unpack(w, e):
        raw = w[e].view(torch.uint8)
        return torch.stack((raw_lut[(raw & 15).long()], raw_lut[(raw >> 4).long()]), -1).flatten(1) / 128

    refs = torch.zeros(x.shape[0], HIDDEN, dtype=torch.float32, device=x.device)
    for t in range(x.shape[0]):
        for ge in routes[t].tolist():
            if ge >= LOCAL_EXPERTS:
                continue
            gate, up = (x[t].float() @ unpack(w13, ge).T).chunk(2)
            refs[t] += (up * torch.nn.functional.silu(gate)) @ unpack(w2, ge).T / TOPK
    return refs


def run_case(batch: int, args):
    pack, fused = _check_contract()
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    x = (torch.randn(batch, HIDDEN, device=device, dtype=torch.bfloat16) * 0.1).contiguous()
    routes = _routes(args.route_file, batch, args.seed, device)
    # A tiny batch can randomly miss this EP rank entirely; force one local
    # assignment so correctness is not vacuous while preserving six uniques.
    if not (routes < LOCAL_EXPERTS).any().item():
        routes[0, 0] = 0
    weights = torch.full((batch, TOPK), 1 / TOPK, device=device, dtype=torch.float32)
    # -1 tells vLLM that a global expert is remote to this EP rank.
    expert_map = torch.arange(GLOBAL_EXPERTS, device=device, dtype=torch.int32)
    expert_map[LOCAL_EXPERTS:] = -1
    w13, w2, s13, s2 = _packed_weights(args.seed + 17, device)
    w13_m, w2_m, s13_m, s2_m, _, _ = pack(_LayerStub(), w13, w2, s13, s2, None, None)
    scalar_types = __import__("vllm.scalar_type", fromlist=["scalar_types"]).scalar_types
    kwargs = dict(
        hidden_states=x, w1=w13_m, w2=w2_m, bias1=None, bias2=None,
        w1_scale=s13_m, w2_scale=s2_m, topk_weights=weights,
        topk_ids=routes, quant_type_id=scalar_types.float4_e2m1f.id,
        global_num_experts=GLOBAL_EXPERTS, expert_map=expert_map,
        input_dtype=None,
    )
    # Keep this adapter compatible with minor upstream signature additions,
    # while refusing to silently omit a required known argument.
    sig = inspect.signature(fused).parameters
    missing = [k for k, p in sig.items() if p.default is inspect.Parameter.empty and k not in kwargs]
    if missing:
        raise RuntimeError(f"unsupported fused_marlin_moe signature; missing {missing}")
    if "expert_map" not in sig:
        raise RuntimeError("installed fused_marlin_moe lacks expert_map; cannot honor EP4")
    kwargs = {k: v for k, v in kwargs.items() if k in sig}
    y_warm = fused(**kwargs)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            y_graph = fused(**kwargs)
    graph.replay()
    torch.cuda.synchronize(device)
    y = y_graph if isinstance(y_graph, torch.Tensor) else y_warm
    sample = y.float()
    active = (routes < LOCAL_EXPERTS).any(dim=1).nonzero(as_tuple=False).flatten()
    check_ids = active[:4].tolist() + active[-4:].tolist()
    check_ids = sorted(set(check_ids))
    if not check_ids:
        raise AssertionError("route set has no local EP4 assignments")
    ref = _reference(x[check_ids], routes[check_ids], w13, w2)
    rel = ((sample[check_ids] - ref).norm() / ref.norm().clamp_min(1e-12)).item()
    if not torch.isfinite(sample).all().item():
        raise AssertionError("Marlin output is non-finite")
    if rel > args.tolerance:
        raise AssertionError(f"relative L2 {rel:.6g} exceeds {args.tolerance}")
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    samples = []
    for _ in range(7):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 10)
    row = dict(backend="marlin_mxfp4_w4a16", batch=batch, experts=LOCAL_EXPERTS,
               global_experts=GLOBAL_EXPERTS, topk=TOPK, hidden=HIDDEN,
               intermediate=INTERMEDIATE, relative_l2=rel,
               us=sorted(samples)[len(samples) // 2],
               active_tokens=len(active),
               active_assignments=int((routes < LOCAL_EXPERTS).sum().item()),
               weight_seed=args.seed + 17, route_file=args.route_file,
               graph=True, status="ok")
    print(json.dumps(row), flush=True)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="store_true", help="run CUDA cases; default only validates arguments")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32, 128, 256])
    p.add_argument("--route-file")
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--tolerance", type=float, default=.05)
    p.add_argument("--output")
    args = p.parse_args()
    if not args.run:
        print(json.dumps(dict(status="validated", batches=args.batches, hidden=HIDDEN,
                              intermediate=INTERMEDIATE, local_experts=LOCAL_EXPERTS,
                              global_experts=GLOBAL_EXPERTS, topk=TOPK,
                              message="pass --run to launch CUDA")))
        return
    rows = []
    for batch in args.batches:
        try:
            rows.append(run_case(batch, args))
        except Exception as exc:
            row = dict(backend="marlin_mxfp4_w4a16", batch=batch,
                       status="error", error=f"{type(exc).__name__}: {exc}")
            rows.append(row)
            print(json.dumps(row), flush=True)
        if args.output:
            Path(args.output).write_text(json.dumps({"rows": rows}, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
