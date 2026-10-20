"""Bounded DeepGEMM FP4 EP4 local-MoE benchmark.

This follows the installed vLLM DeepGemmFP4Experts.apply path.  It uses fixed
synthetic weights/routes shared in shape with compare_marlin_candidate.py and
does not start any work unless --run is supplied.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")

import torch

HIDDEN = 5120
INTERMEDIATE = 2304
GLOBAL_EXPERTS = 384
LOCAL_EXPERTS = 96
TOPK = 6
ACT_BLOCK_K = 128
FP4_LUT = (0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6)


def routes_for(batch: int, seed: int, device: torch.device, path: str | None):
    if path:
        payload = json.loads(Path(path).read_text())
        values = payload.get("topk_ids", payload.get("routes", payload))
        ids = torch.tensor(values, dtype=torch.int32, device=device)
        if tuple(ids.shape) != (batch, TOPK):
            raise ValueError(f"route file shape must be {(batch, TOPK)}, got {tuple(ids.shape)}")
    else:
        gen = torch.Generator(device=device).manual_seed(seed + 101)
        ids = torch.rand((batch, GLOBAL_EXPERTS), device=device, generator=gen).topk(TOPK, dim=-1).indices.to(torch.int32)
    if ids.min().item() < 0 or ids.max().item() >= GLOBAL_EXPERTS:
        raise ValueError("route expert IDs must be in [0, 384)")
    if any(len(set(row)) != TOPK for row in ids.tolist()):
        raise ValueError("routes must contain six distinct experts per token")
    if not (ids < LOCAL_EXPERTS).any().item():
        ids[0, 0] = 0
    return ids.contiguous()


def event_time(torch, graph, device, repeats=10, trials=7):
    values = []
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(trials):
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record(); end.synchronize()
        values.append(start.elapsed_time(end) * 1000 / (repeats * 10))
    return {"median_us": statistics.median(values), "min_us": min(values), "max_us": max(values)}


def run_case(batch, args):
    from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import DeepGemmFP4Experts
    from vllm.model_executor.layers.fused_moe.deep_gemm_utils import compute_aligned_M_and_alignment
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.utils.deep_gemm import (DeepGemmQuantScaleFMT,
                                       get_mk_alignment_for_contiguous_layout)
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import _pack_deepgemm_mxfp4_scales

    device = torch.device(f"cuda:{args.device}"); torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    x = (torch.randn((batch, HIDDEN), device=device, dtype=torch.bfloat16) * 0.1).contiguous()
    from compare_marlin_candidate import _packed_weights, _routes, _reference
    routes = _routes(args.route_file, batch, args.seed, device)
    if not (routes < LOCAL_EXPERTS).any().item():
        routes[0, 0] = 0
    weights = torch.full((batch, TOPK), 1 / TOPK, device=device, dtype=torch.float32)
    expert_map = torch.arange(GLOBAL_EXPERTS, device=device, dtype=torch.int32); expert_map[LOCAL_EXPERTS:] = -1
    w1, w2, s1_raw, s2_raw = _packed_weights(args.seed + 17, device)
    scale_fmt = DeepGemmQuantScaleFMT.from_oracle()
    # Reuse the installed production post-load conversion, including scale
    # packing and the grouped kernel's required memory layout.
    s1, s2 = _pack_deepgemm_mxfp4_scales(w1, w2, s1_raw, s2_raw)
    # Marlin's packed weights/routes are the shared synthetic fixture.  The
    # apply method only needs these fields plus two small helper methods.
    obj = SimpleNamespace(w1_scale=s1, w2_scale=s2, gemm1_clamp_limit=None,
                          _ACT_BLOCK_K=128, _WEIGHT_BLOCK_K=32)
    obj.adjust_N_for_activation = lambda n, activation: n // 2
    obj._act_mul_quant = lambda input, output, activation: DeepGemmFP4Experts._act_mul_quant(obj, input, output, activation)
    align = get_mk_alignment_for_contiguous_layout()[0]
    m_sum, align_used = compute_aligned_M_and_alignment(batch, TOPK, LOCAL_EXPERTS, align, None)
    workspace13 = torch.empty((m_sum, max(2 * INTERMEDIATE, HIDDEN)), device=device, dtype=torch.bfloat16)
    workspace2 = torch.empty((m_sum, max(INTERMEDIATE, HIDDEN)), device=device, dtype=torch.bfloat16)
    output = torch.empty((batch, HIDDEN), device=device, dtype=torch.bfloat16)

    def apply(aq, aq_scale):
        return DeepGemmFP4Experts.apply(obj, output, aq, w1, w2, weights, routes,
                                        MoEActivation.SILU, GLOBAL_EXPERTS, expert_map,
                                        aq_scale, None, workspace13, workspace2, None, False)

    def chain():
        # The FP32-scale scatter path requires contiguous per-token scales.
        # use_ue8m0 rounds values to powers of two but retains FP32 storage.
        use_ue8m0 = DeepGemmQuantScaleFMT.from_oracle() == DeepGemmQuantScaleFMT.UE8M0
        aq, aq_scale = per_token_group_quant_fp8(
            x, ACT_BLOCK_K, column_major_scales=False, use_ue8m0=use_ue8m0
        )
        return apply(aq, aq_scale)

    chain(); torch.cuda.synchronize(device)
    active = (routes < LOCAL_EXPERTS).any(dim=1)
    active_indices = active.nonzero(as_tuple=False).flatten().tolist()
    check = sorted(set(active_indices[:4] + active_indices[-4:]))
    ref = _reference(x[check], routes[check], w1, w2)
    eager_output = output.float().clone()
    eager_rel = ((eager_output[check] - ref).norm() / ref.norm().clamp_min(1e-12)).item()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            chain()
    graph.replay(); torch.cuda.synchronize(device)
    sample = output.float()
    rel = ((sample[check] - ref).norm() / ref.norm().clamp_min(1e-12)).item()
    if not torch.isfinite(sample).all().item():
        raise AssertionError("DeepGEMM output is non-finite")
    if eager_rel > args.tolerance or rel > args.tolerance:
        raise AssertionError(f"relative L2 eager={eager_rel:.6g}, graph={rel:.6g} exceeds {args.tolerance}")
    row = dict(backend="deepgemm_fp4_ep4", batch=batch, global_experts=GLOBAL_EXPERTS,
               local_experts=LOCAL_EXPERTS, topk=TOPK, hidden=HIDDEN, intermediate=INTERMEDIATE,
               act_block_k=ACT_BLOCK_K, m_sum=m_sum, align_used=align_used,
               active_tokens=int(active.sum().item()), active_assignments=int((routes < LOCAL_EXPERTS).sum().item()),
               relative_l2=rel, eager_relative_l2=eager_rel,
               timing=event_time(torch, graph, device), graph=True,
               route_file=args.route_file, weight_seed=args.seed + 17, status="ok")
    print(json.dumps(row), flush=True)
    return row


def main():
    p = argparse.ArgumentParser(); p.add_argument("--run", action="store_true")
    p.add_argument("--device", type=int, default=0); p.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32, 128, 256])
    p.add_argument("--route-file"); p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--tolerance", type=float, default=.12); p.add_argument("--output")
    args = p.parse_args()
    if not args.run:
        print(json.dumps(dict(status="validated", batches=args.batches, hidden=HIDDEN, intermediate=INTERMEDIATE,
                              global_experts=GLOBAL_EXPERTS, local_experts=LOCAL_EXPERTS, topk=TOPK,
                              message="pass --run to launch CUDA"))); return
    rows = []
    for batch in args.batches:
        try: rows.append(run_case(batch, args))
        except Exception as exc:
            rows.append(dict(backend="deepgemm_fp4_ep4", batch=batch, status="error", error=f"{type(exc).__name__}: {exc}"))
            print(json.dumps(rows[-1]), flush=True)
        if args.output: Path(args.output).write_text(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__": main()
