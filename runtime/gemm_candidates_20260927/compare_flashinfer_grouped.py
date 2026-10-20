"""Bounded FlashInfer SM120 MXFP8 x MXFP4 grouped-MoE candidate.

This measures the local expert chain only.  Routing, expert sorting/padding and
scatter/reduce are intentionally explicit so their cost and correctness are
visible.  The GEMM API itself is FlashInfer's grouped operator, not the fused
MoE wrapper.  Operands use the same synthetic MXFP4 weights as the Marlin
candidate and are suitable for a later route-file replay.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")

import torch

from compare_marlin_candidate import (
    GLOBAL_EXPERTS,
    HIDDEN,
    INTERMEDIATE,
    LOCAL_EXPERTS,
    SCALE_BYTE,
    TOPK,
    _packed_weights,
    _reference,
    _routes,
)


def _quantize(x: torch.Tensor):
    from flashinfer import mxfp8_quantize

    params = inspect.signature(mxfp8_quantize).parameters
    kwargs = {"is_sf_swizzled_layout": False}
    if "enable_pdl" in params:
        kwargs["enable_pdl"] = False
    values, scales = mxfp8_quantize(x, **kwargs)
    if scales.ndim == 1:
        scales = scales.view(x.shape[0], x.shape[1] // 32)
    return values, scales


def _grouped(a, b, a_sf, b_sf, indptr, tile_n):
    from flashinfer.gemm import group_gemm_mxfp8_mxfp4_nt_groupwise

    params = inspect.signature(group_gemm_mxfp8_mxfp4_nt_groupwise).parameters
    kwargs = {"mma_sm": 1, "tile_m": 128, "tile_n": tile_n, "tile_k": 128,
              "swap_ab": True, "out_dtype": torch.bfloat16}
    kwargs = {k: v for k, v in kwargs.items() if k in params}
    return group_gemm_mxfp8_mxfp4_nt_groupwise(a, b, a_sf, b_sf, indptr, **kwargs)


def _route_plan(x, routes):
    """Return sorted local assignments, padded expert indptr and scatter maps."""
    pairs = []
    for token, row in enumerate(routes.tolist()):
        for slot, expert in enumerate(row):
            if expert < LOCAL_EXPERTS:
                pairs.append((expert, token, slot))
    if not pairs:
        raise ValueError("route set has no local EP4 assignments")
    pairs.sort()
    sorted_experts = []
    sorted_tokens = []
    sorted_slots = []
    sorted_valid = []
    group_counts = []
    group_padded = []
    indptr = [0]
    cursor = 0
    for expert in range(LOCAL_EXPERTS):
        group = [p for p in pairs if p[0] == expert]
        group_counts.append(len(group))
        sorted_experts.extend([expert] * len(group))
        sorted_tokens.extend([p[1] for p in group])
        sorted_slots.extend([p[2] for p in group])
        sorted_valid.extend([True] * len(group))
        cursor += len(group)
        padded = (cursor + 3) // 4 * 4 - cursor
        sorted_experts.extend([expert] * padded)
        sorted_tokens.extend([0] * padded)
        sorted_slots.extend([0] * padded)
        sorted_valid.extend([False] * padded)
        cursor += padded
        group_padded.append(len(group) + padded)
        indptr.append(cursor)
    return (
        torch.tensor(sorted_experts, device=x.device, dtype=torch.int32),
        torch.tensor(sorted_tokens, device=x.device, dtype=torch.int64),
        torch.tensor(sorted_slots, device=x.device, dtype=torch.int64),
        torch.tensor(indptr, device=x.device, dtype=torch.int32),
        torch.tensor(sorted_valid, device=x.device, dtype=torch.bool),
        tuple(group_counts), tuple(group_padded),
    )


def _make_b(raw, sf):
    from flashinfer.fp4_quantization import _pad_scale_factors, get_fp4_quantization_module
    from flashinfer.utils import get_compute_capability
    major, minor = get_compute_capability(raw.device)
    n, k = raw.shape[1], raw.shape[2] * 2
    sf = _pad_scale_factors(sf.contiguous(), n, k, 32)
    sf = get_fp4_quantization_module(f"{major}{minor}").block_scale_interleave_sm100(sf)
    # Official test uses contiguous [E,N,K//2] packed B storage.
    return raw.contiguous(), sf.view(sf.shape)


def _build_a_scale_map(plan, kblocks, device):
    """Build the official 128x4 interleave map once, outside graph capture."""
    _, _, _, indptr, _, _, group_padded = plan
    ktiles = (kblocks + 3) // 4
    src, dst = [], []
    src_base = dst_base = 0
    for expert, token_rows in enumerate(group_padded):
        m0 = int(indptr[expert].item())
        m1 = int(indptr[expert + 1].item())
        sf0 = (m0 + expert * 127) // 128 * 128
        sf1 = (m1 + (expert + 1) * 127) // 128 * 128
        sf_rows = sf1 - sf0
        for row in range(token_rows):
            for col in range(kblocks):
                local = ((row // 128 * ktiles + col // 4) * 512
                         + (row % 32) * 16 + ((row % 128) // 32) * 4 + col % 4)
                src.append((src_base + row) * kblocks + col)
                dst.append(dst_base + local)
        src_base += token_rows
        # Destination indices are flattened scale elements, not rows.
        dst_base += sf_rows * kblocks
    return (torch.tensor(src, device=device, dtype=torch.int64),
            torch.tensor(dst, device=device, dtype=torch.int64),
            dst_base, kblocks)


def _chain(x, w13, w13_sf, w2, w2_sf, plan, tile_n, scale_map1, scale_buf1,
           scale_map2, scale_buf2, phase_events=None):
    (experts, token_ids, slots, indptr, valid,
     group_counts, group_padded) = plan
    gathered = x.index_select(0, token_ids)
    gathered = gathered * valid[:, None]
    if phase_events is not None:
        phase_events["pack1"] = [torch.cuda.Event(True), torch.cuda.Event(True)]
        phase_events["pack1"][0].record()
    a1, a1_sf = _quantize(gathered)
    _, dst_idx, _, kblocks = scale_map1
    scale_buf1.zero_()
    scale_buf1.scatter_(0, dst_idx, a1_sf.flatten())
    a1_sf = scale_buf1.view(-1, kblocks)
    if phase_events is not None:
        phase_events["pack1"][1].record()
        phase_events["gemm1"] = [torch.cuda.Event(True), torch.cuda.Event(True)]
        phase_events["gemm1"][0].record()
    fc1 = _grouped(a1, w13, a1_sf, w13_sf, indptr, tile_n)
    if phase_events is not None:
        phase_events["gemm1"][1].record()
        phase_events["activation_pack2"] = [torch.cuda.Event(True), torch.cuda.Event(True)]
        phase_events["activation_pack2"][0].record()
    n = INTERMEDIATE
    gate, up = fc1[:, :n], fc1[:, n:]
    intermediate = torch.nn.functional.silu(gate) * up
    a2, a2_sf = _quantize(intermediate)
    _, dst_idx, _, kblocks = scale_map2
    scale_buf2.zero_()
    scale_buf2.scatter_(0, dst_idx, a2_sf.flatten())
    a2_sf = scale_buf2.view(-1, kblocks)
    if phase_events is not None:
        phase_events["activation_pack2"][1].record()
        phase_events["gemm2"] = [torch.cuda.Event(True), torch.cuda.Event(True)]
        phase_events["gemm2"][0].record()
    fc2 = _grouped(a2, w2, a2_sf, w2_sf, indptr, tile_n)
    if phase_events is not None:
        phase_events["gemm2"][1].record()
    out = torch.zeros(x.shape[0], HIDDEN, device=x.device, dtype=torch.float32)
    weights = 1.0 / TOPK
    out.index_add_(0, token_ids, fc2.float() * valid[:, None] * weights)
    return out, plan


def run_case(batch: int, tile_n: int, args):
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    x = (torch.randn(batch, HIDDEN, device=device, dtype=torch.bfloat16) * .1).contiguous()
    routes = _routes(args.route_file, batch, args.seed, device)
    if not (routes < LOCAL_EXPERTS).any().item():
        routes[0, 0] = 0
    raw_w13, raw_w2, sf13, sf2 = _packed_weights(args.seed + 17, device)
    plan = _route_plan(x, routes)
    w13, w13_sf = _make_b(raw_w13, sf13)
    w2, w2_sf = _make_b(raw_w2, sf2)
    scale_map1 = _build_a_scale_map(plan, HIDDEN // 32, device)
    scale_map2 = _build_a_scale_map(plan, INTERMEDIATE // 32, device)
    scale_buf1 = torch.empty(scale_map1[2], dtype=torch.uint8, device=device)
    scale_buf2 = torch.empty(scale_map2[2], dtype=torch.uint8, device=device)

    def step():
        return _chain(x, w13, w13_sf, w2, w2_sf, plan, tile_n,
                      scale_map1, scale_buf1, scale_map2, scale_buf2)[0]

    warm = step()
    torch.cuda.synchronize(device)
    phase_events = {}
    _chain(x, w13, w13_sf, w2, w2_sf, plan, tile_n,
           scale_map1, scale_buf1, scale_map2, scale_buf2, phase_events)
    torch.cuda.synchronize(device)
    phase_us = {name: start.elapsed_time(end) * 1000
                for name, (start, end) in phase_events.items()}
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            graph_out = step()
    graph.replay()
    torch.cuda.synchronize(device)
    active = (routes < LOCAL_EXPERTS).any(dim=1).nonzero(as_tuple=False).flatten()
    ids = sorted(set(active[:4].tolist() + active[-4:].tolist()))
    ref = _reference(x[ids], routes[ids], raw_w13, raw_w2)
    rel = ((graph_out[ids] - ref).norm() / ref.norm().clamp_min(1e-12)).item()
    if not torch.isfinite(graph_out).all().item():
        raise AssertionError("FlashInfer grouped output is non-finite")
    if rel > args.tolerance:
        raise AssertionError(f"relative L2 {rel:.6g} exceeds {args.tolerance}")

    times = []
    for _ in range(7):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(10):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / 100)
    row = dict(backend="flashinfer_grouped_mxfp8_mxfp4", batch=batch,
               tile_n=tile_n, local_experts=LOCAL_EXPERTS,
               global_experts=GLOBAL_EXPERTS, topk=TOPK, hidden=HIDDEN,
               intermediate=INTERMEDIATE, relative_l2=rel,
               us=sorted(times)[len(times) // 2], active_tokens=len(active),
               active_assignments=int((routes < LOCAL_EXPERTS).sum().item()),
               phase_us=phase_us,
               padded_rows=int(plan[3][-1].item()), graph=True,
               route_file=args.route_file, status="ok")
    print(json.dumps(row), flush=True)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="store_true")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32, 128, 256])
    p.add_argument("--tiles", nargs="+", type=int, default=[32, 64, 128])
    p.add_argument("--route-file")
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--tolerance", type=float, default=.12)
    p.add_argument("--output")
    args = p.parse_args()
    if not args.run:
        print(json.dumps(dict(status="validated", batches=args.batches,
                              tiles=args.tiles, hidden=HIDDEN,
                              intermediate=INTERMEDIATE, local_experts=LOCAL_EXPERTS,
                              global_experts=GLOBAL_EXPERTS, topk=TOPK,
                              message="pass --run to launch CUDA")))
        return
    rows = []
    for tile_n in args.tiles:
        if tile_n not in (32, 64, 128):
            raise ValueError("SM120 tile_n must be 32, 64, or 128")
        for batch in args.batches:
            try:
                rows.append(run_case(batch, tile_n, args))
            except Exception as exc:
                row = dict(backend="flashinfer_grouped_mxfp8_mxfp4", batch=batch,
                           tile_n=tile_n, status="error",
                           error=f"{type(exc).__name__}: {exc}")
                rows.append(row)
                print(json.dumps(row), flush=True)
            if args.output:
                Path(args.output).write_text(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
