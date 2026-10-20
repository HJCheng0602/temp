#!/usr/bin/env python3
"""Local mHC prenorm GEMM probe.

This is intentionally independent of the service launcher.  It measures the
full mHC GEMM semantic value (split-K partials and row square sums), rather
than treating a projection-only matmul as the workload.  It can consume a
captured ``.pt``/``.pth`` dictionary or generate explicitly labelled inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import statistics
from pathlib import Path
from typing import Any, Callable

import torch


DSV41_MHC_SCHEMA = "dsv41_mhc_probe_v1"
DSV41_MHC_DEFAULTS = {
    "full": (20480, 24),
    "first": (5120, 24),
}


def dsv41_mhc_boundaries(k: int, split_k: int) -> list[tuple[int, int]]:
    split_k = max(1, min(int(split_k), k))
    return [
        (index * k // split_k, (index + 1) * k // split_k)
        for index in range(split_k)
        if (index + 1) * k // split_k > index * k // split_k
    ]


def dsv41_mhc_tensor_digest(value: torch.Tensor) -> str:
    """Hash bytes without requiring NumPy support for BF16 tensors."""
    raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def dsv41_mhc_reference(
    x: torch.Tensor, weight_k_n: torch.Tensor, split_k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """FP32 accumulation reference retaining the split-K output contract."""
    if x.dtype != torch.bfloat16 or weight_k_n.dtype != torch.float32:
        raise TypeError("mHC probe requires BF16 activations and FP32 weights")
    m, k = x.shape
    if weight_k_n.shape[0] != k:
        raise ValueError(f"weight K={weight_k_n.shape[0]} does not match x K={k}")
    boundaries = dsv41_mhc_boundaries(k, split_k)
    partials: list[torch.Tensor] = []
    square_parts: list[torch.Tensor] = []
    x_float = x.float()
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
      for begin, end in boundaries:
        partials.append(torch.mm(x_float[:, begin:end], weight_k_n[begin:end]))
        square_parts.append(x_float[:, begin:end].square().sum(dim=1))
    finally:
      torch.backends.cuda.matmul.allow_tf32 = previous
    partial = torch.stack(partials, dim=0)
    square_sum = torch.stack(square_parts, dim=0)
    return partial, square_sum, partial.sum(dim=0), square_sum.sum(dim=0)


def dsv41_mhc_pytorch_candidate(
    x: torch.Tensor, weight_k_n: torch.Tensor, split_k: int, tf32: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch matmul candidate with the same split-K and reduction semantics."""
    split_k = max(1, min(int(split_k), x.shape[1]))
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    try:
        boundaries = dsv41_mhc_boundaries(x.shape[1], split_k)
        partials = []
        square_parts = []
        x_float = x.float()
        for begin, end in boundaries:
            partials.append(torch.mm(x_float[:, begin:end], weight_k_n[begin:end]))
            square_parts.append(x_float[:, begin:end].square().sum(dim=1))
        partial = torch.stack(partials, dim=0)
        square_sum = torch.stack(square_parts, dim=0)
        output = partial.sum(dim=0)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    return partial, square_sum, output, square_sum.sum(dim=0)


def dsv41_mhc_vllm_candidate(
    x: torch.Tensor, fn_n_k: torch.Tensor, split_k: int, mode: str,
    tile_n: int = 12, n_thr: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Call repository DeepGEMM or TileLang through their public local helpers."""
    fn = fn_n_k
    output = torch.empty(split_k, x.shape[0], fn.shape[0], dtype=torch.float32, device=x.device)
    square_sum = torch.empty(split_k, x.shape[0], dtype=torch.float32, device=x.device)
    if mode == "deep_gemm":
        helper = importlib.import_module("vllm.utils.deep_gemm")
        helper.tf32_hc_prenorm_gemm(x, fn, output, square_sum, split_k)
    elif mode == "tilelang":
        if (x.shape[1] // split_k) % n_thr:
            raise ValueError(
                f"TileLang requires (K/split_k)%n_thr==0, got K={x.shape[1]}, "
                f"split_k={split_k}, n_thr={n_thr}"
            )
        helper = importlib.import_module(
            "vllm.model_executor.kernels.mhc.tilelang_kernels"
        )
        helper._HC_PRENORM_GEMM_TILELANG_KERNEL(
            x, fn, output, square_sum, x.shape[1], 1,
            tile_n=tile_n, n_thr=n_thr, n_splits=split_k
        )
    else:
        raise ValueError(f"unknown repository candidate {mode}")
    return output, square_sum, output.sum(dim=0), square_sum.sum(dim=0)


def dsv41_mhc_load_cases(
    path: Path | None, mode: str, batch: int, seed: int
) -> tuple[
    dict[str, tuple[torch.Tensor, torch.Tensor]],
    str,
    dict[str, Any],
    dict[str, tuple[torch.Tensor, torch.Tensor]],
]:
    """Load a real capture or create a visibly labelled deterministic fixture."""
    captured: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    if path is not None:
        payload = torch.load(path, map_location="cuda", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError("capture must be a dictionary")
        source = "real_capture"
        metadata = {"capture_path": str(path), "capture_keys": sorted(map(str, payload))}
        cases: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        requested = ("full", "first") if mode == "both" else (mode,)
        # capture_hook.py stores the real call as args=(x, fn, out, sqrsum,
        # n_splits), where fn is [24,K], unlike this probe's [K,24] form.
        if "args" in payload and isinstance(payload["args"], (tuple, list)):
            args = payload["args"]
            if len(args) < 2 or not isinstance(args[0], torch.Tensor) or not isinstance(args[1], torch.Tensor):
                raise ValueError("mHC hook capture args must begin with tensors x and fn")
            x = args[0]
            fn = args[1]
            inferred = "full" if x.shape[1] == DSV41_MHC_DEFAULTS["full"][0] else "first"
            requested = (inferred,)
            metadata["capture_format"] = "capture_hook_args_x_fn_out_sqrsum_nsplit"
            metadata["captured_output_present"] = isinstance(payload.get("output"), torch.Tensor)
            metadata["captured_n_splits"] = int(args[4]) if len(args) > 4 and isinstance(args[4], int) else None
            cases[inferred] = (x.cuda().contiguous(), fn.t().cuda().contiguous())
            if isinstance(args[2], torch.Tensor) and isinstance(args[3], torch.Tensor):
                captured[inferred] = (args[2].cuda().contiguous(), args[3].cuda().contiguous())
                metadata["captured_buffers"] = True
            return cases, source, metadata, captured
        for case in requested:
            candidate = payload.get(case, payload)
            if not isinstance(candidate, dict):
                raise TypeError(f"capture entry {case!r} must be a dictionary")
            x = candidate.get("x", candidate.get("a", candidate.get("hidden_states")))
            weight = candidate.get("weight", candidate.get("b", candidate.get("projection_weight")))
            if x is None or weight is None:
                raise KeyError(f"capture {case!r} needs x/a/hidden_states and weight/b")
            cases[case] = (x.cuda().contiguous(), weight.cuda().contiguous())
        return cases, source, metadata, captured

    generator = torch.Generator(device="cuda").manual_seed(seed)
    selected = ("full", "first") if mode == "both" else (mode,)
    cases = {}
    for case in selected:
        k, n = DSV41_MHC_DEFAULTS[case]
        x = torch.randn((batch, k), device="cuda", dtype=torch.float32, generator=generator).to(torch.bfloat16)
        weight = torch.randn((k, n), device="cuda", dtype=torch.float32, generator=generator)
        cases[case] = (x, weight)
    return cases, "synthetic_fixture", {"seed": seed, "batch": batch}, captured


def dsv41_mhc_error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    delta = (actual.float() - expected.float()).abs()
    denom = expected.float().abs().clamp_min(1e-8)
    return {
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "relative_max": float((delta / denom).max().item()),
        "relative_l2": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(expected.float()).clamp_min(1e-8)),
    }


def dsv41_mhc_measure(
    fn: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    warmup: int,
    samples: int,
    replays: int,
    displacement_mib: int,
) -> dict[str, Any]:
    """Return cold launch and CUDA Graph hot timings in milliseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph_result: dict[str, Any] = {"available": False}
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()
        cold_graph: list[float] = []
        hot: list[float] = []
        for _ in range(samples):
            if displacement_mib:
                scratch = torch.empty(displacement_mib * 1024 * 1024 // 2, dtype=torch.bfloat16, device="cuda")
                scratch.fill_(0)
            marker = torch.cuda.Event(enable_timing=True)
            done = torch.cuda.Event(enable_timing=True)
            marker.record()
            graph.replay()
            done.record()
            done.synchronize()
            cold_graph.append(float(marker.elapsed_time(done)))
            if displacement_mib:
                del scratch
            marker = torch.cuda.Event(enable_timing=True)
            done = torch.cuda.Event(enable_timing=True)
            marker.record()
            for _ in range(replays):
                graph.replay()
            done.record()
            done.synchronize()
            hot.append(float(marker.elapsed_time(done)) / replays)
        graph_result = {
            "available": True,
            "cold_graph_displaced_samples_ms": cold_graph,
            "cold_graph_displaced_median_ms": statistics.median(cold_graph),
            "samples_ms_per_replay": hot,
            "median_ms_per_replay": statistics.median(hot),
            "replays_per_sample": replays,
        }
    except Exception as exc:  # candidate-specific graph support is optional
        graph_result = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "timing_contract": "cold and hot are CUDA Graph replays; cold optionally follows allocator displacement",
        "hot_graph": graph_result,
    }


def dsv41_mhc_candidates(
    names: list[str], x: torch.Tensor, weight: torch.Tensor,
    tile_n: int, n_thr: int,
) -> dict[str, Callable[..., Any]]:
    candidates: dict[str, Callable[..., Any]] = {
        "pytorch_tf32": lambda x, w, s: dsv41_mhc_pytorch_candidate(x, w, s, True),
        "pytorch_fp32": lambda x, w, s: dsv41_mhc_pytorch_candidate(x, w, s, False),
    }
    if names == ["all"] or "deep_gemm" in names:
        fn = weight.t().contiguous()
        candidates["deep_gemm"] = lambda x, _w, s, fn=fn: dsv41_mhc_vllm_candidate(x, fn, s, "deep_gemm")
    if names == ["all"] or "tilelang" in names:
        fn = weight.t().contiguous()
        candidates["tilelang"] = lambda x, _w, s, fn=fn: dsv41_mhc_vllm_candidate(
            x, fn, s, "tilelang", tile_n=tile_n, n_thr=n_thr
        )
    if names != ["all"]:
        candidates = {name: candidates[name] for name in names if name in candidates}
    return candidates


def dsv41_mhc_run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("mHC probe requires a CUDA device; no GPU execution is performed by this source audit")
    cases, source, metadata, captured = dsv41_mhc_load_cases(
        args.input_capture, args.mode, args.batch, args.seed
    )
    result: dict[str, Any] = {
        "schema": DSV41_MHC_SCHEMA,
        "source_kind": source,
        "metadata": metadata,
        "semantics": {
            "input": "BF16 [tokens,K]",
            "weight": "FP32 [K,24]",
            "outputs": "FP32 split-K partials [splits,tokens,24] and square_sums [splits,tokens], both reduced",
            "reference": "FP32 accumulation per K slice, then required split-K reductions",
        },
        "cases": {},
    }
    for case, (x, weight) in cases.items():
        candidates = dsv41_mhc_candidates(args.candidate, x, weight, args.tile_n, args.n_thr)
        split_k = args.split_k if args.split_k else (
            int(metadata.get("captured_n_splits"))
            if metadata.get("captured_n_splits")
            else (4 if case == "full" else 1)
        )
        reference = dsv41_mhc_reference(x, weight, split_k)
        entry: dict[str, Any] = {
            "shape": {"batch": x.shape[0], "k": x.shape[1], "n": weight.shape[1], "split_k": split_k},
            "input_digest": {"x": dsv41_mhc_tensor_digest(x), "weight": dsv41_mhc_tensor_digest(weight)},
            "reference_output_digest": dsv41_mhc_tensor_digest(reference[2]),
            "candidates": {},
        }
        for name, candidate in candidates.items():
            try:
                output = candidate(x, weight, split_k)
                errors = {
                    "partials": dsv41_mhc_error(output[0], reference[0]),
                    "square_sums": dsv41_mhc_error(output[1], reference[1]),
                    "reduced_output": dsv41_mhc_error(output[2], reference[2]),
                    "reduced_square_sum": dsv41_mhc_error(output[3], reference[3]),
                }
                captured_errors = None
                if case in captured:
                    captured_mix, captured_square = captured[case]
                    captured_errors = {
                        "reduced_output": dsv41_mhc_error(output[2], captured_mix.sum(dim=0)),
                        "reduced_square_sum": dsv41_mhc_error(output[3], captured_square.sum(dim=0)),
                    }
                timing = dsv41_mhc_measure(
                    lambda candidate=candidate: candidate(x, weight, split_k),
                    args.warmup,
                    args.samples,
                    args.replays,
                    args.displacement_mib,
                )
                entry["candidates"][name] = {
                    "status": "ok",
                    "outputs_finite": all(
                        bool(torch.isfinite(value).all().item()) for value in output
                    ),
                    "output_digest": dsv41_mhc_tensor_digest(output[2]),
                    "errors_vs_fp32_reference": errors,
                    "errors_vs_captured_buffers": captured_errors,
                    "timing": timing,
                }
            except Exception as exc:
                entry["candidates"][name] = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
        result["cases"][case] = entry
    return result


def dsv41_mhc_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-capture", type=Path, help=".pt/.pth dict with x/a and weight/b, optionally nested under full/first")
    parser.add_argument("--mode", choices=("full", "first", "both"), default="both")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=4101)
    parser.add_argument("--split-k", type=int, default=0, help="override split-K; default is 4 for full and 1 for first")
    parser.add_argument("--candidate", nargs="+", choices=("all", "pytorch_tf32", "pytorch_fp32", "deep_gemm", "tilelang"), default=["all"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--displacement-mib", type=int, default=256, help="BF16 allocator displacement before each cold graph replay; 0 disables it")
    parser.add_argument("--tile-n", type=int, choices=(4, 8, 12), default=12)
    parser.add_argument("--n-thr", type=int, choices=(256, 512), default=512)
    parser.add_argument("--output", type=Path, default=Path("mhc_probe.json"))
    return parser.parse_args()


def main() -> None:
    args = dsv41_mhc_parse_args()
    result = dsv41_mhc_run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "schema": DSV41_MHC_SCHEMA, "source_kind": result["source_kind"]}))


if __name__ == "__main__":
    main()
