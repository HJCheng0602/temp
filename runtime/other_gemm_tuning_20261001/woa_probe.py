"""Measure the grouped BF16 ``wo_a`` GEMM without changing production code.

The default tensors reproduce the deployment layout exactly:
``[2, B, 4096] @ [2, 4096, 1024] -> [2, B, 1024]``.  A payload made by a
capture hook may be supplied as a ``torch.save`` file containing ``a`` and
``b`` (and optionally ``out``); all input strides are checked and reported.

The cuBLASLt candidate is optional and must be a prebuilt extension exposing
``bmm_out(a, b, out, algorithm)``.  It is warmed before graph capture so plan
creation cannot be hidden inside the measured path.  Captured contiguous A
(``[262144,4096,1]`` at B64) is supported in addition to the synthetic
interleaved A layout.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
from typing import Any, Callable

import torch


EXPECTED_A_STRIDE = (4096, 8192, 1)
EXPECTED_B_STRIDE = (4194304, 1, 4096)


def _load_payload(path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    value = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(value, dict):
        raise ValueError(f"payload must be a dictionary: {path}")
    if "a" in value and "b" in value:
        a, b = value["a"], value["b"]
    elif "args" in value:
        # other_gemm_tuning/capture_hook.py stores cpu(args), usually the two
        # bmm operands, plus the post-call output and call metadata.
        args = value["args"]
        tensors = [item for item in (args if isinstance(args, (tuple, list)) else ())
                   if isinstance(item, torch.Tensor)]
        if len(tensors) < 2:
            raise ValueError(f"capture args do not contain two tensor operands: {path}")
        a, b = tensors[:2]
    else:
        raise ValueError(f"payload must contain a/b or captured args: {path}")
    output = value.get("output")
    if output is not None and not isinstance(output, torch.Tensor):
        output = None
    def preserve_layout(tensor: torch.Tensor) -> torch.Tensor:
        result = torch.empty_strided(tuple(tensor.shape), tuple(tensor.stride()),
                                     device=device, dtype=tensor.dtype)
        return result.copy_(tensor)

    return preserve_layout(a), preserve_layout(b), output


def _synthetic(batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    # Transposes intentionally preserve the production non-contiguous strides.
    a = torch.randn((batch, 2, 4096), device=device, dtype=torch.bfloat16).transpose(0, 1)
    b = torch.randn((2, 1024, 4096), device=device, dtype=torch.bfloat16).transpose(1, 2)
    return a, b


def _check_layout(a: torch.Tensor, b: torch.Tensor) -> None:
    if a.shape[0] != 2 or a.shape[2] != 4096 or b.shape != (2, 4096, 1024):
        raise ValueError(f"expected [2,B,4096]@[2,4096,1024], got {tuple(a.shape)} {tuple(b.shape)}")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError(f"expected BF16 operands, got {a.dtype} and {b.dtype}")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("wo_a probe requires CUDA tensors")


def _matrix_layout_supported(tensor: torch.Tensor) -> bool:
    """Match the row/column layout cases encoded in cublaslt_fast.cu."""
    if tensor.ndim != 3 or tensor.stride(2) <= 0 or tensor.stride(1) <= 0:
        return False
    # The bridge uses row order when the last dimension is contiguous, and
    # column order otherwise. Its leading stride must cover one matrix row or
    # column respectively; arbitrary overlapping views are rejected.
    row = tensor.stride(2) == 1 and tensor.stride(1) >= tensor.shape[2]
    col = tensor.stride(1) == 1 and tensor.stride(2) >= tensor.shape[1]
    return bool(row or col)


def _extension_layout_supported(a: torch.Tensor, b: torch.Tensor,
                                out: torch.Tensor) -> bool:
    # cublaslt_fast.cu requires row-major output but accepts row/column input
    # matrices and their explicit strided-batch offsets.
    return (_matrix_layout_supported(a) and _matrix_layout_supported(b)
            and out.stride(2) == 1 and out.stride(1) >= out.shape[2])


def _load_extension(path: Path) -> Any:
    # The .so was compiled with TORCH_EXTENSION_NAME=dsv4_wo_a_cublaslt;
    # Python's loader name must match its PyInit symbol exactly.
    spec = importlib.util.spec_from_file_location("dsv4_wo_a_cublaslt", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load extension {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "bmm_out"):
        raise TypeError(f"extension has no bmm_out(a,b,out,algorithm): {path}")
    return module


def _relative_error(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float | int | bool]:
    actual_finite = bool(torch.isfinite(actual).all().item())
    reference_finite = bool(torch.isfinite(reference).all().item())
    if not actual_finite or not reference_finite:
        return {
            "actual_finite": actual_finite,
            "reference_finite": reference_finite,
            "nonfinite_actual": int((~torch.isfinite(actual)).sum().item()),
            "nonfinite_reference": int((~torch.isfinite(reference)).sum().item()),
            "relative_l2": float("nan"),
            "max_abs": float("nan"),
        }
    delta = actual.float() - reference
    denom = reference.norm().item()
    return {
        "relative_l2": float(delta.norm().item() / denom) if denom else 0.0,
        "max_abs": float(delta.abs().max().item()),
        "actual_finite": actual_finite,
        "reference_finite": reference_finite,
        "nonfinite_actual": 0,
        "nonfinite_reference": 0,
    }


def _eager_hot_timed(fn: Callable[[], Any], repeats: int, samples: int) -> dict[str, Any]:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    hot: list[float] = []
    for _ in range(samples):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        end.synchronize()
        hot.append(start.elapsed_time(end) * 1000.0 / repeats)

    return {
        "hot_median_us": statistics.median(hot),
        "hot_samples": hot,
    }


def _graph_timed(fn: Callable[[], Any], device: torch.device,
                 repeats: int, samples: int) -> dict[str, Any]:
    # Capture exactly one invocation. Repeating inside one graph would hide
    # per-replay launch/cache behavior in a cold measurement.
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    hot: list[float] = []
    for _ in range(samples):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        hot.append(start.elapsed_time(end) * 1000.0 / repeats)

    # A device-side write displaces cache state before the timed replay. The
    # fill and synchronization are deliberately outside the event interval.
    displacement = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    cold: list[float] = []
    for _ in range(samples):
        # Queue displacement and immediately place the timed events on the
        # same stream. No host synchronize belongs between displacement and
        # start; the end event still provides completion before the next row.
        displacement.fill_(0)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        cold.append(start.elapsed_time(end) * 1000.0)
    return {
        "graph_hot_median_us": statistics.median(hot),
        "graph_hot_samples": hot,
        "graph_cold_median_us": statistics.median(cold),
        "graph_cold_samples": cold,
        "graph_cold_l2_displacement_bytes": displacement.numel(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, help="torch.save dict with CUDA/CPU tensors a and b")
    parser.add_argument("--input-capture", type=Path,
                        help="alias for --payload; accepts other_gemm_tuning capture format")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--extension", type=Path, help="optional prebuilt cuBLASLt .so")
    parser.add_argument("--algorithm", type=int,
                        help="single cuBLASLt heuristic index (legacy alias)")
    parser.add_argument("--algorithms", default=None,
                        help="comma-separated cuBLASLt indices; defaults to 0..7")
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile-kernels", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this script does not silently run a CPU substitute")
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.manual_seed(1729)
    payload_path = args.input_capture or args.payload
    captured_output = None
    if payload_path:
        a, b, captured_output = _load_payload(payload_path, device)
    else:
        a, b = _synthetic(args.batch, device)
    _check_layout(a, b)
    # The deployment output is the same transposed group buffer as the input.
    if captured_output is not None:
        if captured_output.shape != (2, a.shape[1], b.shape[2]):
            raise ValueError(f"captured output shape mismatch: {tuple(captured_output.shape)}")
        out = torch.empty_strided(tuple(captured_output.shape), tuple(captured_output.stride()),
                                  device=device, dtype=torch.bfloat16)
    else:
        out = torch.empty((a.shape[1], 2, b.shape[2]), device=device,
                          dtype=torch.bfloat16).transpose(0, 1)
    if tuple(a.stride()) != EXPECTED_A_STRIDE or tuple(b.stride()) != EXPECTED_B_STRIDE:
        print("WARNING: captured strides differ from production layout", flush=True)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        reference = torch.bmm(a.float(), b.float())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    captured_reference = (captured_output.to(device=device, dtype=torch.bfloat16)
                          if captured_output is not None else None)
    extension = _load_extension(args.extension) if args.extension else None
    rows: list[dict[str, Any]] = []

    def add(name: str, fn: Callable[[], Any]) -> None:
        result = fn()
        torch.cuda.synchronize()
        correctness = {"fp32_reference": _relative_error(out, reference)}
        if captured_reference is not None:
            correctness["captured_baseline"] = _relative_error(out, captured_reference)
        timing = _eager_hot_timed(fn, args.repeats, args.samples)
        graph = _graph_timed(fn, device, args.repeats, args.samples)
        entry = dict(name=name, correctness=correctness, timing={**timing, **graph})
        if args.profile_kernels:
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured):
                fn()
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profile:
                captured.replay()
                torch.cuda.synchronize()
            entry['profile_cuda_events'] = [
                dict(name=event.name, duration_us=event.device_time_total)
                for event in profile.events()
                if event.device_type == torch.autograd.DeviceType.CUDA
            ]
        rows.append(entry)

    add("torch.bmm", lambda: torch.bmm(a, b, out=out))
    def mm_per_group() -> torch.Tensor:
        for group in range(2):
            torch.mm(a[group], b[group], out=out[group])
        return out

    def mm_per_group_full_bf16_accumulation() -> torch.Tensor:
        flag = "allow_bf16_reduced_precision_reduction"
        previous = getattr(torch.backends.cuda.matmul, flag)
        setattr(torch.backends.cuda.matmul, flag, False)
        try:
            for group in range(2):
                torch.mm(a[group], b[group], out=out[group])
        finally:
            setattr(torch.backends.cuda.matmul, flag, previous)
        return out

    add("torch.mm_per_group", mm_per_group)
    add("torch.mm_per_group_full_bf16_accumulation",
        mm_per_group_full_bf16_accumulation)
    if extension is not None:
        if args.algorithms is not None:
            algorithms = [int(item.strip()) for item in args.algorithms.split(",") if item.strip()]
        elif args.algorithm is not None:
            algorithms = [args.algorithm]
        else:
            algorithms = list(range(8))
        if not _extension_layout_supported(a, b, out):
            rows.append(dict(name="cublasLt", status="unsupported",
                             error="captured strides are outside cublaslt_fast.cu row/column layout"))
        for algorithm in algorithms if _extension_layout_supported(a, b, out) else ():
            try:
                # This first call creates the cuBLASLt plan outside graph capture.
                extension.bmm_out(a, b, out, algorithm)
                torch.cuda.synchronize()
                add(f"cublasLt_algo_{algorithm}",
                    lambda algorithm=algorithm: extension.bmm_out(a, b, out, algorithm))
            except Exception as exc:
                rows.append(dict(name=f"cublasLt_algo_{algorithm}", status="unsupported",
                                 error=f"{type(exc).__name__}: {exc}"))

    result = {
        "shape": [list(a.shape), list(b.shape), list(out.shape)],
        "dtype": str(a.dtype),
        "strides": [list(a.stride()), list(b.stride()), list(out.stride())],
        "reference": "torch.bmm(a.float(), b.float())",
        "captured_output_present": captured_output is not None,
        "rows": rows,
        "caveat": "Timing is isolated GEMM time; no claim about end-to-end serving impact.",
    }
    text = json.dumps(result, indent=2)
    print(text, flush=True)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
