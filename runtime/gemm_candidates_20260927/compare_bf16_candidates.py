"""Bounded BF16 wo_a batched-GEMM comparison.

The production wo_a path has two independent groups and computes
``[2, B, 4096] @ [2, 4096, 1024]``.  This benchmark keeps that layout, checks
every candidate against an FP32 reference before timing, and uses a CUDA
Graph to remove launch jitter from the reported steady-state timings.

The optional cuBLASLt implementation is built only when ``--cublaslt`` is
requested on a CUDA host.  The benchmark remains useful with only PyTorch:

    python microbench/compare_bf16_candidates.py --output /tmp/wo_a.json
    python microbench/compare_bf16_candidates.py --cublaslt --output /tmp/wo_a.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
from typing import Any, Callable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

GROUPS = 2
K = 4096
N = 1024
BATCHES = (1, 8, 32, 128, 256)


def graph_time(torch: Any, fn: Callable[[], Any], repeats: int = 20, trials: int = 7) -> dict[str, Any]:
    """Time graph replay in microseconds per operation."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            fn()
    values = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000.0 / repeats)
    return {
        "median_us": statistics.median(values),
        "min_us": min(values),
        "max_us": max(values),
        "repeats_per_event": repeats,
        "trials": trials,
        "timing": "CUDA Graph replay; per call",
    }


def check(torch: Any, got: Any, a: Any, b: Any) -> dict[str, Any]:
    """Compare BF16 output to an FP32 accumulation reference."""
    ref = torch.bmm(a.float(), b.float())
    try:
        torch.testing.assert_close(got.float(), ref, rtol=2e-2, atol=2e-2)
    except Exception as exc:  # keep all candidates in the output file
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    rel = (got.float() - ref).norm() / ref.norm().clamp_min(1e-8)
    return {"status": "ok", "relative_l2": float(rel.item()), "rtol": 2e-2, "atol": 2e-2}


def load_cublaslt(torch: Any):
    """Build the tiny extension lazily; importing this script stays CPU-safe."""
    from torch.utils.cpp_extension import load

    source = Path(__file__).with_name("cublaslt_bench.cu")
    cuda_home = os.environ.get("CUDA_HOME")
    include_dirs = [str(Path(cuda_home) / "include")] if cuda_home else []
    # CUDA 13 pip runtimes commonly ship only versioned libcublasLt.so.13
    # under lib/ (and do not provide the usual unversioned -lcublasLt link).
    library_candidates = []
    if cuda_home:
        for directory in (Path(cuda_home) / "lib", Path(cuda_home) / "lib64"):
            library_candidates.extend(sorted(directory.glob("libcublasLt.so*")))
    try:
        import nvidia.cublas
        library_candidates.extend(sorted(Path(nvidia.cublas.__path__[0]).glob("lib/libcublasLt.so*")))
    except Exception:
        pass
    library = next((path for path in library_candidates if path.is_file()), None)
    return load(
        name="wo_a_cublaslt_bench",
        sources=[str(source)],
        extra_include_paths=include_dirs,
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
        extra_ldflags=[str(library)] if library else ["-lcublasLt"],
        verbose=False,
    )


def metadata(torch: Any) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": props.name,
        "compute_capability": [props.major, props.minor],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "BF16 GEMM only; output BF16, reference FP32",
        "shape": f"[{GROUPS},B,{K}] @ [{GROUPS},{K},{N}]",
        "cpu_threads": 1,
        "gpu_state": subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,clocks.sm,power.limit,temperature.gpu", "--format=csv"],
            capture_output=True, text=True, check=False,
        ).stdout,
    }


def main() -> None:
    global GROUPS, K, N
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batches", nargs="+", type=int, choices=BATCHES, default=list(BATCHES))
    parser.add_argument("--cublaslt", action="store_true", help="build and search the optional cuBLASLt extension")
    parser.add_argument("--algorithms", type=int, default=8, help="maximum heuristic algorithms per shape")
    parser.add_argument("--layout", choices=("contiguous", "production"), default="contiguous",
                        help="contiguous tensors or token-major A plus column-major transposed B")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--weight-ring-mib", type=int, default=0)
    parser.add_argument("--kind", choices=("wo_a", "lm_head", "router"), default="wo_a")
    args = parser.parse_args()
    GROUPS, K, N = {"wo_a": (2, 4096, 1024), "lm_head": (1, 5120, 32320),
                    "router": (1, 5120, 384)}[args.kind]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch
    except Exception as exc:
        payload = {"status": "environment_error", "reason": f"{type(exc).__name__}: {exc}"}
        out_path.write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload), flush=True)
        return
    if not torch.cuda.is_available():
        payload = {"status": "environment_error", "reason": "CUDA is unavailable; benchmark requires a CUDA device"}
        out_path.write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload), flush=True)
        return
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("highest")

    extension = None
    extension_error = None
    if args.cublaslt:
        try:
            extension = load_cublaslt(torch)
        except Exception as exc:
            extension_error = f"{type(exc).__name__}: {exc}"

    payload: dict[str, Any] = {"status": "running", "metadata": metadata(torch), "rows": []}
    payload["metadata"]["weight_ring_mib"] = args.weight_ring_mib
    payload["metadata"]["kind"] = args.kind
    if extension_error:
        payload["cublaslt_build_error"] = extension_error
    out_path.write_text(json.dumps(payload, indent=2))
    for batch in args.batches:
        # Keep the deployment's group-major tensors.  Candidate conversion is
        # deliberately inside the timed callable when a strided input needs it.
        if args.layout == "contiguous":
            a = torch.randn((GROUPS, batch, K), device="cuda", dtype=torch.bfloat16) * 0.05
            b = torch.randn((GROUPS, K, N), device="cuda", dtype=torch.bfloat16) * 0.05
        else:
            # Match the deployment's token-major activation and transposed
            # weight storage without materializing either operand.
            a = (torch.randn((batch, GROUPS, K), device="cuda", dtype=torch.bfloat16) * 0.05).transpose(0, 1)
            b = (torch.randn((GROUPS, N, K), device="cuda", dtype=torch.bfloat16) * 0.05).transpose(1, 2)
        row: dict[str, Any] = {"batch": batch, "shape": [[GROUPS, batch, K], [GROUPS, K, N]],
                               "layout": args.layout, "a_stride": list(a.stride()),
                               "b_stride": list(b.stride()), "candidates": {}}
        ring_count = max(1, math.ceil(args.weight_ring_mib * 1024**2 / (b.numel() * b.element_size())))
        b_ring = [b] + [b.clone() for _ in range(ring_count - 1)]
        row["weight_ring_count"] = ring_count
        row["weight_ring_bytes"] = ring_count * b.numel() * b.element_size()
        cursor = 0
        def pick_b():
            nonlocal cursor
            wb = b_ring[cursor % ring_count]
            cursor += 1
            return wb

        def baseline() -> Any:
            wb = pick_b()
            return torch.mm(a[0], wb[0]).unsqueeze(0) if GROUPS == 1 else torch.bmm(a, wb)

        out = baseline()
        result = {"status": "ok", "correctness": check(torch, out, a, b), "input_layout": "native"}
        if result["correctness"]["status"] == "ok":
            result["timing"] = graph_time(torch, baseline, repeats=ring_count if ring_count > 1 else 20)
            result["flops"] = 2 * GROUPS * batch * K * N
            result["tflops"] = result["flops"] / result["timing"]["median_us"] / 1e6
        else:
            result["status"] = "numerical_error"
        row["candidates"]["aten.mm" if GROUPS == 1 else "aten.bmm"] = result

        if extension is not None:
            for algorithm in range(max(0, args.algorithms)):
                candidate = {"algorithm": algorithm}
                # The extension supports native row-major strided tensors.  If
                # production data has another layout, account for contiguous()
                # in every graph replay rather than hiding its cost.
                # Row-major A and either row- or column-major B are passed
                # directly.  This covers token-major A and transposed weights.
                native_layout = (a.stride(2) == 1 or a.stride(1) == 1) and (b.stride(2) == 1 or b.stride(1) == 1)
                needs_conversion = not native_layout

                def cublaslt_call(algorithm: int = algorithm) -> Any:
                    aa = a.contiguous() if needs_conversion else a
                    wb = pick_b()
                    bb = wb.contiguous() if needs_conversion else wb
                    return extension.bmm(aa, bb, algorithm)

                try:
                    got = cublaslt_call()
                    candidate["correctness"] = check(torch, got, a, b)
                    if candidate["correctness"]["status"] != "ok":
                        candidate["status"] = "numerical_error"
                    else:
                        candidate["status"] = "ok"
                        candidate["input_layout"] = "contiguous_conversion_in_graph" if needs_conversion else "native"
                        candidate["timing"] = graph_time(torch, cublaslt_call, repeats=ring_count if ring_count > 1 else 20)
                        candidate["flops"] = 2 * GROUPS * batch * K * N
                        candidate["tflops"] = candidate["flops"] / candidate["timing"]["median_us"] / 1e6
                except Exception as exc:
                    candidate.update(status="unsupported", reason=f"{type(exc).__name__}: {exc}")
                row["candidates"][f"cublasLt_algo_{algorithm}"] = candidate
        payload["rows"].append(row)
        out_path.write_text(json.dumps(payload, indent=2))
        print(json.dumps(row, default=str), flush=True)
    payload["status"] = "done"
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"status": "done", "rows": len(payload["rows"])}), flush=True)


if __name__ == "__main__":
    main()
