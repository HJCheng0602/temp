"""Compare FlashInfer dense MXFP8 candidates without launching a benchmark here.

The process is intentionally single-threaded on the CPU.  Run on an SM120/121
developer machine with a CUDA-enabled FlashInfer installation:

  python compare_dense_candidates.py --output results/dense_candidates.json

Each candidate sees the same quantized A/B tensors and 128x4 swizzled scales.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# N/K pairs copied from analysis.json shape_ledger entries 6, 11, and 17.
# Entry 6 is the principal dense MXFP8 family; entries 11 and 17 are the
# shared-expert and sparse-indexer dense projections.
SHAPES = (
    (1792, 5120), (8192, 1280), (5120, 2048), (1152, 5120), (25600, 6144),
    (5120, 576), (4096, 1280),
)
BATCHES = (1, 8, 32, 128, 256)
BACKENDS = ("cutlass", "b12x", "cudnn")


def event_graph_time(torch, fn, repeats=10, trials=7):
    """Return per-call GPU event times after replaying a captured graph."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    values = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000.0 / repeats)
    return {"median_us": statistics.median(values), "min_us": min(values),
            "max_us": max(values), "repeats_per_event": repeats,
            "trials": trials}


def quantize_pair(flashinfer, a_bf16, w_bf16, swizzled):
    quant = flashinfer.mxfp8_quantize
    a, a_sf = quant(input=a_bf16, is_sf_swizzled_layout=swizzled)
    w, w_sf = quant(input=w_bf16, is_sf_swizzled_layout=swizzled)
    return a, w.t(), a_sf, w_sf


def decode_ue8m0(torch, scale):
    if scale.dtype == torch.uint8:
        return torch.ldexp(torch.ones_like(scale, dtype=torch.float32), scale.to(torch.int32) - 127)
    return scale.float()


def dequant_linear(torch, codes, scale):
    """Independent reference for linear (non-swizzled) scales from the same quantizer."""
    if scale.ndim == 1 and scale.numel() == codes.shape[0] * (codes.shape[1] // 32):
        scale = scale.reshape(codes.shape[0], codes.shape[1] // 32)
    if scale.ndim != 2 or codes.ndim != 2 or codes.shape[1] % 32:
        raise ValueError(f"unexpected linear MXFP8 tensors: codes={tuple(codes.shape)} scale={tuple(scale.shape)}")
    return codes.float() * decode_ue8m0(torch, scale).repeat_interleave(32, dim=1)


def correctness(torch, out, a_ref, b_ref):
    ref = a_ref.float() @ b_ref.float().t()
    got = out.float()
    try:
        torch.testing.assert_close(got, ref, rtol=0.08, atol=0.08)
    except Exception as exc:
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    rel = (got - ref).norm() / ref.norm().clamp_min(1e-8)
    rel_value = float(rel.item())
    if rel_value >= 0.01:
        return {"status": "failed", "relative_l2": rel_value,
                "reason": "relative L2 is not below 0.01", "rtol": 0.08, "atol": 0.08}
    return {"status": "ok", "relative_l2": rel_value, "rtol": 0.08, "atol": 0.08}


def metadata(torch, flashinfer):
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "flashinfer": getattr(flashinfer, "__version__", "unknown"),
        "gpu": props.name, "compute_capability": [props.major, props.minor],
        "sm_count": props.multi_processor_count,
        "cpu_threads": 1, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "dense MXFP8; GEMM-only and quant+GEMM; no allocation/communication",
        "shape_source": "results/kernel_top30_b32_20260926/analysis.json shape_ledger entries 6, 11, 17",
        "api": "flashinfer.mm_mxfp8(A[M,K], B[K,N], A_scale, B_scale, backend=...)",
        "gpu_state": subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,clocks.sm,power.limit,temperature.gpu",
             "--format=csv"], capture_output=True, text=True).stdout,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="one shape and B=1,8")
    parser.add_argument("--backend", choices=BACKENDS, action="append", dest="backends",
                        help="repeat to select candidates (default: all)")
    parser.add_argument("--batches", type=int, nargs="+", choices=BATCHES, default=None,
                        help="batch sizes (default: 1 8 32 128 256)")
    parser.add_argument("--weight-ring-mib", type=int, default=0,
                        help="rotate quantized weight buffers to limit same-weight cache reuse")
    args = parser.parse_args()
    try:
        import torch
        import flashinfer
        from flashinfer import mm_mxfp8
    except Exception as exc:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({"status": "environment_error", "reason": f"{type(exc).__name__}: {exc}"}, indent=2))
        print(json.dumps({"status": "environment_error", "reason": str(exc)}), flush=True)
        return

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(args.device)
    torch.manual_seed(1729)
    torch.set_float32_matmul_precision("highest")
    shapes = SHAPES[:1] if args.smoke else SHAPES
    backends = tuple(args.backends or BACKENDS)
    batches = tuple(args.batches or (BATCHES[:2] if args.smoke else BATCHES))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {"metadata": metadata(torch, flashinfer), "rows": []}
    data["metadata"]["weight_ring_mib"] = args.weight_ring_mib
    for n, k in shapes:
        for m in batches:
            case = {"m": m, "n": n, "k": k, "backends": {}}
            # Quantization is performed once so all backends consume identical bytes/layout.
            try:
                a_bf16 = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.05
                w_bf16 = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
                a, b, a_sf, b_sf = quantize_pair(flashinfer, a_bf16, w_bf16, True)
                # The reference uses the quantizer's linear scales, independently
                # dequantized as UE8M0 blocks; candidate tensors remain swizzled.
                a_ref_code, w_ref_code_t, a_ref_sf, w_ref_sf = quantize_pair(flashinfer, a_bf16, w_bf16, False)
                a_ref = dequant_linear(torch, a_ref_code, a_ref_sf)
                b_ref = dequant_linear(torch, w_ref_code_t.t(), w_ref_sf)
                ring_count = max(1, math.ceil(args.weight_ring_mib * 1024**2 / b.numel()))
                weight_ring = [(b, b_sf)] + [(b.clone(), b_sf.clone()) for _ in range(ring_count - 1)]
                case["weight_ring_count"] = ring_count
                case["weight_ring_bytes"] = sum(w.numel() * w.element_size() + s.numel() * s.element_size()
                                                for w, s in weight_ring)
                for backend in backends:
                    row = {"status": "unknown", "backend": backend, "m": m, "n": n, "k": k}
                    try:
                        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
                        cursor = 0
                        def next_weight():
                            nonlocal cursor
                            pair = weight_ring[cursor % ring_count]
                            cursor += 1
                            return pair
                        def gemm():
                            wb, sb = next_weight()
                            return mm_mxfp8(a, wb, a_sf, sb, out=out, out_dtype=torch.bfloat16, backend=backend)
                        result = gemm()
                        row["correctness"] = correctness(torch, result, a_ref, b_ref)
                        if row["correctness"]["status"] != "ok":
                            row.update(status="numerical_error", reason="correctness check failed")
                            case["backends"][backend] = row
                            continue
                        row["gemm_only"] = event_graph_time(torch, gemm, repeats=ring_count if ring_count > 1 else 10)
                        # Static weight codes/scales are reused; only activations
                        # are quantized in this path, matching decode inference.
                        def quant_gemm():
                            aq, aq_sf = flashinfer.mxfp8_quantize(input=a_bf16, is_sf_swizzled_layout=True)
                            wb, sb = next_weight()
                            return mm_mxfp8(aq, wb, aq_sf, sb, out=out, out_dtype=torch.bfloat16, backend=backend)
                        row["quant_plus_gemm"] = event_graph_time(torch, quant_gemm, repeats=ring_count if ring_count > 1 else 3)
                        row.update(status="ok", flops=2 * m * n * k,
                                   tflops=2 * m * n * k / row["gemm_only"]["median_us"] / 1e6)
                    except Exception as exc:
                        row.update(status="unsupported", reason=f"{type(exc).__name__}: {exc}")
                    case["backends"][backend] = row
                del a_bf16, w_bf16, a, b, a_sf, b_sf, a_ref, b_ref, weight_ring
            except Exception as exc:
                case["quantization_error"] = f"{type(exc).__name__}: {exc}"
            data["rows"].append(case)
            output.write_text(json.dumps(data, indent=2))
            print(json.dumps(case, default=str), flush=True)
    print(json.dumps({"status": "done", "cases": len(data["rows"])}), flush=True)


if __name__ == "__main__":
    main()
