"""SM120 dense MXFP8 CUDA-graph microbenchmark.

The input contract is a ``torch.save`` dictionary keyed by ``"KxN"`` or a
dictionary with a ``cases`` member. Each case contains FP8 ``a`` [M,K], FP8
``b`` [K,N], and UE8M0 scale bytes. Captures from ``capture_hook.py`` retain
FlashInfer's flat F8_128x4 scale ABI and the original tensor strides. Missing
cases are generated deterministically, and are marked synthetic in the JSON.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


SHAPES = (
    (5120, 1792), (1280, 8192), (2048, 5120), (1280, 4096),
    (6144, 25600), (5120, 1152), (576, 5120),
)


def _case_key(k, n):
    return f"{k}x{n}"


def _to_device_preserve(value, device):
    """Copy a captured tensor without changing its shape or strides."""
    import torch

    if not isinstance(value, torch.Tensor):
        return value
    out = torch.empty_strided(tuple(value.shape), tuple(value.stride()),
                              dtype=value.dtype, device=device)
    out.copy_(value)
    return out


def _capture_tensor_args(payload):
    import torch

    args = payload.get("args")
    if not isinstance(args, (tuple, list)) or len(args) < 4:
        raise ValueError("capture must contain args=(a,b,a_scale,b_scale,...)")
    if not all(isinstance(args[i], torch.Tensor) for i in range(4)):
        raise ValueError("capture args[0:4] must be tensors")
    return args[0], args[1], args[2], args[3]


def _load_capture(path, device, m, k, n):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    a, b, a_scale, b_scale = _capture_tensor_args(payload)
    if tuple(a.shape) != (m, k) or tuple(b.shape) != (k, n):
        raise ValueError(f"capture shapes are A={tuple(a.shape)} B={tuple(b.shape)}, "
                         f"expected {(m, k)} and {(k, n)}")
    return {
        "a": _to_device_preserve(a, device),
        "b": _to_device_preserve(b, device),
        "a_scale": _to_device_preserve(a_scale, device),
        "b_scale": _to_device_preserve(b_scale, device),
        "provenance": "captured_flashinfer_rank0",
        "capture_path": str(path),
        "capture_output": _to_device_preserve(payload.get("output"), device),
        "capture_metadata": payload.get("metadata", {}),
        "capture_backend": (payload.get("kwargs", {}) or {}).get("backend"),
    }


def _capture_files(directory, m, k, n):
    """Prefer rank 0 and select one capture by tensor metadata, not filename."""
    import torch

    candidates = []
    for path in sorted(Path(directory).glob("fp8_rank*_*.pt")):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            a, b, _, _ = _capture_tensor_args(payload)
            if tuple(a.shape) == (m, k) and tuple(b.shape) == (k, n):
                rank = int(payload.get("metadata", {}).get("rank", 999))
                candidates.append((rank != 0, rank, path))
        except Exception:
            continue
    if not candidates:
        raise FileNotFoundError(f"no captured fp8 case for A=({m},{k}) B=({k},{n})")
    candidates.sort(key=lambda item: (item[0], item[1], str(item[2])))
    return candidates[0][2]


def _synthetic(k, n, m, device, seed):
    import torch

    gen = torch.Generator(device=device).manual_seed(seed)
    a_float = torch.randn((m, k), generator=gen, device=device) * 0.25
    b_float = torch.randn((k, n), generator=gen, device=device) * 0.25
    fp8 = torch.float8_e4m3fn
    a = a_float.to(fp8)
    b = b_float.to(fp8)
    # Unit UE8M0 scales are a valid exact MXFP8 operand contract.
    a_scale = torch.full((m, k // 32), 127, dtype=torch.uint8, device=device)
    b_scale = torch.full((n, k // 32), 127, dtype=torch.uint8, device=device)
    return {"a": a, "b": b, "a_scale": a_scale, "b_scale": b_scale,
            "provenance": "synthetic_unit_scale"}


def _load_case(path, capture_dir, k, n, m, device, seed):
    import torch

    if capture_dir is not None:
        capture = _capture_files(capture_dir, m, k, n)
        return _load_capture(capture, device, m, k, n)
    if path is None:
        return _synthetic(k, n, m, device, seed)
    payload = torch.load(path, map_location=device, weights_only=False)
    cases = payload.get("cases", payload) if isinstance(payload, dict) else {}
    item = cases.get(_case_key(k, n))
    if item is None:
        raise KeyError(f"input payload has no {_case_key(k, n)} case")
    required = ("a", "b", "a_scale", "b_scale")
    if any(key not in item for key in required):
        raise KeyError(f"{_case_key(k, n)} requires {required}")
    return {key: _to_device_preserve(item[key], device) for key in required} | {
        "provenance": "captured_payload",
    }


def _scale_bytes(scale):
    import torch

    if scale.dtype == torch.uint8:
        return scale
    # float8_e8m0fnu is numerically the scale value, but callers may provide
    # its raw byte view; converting through uint8 is only valid for the latter.
    if str(scale.dtype).endswith("float8_e8m0fnu"):
        return scale.view(torch.uint8)
    raise TypeError(f"MXFP8 scales must be uint8 or float8_e8m0fnu, got {scale.dtype}")


def _dequant_a(a, scale):
    import torch

    raw = _scale_bytes(scale).to(torch.int16)
    if tuple(raw.shape) != (a.shape[0], a.shape[1] // 32):
        raise ValueError(f"A scale shape {tuple(raw.shape)} != {(a.shape[0], a.shape[1] // 32)}")
    factor = torch.pow(torch.tensor(2.0, device=a.device), (raw - 127).float())
    return a.float() * factor.repeat_interleave(32, dim=1)


def _dequant_b(b, scale):
    import torch

    raw = _scale_bytes(scale).to(torch.int16)
    k, n = b.shape
    if tuple(raw.shape) == (n, k // 32):
        factor = torch.pow(torch.tensor(2.0, device=b.device), (raw - 127).float())
        factor = factor.repeat_interleave(32, dim=1).transpose(0, 1)
    elif tuple(raw.shape) == (k // 32, n):
        factor = torch.pow(torch.tensor(2.0, device=b.device), (raw - 127).float())
        factor = factor.repeat_interleave(32, dim=0)
    else:
        raise ValueError(f"B scale shape {tuple(raw.shape)} incompatible with {tuple(b.shape)}")
    return b.float() * factor


def _unswizzle_scale(scale, rows, k):
    """Invert vLLM/FlashInfer's F8_128x4 scale mapping exactly."""
    import torch

    raw = _scale_bytes(scale).contiguous()
    if raw.ndim != 1 or k % 32:
        raise ValueError("F8_128x4 scale must be a flat uint8 buffer and K%32==0")
    mt = (rows + 127) // 128
    kt = (k + 127) // 128
    expected = mt * 128 * kt * 4
    if raw.numel() != expected:
        raise ValueError(f"F8_128x4 scale length {raw.numel()} != {expected}")
    # swizzle_mxfp8_scale: [mt,4,32,kt,4] -> transpose(1,3) -> flat.
    packed = raw.view(mt, kt, 32, 4, 4)
    linear = packed.transpose(1, 3).contiguous().view(mt * 128, kt * 4)
    return linear[:rows, : k // 32]


def _swizzle_scale(linear, rows, k):
    """Pack row-major [rows,K/32] scales into F8_128x4."""
    import torch

    mt, kt = (rows + 127) // 128, (k + 127) // 128
    padded = torch.full((mt * 128, kt * 4), 127, dtype=torch.uint8,
                        device=linear.device)
    padded[:rows, : linear.shape[1]].copy_(linear)
    return padded.view(mt, 4, 32, kt, 4).transpose(1, 3).contiguous().view(-1)


def _make_k_padded_case(case, padded_k=640):
    """Build static B/K padding; A and A scales are refreshed per invocation."""
    import torch

    a, b = case["a"], case["b"]
    k, n = b.shape
    if k != 576 or padded_k != 640:
        raise ValueError("the padded candidate is defined only for K=576 -> 640")
    a_pad = torch.empty((a.shape[0], padded_k), dtype=a.dtype, device=a.device)
    a_pad.zero_()
    b_pad = torch.empty_strided((padded_k, n), (1, padded_k), dtype=b.dtype,
                                device=b.device)
    b_pad.zero_()
    b_pad[:k].copy_(b)
    a_linear = _unswizzle_scale(case["a_scale"], a.shape[0], k)
    b_linear = _unswizzle_scale(case["b_scale"], n, k)
    a_scale_pad = _swizzle_scale(
        torch.cat((a_linear, torch.full((a_linear.shape[0], (padded_k-k)//32),
                                        127, dtype=torch.uint8, device=a.device)), dim=1),
        a.shape[0], padded_k)
    b_scale_pad = _swizzle_scale(
        torch.cat((b_linear, torch.full((b_linear.shape[0], (padded_k-k)//32),
                                        127, dtype=torch.uint8, device=b.device)), dim=1),
        n, padded_k)
    return {**case, "a_pad": a_pad, "b_pad": b_pad,
            "a_scale_pad": a_scale_pad, "a_scale_source": a_scale_pad.clone(),
            "b_scale_pad": b_scale_pad,
            "pad_k": padded_k}


def _reference(case):
    import torch

    a_scale, b_scale = case["a_scale"], case["b_scale"]
    if a_scale.ndim == 2 and b_scale.ndim == 2:
        return (_dequant_a(case["a"], a_scale) @ _dequant_b(case["b"], b_scale),
                "fp32_dequant_linear_scales")
    a_linear = _unswizzle_scale(a_scale, case["a"].shape[0], case["a"].shape[1])
    b_linear = _unswizzle_scale(b_scale, case["b"].shape[1], case["b"].shape[0])
    a_factor = torch.ldexp(torch.ones_like(a_linear, dtype=torch.float32),
                           a_linear.to(torch.int32) - 127)
    b_factor = torch.ldexp(torch.ones_like(b_linear, dtype=torch.float32),
                           b_linear.to(torch.int32) - 127)
    a_ref = case["a"].float() * a_factor.repeat_interleave(32, dim=1)
    b_ref = case["b"].float() * b_factor.repeat_interleave(32, dim=1).transpose(0, 1)
    return a_ref @ b_ref, "fp32_dequant_f8_128x4_scales"


def _invoke_factory(case, backend):
    import flashinfer
    import torch

    fn = getattr(flashinfer, "mm_mxfp8", None)
    if fn is None:
        from flashinfer.gemm import mm_mxfp8 as fn
    if backend == "b12x_padded_k576" and case["b"].shape[0] == 576:
        padded = case.get("padded_case")
        if padded is None:
            padded = _make_k_padded_case(case)
            case["padded_case"] = padded

        def invoke():
            # Keep the conversion in the timed graph: this candidate pays the
            # activation and activation-scale repack cost on every invocation.
            padded["a_pad"].zero_()
            padded["a_pad"][:, :576].copy_(case["a"])
            padded["a_scale_pad"].copy_(padded["a_scale_source"])
            return fn(padded["a_pad"], padded["b_pad"], padded["a_scale_pad"],
                      padded["b_scale_pad"], backend="b12x", out_dtype=torch.bfloat16)

        return invoke
    a, b = case["a"], case["b"]

    def invoke():
        return fn(a, b, case["a_scale"], case["b_scale"],
                  backend=backend, out_dtype=torch.bfloat16)

    return invoke


def _measure(invoke, cold_bytes, samples, replays):
    import torch

    for _ in range(10):
        invoke()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = invoke()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    hot = []
    for _ in range(samples):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        hot.append(start.elapsed_time(end) * 1000.0 / replays)
    cold_buffer = torch.empty(cold_bytes, dtype=torch.uint8, device=output.device)
    cold_buffer.fill_(1)
    cold = []
    for _ in range(samples):
        starts, ends = [], []
        for _ in range(replays):
            cold_buffer.add_(1)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            starts.append(start)
            ends.append(end)
        ends[-1].synchronize()
        cold.append(statistics.mean(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)))
    return output.clone(), {"hot_us": hot, "cold_us": cold,
                            "hot_median_us": statistics.median(hot),
                            "cold_median_us": statistics.median(cold),
                            "samples": samples, "replays": replays,
                            "cold_bytes": cold_bytes}


def _error(value, reference):
    import torch

    delta = value.float() - reference
    return {"finite": bool(torch.isfinite(value).all().item()),
            "max_abs": float(delta.abs().max().item()),
            "rel_l2": float(delta.norm().item() /
                             reference.norm().clamp_min(1e-12).item())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--capture-dir", type=Path, default=None,
                        help="directory written by capture_hook.py; rank 0 is preferred")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--backends", nargs="+", default=["b12x", "cutlass", "cudnn"])
    parser.add_argument("--shape", nargs=2, type=int, metavar=("K", "N"))
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--cold-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    shapes = [tuple(args.shape)] if args.shape else SHAPES
    result = {"schema": "sm120_dense_mxfp8_probe_v1", "m": args.m,
              "device": str(device), "shapes": {}}
    for k, n in shapes:
        if k % 32:
            raise ValueError(f"K={k} must be divisible by MXFP8 block size 32")
        case = _load_case(args.input, args.capture_dir, k, n, args.m, device,
                          args.seed + k + n)
        reference, reference_kind = _reference(case)
        row = {"provenance": case["provenance"],
               "reference_kind": reference_kind,
               "capture_path": case.get("capture_path"),
               "captured_backend": case.get("capture_backend"),
               "input_shapes": {key: list(case[key].shape) for key in ("a", "b")},
               "input_strides": {key: list(case[key].stride()) for key in ("a", "b")},
               "backends": {}}
        for backend in args.backends:
            try:
                if backend == "b12x_padded_k576" and k != 576:
                    raise ValueError("b12x_padded_k576 applies only to K=576")
                output, timing = _measure(
                    _invoke_factory(case, backend), args.cold_bytes,
                    args.samples, args.replays,
                )
                row["backends"][backend] = {
                    "status": "ok", "timing": timing,
                    "reference_error": _error(output, reference),
                    "output_shape": list(output.shape),
                }
            except Exception as exc:
                row["backends"][backend] = {"status": "unavailable", "error": repr(exc)}
        result["shapes"][_case_key(k, n)] = row
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
