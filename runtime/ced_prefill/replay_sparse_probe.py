"""Replay a captured sparse MLA call on a CUDA device, without model weights."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def _dtype(torch, text):
    name = str(text).split(".")[-1]
    value = getattr(torch, name, None)
    if value is None:
        raise ValueError(f"Unsupported captured dtype: {text}")
    return value


def _allocate(torch, cpu_value, meta, device):
    shape = tuple(cpu_value.shape)
    stride = tuple(meta.get("stride", cpu_value.stride()))
    dtype = _dtype(torch, meta["dtype"])
    storage = 0
    for size, step in zip(shape, stride):
        if size:
            storage += (size - 1) * step
    backing = torch.zeros((storage + 1,), dtype=dtype, device=device)
    result = torch.as_strided(backing, size=shape, stride=stride)
    result.copy_(cpu_value, non_blocking=False)
    return result


def _diff(torch, first, current):
    if first is None or current is None:
        return {"status": "missing"}
    if tuple(first.shape) != tuple(current.shape):
        return {"status": "shape_mismatch", "first_shape": list(first.shape),
                "current_shape": list(current.shape)}
    if torch.equal(first, current):
        return {"status": "equal", "max_abs": 0.0, "rmse": 0.0}
    delta = first.float() - current.float()
    return {"status": "different", "max_abs": float(delta.abs().max()),
            "rmse": float(delta.square().mean().sqrt())}


def _load_function():
    import importlib

    try:
        return importlib.import_module(
            "vllm.models.deepseek_v41.nvidia.flashinfer_sparse"
        ).flashinfer_trtllm_batch_decode_sparse_mla_dsv4
    except (ImportError, AttributeError):
        return importlib.import_module("vllm.utils.flashinfer").flashinfer_trtllm_batch_decode_sparse_mla_dsv4


def _configure_environment(gpu):
    deployment = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(deployment))
    from parallel_matrix import environment

    configured = environment()
    configured["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.update(configured)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")

    _configure_environment(args.gpu)
    if os.environ.get("CED_SPARSE_PROBE_BOOTSTRAPPED") != "1":
        os.environ["CED_SPARSE_PROBE_BOOTSTRAPPED"] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    metadata = json.loads(args.metadata.read_text())
    payload = torch.load(args.capture, map_location="cpu", weights_only=True)
    device = torch.device("cuda")
    inputs = payload.get("inputs", {})
    tensor_meta = metadata.get("tensor_meta", {})
    kwargs = {}
    for key, cpu_value in inputs.items():
        if not hasattr(cpu_value, "shape"):
            if cpu_value is not None:
                kwargs[key] = cpu_value
            continue
        kwargs[key] = _allocate(torch, cpu_value, tensor_meta.get(key, {}), device)
    for key, value in metadata.get("scalar_kwargs", {}).items():
        if key not in kwargs and value is not None and key not in {"out", "workspace_buffer"}:
            kwargs[key] = value

    workspace = metadata.get("workspace_meta", {})
    workspace_shape = tuple(workspace.get("shape", []))
    if not workspace_shape:
        raise ValueError("Capture metadata has no workspace shape")
    kwargs["workspace_buffer"] = torch.zeros(workspace_shape, dtype=torch.uint8, device=device)

    output_cpu = payload.get("outputs_first", {}).get("output")
    if output_cpu is None:
        raise ValueError("Capture payload has no outputs_first['output'] tensor")
    output_meta = metadata.get("output_meta", {}).get("output", {})
    output = _allocate(torch, output_cpu, output_meta or {"dtype": str(output_cpu.dtype)}, device)
    kwargs["out"] = output
    function = _load_function()
    cache_keys = [key for key in kwargs if "cache" in key]
    index_keys = [key for key in kwargs if "indices" in key]
    first_output = None
    rows = []
    for repeat in range(args.repeats):
        before_cache = {key: kwargs[key].clone() for key in cache_keys}
        before_indices = {key: kwargs[key].clone() for key in index_keys}
        result = function(**kwargs)
        torch.cuda.synchronize()
        observed = kwargs["out"] if kwargs.get("out") is not None else result
        observed_cpu = observed.detach().cpu().clone() if hasattr(observed, "detach") else None
        if first_output is None and observed_cpu is not None:
            first_output = observed_cpu
        rows.append(dict(
            repeat=repeat + 1,
            vs_first=_diff(torch, first_output, observed_cpu),
            vs_captured=_diff(torch, output_cpu, observed_cpu),
            cache_changed={key: not torch.equal(value, kwargs[key])
                           for key, value in before_cache.items()},
            indices_changed={key: not torch.equal(value, kwargs[key])
                             for key, value in before_indices.items()},
        ))
    report = dict(capture=str(args.capture), metadata=str(args.metadata), gpu=str(args.gpu),
                  repeats=args.repeats, rows=rows)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, separators=(",", ":")))


if __name__ == "__main__":
    main()
