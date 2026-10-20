"""Reproduce sparse-index order sensitivity from two captured calls."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def _dtype(torch, text):
    return getattr(torch, str(text).split(".")[-1])


def _allocate(torch, value, meta, device):
    shape = tuple(value.shape)
    stride = tuple(meta.get("stride", value.stride()))
    storage = sum((size - 1) * step for size, step in zip(shape, stride) if size) + 1
    backing = torch.zeros((storage,), dtype=_dtype(torch, meta["dtype"]), device=device)
    result = torch.as_strided(backing, size=shape, stride=stride)
    result.copy_(value)
    return result


def _load_capture(torch, capture, metadata, device):
    payload = torch.load(capture, map_location="cpu", weights_only=True)
    meta = json.loads(Path(metadata).read_text())
    kwargs = {}
    for key, value in payload.get("inputs", {}).items():
        if hasattr(value, "shape"):
            kwargs[key] = _allocate(torch, value, meta.get("tensor_meta", {}).get(key, {}), device)
        elif value is not None:
            kwargs[key] = value
    for key, value in meta.get("scalar_kwargs", {}).items():
        if key not in kwargs and value is not None and key not in {"out", "workspace_buffer"}:
            kwargs[key] = value
    workspace = meta.get("workspace_meta", {})
    kwargs["workspace_buffer"] = torch.zeros(tuple(workspace["shape"]), dtype=torch.uint8, device=device)
    output_cpu = payload["outputs_first"]["output"]
    output_meta = meta.get("output_meta", {}).get("output", {"dtype": str(output_cpu.dtype)})
    kwargs["out"] = _allocate(torch, output_cpu, output_meta, device)
    return kwargs, payload, meta


def _byte_equal(torch, left, right):
    if tuple(left.shape) != tuple(right.shape):
        return False
    return torch.equal(left.detach().contiguous().view(torch.uint8),
                       right.detach().contiguous().view(torch.uint8))


def _canonical(torch, indices):
    values = indices.clone()
    valid = values >= 0
    high = torch.iinfo(values.dtype).max
    values = torch.where(valid, values, torch.full_like(values, high))
    values = values.sort(dim=-1).values
    return torch.where(values == high, torch.full_like(values, -1), values)


def _output_diff(torch, first, current):
    if first is None or current is None:
        return {"status": "missing"}
    delta = first.float() - current.float()
    return {"status": "equal" if torch.equal(first, current) else "different",
            "max_abs": float(delta.abs().max()),
            "rmse": float(delta.square().mean().sqrt()),
            "last_row_max_abs": float(delta[-1:].abs().max()),
            "last_row_rmse": float(delta[-1:].square().mean().sqrt())}


def _run(torch, function, kwargs):
    result = function(**kwargs)
    torch.cuda.synchronize()
    output = kwargs["out"] if kwargs.get("out") is not None else result
    return output.detach().cpu().clone() if hasattr(output, "detach") else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-capture", type=Path, required=True)
    parser.add_argument("--left-metadata", type=Path, required=True)
    parser.add_argument("--right-capture", type=Path, required=True)
    parser.add_argument("--right-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    deployment = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(deployment))
    from replay_sparse_probe import _configure_environment

    _configure_environment(args.gpu)
    if os.environ.get("CED_SPARSE_ORDER_BOOTSTRAPPED") != "1":
        os.environ["CED_SPARSE_ORDER_BOOTSTRAPPED"] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)
    import torch
    from replay_sparse_probe import _load_function

    torch.set_num_threads(2)
    device = torch.device("cuda")
    left, left_payload, left_meta = _load_capture(torch, args.left_capture, args.left_metadata, device)
    right, right_payload, right_meta = _load_capture(torch, args.right_capture, args.right_metadata, device)
    function = _load_function()
    original_left = _run(torch, function, left)
    original_right = _run(torch, function, right)

    cache_compare = {}
    for key in ("swa_kv_cache", "compressed_kv_cache"):
        if key in left and key in right:
            cache_compare[key] = _byte_equal(torch, left[key], right[key])
    index_compare = {}
    canonical_left, _, _ = _load_capture(torch, args.left_capture, args.left_metadata, device)
    canonical_right, _, _ = _load_capture(torch, args.right_capture, args.right_metadata, device)
    for key in ("sparse_indices", "extra_sparse_indices"):
        if key in left and key in right:
            index_compare[key] = dict(
                raw_equal=torch.equal(left[key], right[key]),
                canonical_equal=torch.equal(_canonical(torch, left[key]), _canonical(torch, right[key])),
            )
            canonical_left[key] = _canonical(torch, left[key])
            canonical_right[key] = _canonical(torch, right[key])
    canonical_left["out"] = torch.zeros_like(left["out"])
    canonical_right["out"] = torch.zeros_like(right["out"])
    canonical_output_left = _run(torch, function, canonical_left)
    canonical_output_right = _run(torch, function, canonical_right)
    captured_left = left_payload["outputs_first"]["output"]
    captured_right = right_payload["outputs_first"]["output"]
    report = dict(
        gpu=str(args.gpu), cache_byte_equal=cache_compare, index_sets=index_compare,
        original=_output_diff(torch, original_left, original_right),
        left_vs_captured=_output_diff(torch, captured_left, original_left),
        right_vs_captured=_output_diff(torch, captured_right, original_right),
        canonical=_output_diff(torch, canonical_output_left, canonical_output_right),
        canonical_left_vs_captured=_output_diff(torch, captured_left, canonical_output_left),
        canonical_right_vs_captured=_output_diff(torch, captured_right, canonical_output_right),
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, separators=(",", ":")))


if __name__ == "__main__":
    main()
