"""One-call sparse MLA input/output probe; torch is imported lazily."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path


TARGET = "vllm.models.deepseek_v41.nvidia.flashinfer_sparse"
CAPTURE_KEYS = (
    "query", "sparse_indices", "swa_topk_lens", "extra_sparse_indices",
    "extra_sparse_topk_lens", "sinks", "swa_kv_cache", "compressed_kv_cache",
)


def _leaves(value, prefix="value"):
    if hasattr(value, "detach") and hasattr(value, "shape"):
        yield prefix, value
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _leaves(item, f"{prefix}[{index}]")


def _cpu_clone(value):
    return value.detach().to(device="cpu").clone()


def _output_stats(first, second):
    import torch

    left = first if isinstance(first, dict) else dict(_leaves(first, "output"))
    right = second if isinstance(second, dict) else dict(_leaves(second, "output"))
    rows = []
    for key in list(dict.fromkeys(list(left) + list(right))):
        a, b = left.get(key), right.get(key)
        row = {"key": key}
        if a is None or b is None:
            row["status"] = "missing"
        elif tuple(a.shape) != tuple(b.shape):
            row.update(status="shape_mismatch", left_shape=list(a.shape),
                       right_shape=list(b.shape))
        elif torch.equal(a, b):
            row.update(status="equal", max_abs=0.0, rmse=0.0)
        else:
            difference = a.float() - b.float()
            row.update(status="different", max_abs=float(difference.abs().max()),
                       rmse=float(difference.square().mean().sqrt()))
        rows.append(row)
    return rows


def _meta(tensor):
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype), "device": str(tensor.device)}


def _snapshot_kwargs(kwargs):
    snapshots = {}
    metadata = {}
    for key in CAPTURE_KEYS:
        value = kwargs.get(key)
        if value is None:
            continue
        if not hasattr(value, "detach"):
            snapshots[key] = value if isinstance(value, (int, float, bool, str)) else repr(value)
            continue
        if "cache" in key:
            original_meta = _meta(value)
            index_key = "sparse_indices" if key == "swa_kv_cache" else "extra_sparse_indices"
            indices = kwargs.get(index_key)
            max_slot = int(indices.max().item()) if indices is not None and indices.numel() else -1
            page_width = int(value.shape[1]) if value.ndim > 1 else int(value.shape[0])
            pages = max(1, max_slot // page_width + 1)
            if pages > value.shape[0]:
                raise RuntimeError(f"Sparse index exceeds cache capacity: {key}: {max_slot}")
            value = value[:pages]
            metadata[key] = dict(original_meta, captured_shape=list(value.shape), max_valid_slot=max_slot,
                                 page_width=page_width, pages=pages)
        else:
            metadata[key] = _meta(value)
        snapshots[key] = _cpu_clone(value)
    scalars = {}
    for key, value in kwargs.items():
        if isinstance(value, (int, float, bool, str)) or value is None:
            scalars[key] = value
    return snapshots, metadata, scalars


@contextmanager
def capture_sparse_calls(output_path, label):
    """Capture the first sparse prefill call with a compressed cache."""
    import importlib
    import torch

    module = importlib.import_module(TARGET)
    original = module.flashinfer_trtllm_batch_decode_sparse_mla_dsv4
    state = {"calls": 0, "captured": False, "payload": None, "metadata": None}

    def wrapper(*args, **kwargs):
        eligible = kwargs.get("compressed_kv_cache") is not None
        if state["captured"] or not eligible:
            return original(*args, **kwargs)
        state["calls"] += 1
        first = original(*args, **kwargs)
        first_out = kwargs.get("out") if kwargs.get("out") is not None else first
        first_cpu = {key: _cpu_clone(value) for key, value in _leaves(first_out, "output")}
        if not first_cpu:
            raise RuntimeError("Sparse probe did not capture an output tensor")
        second = original(*args, **kwargs)
        snapshots, tensor_meta, scalars = _snapshot_kwargs(kwargs)
        state["captured"] = True
        second_out = kwargs.get("out") if kwargs.get("out") is not None else second
        output_second = {key: _cpu_clone(value)
                         for key, value in _leaves(second_out, "output")}
        state["payload"] = dict(label=label, inputs=snapshots,
                                outputs_first=first_cpu,
                                outputs_second=output_second)
        state["metadata"] = dict(label=label, calls_before_capture=state["calls"],
                                  tensor_meta=tensor_meta, scalar_kwargs=scalars,
                                  workspace_meta=_meta(kwargs["workspace_buffer"]),
                                  output_meta={key: _meta(value) for key, value in output_second.items()},
                                  output_comparison=_output_stats(first_cpu, state["payload"]["outputs_second"]))
        return second

    module.flashinfer_trtllm_batch_decode_sparse_mla_dsv4 = wrapper
    try:
        yield state
    finally:
        module.flashinfer_trtllm_batch_decode_sparse_mla_dsv4 = original
        path = Path(output_path)
        if path.is_dir() or not path.suffix:
            path = path / str(label)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = state["metadata"] or dict(label=label, captured=False,
                                             calls_before_capture=state["calls"])
        if state["captured"]:
            capability = None
            if torch.cuda.is_available():
                capability = list(torch.cuda.get_device_capability())
            metadata["device_capability"] = capability
            metadata["captured"] = True
            path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
            torch.save(state["payload"], path.with_suffix(".pt"))
        else:
            path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
