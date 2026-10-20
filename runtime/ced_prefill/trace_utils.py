"""Small observational forward-trace helpers for CED debugging.

Torch is imported lazily so this module can be imported by result tooling and
the coordinator without initializing a runtime or CUDA.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any


def _tensor_leaves(value, prefix="value"):
    if value is None:
        return
    if hasattr(value, "detach") and hasattr(value, "shape"):
        yield prefix, value
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _tensor_leaves(item, f"{prefix}[{index}]")


def _last_row(value):
    import torch

    tensor = value.detach()
    if tensor.ndim == 0:
        tensor = tensor.reshape(1)
    else:
        tensor = tensor[-1:]
    return tensor.to(device="cpu").clone()


def _input_tensor(args, kind):
    index = 1 if kind == "attn" and len(args) > 1 else 0
    if len(args) <= index:
        return None
    for _, tensor in _tensor_leaves(args[index], "input"):
        return tensor
    return None


def _module_kind(name):
    leaf = name.rsplit(".", 1)[-1].lower()
    if leaf == "attn" or leaf.endswith("attention"):
        return "attn"
    if leaf in {"ffn", "mlp", "feed_forward"} or "ffn" in leaf or "mlp" in leaf:
        return "ffn"
    return "layer"


class TraceRecorder:
    def __init__(self, max_layers=40):
        self.max_layers = max_layers
        self.records: dict[str, list[dict[str, Any]]] = {}
        self.events: list[tuple[str, int]] = []
        self._handles = []

    def _record(self, name, kind, inputs, outputs):
        if not inputs and not outputs:
            return
        rows = self.records.setdefault(name, [])
        call = len(rows)
        rows.append(dict(call=call, kind=kind, inputs=inputs, outputs=outputs))
        self.events.append((name, call))

    def attach(self, backbone):
        layers = getattr(backbone, "layers", None)
        if layers is None:
            model = getattr(backbone, "model", None)
            layers = getattr(model, "layers", None)
        if layers is None:
            raise ValueError("backbone has no layers or model.layers")
        for index, layer in enumerate(layers):
            if index >= self.max_layers:
                break
            if layer is None or layer.__class__.__name__ == "PPMissingLayer":
                continue
            modules = [(f"layers.{index}", layer)]
            for child_name in ("attn", "attention", "ffn", "mlp", "feed_forward"):
                child = getattr(layer, child_name, None)
                if child is not None:
                    modules.append((f"layers.{index}.{child_name}", child))
            for name, module in modules:
                kind = _module_kind(name)
                self._handles.append(module.register_forward_pre_hook(
                    lambda mod, args, name=name, kind=kind: self._pre(name, kind, args)
                ))
                self._handles.append(module.register_forward_hook(
                    lambda mod, args, output, name=name, kind=kind:
                    self._post(name, kind, args, output)
                ))
        return self

    def _pre(self, name, kind, args):
        selected = _input_tensor(args, kind)
        inputs = {}
        if selected is not None:
            inputs["selected"] = _last_row(selected)
        self._record(name + ".pre", kind, inputs, {})

    def _post(self, name, kind, args, output):
        outputs = {key: _last_row(value)
                   for key, value in _tensor_leaves(output, "output")}
        self._record(name + ".post", kind, {}, outputs)

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def summary(self):
        return {name: [
            {"call": row["call"], "kind": row["kind"],
             "inputs": {key: _tensor_meta(value) for key, value in row["inputs"].items()},
             "outputs": {key: _tensor_meta(value) for key, value in row["outputs"].items()}}
            for row in rows
        ] for name, rows in self.records.items()}

    def save_raw(self, path):
        import torch

        torch.save(self.records, Path(path))


def _tensor_meta(tensor):
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "device": str(tensor.device), "numel": tensor.numel()}


@contextmanager
def capture_backbone(backbone, max_layers=40):
    recorder = TraceRecorder(max_layers=max_layers).attach(backbone)
    try:
        yield recorder
    finally:
        recorder.close()


def compare_traces(left, right):
    """Compare two recorders or raw ``records`` dictionaries."""
    import torch

    left_records = getattr(left, "records", left)
    right_records = getattr(right, "records", right)
    left_events = getattr(left, "events", None)
    right_events = getattr(right, "events", None)
    if left_events is None:
        left_events = [(name, row["call"]) for name, rows in left_records.items()
                       for row in rows]
    if right_events is None:
        right_events = [(name, row["call"]) for name, rows in right_records.items()
                        for row in rows]
    events = list(dict.fromkeys(list(left_events) + list(right_events)))
    comparisons = []
    first_mismatch = None

    def compare_tensor(module, call, side, key, a, b):
        nonlocal first_mismatch
        if a is None or b is None:
            result = {"module": module, "call": call, "side": side,
                      "key": key, "status": "missing"}
        elif tuple(a.shape) != tuple(b.shape):
            result = {"module": module, "call": call, "side": side,
                      "key": key, "status": "shape_mismatch",
                      "left_shape": list(a.shape), "right_shape": list(b.shape)}
        elif torch.equal(a, b):
            result = {"module": module, "call": call, "side": side,
                      "key": key, "status": "equal", "max_abs": 0.0,
                      "rmse": 0.0, "relative_l2": 0.0}
        else:
            difference = a.float() - b.float()
            denom = b.float().norm().clamp_min(1e-30)
            result = {"module": module, "call": call, "side": side,
                      "key": key, "status": "different",
                      "max_abs": float(difference.abs().max()),
                      "rmse": float(difference.square().mean().sqrt()),
                      "relative_l2": float(difference.norm() / denom)}
        comparisons.append(result)
        if first_mismatch is None and result["status"] != "equal":
            first_mismatch = result

    for module, call in events:
        left_rows, right_rows = left_records.get(module, []), right_records.get(module, [])
        lrow = left_rows[call] if call < len(left_rows) else {}
        rrow = right_rows[call] if call < len(right_rows) else {}
        for side in ("inputs", "outputs"):
            keys = list(dict.fromkeys(list(lrow.get(side, {})) + list(rrow.get(side, {}))))
            for key in keys:
                compare_tensor(module, call, side, key,
                               lrow.get(side, {}).get(key), rrow.get(side, {}).get(key))
    return {"first_mismatch": first_mismatch, "comparisons": comparisons}


def json_summary(recorder):
    """Return a JSON-serializable shape/call summary for a recorder."""
    return recorder.summary() if hasattr(recorder, "summary") else {
        "records": "raw trace supplied; use compare_traces for numeric comparison"
    }


def write_summary(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")
