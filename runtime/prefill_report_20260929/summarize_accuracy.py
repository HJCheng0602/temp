"""CPU-only aggregation for independent CED accuracy runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _torch():
    import torch

    torch.set_num_threads(1)
    return torch


def _json_files(root: Path):
    return sorted(path for path in root.rglob("*.json")
                  if path.name not in {"configuration.json", "status.json"})


def _pt(root: Path, pattern: str):
    return sorted(root.rglob(pattern))


def _sample(value: Any):
    if isinstance(value, dict):
        return value.get("token_ids", value)
    return value


def _metrics(torch, lhs, rhs):
    if lhs is None or rhs is None:
        return {"status": "missing"}
    if tuple(lhs.shape) != tuple(rhs.shape):
        return {"status": "shape_mismatch", "lhs": list(lhs.shape),
                "rhs": list(rhs.shape)}
    if lhs.numel() == 0:
        return {"status": "empty"}
    a, b = lhs.float(), rhs.float()
    d = a - b
    return {"status": "ok", "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
            "shape": list(lhs.shape), "max_abs": float(d.abs().max()),
            "rmse": float(d.square().mean().sqrt()),
            "relative_l2": float(d.norm() / b.norm().clamp_min(1e-12)),
            "torch_equal": bool(torch.equal(lhs, rhs)),
            "argmax_equal": bool(torch.equal(lhs.argmax(-1), rhs.argmax(-1)))}


def _kl(torch, lhs, rhs):
    if lhs is None or rhs is None or tuple(lhs.shape) != tuple(rhs.shape) or not lhs.numel():
        return None
    p = torch.log_softmax(lhs.float(), -1)
    q = torch.log_softmax(rhs.float(), -1)
    return float(torch.nn.functional.kl_div(q, p.exp(), reduction="batchmean"))


def _load_pt(torch, path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # preserve missing/invalid evidence in report
        return {"_load_error": str(exc), "_path": str(path)}


def _payloads(torch, root: Path):
    rows = {}
    for path in _pt(root, "*.pt"):
        value = _load_pt(torch, path)
        if isinstance(value, dict) and "logits" in value and "continuations" in value:
            rows[path.stem] = (path, value)
    return rows


def _continuation(torch, ced, reference):
    left, right = ced.get("continuations"), reference.get("continuations")
    if not isinstance(left, list) or not isinstance(right, list):
        return {"status": "missing"}
    rows = []
    history = _sample(ced.get("sample")) == _sample(reference.get("sample"))
    for index, (a, b) in enumerate(zip(left, right)):
        a_sample, b_sample = _sample(a.get("sample")), _sample(b.get("sample"))
        item = {"step": index, "same_history": history,
                "sample_equal": a_sample == b_sample}
        if history:
            item["logits"] = _metrics(torch, a.get("logits"), b.get("logits"))
            item["kl"] = _kl(torch, a.get("logits"), b.get("logits"))
        else:
            item["logits"] = {"status": "not_compared_history_diverged"}
            item["kl"] = None
        rows.append(item)
        history = history and item["sample_equal"]
    return {"status": "ok", "steps": rows, "all_samples_equal": bool(rows) and history}


def _accuracy_json(torch, root: Path):
    rows = []
    for path in _json_files(root):
        try:
            value = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(value, dict) and ("comparisons" in value or "continuation" in value):
            rows.append((path, value))
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            rows.extend((path, item) for item in value)
    return rows


def _finite_cache_summary(root: Path):
    """Extract rank-local evidence from the authoritative accuracy.json schema.

    ``accuracy.json`` is a list of request/mode entries.  The useful finite and
    cache fields live below ``entry.comparisons[*].result``; looking only at the
    top-level list item silently reports ``null`` and can look like coverage.
    Keep missing/error records explicit and retain pair names for auditability.
    """
    result = []
    for path in _json_files(root):
        try:
            value = json.loads(path.read_text())
        except Exception:
            continue
        entries = value if isinstance(value, list) else [value]
        for entry_index, entry in enumerate(entries):
            if not isinstance(entry, dict) or not isinstance(entry.get("comparisons"), list):
                continue
            for rank, comparison in enumerate(entry["comparisons"]):
                if not isinstance(comparison, dict):
                    continue
                result_value = comparison.get("result")
                if not isinstance(result_value, dict):
                    result.append({
                        "path": str(path), "entry": entry_index, "length": entry.get("length"),
                        "rank": rank, "status": comparison.get("status", "missing_result"),
                    })
                    continue
                pair_rows = {}
                for pair, pair_value in (result_value.get("comparisons") or {}).items():
                    if not isinstance(pair_value, dict):
                        pair_rows[pair] = {"status": "invalid_pair_record"}
                        continue
                    cache = pair_value.get("cache")
                    pair_rows[pair] = {
                        "status": pair_value.get("status", "ok"),
                        "logits": {
                            "status": pair_value.get("logits", {}).get("status")
                            if isinstance(pair_value.get("logits"), dict) else "missing",
                            "finite": pair_value.get("logits", {}).get("finite")
                            if isinstance(pair_value.get("logits"), dict) else None,
                        },
                        "cache": cache if isinstance(cache, dict) else {"status": "missing"},
                    }
                result.append({
                    "path": str(path), "entry": entry_index,
                    "request_id": entry.get("request_id"), "length": entry.get("length"),
                    "mode": entry.get("mode"), "rank": rank,
                    "status": comparison.get("status", "ok"),
                    "finite": result_value.get("finite"),
                    "pairs": pair_rows,
                })
    return result


def _head_compare(torch, root: Path, case: str, left: str, right: str):
    result = {"case": case, "left": left, "right": right, "kinds": {}}
    for kind in ("normalized_source20_input", "compressor_latent", "index_pre"):
        lhs_paths = _pt(root, f"{case}_{left}_{kind}_*.pt")
        rhs_paths = _pt(root, f"{case}_{right}_{kind}_*.pt")
        lhs = [_load_pt(torch, path) for path in lhs_paths]
        rhs = [_load_pt(torch, path) for path in rhs_paths]
        lhs = torch.cat(lhs, 0) if lhs else None
        rhs = torch.cat(rhs, 0) if rhs else None
        result["kinds"][kind] = _metrics(torch, lhs, rhs)
        result["kinds"][kind]["files"] = {"left": len(lhs_paths), "right": len(rhs_paths)}
    return result


def _page_digest(path: Path):
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data)}


def _page_summary(root: Path):
    pages = []
    for path in _pt(root, "*.pages.pt"):
        pages.append(_page_digest(path))
    return {
        "files": pages,
        "numeric_decoded_main": {
            "status": "unavailable",
            "reason": "saved_pt_does_not_preserve_verified_original_storage_stride"
        },
    }


def _page_tensor_comparison(torch, root: Path, case: str,
                            left_mode: str, right_mode: str):
    """Compare saved page payloads, excluding torch serialization metadata."""
    left = {path.name.split(f"_{left_mode}_", 1)[1]: path
            for path in _pt(root, f"{case}_{left_mode}_*.pages.pt")
            if f"_{left_mode}_" in path.name}
    right = {path.name.split(f"_{right_mode}_", 1)[1]: path
             for path in _pt(root, f"{case}_{right_mode}_*.pages.pt")
             if f"_{right_mode}_" in path.name}
    rows = {}
    for name in sorted(set(left) | set(right)):
        if name not in left or name not in right:
            rows[name] = {"status": "missing", "left": name in left,
                          "right": name in right}
            continue
        a, b = _load_pt(torch, left[name]), _load_pt(torch, right[name])
        if not hasattr(a, "shape") or not hasattr(b, "shape") or tuple(a.shape) != tuple(b.shape):
            rows[name] = {"status": "shape_mismatch",
                          "left_shape": list(getattr(a, "shape", ())),
                          "right_shape": list(getattr(b, "shape", ()))}
            continue
        a = a.detach().cpu().contiguous().view(torch.uint8)
        b = b.detach().cpu().contiguous().view(torch.uint8)
        changed = a.ne(b)
        rows[name] = {"status": "ok", "bytes": a.numel(),
                      "byte_equal": bool(torch.equal(a, b)),
                      "different_byte_fraction": float(changed.float().mean())}
    return rows


def _recompute_references(torch, root: Path):
    payloads = _payloads(torch, root)
    references = []
    for key, (path, value) in sorted(payloads.items()):
        for mode in ("native_bounded", "canonical_full", "original_full"):
            suffix = f"_{mode}"
            if not key.endswith(suffix):
                continue
            ced_key = key[:-len(suffix)] + "_ced"
            if ced_key in payloads:
                ced_path, ced = payloads[ced_key]
                references.append({"mode": mode, "reference_file": str(path),
                    "ced_file": str(ced_path),
                    "first_sample_equal": _sample(ced.get("sample")) == _sample(value.get("sample")),
                    "first_logits": _metrics(torch, ced.get("logits"), value.get("logits")),
                    "first_kl": _kl(torch, ced.get("logits"), value.get("logits")),
                    "continuation": _continuation(torch, ced, value)})
    return references


def summarize(run_a: Path, run_b: Path) -> dict[str, Any]:
    torch = _torch()
    a_payloads, b_payloads = _payloads(torch, run_a), _payloads(torch, run_b)
    head = {"run_a": [], "run_b": []}
    pages = {"run_a": [], "run_b": []}
    for root, key in ((run_a, "run_a"), (run_b, "run_b")):
        root_payloads = _payloads(torch, root)
        root_cases = sorted({name[:-len("_native_bounded")] for name in root_payloads
                             if name.endswith("_native_bounded")})
        for case in root_cases:
            for mode in ("native_bounded", "canonical_full", "original_full"):
                head[key].append(_head_compare(torch, root, case, "ced", mode))
                if mode == "native_bounded":
                    pages[key].append({"case": case, "mode": mode,
                        "comparisons": {
                            "transferred_vs_native": _page_tensor_comparison(
                                torch, root, case, "transferred", mode),
                            "ced_vs_native": _page_tensor_comparison(
                                torch, root, case, "ced", mode),
                        }})
    return {
        "run_a": str(run_a), "run_b": str(run_b),
        "payload_counts": {"run_a": len(a_payloads), "run_b": len(b_payloads)},
        "same_history_recomputed": {"run_a": _recompute_references(torch, run_a),
                                     "run_b": _recompute_references(torch, run_b)},
        "accuracy_json_coverage": {"run_a": _finite_cache_summary(run_a),
                                    "run_b": _finite_cache_summary(run_b)},
        "head_diagnostics": head,
        "page_tensor_comparisons": pages,
        "page_payloads": {"run_a": _page_summary(run_a), "run_b": _page_summary(run_b)},
        "limitations": [
            "Continuation logits are compared only while sampled token history remains equal.",
            "Rank-local JSON evidence is reported as found; absence of a rank is not pass.",
            "Packed SM120 numeric dequantization is withheld unless original storage stride is preserved.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.run_a, args.run_b), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
