"""Probe prefill indexer top-k repeatability without changing model outputs.

The probe is opt-in.  ``install_import_hook`` patches the already installed
``sparse_attn_indexer`` module when imported, while ``install`` is useful for
tests that already hold the module object.  Only the first eligible call for
each target wave label is repeated.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import sys


_PROBED_LABELS: set[str] = set()
_PATCHED = "_dsv41_prefill_topk_probe"


def _wave_label() -> str | None:
    control = os.environ.get("DSV41_WAVE_CONTROL")
    if not control:
        return None
    try:
        value = json.loads(Path(control).read_text())
    except (OSError, ValueError):
        return None
    log = value.get("log")
    if not isinstance(log, str):
        return None
    name = Path(log).name
    if not name.startswith("b1_prompt2048_"):
        return None
    for suffix in ("_free.jsonl", "_repeat.jsonl", "_free_again.jsonl"):
        if name.endswith(suffix):
            return name[: -len(".jsonl")]
    return None


def _probe_root() -> Path:
    capture_dir = os.environ.get("DSV41_PRECISION_DIR")
    if not capture_dir:
        raise RuntimeError("DSV41_PRECISION_DIR is required for the prefill probe")
    root = Path(capture_dir).parent / "prefill_topk_probe"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _valid_bounds(starts, ends, logits) -> bool:
    import torch

    if not isinstance(starts, torch.Tensor) or not isinstance(ends, torch.Tensor):
        return False
    if starts.ndim != 1 or ends.ndim != 1 or starts.shape != ends.shape:
        return False
    if starts.numel() == 0 or starts.shape[0] != logits.shape[0]:
        return False
    if starts.device != logits.device or ends.device != logits.device:
        return False
    if starts.dtype not in (torch.int32, torch.int64) or ends.dtype != starts.dtype:
        return False
    if bool((starts < 0).any().item()) or bool((ends < starts).any().item()):
        return False
    if bool((ends > logits.shape[1]).any().item()):
        return False
    if starts.numel() > 1 and bool((starts[1:] < starts[:-1]).any().item()):
        return False
    return True


def _sha256_tensor(value) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _write_probe(label, logits, starts, ends, first, repeats, k):
    import torch

    root = _probe_root()
    safe_label = "".join(c if c.isalnum() or c in "_-" else "_" for c in label)
    row = logits.shape[0] - 1
    # The saved row is deliberately small and is only used for offline score
    # inspection.  Full output tensors remain on GPU during the probe.
    row_logits = logits[row].detach().float().cpu()
    row_indices = [value[row].detach().cpu() for value in repeats]
    row_indices.insert(0, first[row].detach().cpu())
    torch.save(
        dict(
            logits=row_logits,
            starts=starts.detach().cpu(),
            ends=ends.detach().cpu(),
            indices=torch.stack(row_indices),
        ),
        root / f"{safe_label}.pt",
    )

    valid = ends - starts
    # The top-k boundary score is computed only for rows with >= k valid
    # entries.  This is diagnostic data; it never feeds the model.
    eligible = valid >= k
    boundary = torch.full((logits.shape[0],), float("nan"), device=logits.device)
    rows = torch.nonzero(eligible, as_tuple=False).flatten()
    if rows.numel():
        # The operator output is the valid local row top-k, so its kth value
        # gives the boundary without reading logits beyond each row's end.
        boundary[rows] = logits[rows].gather(1, first[rows, :k]).min(dim=1).values
    tie_counts = torch.zeros_like(valid, dtype=torch.int32)
    if rows.numel():
        # Real workloads use start==0; this count remains correct for every
        # valid row and is kept on device until the small summary is written.
        offsets = torch.arange(logits.shape[1], device=logits.device)[None, :]
        valid_mask = (offsets >= starts[rows, None]) & (offsets < ends[rows, None])
        score = boundary[rows, None]
        tie_counts[rows] = ((logits[rows] == score) & valid_mask).sum(dim=1).to(torch.int32)

    raw_equal = []
    set_equal = []
    differing_rows = []
    first_cpu = first.detach()
    for current in repeats:
        raw_equal.append(bool(torch.equal(first_cpu, current)))
        lhs = torch.sort(first_cpu, dim=1).values
        rhs = torch.sort(current, dim=1).values
        equal_rows = torch.all(lhs == rhs, dim=1)
        set_equal.append(bool(equal_rows.all().item()))
        differing_rows.append(torch.nonzero(~equal_rows, as_tuple=False).flatten()[:16].cpu().tolist())

    # Independent stable sort is intentionally restricted to rows whose local
    # interval starts at zero.  No assumption about a nonzero local offset is
    # made by this probe.
    stable_rows = torch.nonzero(
        (starts == 0) & (ends == logits.shape[1]) & eligible, as_tuple=False
    ).flatten()
    stable_equal = []
    if stable_rows.numel():
        stable = torch.argsort(logits[stable_rows], dim=1, descending=True, stable=True)[:, :k]
        stable_equal = torch.all(stable == first_cpu[stable_rows], dim=1).cpu().tolist()

    summary = dict(
        label=label,
        rows=int(logits.shape[0]),
        width=int(logits.shape[1]),
        topk=int(k),
        starts_zero_rows=int((starts == 0).sum().item()),
        eligible_rows=int(eligible.sum().item()),
        tie_count_min=int(tie_counts[eligible].min().item()) if bool(eligible.any().item()) else 0,
        tie_count_max=int(tie_counts[eligible].max().item()) if bool(eligible.any().item()) else 0,
        tie_count_nontrivial=int((tie_counts[eligible] > 1).sum().item()) if bool(eligible.any().item()) else 0,
        frozen_metadata_sha256={
            "starts": _sha256_tensor(starts),
            "ends": _sha256_tensor(ends),
        },
        cross_phase_last_row_logits_sha256=_sha256_tensor(row_logits),
        last_row_raw_sha256=[_sha256_tensor(value[row]) for value in [first, *repeats]],
        last_row_set_sha256=[
            _sha256_tensor(torch.sort(value[row]).values)
            for value in [first, *repeats]
        ],
        raw_equal=raw_equal,
        set_equal=set_equal,
        differing_rows=differing_rows,
        stable_sort_equal_on_start_zero=stable_equal,
    )
    (root / f"{safe_label}.json").write_text(json.dumps(summary, indent=2) + "\n")


def _wrap_topk(original):
    @functools.wraps(original)
    def wrapped(logits, cu_seqlen_ks, cu_seqlen_ke, topk_indices, num_rows,
                logits_stride_0, logits_stride_1, topk_tokens):
        import torch

        label = _wave_label()
        try:
            import torch.distributed as dist
            rank_zero = dist.is_initialized() and dist.get_rank() == 0
        except (ImportError, RuntimeError):
            rank_zero = False
        if (
            label is None
            or label in _PROBED_LABELS
            or not rank_zero
            or torch.cuda.is_current_stream_capturing()
            or not isinstance(logits, torch.Tensor)
            or not isinstance(topk_indices, torch.Tensor)
            or topk_indices.ndim != 2
            or topk_indices.shape[0] != logits.shape[0]
            or topk_indices.shape[1] < int(topk_tokens)
            or not _valid_bounds(cu_seqlen_ks, cu_seqlen_ke, logits)
            or int(num_rows) != logits.shape[0]
            or int(topk_tokens) <= 0
            or int(topk_tokens) > logits.shape[1]
        ):
            return original(logits, cu_seqlen_ks, cu_seqlen_ke, topk_indices,
                            num_rows, logits_stride_0, logits_stride_1, topk_tokens)

        if bool((cu_seqlen_ks != 0).any().item()):
            return original(logits, cu_seqlen_ks, cu_seqlen_ke, topk_indices,
                            num_rows, logits_stride_0, logits_stride_1, topk_tokens)

        _PROBED_LABELS.add(label)
        original(logits, cu_seqlen_ks, cu_seqlen_ke, topk_indices, num_rows,
                 logits_stride_0, logits_stride_1, topk_tokens)
        first = topk_indices.clone()
        repeats = []
        for _ in range(4):
            original(logits, cu_seqlen_ks, cu_seqlen_ke, topk_indices, num_rows,
                     logits_stride_0, logits_stride_1, topk_tokens)
            repeats.append(topk_indices.clone())
        topk_indices.copy_(first)
        _write_probe(label, logits, cu_seqlen_ks, cu_seqlen_ke, first, repeats,
                     int(topk_tokens))

    return wrapped


def install(module) -> None:
    """Patch the sparse indexer module's prefill top-k op once."""
    if getattr(module, _PATCHED, False):
        return
    module.ops.top_k_per_row_prefill = _wrap_topk(module.ops.top_k_per_row_prefill)
    setattr(module, _PATCHED, True)


def install_import_hook() -> None:
    target = "vllm.model_executor.layers.sparse_attn_indexer"

    class Finder:
        def find_spec(self, fullname, path=None, target_module=None):
            if fullname != target:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return None
            loader = spec.loader

            class WrappedLoader:
                def create_module(self, module_spec):
                    return loader.create_module(module_spec) if hasattr(loader, "create_module") else None

                def exec_module(self, module):
                    loader.exec_module(module)
                    install(module)

            spec.loader = WrappedLoader()
            return spec

    sys.meta_path.insert(0, Finder())
    if target in sys.modules:
        install(sys.modules[target])
