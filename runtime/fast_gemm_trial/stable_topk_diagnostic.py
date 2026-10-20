"""Opt-in deterministic reference selectors for sparse-indexer diagnosis.

This module is deliberately slower than the production kernels and is never
enabled unless ``DSV41_STABLE_TOPK_DIAGNOSTIC=1``.  It provides a CUDA-graph
safe tensor implementation for the small prefill chunks used by the probe and
for the current one-token decode path.  Ties are resolved by logical column
index because stable descending sort preserves the original column order.
"""

from __future__ import annotations

import functools
import importlib.machinery
import os
import sys


def _enabled() -> bool:
    return os.environ.get("DSV41_STABLE_TOPK_DIAGNOSTIC") == "1"


def _stable_rows(logits, starts, ends, k):
    import torch

    width = logits.shape[1]
    used_k = min(int(k), width)
    columns = torch.arange(width, device=logits.device, dtype=torch.int64)[None, :]
    # The production prefill operator returns positions relative to row start.
    # Shift valid entries to the left before sorting so invalid -inf padding
    # cannot win ties against a valid -inf entry in a later request.
    absolute = (columns + starts[:, None]).clamp(max=max(width - 1, 0))
    valid = columns < (ends - starts)[:, None]
    scores = logits.gather(1, absolute)
    masked = scores.masked_fill(~valid, -float("inf"))
    ordered = torch.argsort(masked, dim=1, descending=True, stable=True)[:, :used_k]
    valid_count = (ends - starts).clamp(max=used_k)
    ranks = torch.arange(used_k, device=logits.device, dtype=valid_count.dtype)[None, :]
    return torch.where(ranks < valid_count[:, None], ordered, -1).to(torch.int32)


def _prefill_reference(logits, starts, ends, out, k):
    import torch

    out.fill_(-1)
    used_k = min(int(k), logits.shape[1], out.shape[1])
    if used_k:
        out[:, :used_k].copy_(_stable_rows(logits, starts, ends, used_k))


def _decode_reference(logits, seq_lens, next_n, out, k, max_seq_len):
    import torch

    # The installed decoder selector receives one row per request for the
    # current validation.  Refuse native speculative/multi-token layouts.
    torch._assert(int(next_n) == 1, "stable diagnostic only supports next_n=1")
    torch._assert((seq_lens.ndim == 1) | (seq_lens.ndim == 2),
                  "stable diagnostic requires 1D/2D decode lengths")
    ends = seq_lens if seq_lens.ndim == 1 else seq_lens[:, 0]
    starts = torch.zeros_like(ends)
    torch._assert(ends.ndim == 1, "stable diagnostic requires 1D decode lengths")
    torch._assert(ends.shape[0] == logits.shape[0], "decode rows/lengths mismatch")
    torch._assert_async(torch.all((ends >= 0) & (ends <= logits.shape[1])), "decode length out of bounds")
    _prefill_reference(logits, starts, ends, out, k)


def _wrap_prefill(original):
    @functools.wraps(original)
    def wrapped(logits, starts, ends, out, num_rows, stride0, stride1, k):
        if not _enabled():
            return original(logits, starts, ends, out, num_rows, stride0, stride1, k)
        import torch

        torch._assert(int(num_rows) == logits.shape[0], "prefill rows mismatch")
        torch._assert(starts.ndim == 1, "prefill starts must be 1D")
        torch._assert(ends.ndim == 1, "prefill ends must be 1D")
        torch._assert(starts.shape == ends.shape, "prefill bounds shape mismatch")
        torch._assert_async(torch.all((starts >= 0) & (ends >= starts) & (ends <= logits.shape[1])),
                      "prefill bounds out of range")
        for begin in range(0, int(num_rows), 64):
            stop = min(begin + 64, int(num_rows))
            _prefill_reference(logits[begin:stop], starts[begin:stop], ends[begin:stop],
                               out[begin:stop], int(k))

    return wrapped


def _wrap_decode_factory(original_factory):
    @functools.wraps(original_factory)
    def factory(backend):
        original = original_factory(backend)

        @functools.wraps(original)
        def wrapped(logits, seq_lens, next_n, out, k, max_seq_len):
            if not _enabled():
                return original(logits, seq_lens, next_n, out, k, max_seq_len)
            _decode_reference(logits, seq_lens, next_n, out, int(k), max_seq_len)

        return wrapped

    return factory


def install(module) -> None:
    """Patch both selector entry points on the sparse indexer module."""
    marker = "_dsv41_stable_topk_diagnostic"
    if getattr(module, marker, False):
        return
    module.ops.top_k_per_row_prefill = _wrap_prefill(module.ops.top_k_per_row_prefill)
    if hasattr(module, "SparseIndexerTopk"):
        cls = module.SparseIndexerTopk
        if not getattr(cls, "_dsv41_stable_topk_diagnostic", False):
            original_forward = cls.forward

            @functools.wraps(original_forward)
            def forward(self, logits, seq_lens, next_n, topk_indices, topk_tokens, max_seq_len):
                if not _enabled():
                    return original_forward(self, logits, seq_lens, next_n,
                                            topk_indices, topk_tokens, max_seq_len)
                _decode_reference(logits, seq_lens, next_n, topk_indices,
                                  int(topk_tokens), max_seq_len)

            cls.forward = forward
            setattr(cls, "_dsv41_stable_topk_diagnostic", True)
    elif hasattr(module, "get_indexer_topk"):
        module.get_indexer_topk = _wrap_decode_factory(module.get_indexer_topk)
    setattr(module, marker, True)
    print('DSV41_STABLE_TOPK_DIAGNOSTIC_INSTALLED', flush=True)


def self_test() -> None:
    """Small CPU-only semantic checks for ties, masks and -1 padding."""
    import torch

    logits = torch.tensor([[2.0, 5.0, 5.0, -3.0], [9.0, 1.0, 1.0, 1.0]])
    starts = torch.tensor([0, 1])
    ends = torch.tensor([3, 4])
    columns = torch.arange(4)[None, :]
    valid = (columns >= starts[:, None]) & (columns < ends[:, None])
    ordered = torch.argsort(logits.masked_fill(~valid, -float("inf")), dim=1,
                            descending=True, stable=True)
    got = torch.where(torch.arange(2)[None, :] < (ends - starts).clamp(max=2)[:, None],
                      ordered[:, :2], -1)
    assert torch.equal(got, torch.tensor([[1, 2], [1, 2]]))
    assert torch.equal(_stable_rows(logits, starts, ends, 2),
                       torch.tensor([[1, 2], [0, 1]], dtype=torch.int32))


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


if __name__ == "__main__":
    self_test()
