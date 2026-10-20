"""Independent native full/bounded reference adapter for CED reports.

The adapter is intentionally thin: model construction, request scheduling and
the original sparse callable are owned by the caller.  It only installs hooks
around one real ``RequestRunner.step`` invocation.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from accuracy_checks import native_bounded_swa


def _torch():
    import torch

    return torch


def _slot_to_position(metadata: Any, block_size: int, *, active_tokens: int,
                      expected_blocks: Any = None):
    """Build a physical-slot mapper from native block-table metadata."""
    torch = _torch()
    table = getattr(metadata, "block_table", None)
    if table is None:
        raise ValueError("native SWA metadata has no block_table")
    if table.ndim < 2:
        raise ValueError("native SWA block_table must be batched")
    active_blocks = (int(active_tokens) + block_size - 1) // block_size
    table = table[:1, :active_blocks]
    flat = table.reshape(-1).to(dtype=torch.long)
    flat = flat[flat >= 0]
    if flat.numel() == 0:
        raise ValueError("native SWA block_table is empty")
    if expected_blocks is not None:
        expected = set(int(value) for value in expected_blocks)
        actual = set(int(value) for value in flat.cpu().tolist())
        if not actual.issubset(expected):
            raise ValueError(
                f"native block_table IDs disagree with runner ownership: "
                f"unexpected={sorted(actual - expected)}"
            )
    max_block = int(flat.max().item())
    positions = torch.full(
        ((max_block + 1) * block_size,), -1, dtype=torch.long,
        device=table.device,
    )
    # This experiment is batch one. Padding entries beyond active_blocks are
    # excluded above; physical page zero is therefore a valid page ID.
    page_numbers = torch.arange(table.shape[-1], device=table.device)
    for row in table.reshape(-1, table.shape[-1]):
        valid = row >= 0
        pages = row[valid]
        logical = page_numbers[valid]
        offsets = torch.arange(block_size, device=table.device)
        positions[pages[:, None] * block_size + offsets[None, :]] = (
            logical[:, None] * block_size + offsets[None, :]
        )

    def mapper(slots):
        result = torch.full_like(slots, -1, dtype=torch.long)
        valid = (slots >= 0) & (slots < positions.numel())
        result[valid] = positions[slots[valid]]
        return result

    return mapper


@contextmanager
def native_reference(runner: Any, mode: str, *, prompt_length: int,
                     original_sparse: Any = None, window: int = 128
                     ) -> Iterator[dict[str, Any]]:
    """Install one native reference mode around ``runner.step``.

    ``mode`` is ``original_full``, ``canonical_full`` or
    ``native_bounded``.  The caller must retain the original sparse function
    before CED runtime installation and pass it for ``original_full``.
    The context is entered before ``runner.step`` and exits after the forward;
    it does not run a step itself.
    """
    if mode not in {"original_full", "canonical_full", "native_bounded"}:
        raise ValueError(f"unknown native reference mode: {mode}")
    torch = _torch()
    backbone = runner.backbone
    layer20 = backbone.layers[20]
    layer39 = backbone.layers[39]
    state: dict[str, Any] = {
        "mode": mode, "prompt_length": int(prompt_length),
        "window": int(window), "finite": True,
    }
    sparse_module = None
    previous_sparse = None
    bounded_cm = None
    handles = []

    if mode == "original_full":
        if original_sparse is None:
            raise ValueError("original_full requires original_sparse")
        from vllm.models.deepseek_v41.nvidia import flashinfer_sparse
        sparse_module = flashinfer_sparse
        previous_sparse = sparse_module.flashinfer_trtllm_batch_decode_sparse_mla_dsv4
        sparse_module.flashinfer_trtllm_batch_decode_sparse_mla_dsv4 = original_sparse

    def before_layer(_module, args):
        nonlocal bounded_cm
        positions = args[1] if len(args) > 1 else None
        state["layer20_input"] = (
            args[0][-128:].detach().float().cpu().clone()
            if hasattr(args[0], "detach") else None
        )
        if positions is None:
            raise ValueError("layer-20 hook did not receive positions")
        state["positions"] = positions[-128:].detach().cpu().clone()
        if mode != "native_bounded":
            return
        from vllm.forward_context import get_forward_context
        metadata = get_forward_context().attn_metadata
        prefixes = {
            layer.attn.swa_cache_layer.prefix
            for layer in backbone.layers[20:40]
        }
        if not isinstance(metadata, dict):
            raise ValueError("native reference requires dict attention metadata")
        metadata = {key: value for key, value in metadata.items()
                    if key in prefixes}
        objects = list(metadata.values())
        objects = [obj for obj in objects
                   if getattr(obj, "prefill_swa_indices", None) is not None]
        if not objects:
            raise ValueError("native reference found no decoder SWA prefill metadata")
        block_size = int(getattr(objects[0], "block_size"))
        prefix = next(key for key, value in metadata.items() if value is objects[0])
        group = runner.layer_groups.get(prefix)
        expected_blocks = None if group is None else runner.blocks[group]
        mapper = _slot_to_position(
            objects[0], block_size, active_tokens=int(positions.max().item()) + 1,
            expected_blocks=expected_blocks,
        )
        bounded_cm = native_bounded_swa(
            metadata, positions, window_start=max(0, prompt_length - window),
            slot_to_position=mapper,
        )
        state["bounded"] = bounded_cm.__enter__()

    def after_layer(_module, _args, output):
        nonlocal bounded_cm
        value = output[0] if isinstance(output, (tuple, list)) and output else output
        if hasattr(value, "detach"):
            state["layer39_output"] = value[-1:].detach().float().cpu().clone()
        if bounded_cm is not None:
            bounded_cm.__exit__(None, None, None)
            bounded_cm = None

    handles.append(layer20.register_forward_pre_hook(before_layer))
    handles.append(layer39.register_forward_hook(after_layer))
    try:
        yield state
        logits = getattr(runner, "last_logits", None)
        if logits is not None:
            state["logits"] = logits.detach().float().cpu().clone()
            state["finite"] = bool(torch.isfinite(state["logits"]).all())
    finally:
        if bounded_cm is not None:
            bounded_cm.__exit__(None, None, None)
        for handle in reversed(handles):
            handle.remove()
        if sparse_module is not None:
            sparse_module.flashinfer_trtllm_batch_decode_sparse_mla_dsv4 = previous_sparse
