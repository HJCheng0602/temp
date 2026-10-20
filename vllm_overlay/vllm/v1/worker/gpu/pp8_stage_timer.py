"""Opt-in asynchronous CUDA-event timing around one PP stage forward."""

import json
import os
from pathlib import Path
import time
from collections import deque

_pending = deque()
_stream = None
_sequence = 0
_rank_info = {}


def start(input_batch, descriptor):
    global _stream, _sequence, _rank_info
    directory = os.environ.get("DSV41_PP_TIMING_DIR")
    if not directory:
        return None
    import torch
    from vllm.distributed import get_pp_group, get_tp_group

    if _stream is None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        rank = get_pp_group().rank_in_group
        _rank_info = dict(pp_rank=rank, tp_rank=get_tp_group().rank_in_group,
                          global_rank=torch.distributed.get_rank())
        name = (f"rank_{_rank_info['global_rank']}_pp{rank}_tp{_rank_info['tp_rank']}.jsonl"
                if os.environ.get("DSV41_PP_TIMING_PER_RANK") == "1"
                else f"stage_{rank}.jsonl")
        _stream = (path / name).open("a", buffering=1)
    while _pending and _pending[0][1].query():
        begin, end, row = _pending.popleft()
        row["gpu_ms"] = begin.elapsed_time(end)
        _stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    _sequence += 1
    row = dict(sequence=_sequence, n=input_batch.num_tokens,
               requests=input_batch.num_reqs, prefill=bool(input_batch.has_prefill),
               graph_mode=descriptor.cg_mode.name, request_ids=input_batch.req_ids,
               padded_tokens=descriptor.num_tokens, **_rank_info,
               cpu_start=time.monotonic())
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    return begin, end, row


def stop(ticket, output):
    if ticket is None:
        return
    begin, end, row = ticket
    end.record()
    row["cpu_end"] = time.monotonic()
    tensors = getattr(output, "tensors", {})
    row["source_outputs"] = {
        name: dict(shape=list(value.shape), address=value.data_ptr())
        for name, value in tensors.items() if name.startswith("pp_source_input_")
    }
    _pending.append(ticket)
