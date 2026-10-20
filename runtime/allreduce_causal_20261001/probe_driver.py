"""Launcher helpers for the standalone CUDA IPC dependency probe.

The caller owns process-group setup and should invoke ``run`` once per case.
Each run clears the local flag slab and initializes the local payload slab,
then synchronizes the group before launching the diagnostic kernel.
"""

from __future__ import annotations

import os
import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load

from flashinfer.comm.cuda_ipc import create_shared_buffer, free_shared_buffer


_ROOT = Path(__file__).resolve().parent
_SOURCE = _ROOT / "probe_kernel.cu"


def load_probe(group: Optional[Any] = None):
    """Build/load the extension, with rank 0 doing the first build."""
    os.environ.setdefault("MAX_JOBS", "2")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    initialized = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank(group=group) if initialized else 0
    name = "allreduce_causal_probe_20261001"
    kwargs = dict(
        name=name,
        sources=[str(_SOURCE)],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    if rank == 0:
        module = load(**kwargs)
        if initialized:
            dist.barrier(group=group)
        return module
    if initialized:
        dist.barrier(group=group)
    return load(**kwargs)


class ProbeBuffers:
    """Two caller-owned IPC slabs and their GPU pointer tables."""

    def __init__(self, group: Any, world: int, payload_words: int, rounds: int, repeats: int, protocol: int):
        if protocol not in (0, 1):
            raise ValueError("protocol must be 0 (BF16-sized) or 1 (int32)")
        self.group = group
        self.world = int(world)
        self.rank = dist.get_rank(group=group)
        self.payload_words = int(payload_words)
        self.rounds = int(rounds)
        self.repeats = int(repeats)
        self.protocol = int(protocol)
        self.word_bytes = 2 if protocol == 0 else 4
        self.slots = self.world * self.rounds * self.repeats
        self.device = torch.device("cuda", torch.cuda.current_device())
        self._data = create_shared_buffer(
            self.slots * self.payload_words * self.word_bytes, group=group
        )
        self._flags = create_shared_buffer(self.slots * 4, group=group)
        self.data_ptrs = torch.tensor(self._data, dtype=torch.int64, device=self.device)
        self.flag_ptrs = torch.tensor(self._flags, dtype=torch.int64, device=self.device)

    def prepare(self, module, threads: int = 256) -> None:
        module.probe_clear_flags(self.flag_ptrs, self.rank, self.world, self.slots, threads)
        module.probe_initialize(
            self.data_ptrs,
            self.rank,
            self.world,
            self.payload_words,
            self.rounds,
            self.repeats,
            self.protocol,
            threads,
        )
        torch.cuda.current_stream().synchronize()
        dist.barrier(group=self.group)

    def run(self, module, mode: int, threads: int = 256):
        if mode not in (0, 1, 2):
            raise ValueError("mode must be 0 (independent), 1 (dependent), or 2 (batched publication)")
        self.prepare(module, threads)
        phases = torch.empty(self.repeats * (3 + 2 * self.rounds), dtype=torch.int64, device=self.device)
        output_dtype = torch.int16 if self.protocol == 0 else torch.int32
        final_output = torch.empty(self.payload_words, dtype=output_dtype, device=self.device)
        module.probe_launch(
            self.data_ptrs,
            self.flag_ptrs,
            self.rank,
            self.world,
            self.payload_words,
            self.rounds,
            self.repeats,
            mode,
            self.protocol,
            threads,
            phases,
            final_output,
        )
        torch.cuda.current_stream().synchronize()
        return phases, final_output

    def close(self) -> None:
        free_shared_buffer(self._data, group=self.group)
        free_shared_buffer(self._flags, group=self.group)


def _percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] + fraction * (ordered[high] - ordered[low])


def _expected_output(rank, world, rounds, words, protocol, mode):
    dependent = mode == 1
    origin = (rank - rounds) % world if dependent else (rank - 1) % world
    seed_round = 0 if dependent else rounds - 1
    values = [origin * 1000003 + seed_round * 1009 + i for i in range(words)]
    if protocol == 0:
        values = [((value + 32768) % 65536) - 32768 for value in values]
        return torch.tensor(values, dtype=torch.int16)
    return torch.tensor(values, dtype=torch.int32)


def _run_cli(args):
    if not dist.is_available():
        raise RuntimeError("torch.distributed is required")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", init_method="env://")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 4:
        raise RuntimeError(f"probe CLI requires exactly 4 ranks, got {world}")
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    group = dist.group.WORLD
    module = load_probe(group)
    output_path = Path(args.output)
    if rank == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("")

    for protocol in args.protocols:
        word_bytes = 2 if protocol == 0 else 4
        for payload_bytes in args.payload_bytes:
            if payload_bytes % word_bytes:
                continue
            words = payload_bytes // word_bytes
            for rounds in args.rounds:
                buffers = ProbeBuffers(group, world, words, rounds, args.repeats, protocol)
                try:
                    for mode in args.modes:
                        for threads in args.threads:
                            for launch_index in range(args.launch_repeats):
                                phases, final_output = buffers.run(module, mode, threads)
                                expected = _expected_output(rank, world, rounds, words, protocol, mode)
                                actual = final_output.detach().cpu()
                                if not torch.equal(actual, expected):
                                    raise AssertionError(
                                        f"rank {rank} output mismatch protocol={protocol} bytes={payload_bytes} "
                                        f"rounds={rounds} mode={mode} threads={threads}"
                                    )
                                phase = phases.detach().cpu().view(args.repeats, 3 + 2 * rounds)
                                durations = (phase[:, -1] - phase[:, 0]).tolist()
                                kept = durations[8:-1]
                                waits = None if mode == 2 else []
                                receive_timestamps = []
                                for row in phase.tolist():
                                    receive_timestamps.append([row[2 + 2 * r] for r in range(rounds)])
                                    if waits is not None:
                                        waits.append([row[2 + 2 * r] - row[1 + 2 * r] for r in range(rounds)])
                                local_record = {
                                    "rank": rank,
                                    "durations": durations,
                                    "kept_durations": kept,
                                    "round_waits": waits,
                                    "round_receive_timestamps": receive_timestamps,
                                }
                                gathered = [None] * world
                                dist.all_gather_object(gathered, local_record, group=group)
                                if rank == 0:
                                    rank_medians = [statistics.median(item["kept_durations"]) for item in gathered]
                                    record = {
                                        "protocol": "bf16_sized" if protocol == 0 else "int32",
                                        "payload_bytes": payload_bytes,
                                        "rounds": rounds,
                                        "mode": {0: "independent", 1: "dependent", 2: "batched_publication"}[mode],
                                        "threads": threads,
                                        "launch": launch_index,
                                        "repeats": args.repeats,
                                        "discard": {"first": 8, "last": 1},
                                        "rank_medians": rank_medians,
                                        "median_of_rank_medians": statistics.median(rank_medians),
                                        "p50": _percentile([x for item in gathered for x in item["kept_durations"]], 0.50),
                                        "p95": _percentile([x for item in gathered for x in item["kept_durations"]], 0.95),
                                        "samples": gathered,
                                    }
                                    with output_path.open("a") as stream:
                                        stream.write(json.dumps(record) + "\n")
                finally:
                    buffers.close()
    dist.barrier(group=group)


def main():
    parser = argparse.ArgumentParser(description="Run the four-rank CUDA IPC dependency probe")
    parser.add_argument("--output", default=str(_ROOT / "probe_results.jsonl"))
    parser.add_argument("--repeats", type=int, default=64)
    parser.add_argument("--launch-repeats", type=int, default=5)
    parser.add_argument("--rounds", nargs="+", type=int, default=[1, 2, 3, 6])
    parser.add_argument("--modes", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--protocols", nargs="+", type=int, default=[1])
    parser.add_argument("--threads", nargs="+", type=int, default=[64, 256])
    parser.add_argument("--payload-bytes", nargs="+", type=int, default=[256, 2560, 10240])
    _run_cli(parser.parse_args())


if __name__ == "__main__":
    main()
