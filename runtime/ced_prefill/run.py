"""Coordinate the four-GPU CED prefill/decode experiment.

The parent process deliberately does not import torch or vLLM.  Workers are
independent TP4 processes and communicate through multiprocessing pipes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from multiprocessing.connection import wait as wait_connections
import os
from pathlib import Path
import signal
import sys
import time
import traceback


DEFAULT_SIZES = (2048, 4096, 8192, 16384, 32768, 65536, 131072)
DEFAULT_MODEL = "/mnt/fuse/models/DeepSeek-V4.1-Flash-local-engram"


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def worker_entry(connection, settings, role, rank, gpu_ids, port):
    # Import worker only after spawn: importing it in the coordinator would
    # pull torch into the parent and can initialize CUDA before assignment.
    worker_dir = str(Path(__file__).resolve().parent)
    if worker_dir not in sys.path:
        sys.path.insert(0, worker_dir)
    from worker import worker_main

    worker_main(connection, settings, role, rank, gpu_ids, port)


class Group:
    def __init__(self, ctx, settings, role, gpus, port, timeout):
        self.role = role
        self.timeout = timeout
        self.children = []
        self.connections = []
        gpu_ids = tuple(gpus)
        visible = ",".join(map(str, gpu_ids))
        for rank, gpu in enumerate(gpu_ids):
            parent, child = ctx.Pipe()
            process = ctx.Process(
                target=worker_entry,
                args=(child, settings, role, rank, visible, port),
                name=f"ced-{role}-rank{rank}",
            )
            process.daemon = False
            process.start()
            child.close()
            self.children.append(process)
            self.connections.append(parent)

    def command(self, op, args=None):
        for connection in self.connections:
            connection.send({"op": op, "args": args or {}})
        deadline = time.monotonic() + self.timeout
        pending = set(self.connections)
        replies = [None] * len(self.connections)
        while pending:
            if any(not process.is_alive() and process.exitcode not in (None, 0)
                   for process in self.children):
                raise RuntimeError(f"{self.role} worker exited during {op}: "
                                   f"{[p.exitcode for p in self.children]}")
            remaining = max(0.0, deadline - time.monotonic())
            if not remaining:
                raise TimeoutError(f"{self.role} timed out during {op}")
            ready = wait_connections(pending, min(remaining, 1.0))
            for connection in ready:
                rank = self.connections.index(connection)
                reply = connection.recv()
                pending.remove(connection)
                replies[rank] = reply
                if reply.get("status") != "ok" and reply.get("status") not in {"ready", "stopped"}:
                    raise RuntimeError(f"{self.role} rank {rank} {op} failed: {reply.get('error')}")
        return replies

    def command_ranks(self, op, args_by_rank):
        for rank, connection in enumerate(self.connections):
            connection.send({"op": op, "args": args_by_rank[rank]})
        deadline = time.monotonic() + self.timeout
        pending = set(self.connections)
        replies = [None] * len(self.connections)
        while pending:
            if any(not process.is_alive() and process.exitcode not in (None, 0)
                   for process in self.children):
                raise RuntimeError(f"{self.role} worker exited during {op}: "
                                   f"{[p.exitcode for p in self.children]}")
            ready = wait_connections(pending, min(max(0.0, deadline - time.monotonic()), 1.0))
            if not ready and time.monotonic() >= deadline:
                raise TimeoutError(f"{self.role} timed out during {op}")
            for connection in ready:
                rank = self.connections.index(connection)
                reply = connection.recv()
                pending.remove(connection)
                if reply.get("status") != "ok":
                    raise RuntimeError(f"{self.role} rank {rank} {op} failed: {reply.get('error')}")
                replies[rank] = reply
        return replies

    def ready(self):
        deadline = time.monotonic() + self.timeout
        pending = set(self.connections)
        replies = [None] * len(self.connections)
        while pending:
            if any(not process.is_alive() and process.exitcode not in (None, 0)
                   for process in self.children):
                raise RuntimeError(f"{self.role} worker exited during startup: "
                                   f"{[p.exitcode for p in self.children]}")
            ready = wait_connections(pending, min(max(0.0, deadline - time.monotonic()), 1.0))
            if not ready and time.monotonic() >= deadline:
                raise TimeoutError(f"{self.role} did not become ready")
            for connection in ready:
                rank = self.connections.index(connection)
                reply = connection.recv()
                pending.remove(connection)
                replies[rank] = reply
                if reply.get("status") != "ready":
                    raise RuntimeError(f"{self.role} rank {rank} failed to start: {reply.get('error')}")
        return replies

    def stop(self):
        for connection in self.connections:
            try:
                connection.send({"op": "stop"})
            except (BrokenPipeError, EOFError, OSError):
                pass
        for rank, connection in enumerate(self.connections):
            try:
                if connection.poll(10):
                    connection.recv()
            except (EOFError, OSError):
                pass

    def terminate(self):
        for process in self.children:
            if process.is_alive():
                process.terminate()
        for process in self.children:
            process.join(10)
            if process.is_alive():
                process.kill()
                process.join(5)
        for connection in self.connections:
            connection.close()

    def pids(self):
        return [process.pid for process in self.children]


def load_workload(path: Path) -> list[int]:
    payload = json.loads(path.read_text())
    tokens = payload.get("unit_token_ids")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError(f"workload has no unit_token_ids: {path}")
    return [int(token) for token in tokens]


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    hashes = {}
    for path in sorted(root.rglob("*.py")):
        if "runtime_snapshot" in path.parts:
            continue
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def phase_result(replies, key):
    return [reply.get("result", {}) for reply in replies]


def remove_handoff_files(directory: Path) -> None:
    for rank in range(4):
        try:
            (directory / f"rank{rank}.pt").unlink()
        except FileNotFoundError:
            pass


def run_case(prefill, decode, tokens, length, chunk, handoff_dir, request_id, mode):
    prompt = tokens * (length // len(tokens) + 1)
    prompt = prompt[:length]
    handoff_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    prefill_replies = prefill.command(
        "prefill", {"request_id": request_id, "tokens": prompt, "chunk": chunk}
    )
    prefill_done = time.perf_counter()
    export_args = [{"path": str(handoff_dir / f"rank{rank}.pt")} for rank in range(4)]
    exports = [reply.get("result", {}) for reply in
               prefill.command_ranks("export", export_args)]
    export_done = time.perf_counter()
    if mode == "smoke-p":
        return dict(
            request_id=request_id, length=length, mode=mode,
            prefill_ms=(prefill_done - started) * 1000,
            export_ms=(export_done - prefill_done) * 1000,
            workers=phase_result(prefill_replies, "result"), exports=exports,
        )
    restore_started = time.perf_counter()
    restore_args = [{"path": str(handoff_dir / f"rank{rank}.pt"),
                     "request_id": request_id, "tokens": prompt}
                    for rank in range(4)]
    restores = [reply.get("result", {}) for reply in
                decode.command_ranks("restore", restore_args)]
    restore_done = time.perf_counter()
    replay_replies = decode.command("replay")
    replay_done = time.perf_counter()
    rank_results = [reply.get("result", {}).get("result", {})
                    for reply in replay_replies]
    rank_token_ids = [result.get("token_ids", []) for result in rank_results]
    if any(not isinstance(token_ids, list) or len(token_ids) != 1
           or not isinstance(token_ids[0], list) or len(token_ids[0]) != 1
           or not isinstance(token_ids[0][0], int) or token_ids[0][0] < 0
           for token_ids in rank_token_ids):
        raise RuntimeError(f"{request_id}: at least one rank returned no sampled token")
    if any(token_ids != rank_token_ids[0] for token_ids in rank_token_ids[1:]):
        raise RuntimeError(f"{request_id}: sampled token IDs differ across ranks")
    result = rank_results[0]
    token_ids = rank_token_ids[0]
    return dict(
        request_id=request_id, length=length, mode=mode,
        prefill_ms=(prefill_done - started) * 1000,
        export_ms=(export_done - prefill_done) * 1000,
        restore_ms=(restore_done - restore_started) * 1000,
        replay_and_sample_ms=(replay_done - restore_done) * 1000,
        ttft_ms=(replay_done - started) * 1000,
        token_ids=token_ids, token_ids_by_rank=rank_token_ids,
        prefill_workers=phase_result(prefill_replies, "result"),
        exports=exports, restores=restores,
        decode_workers=phase_result(replay_replies, "result"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke-p", "pd"), default="pd")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workload", type=Path,
                        default=Path("/home/admin/dsv41-vllm-20260917/prefill_ttft_sweep/results/baseline_8k_20260927a/workload.json"))
    parser.add_argument("--model", default=os.environ.get("CED_MODEL", DEFAULT_MODEL))
    parser.add_argument("--sizes", default="",
                        help="comma-separated ascending prompt lengths")
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--decode-offload-gb", type=float, default=12.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--p-port", type=int, default=29650)
    parser.add_argument("--d-port", type=int, default=29660)
    parser.add_argument("--handoff-dir", type=Path, default=Path("/dev/shm/ced_prefill"))
    args = parser.parse_args()
    deployment_dir = str(Path(__file__).resolve().parents[1])
    if deployment_dir not in sys.path:
        sys.path.insert(0, deployment_dir)
    import parallel_matrix

    # Keep the worker runtime identical to the established offline benchmark.
    os.environ.update(parallel_matrix.environment())
    if args.chunk <= 0 or args.repeats <= 0 or args.decode_offload_gb < 0:
        raise ValueError("chunk, repeats, and offload must be positive/non-negative")
    sizes = DEFAULT_SIZES if not args.sizes else tuple(int(x) for x in args.sizes.split(",") if x)
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("--sizes must contain positive lengths")
    if tuple(sorted(sizes)) != sizes:
        raise ValueError("--sizes must be ascending")
    if args.mode == "smoke-p" and args.decode_offload_gb:
        # Keep the manifest explicit even though P does not consume this value.
        pass
    tokens = load_workload(args.workload)
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "configuration.json", dict(
        mode=args.mode, sizes=sizes, chunk=args.chunk, repeats=args.repeats,
        max_len=max(sizes), model=args.model, workload=str(args.workload),
        workload_sha256=hashlib.sha256(args.workload.read_bytes()).hexdigest(),
        gpu_groups={"prefill": [0, 1, 2, 3], "decode": [4, 5, 6, 7]},
        decode_offload_gb=args.decode_offload_gb,
        handoff_dir=str(args.handoff_dir),
        source_py_sha256=source_hashes(),
        metric="coordinator request start through decoder replay/sample",
    ))
    ctx = mp.get_context("spawn")
    settings = dict(model=args.model, output=str(args.output), max_len=max(sizes),
                    chunk=args.chunk, decode_offload_gb=args.decode_offload_gb)
    settings["handoff_dir"] = str(args.handoff_dir)
    prefill = decode = None
    records = []
    try:
        save(args.output / "status.json", {"phase": "starting", "time": time.time()})
        prefill = Group(ctx, settings, "prefill", range(4), args.p_port, args.timeout)
        save(args.output / "pids.json", {"prefill": prefill.pids()})
        p_manifest = prefill.ready()
        pids = {"prefill": prefill.pids()}
        d_manifest = None
        if args.mode == "pd":
            decode = Group(ctx, settings, "decode", range(4, 8), args.d_port, args.timeout)
            pids["decode"] = decode.pids()
            save(args.output / "pids.json", pids)
            d_manifest = decode.ready()
        save(args.output / "manifests.json", {"prefill": p_manifest, "decode": d_manifest})
        save(args.output / "status.json", {"phase": "ready", "time": time.time()})
        # Warm the smallest shape first. Larger shapes get their own untimed
        # warmup immediately before their measured repeats.
        initial_size = min(sizes)
        warmup = run_case(prefill, decode, tokens, initial_size, args.chunk,
                          args.handoff_dir, "warmup", args.mode)
        save(args.output / "warmup.json", warmup)
        if args.mode == "pd":
            warmup["reference"] = decode.command("reference", {"chunk": args.chunk})
        save(args.output / "warmup.json", warmup)
        remove_handoff_files(args.handoff_dir)
        save(args.output / "status.json", {"phase": "measuring", "time": time.time()})
        shape_warmups = []
        for length in sizes:
            if length != initial_size:
                shape_warmup = run_case(prefill, decode, tokens, length, args.chunk,
                                        args.handoff_dir, f"warmup_{length}", args.mode)
                shape_warmups.append(shape_warmup)
                save(args.output / "shape_warmups.json", shape_warmups)
                remove_handoff_files(args.handoff_dir)
            for repeat in range(1, args.repeats + 1):
                record = run_case(prefill, decode, tokens, length, args.chunk,
                                  args.handoff_dir, f"measure_{length}_r{repeat}",
                                  args.mode)
                records.append(record)
                save(args.output / "records.json", records)
                remove_handoff_files(args.handoff_dir)
        save(args.output / "COMPLETE.json", {"passed": True, "records": len(records)})
        save(args.output / "status.json", {"phase": "complete", "time": time.time(),
                                            "records": len(records)})
    except BaseException as error:
        save(args.output / "FAILED.json", {"error": repr(error),
                                           "traceback": traceback.format_exc(),
                                           "records": records})
        save(args.output / "status.json", {"phase": "failed", "time": time.time(),
                                            "error": repr(error)})
        raise
    finally:
        for group in (decode, prefill):
            if group is not None:
                try:
                    group.stop()
                except Exception:
                    pass
                group.terminate()
        remove_handoff_files(args.handoff_dir)


if __name__ == "__main__":
    main()
