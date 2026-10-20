"""Profile context growth or benchmark prefill without scheduler chunking."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback

import run_sweep as base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--mode", choices=("profile", "no_chunk"), required=True)
    parser.add_argument("--max-length", type=int, default=131072)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--kv-cache-gib", type=float,
                        help="Explicit per-GPU KV budget for a memory-limited retry")
    args = parser.parse_args()
    work = args.directory.resolve()
    work.mkdir(parents=True, exist_ok=False)
    lengths = tuple(n for n in base.LENGTHS if n <= args.max_length)
    if not lengths or max(lengths) != args.max_length:
        raise ValueError("Maximum length must be one of the baseline lengths")
    base.LENGTHS = lengths
    signal.signal(signal.SIGTERM, base.interrupt)
    signal.signal(signal.SIGINT, base.interrupt)
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", base.PORT)) == 0:
            raise RuntimeError("Benchmark port already occupied")
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if active:
        raise RuntimeError(f"GPUs already active: {active}")
    base.verify_manifest()
    budget = 8192 if args.mode == "profile" else args.max_length + 1
    command = base.build_command(work, budget)
    command += ["--enable-chunked-prefill" if args.mode == "profile"
                else "--no-enable-chunked-prefill"]
    if args.kv_cache_gib is not None:
        command += ["--kv-cache-memory-bytes", str(int(args.kv_cache_gib * 1024**3))]
    if args.mode == "profile":
        strategy = dict(name="pp2_tp4_ep4", pp=2, tp=4, dp=1, ep=True,
                        pp_layers=[20, 20])
        reference = base.matrix.command(strategy, work)
        command += ["--profiler-config", reference[reference.index("--profiler-config") + 1]]
    environment = base.build_environment(work / "rank_timings")
    environment["MAX_JOBS"] = "2"
    workload = json.loads((base.ROOT / "prefill_ttft_sweep/results/"
                           "baseline_8k_20260927a/workload.json").read_text())
    unit = workload["unit_token_ids"]
    tokens = (unit * (args.max_length // len(unit) + 1))[:args.max_length]
    base.save(work / "workload.json", workload)
    base.save(work / "configuration.json", dict(
        mode=args.mode, command=command, strategy="PP2 TP4 EP4", pp_layers=[20, 20],
        batch=1, lengths=lengths, max_length=args.max_length, token_budget=budget,
        enable_chunked_prefill=args.mode == "profile", prefix_cache=False,
        cuda_graph=False, repeats=args.repeats, output_tokens=1,
        kv_cache_gib=args.kv_cache_gib,
        metric="client request start to first streamed token",
        workload_sha256=hashlib.sha256(json.dumps(unit).encode()).hexdigest(),
    ))
    process = None
    records = []

    def status(phase, **extra):
        value = dict(phase=phase, time=time.time(), **extra)
        base.save(work / "status.json", value)
        print(json.dumps(value), flush=True)

    def request(length, label):
        status("request", label=label, length=length)
        record = base.stream_one(tokens[:length], label)
        records.append(record)
        base.save(work / "requests.json", records)
        print(json.dumps(dict(label=label, ttft_ms=record["ttft_ms"])), flush=True)

    try:
        with (work / "server.log").open("w") as log:
            process = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        base.save(work / "server_pid.json", dict(pid=process.pid))
        status("loading", pid=process.pid)
        deadline = time.monotonic() + 1800
        while not base.alive():
            if process.poll() is not None:
                raise RuntimeError(f"Server exited: {process.returncode}")
            if time.monotonic() > deadline:
                raise TimeoutError("Startup exceeded 30 minutes")
            time.sleep(3)
        if args.mode == "profile":
            request(args.max_length, "warmup_profile")
            # Kineto is started only once per server lifetime.
            base.api("/start_profile", {})
            request(args.max_length, "profile_128k")
            base.api("/stop_profile", {})
        else:
            # Complete each length before advancing, preserving smaller cases on OOM.
            for length in lengths:
                request(length, f"warmup_{length}")
                for repeat in range(1, args.repeats + 1):
                    request(length, f"measure_{length}_r{repeat}")
                request(2048, f"flush_after_{length}")
        request(2048, "flush")
        time.sleep(1)
    except BaseException as error:
        base.save(work / "FAILED.json", dict(error=repr(error), traceback=traceback.format_exc()))
        status("failed", error=repr(error))
        raise
    finally:
        if process is not None:
            base.matrix.stop_server(process)
        print("CLEANUP_COMPLETE", flush=True)

    rows = base.read_rank_timings(work / "rank_timings")
    audit = []
    for record in records:
        if not record["label"].startswith(("measure_", "profile_")):
            continue
        rid = record["request_id"]
        for rank in range(8):
            selected = [row for row in rows if row.get("global_rank") == rank
                        and row.get("prefill") and any(
                            item == rid or item.startswith(rid + "-")
                            for item in row.get("request_ids", []))]
            expected = (record["prompt_tokens"] + budget - 1) // budget
            valid = len(selected) == expected and sum(row["n"] for row in selected) == record["prompt_tokens"]
            audit.append(dict(label=record["label"], rank=rank, chunks=len(selected),
                              n=[row["n"] for row in selected], passed=valid))
    base.save(work / "chunk_audit.json", audit)
    if not audit or not all(row["passed"] for row in audit):
        raise RuntimeError("Observed prefill forwards do not match requested execution shape")
    if args.mode == "no_chunk":
        base.save(work / "summary.json", base.summarize(records, rows, budget))
    base.save(work / "COMPLETE.json", dict(passed=True, mode=args.mode, lengths=lengths))
    status("complete")


if __name__ == "__main__":
    main()
