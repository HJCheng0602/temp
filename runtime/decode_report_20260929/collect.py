"""Collect one fast-GEMM decode report with a single PP profiler session."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import statistics
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import parallel_matrix as matrix
from largebatch_pp2 import preemption_snapshot
from run_pp2_graph_comparison import interrupt
from run_pp2_wholebatch import read_waves
from run_pp8_batch_sweep import percentile, summarize_requests

ROOT = matrix.ROOT
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
WARMUP_TOKENS = 16
MEASURE_TOKENS = 96
PROFILE_TOKENS = 16


def _manifest_check() -> None:
    runtime = Path(importlib.metadata.distribution("vllm").locate_file("vllm"))
    manifest = json.loads((ROOT / "pp_deferred_fix/manifest.json").read_text())
    for relative, hashes in manifest["files"].items():
        path = runtime / relative
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != hashes["after"]:
            raise RuntimeError(f"Runtime differs from verified fix: {relative}")


def _command(work: Path, mode: str) -> list[str]:
    strategy = dict(name="pp2_tp4_ep4", pp=2, tp=4, dp=1, ep=True, pp_layers=[20, 20])
    command = matrix.command(strategy, work)
    command.remove("--enforce-eager")
    for key, value in {"--port": 8024, "--max-model-len": 4096,
                       "--max-num-seqs": 256, "--max-num-batched-tokens": 8192,
                       "--gpu-memory-utilization": .85}.items():
        command[command.index(key) + 1] = str(value)
    command += ["--engram-config", '{"cpu_offload":true}', "--async-scheduling",
                "--scheduler-cls", "pp2_fixed_batch_scheduler.FixedBatchScheduler",
                "--compilation-config", json.dumps(
                    dict(mode=0, cudagraph_mode="FULL_DECODE_ONLY",
                         cudagraph_capture_sizes=list(2**i for i in range(9))))]
    if mode == "all":
        command += ["--kernel-config", '{"moe_backend":"marlin"}']
    return command


def _environment(work: Path, mode: str) -> dict[str, str]:
    strategy = dict(name="pp2_tp4_ep4", pp=2, tp=4, dp=1, ep=True, pp_layers=[20, 20])
    env = matrix.environment(strategy)
    for key in list(env):
        if key.startswith("DSV41_"):
            env.pop(key)
    for key in ("CUDA_DEVICE_MAX_CONNECTIONS", "CUDA_DEVICE_MAX_COPY_CONNECTIONS",
                "NCCL_GRAPH_MIXING_SUPPORT", "NCCL_GRAPH_STREAM_ORDERING",
                "VLLM_MARLIN_INPUT_DTYPE"):
        env.pop(key, None)
    startup = Path(__file__).with_name("startup")
    fast = ROOT / "fast_gemm_trial"
    critical = ROOT / "critical_path_hook"
    env.update(
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", MAX_JOBS="2",
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        DSV41_WAVE_BATCH="1", DSV41_WAVE_SIZE="1",
        DSV41_WAVE_CONTROL=str(work / "wave_control.json"),
        DSV41_PP_TIMING_DIR=str(work / "rank_timings"), DSV41_PP_TIMING_PER_RANK="1",
        DSV41_CRITICAL_PATH_DIR=str(work / "graph_dags"),
        PYTHONPATH=os.pathsep.join((str(startup), str(fast), str(critical), str(ROOT),
                                    env.get("PYTHONPATH", ""))),
    )
    if mode != "baseline":
        env["DSV41_FAST_ENABLED"] = "1"
        env["DSV41_FAST_EXTENSION"] = str(fast / "dsv4_wo_a_cublaslt.so")
    return env


def _validate_case(work: Path, label: str, batch: int, output_tokens: int,
                   preemption_before: object) -> dict:
    path = work / "requests" / f"{label}.json"
    records = json.loads(path.read_text())["records"]
    if len(records) != batch or any(sum(len(e.get("token_ids", [])) for e in r.get("events", [])) != output_tokens for r in records):
        raise RuntimeError(f"invalid request records for {label}")
    wave_path = work / "waves" / f"{label}.jsonl"
    waves, request_ids = read_waves(wave_path, batch, output_tokens)
    before = preemption_snapshot()
    if preemption_before is None or before is None or preemption_before != before:
        raise RuntimeError(f"preemption changed for {label}: {preemption_before} -> {before}")
    entry = dict(label=label, batch=batch, request_ids=sorted(request_ids),
                waves=len(waves), output_tokens=output_tokens)
    if output_tokens == MEASURE_TOKENS:
        entry["summary"] = summarize_requests(path)
        entry["summary"].pop("output_ids", None)
        common_start = max(r["events"][8]["t"] for r in records)
        common_end = min(r["events"][-2]["t"] for r in records)
        chosen = [w for w in waves if common_start <= w["wave_start"] <= w["wave_end"] <= common_end]
        if len(chosen) < 32:
            raise RuntimeError(f"Insufficient common whole-batch steps: {label}")
        values = [(w["wave_end"] - w["wave_start"]) * 1000 for w in chosen]
        entry["whole_batch_step"] = dict(p50_ms=statistics.median(values),
            p95_ms=percentile(values, .95), samples=len(values))
        entry["repeat"] = int(label.rsplit("_r", 1)[1])
    return entry


def _rank_validate(work: Path, cases: list[dict]) -> None:
    paths = sorted((work / "rank_timings").glob("rank_*.jsonl"))
    if len(paths) != 8:
        raise RuntimeError(f"Expected eight rank logs, got {len(paths)}")
    expected = {frozenset(map(str, c["request_ids"])): c for c in cases}
    for path in paths:
        counts = {c["label"]: 0 for c in cases}
        for raw in path.read_text().splitlines():
            row = json.loads(raw)
            if row.get("prefill"):
                continue
            ids = set(map(str, row.get("request_ids", [])))
            case = expected.get(frozenset(ids))
            if case is None:
                continue
            counts[case["label"]] += 1
            if row.get("n") != case["batch"] or row.get("padded_tokens") != case["batch"] or row.get("graph_mode") != "FULL":
                raise RuntimeError(f"invalid graph lane row in {path}: {row}")
        for case in cases:
            if counts[case["label"]] != case["output_tokens"] - 1:
                raise RuntimeError(f"Missing/duplicate rank steps: {path.name} {case['label']} {counts[case['label']]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "dense", "all"), default="dense")
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timing-source", type=Path)
    args = parser.parse_args()
    work = args.directory
    command = _command(work, args.mode)
    config = dict(mode=args.mode, batches=BATCHES, warmup_tokens=WARMUP_TOKENS,
                  measurement_tokens=MEASURE_TOKENS, profile_tokens=PROFILE_TOKENS,
                  one_server=True, one_profile_session=True, command=command)
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGALRM, interrupt)
    work.mkdir(parents=True, exist_ok=False)
    matrix.URL = "http://127.0.0.1:8024"
    matrix.REQUEST_SUFFIXES.clear()
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", 8024)) == 0:
            raise RuntimeError("port 8024 is occupied")
    _manifest_check()
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                      "--format=csv,noheader"], text=True).strip()
    if active:
        raise RuntimeError(f"GPUs occupied by existing processes: {active}")
    workload = ROOT / "pp2_graph_experiments/probe_deferred_20260926a/workload.json"
    tokens = json.loads(workload.read_text())["prompt_tokens"]
    if len(tokens) != 2048:
        raise RuntimeError(f"Expected 2048 prompt tokens, got {len(tokens)}")
    config.update(prompt_tokens=len(tokens), pp=2, tp=4, ep=4, pp_layers=[20, 20],
                  source_workload=str(workload), source_workload_sha256=hashlib.sha256(workload.read_bytes()).hexdigest(),
                  accuracy_scope="Performance/profiling only; deterministic precision controls disabled")
    matrix.save(work / "workload.json", dict(prompt_tokens=tokens))
    matrix.save(work / "configuration.json", config)
    matrix.save(work / "wave_control.json", dict(batch=1, size=1, log=str(work / "waves" / "control.jsonl")))
    process = None
    unprofiled = []
    profile_cases = []
    if args.timing_source:
        source_config = json.loads((args.timing_source / "configuration.json").read_text())
        for key in ("mode", "batches", "measurement_tokens", "prompt_tokens", "pp", "tp", "ep", "pp_layers", "source_workload_sha256"):
            if source_config[key] != json.loads(json.dumps(config[key])):
                raise RuntimeError(f"Timing source configuration mismatch: {key}")
        unprofiled = json.loads((args.timing_source / "unprofiled_summary.json").read_text())
        if len(unprofiled) != 27:
            raise RuntimeError("Expected 27 completed timing cases")
        _rank_validate(args.timing_source, unprofiled)
        config["timing_source"] = str(args.timing_source)
        matrix.save(work / "configuration.json", config)
        matrix.save(work / "unprofiled_summary.json", unprofiled)
    def status(phase, **extra):
        value = dict(phase=phase, mode=args.mode, time=time.time(), **extra)
        matrix.save(work / "status.json", value)
        print(json.dumps(value), flush=True)
    try:
        signal.alarm(args.startup_timeout)
        env = _environment(work, args.mode)
        with (work / "server.log").open("w") as log:
            process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        matrix.save(work / "server_pid.json", dict(pid=process.pid, command=command))
        status("starting", server_pid=process.pid)
        deadline = time.monotonic() + args.startup_timeout
        while not matrix.alive():
            if process.poll() is not None:
                raise RuntimeError(f"server exited: {process.returncode}")
            if time.monotonic() >= deadline:
                raise TimeoutError("server startup timeout")
            time.sleep(3)
        signal.alarm(1800)
        for batch in BATCHES:
            label = f"warmup_b{batch}"
            status("warmup", batch=batch)
            matrix.save(work / "wave_control.json", dict(batch=batch, size=batch,
                        log=str(work / "waves" / f"{label}.jsonl")))
            matrix.run_case(work, label, tokens, batch, WARMUP_TOKENS)
        for repeat in (() if args.timing_source else range(1, 4)):
            for batch in BATCHES if repeat % 2 else reversed(BATCHES):
                label = f"b{batch}_r{repeat}"
                status("timing", label=label)
                matrix.save(work / "wave_control.json", dict(batch=batch, size=batch,
                            log=str(work / "waves" / f"{label}.jsonl")))
                before = preemption_snapshot()
                matrix.run_case(work, label, tokens, batch, MEASURE_TOKENS)
                unprofiled.append(_validate_case(work, label, batch, MEASURE_TOKENS, before))
                matrix.save(work / "unprofiled_summary.json", unprofiled)
        status("profile_start")
        matrix.api("/start_profile", {}, timeout=120)
        try:
            for batch in BATCHES:
                label = f"profile_b{batch}"
                status("profiling", batch=batch)
                matrix.save(work / "wave_control.json", dict(batch=batch, size=batch,
                            log=str(work / "waves" / f"{label}.jsonl")))
                before = preemption_snapshot()
                matrix.run_case(work, label, tokens, batch, PROFILE_TOKENS)
                entry = _validate_case(work, label, batch, PROFILE_TOKENS, before)
                entry["profile_session_active"] = True
                profile_cases.append(entry)
                matrix.save(work / "profile_cases.json", profile_cases)
        finally:
            matrix.api("/stop_profile", {}, timeout=600)
        trace_dir = work / "profile_trace"
        trace_dir.mkdir(parents=True, exist_ok=True)
        staging = work / "trace_staging"
        for path in staging.iterdir() if staging.exists() else ():
            if path.is_file():
                shutil.move(str(path), trace_dir / path.name)
        trace_paths = list(trace_dir.glob("*.trace.json.gz"))
        if len(trace_paths) != 8:
            raise RuntimeError(f"Expected eight traces, got {len(trace_paths)}")
        matrix.save(work / "wave_control.json", dict(batch=1, size=1,
                    log=str(work / "waves" / "flush.jsonl")))
        matrix.run_case(work, "flush", tokens, 1, WARMUP_TOKENS)
        matrix.save(work / "unprofiled_summary.json", unprofiled)
        matrix.save(work / "profile_cases.json", profile_cases)
    except BaseException as exc:
        matrix.save(work / "FAILED.json", dict(error=repr(exc), traceback=traceback.format_exc()))
        raise
    finally:
        if process is not None:
            matrix.stop_server(process)
        signal.alarm(0)
        print("CLEANUP_COMPLETE", flush=True)
    _rank_validate(work, profile_cases if args.timing_source else unprofiled + profile_cases)
    matrix.save(work / "COMPLETE.json", dict(passed=True, mode=args.mode,
                unprofiled_cases=len(unprofiled), profile_cases=len(profile_cases)))
    status("complete")


if __name__ == "__main__":
    main()
