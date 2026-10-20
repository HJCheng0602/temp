"""Run one isolated PP2 AllReduce backend trial.

The two comparable backends are NCCL and FlashInfer PCIe IPC.  Both retain
the server's explicit ``--disable-custom-all-reduce`` flag; the backend choice
is controlled by the AllReduce environment variables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

DEPLOYMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOYMENT))

import parallel_matrix as matrix
from decode_report_20260929 import collect
from largebatch_pp2 import preemption_snapshot
from run_pp2_graph_comparison import interrupt
from run_pp2_wholebatch import audit_rank_paths
from run_pp8_batch_sweep import summarize_requests


DEFAULT_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
DEFAULT_PROFILE_BATCHES = (1, 128)
BACKENDS = ("nccl", "flashinfer_pcie_ipc")


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("batches must be positive")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("batches must be unique")
    return result


def _backend_command(work: Path, backend: str, moe_backend: str = "native") -> list[str]:
    return collect._command(work, "all" if moe_backend == "marlin" else "dense")


def _backend_environment(work: Path, backend: str, moe_backend: str = "native",
                         marlin_canonical_order: bool = False) -> dict[str, str]:
    env = collect._environment(work, "all" if moe_backend == "marlin" else "dense")
    # Do not let a shell, inherited trial, or vLLM default override the A/B
    # choice.  AllReduce controls are explicit in both comparable runs.
    for key in list(env):
        if (key.startswith("VLLM_ALLREDUCE_") or key in {
                "VLLM_USE_NCCL_SYMM_MEM", "VLLM_GPU_NIC_PCIE_MAPPING",
                "VLLM_BATCH_INVARIANT"}):
            env.pop(key, None)
    env.update(
        VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC="1" if backend == "flashinfer_pcie_ipc" else "0",
        VLLM_ALLREDUCE_USE_FLASHINFER="0",
        VLLM_ALLREDUCE_USE_SYMM_MEM="0",
        VLLM_USE_NCCL_SYMM_MEM="0",
    )
    if marlin_canonical_order:
        env["DSV41_MARLIN_CANONICAL_ORDER"] = "1"
    return env


def _output_ids(path: Path) -> list[list[int]]:
    records = json.loads(path.read_text())["records"]
    return [[token for event in record.get("events", [])
             for token in event.get("token_ids", [])] for record in records]


def _common_window(path: Path) -> dict[str, float]:
    records = json.loads(path.read_text())["records"]
    return {"start": max(record["events"][8]["t"] for record in records),
            "end": min(record["events"][-2]["t"] for record in records)}


def _assert_host_available() -> None:
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", 8024)) == 0:
            raise RuntimeError("port 8024 is occupied")
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True).strip()
    if active:
        raise RuntimeError(f"GPUs occupied by existing processes: {active}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--moe-backend", choices=("native", "marlin"), default="native")
    parser.add_argument("--marlin-canonical-order", action="store_true",
                        help="enable the existing diagnostic dispatch-order correction")
    parser.add_argument("--batches", type=_csv_ints,
                        default=DEFAULT_BATCHES, help="comma-separated decode batches")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile-batches", type=_csv_ints,
                        default=DEFAULT_PROFILE_BATCHES)
    parser.add_argument("--flashinfer-root", type=Path, default=None,
                        help="shared FlashInfer dependency root added to PYTHONPATH")
    parser.add_argument("--skip-profile", action="store_true",
                        help="skip the optional profiler session")
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.marlin_canonical_order and args.moe_backend != "marlin":
        parser.error("--marlin-canonical-order requires --moe-backend marlin")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if any(batch not in args.batches for batch in args.profile_batches):
        parser.error("--profile-batches must be included in --batches")

    work = args.directory
    if args.flashinfer_root is not None and not args.flashinfer_root.is_dir():
        parser.error(f"--flashinfer-root is not a directory: {args.flashinfer_root}")
    command = _backend_command(work, args.backend, args.moe_backend)
    config = dict(
        backend=args.backend, batches=list(args.batches), repeats=args.repeats,
        moe_backend=args.moe_backend, marlin_canonical_order=args.marlin_canonical_order,
        moe_activation_dtype="bfloat16" if args.moe_backend == "marlin" else "fp8",
        warmup_tokens=collect.WARMUP_TOKENS, measurement_tokens=collect.MEASURE_TOKENS,
        profile_tokens=collect.PROFILE_TOKENS, profile_batches=list(args.profile_batches),
        capture_sizes=list(collect.BATCHES),
        capture_size_policy="FULL_DECODE_ONLY exact captures retain all nine standard buckets even for a subset pilot",
        one_server=True, one_profile_session=True, command=command,
        flashinfer_root=None if args.flashinfer_root is None else str(args.flashinfer_root),
        skip_profile=args.skip_profile,
        allreduce_environment=dict(
            VLLM_ALLREDUCE_USE_FLASHINFER="0",
            VLLM_ALLREDUCE_USE_SYMM_MEM="0",
            VLLM_USE_NCCL_SYMM_MEM="0",
            VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=(
                "1" if args.backend == "flashinfer_pcie_ipc" else "0"),
            VLLM_BATCH_INVARIANT="unset"),
        comparison_note="Output token IDs are retained for comparison only; token equality is not a quality claim.",
    )
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGALRM, interrupt)
    work.mkdir(parents=True, exist_ok=False)
    matrix.URL = "http://127.0.0.1:8024"
    matrix.REQUEST_SUFFIXES.clear()
    collect._manifest_check()
    _assert_host_available()

    workload = matrix.ROOT / "pp2_graph_experiments/probe_deferred_20260926a/workload.json"
    tokens = json.loads(workload.read_text())["prompt_tokens"]
    if len(tokens) != 2048:
        raise RuntimeError(f"Expected 2048 prompt tokens, got {len(tokens)}")
    config.update(
        prompt_tokens=len(tokens), pp=2, tp=4, ep=4, pp_layers=[20, 20],
        source_workload=str(workload),
        source_workload_sha256=hashlib.sha256(workload.read_bytes()).hexdigest(),
        accuracy_scope="Performance/profiling only; deterministic precision controls disabled",
    )
    matrix.save(work / "workload.json", dict(prompt_tokens=tokens))
    matrix.save(work / "configuration.json", config)
    matrix.save(work / "wave_control.json", dict(batch=1, size=1,
                                                  log=str(work / "waves" / "control.jsonl")))

    process = None
    timing_cases: list[dict] = []
    profile_cases: list[dict] = []
    output_ids: dict[str, list[list[int]]] = {}

    def status(phase: str, **extra: object) -> None:
        value = dict(phase=phase, backend=args.backend, time=time.time(), **extra)
        matrix.save(work / "status.json", value)
        print(json.dumps(value), flush=True)

    def run_case(label: str, batch: int, output_tokens: int) -> dict:
        matrix.save(work / "wave_control.json", dict(
            batch=batch, size=batch, log=str(work / "waves" / f"{label}.jsonl")))
        before = preemption_snapshot()
        matrix.run_case(work, label, tokens, batch, output_tokens)
        entry = collect._validate_case(work, label, batch, output_tokens, before)
        output_ids[label] = _output_ids(work / "requests" / f"{label}.json")
        matrix.save(work / "output_ids.json", output_ids)
        if output_tokens == collect.MEASURE_TOKENS:
            entry["common_window"] = _common_window(work / "requests" / f"{label}.json")
        return entry

    try:
        signal.alarm(args.startup_timeout)
        env = _backend_environment(work, args.backend, args.moe_backend,
                                   args.marlin_canonical_order)
        if args.flashinfer_root is not None:
            # Keep startup sitecustomize first, then use the same dependency
            # root for every backend invocation.
            parts = env["PYTHONPATH"].split(os.pathsep)
            env["PYTHONPATH"] = os.pathsep.join((parts[0], str(args.flashinfer_root), *parts[1:]))
        with (work / "server.log").open("w") as log:
            process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        matrix.save(work / "server_pid.json", dict(pid=process.pid, command=command,
                                                    backend=args.backend))
        status("starting", server_pid=process.pid)
        deadline = time.monotonic() + args.startup_timeout
        while not matrix.alive():
            if process.poll() is not None:
                raise RuntimeError(f"server exited: {process.returncode}")
            if time.monotonic() >= deadline:
                raise TimeoutError("server startup timeout")
            time.sleep(3)
        signal.alarm(1800)
        for batch in args.batches:
            status("warmup", batch=batch)
            run_case(f"warmup_b{batch}", batch, collect.WARMUP_TOKENS)
        for repeat in range(1, args.repeats + 1):
            ordered = args.batches if repeat % 2 else tuple(reversed(args.batches))
            for batch in ordered:
                label = f"b{batch}_r{repeat}"
                status("timing", label=label)
                timing_cases.append(run_case(label, batch, collect.MEASURE_TOKENS))
                matrix.save(work / "unprofiled_summary.json", timing_cases)
        if not args.skip_profile:
            status("profile_start")
            matrix.api("/start_profile", {}, timeout=120)
            try:
                for batch in args.profile_batches:
                    label = f"profile_b{batch}"
                    status("profiling", batch=batch)
                    entry = run_case(label, batch, collect.PROFILE_TOKENS)
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
                    path.replace(trace_dir / path.name)
            traces = list(trace_dir.glob("*.trace.json.gz"))
            if len(traces) != 8:
                raise RuntimeError(f"Expected eight traces, got {len(traces)}")
        matrix.save(work / "wave_control.json", dict(batch=1, size=1,
                    log=str(work / "waves" / "flush.jsonl")))
        matrix.run_case(work, "flush", tokens, 1, collect.WARMUP_TOKENS)
        matrix.save(work / "unprofiled_summary.json", timing_cases)
        matrix.save(work / "profile_cases.json", profile_cases)
    except BaseException as exc:
        matrix.save(work / "FAILED.json", dict(error=repr(exc), traceback=traceback.format_exc()))
        raise
    finally:
        if process is not None:
            matrix.stop_server(process)
        signal.alarm(0)
        print("CLEANUP_COMPLETE", flush=True)

    # These checks intentionally run after the server has exited, so the saved
    # rank logs and request records are the final artifacts of this trial.
    collect._rank_validate(work, timing_cases + profile_cases)
    audit_rank_paths(work, timing_cases)
    for case in timing_cases:
        summary = summarize_requests(work / "requests" / f"{case['label']}.json")
        summary.pop("output_ids", None)
        case["request_summary"] = summary
    matrix.save(work / "unprofiled_summary.json", timing_cases)
    matrix.save(work / "summary.json", timing_cases)
    matrix.save(work / "COMPLETE.json", dict(
        passed=True, backend=args.backend, timing_cases=len(timing_cases),
        profile_cases=len(profile_cases), allreduce_quality_claim=False))
    status("complete", timing_cases=len(timing_cases), profile_cases=len(profile_cases))


if __name__ == "__main__":
    main()
