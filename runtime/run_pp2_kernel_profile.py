"""Profile the fixed whole B32 graph, then collect eager shape metadata."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tarfile
import time
import traceback

import parallel_matrix as matrix
from largebatch_pp2 import preemption_snapshot
from run_pp2_graph_comparison import compare_outputs, interrupt
from run_pp2_graph_probe import drain
from run_pp2_wholebatch import read_waves
from run_pp8_batch_sweep import summarize_requests


ROOT = matrix.ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=("graph", "eager"),
                        default=["graph", "eager"])
    parser.add_argument("--critical-path", action="store_true",
                        help="Capture B32 graph topology and CUDA synchronization events")
    args = parser.parse_args()
    if args.critical_path and args.modes != ["graph"]:
        parser.error("--critical-path requires --modes graph (one profiler session)")
    work = args.directory
    work.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGALRM, interrupt)
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", 8024)) == 0:
            raise RuntimeError("Port occupied")
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                      "--format=csv,noheader"], text=True).strip()
    if active:
        raise RuntimeError(f"GPUs active: {active}")
    runtime = Path(importlib.metadata.distribution("vllm").locate_file("vllm"))
    manifest = json.loads((ROOT / "pp_deferred_fix/manifest.json").read_text())
    for relative, hashes in manifest["files"].items():
        if hashlib.sha256((runtime / relative).read_bytes()).hexdigest() != hashes["after"]:
            raise RuntimeError(f"Runtime differs from verified fix: {relative}")

    matrix.URL = "http://127.0.0.1:8024"
    matrix.REQUEST_SUFFIXES.clear()
    matrix.inspect_traces = lambda directory: [dict(file=p.name, bytes=p.stat().st_size)
                                               for p in directory.glob("*.json*")]
    strategy = dict(name="pp2_tp4_ep4", pp=2, tp=4, dp=1, ep=True, pp_layers=[20, 20])
    command = matrix.command(strategy, work)
    command.remove("--enforce-eager")
    for key, value in {"--port": 8024, "--max-model-len": 4096, "--max-num-seqs": 256,
                       "--max-num-batched-tokens": 8192, "--gpu-memory-utilization": .85}.items():
        command[command.index(key) + 1] = str(value)
    profiler = json.loads(command[command.index("--profiler-config") + 1])
    profiler.update(torch_profiler_record_shapes=True, torch_profiler_with_stack=False)
    command[command.index("--profiler-config") + 1] = json.dumps(profiler)
    wave_control, mode_control = work / "control.json", work / "mode.json"
    matrix.save(mode_control, dict(mode="graph"))
    command += ["--engram-config", '{"cpu_offload":true}', "--async-scheduling",
                "--compilation-config", json.dumps(dict(mode=0, cudagraph_mode="FULL_DECODE_ONLY",
                    cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32, 64, 96, 120, 128, 136, 144, 160, 192, 224, 256])),
                "--scheduler-cls", "pp2_fixed_batch_scheduler.FixedBatchScheduler"]
    if args.modes == ["eager"]:
        command.append("--enforce-eager")
    env = matrix.environment(strategy)
    for key in list(env):
        if key.startswith(("DSV41_WAVE_", "DSV41_PP_", "DSV41_GRAPH_PROBE_")):
            env.pop(key)
    for key in ("CUDA_DEVICE_MAX_CONNECTIONS", "CUDA_DEVICE_MAX_COPY_CONNECTIONS",
                "NCCL_GRAPH_MIXING_SUPPORT", "NCCL_GRAPH_STREAM_ORDERING"):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", DSV41_WAVE_BATCH="32",
               DSV41_WAVE_SIZE="32", DSV41_WAVE_CONTROL=str(wave_control),
               DSV41_KERNEL_PROFILE_MODE=str(mode_control),
               DSV41_PP_TIMING_DIR=str(work / "rank_timings"), DSV41_PP_TIMING_PER_RANK="1",
               PYTHONPATH=str(ROOT / "kernel_profile_hook") + os.pathsep + str(ROOT) +
               os.pathsep + env.get("PYTHONPATH", ""))
    if args.critical_path:
        compilation_index = command.index("--compilation-config") + 1
        compilation = json.loads(command[compilation_index])
        compilation["cudagraph_capture_sizes"] = [32]
        command[compilation_index] = json.dumps(compilation)
        env["DSV41_CRITICAL_PATH_DIR"] = str(work / "graph_dags")
        env["PYTHONPATH"] = str(ROOT / "critical_path_hook") + os.pathsep + env["PYTHONPATH"]
    tokens = json.loads((ROOT / "pp2_graph_experiments/wholebatch_20260926a/workload.json").read_text())["prompt_tokens"]
    baseline = summarize_requests(ROOT / "pp2_graph_experiments/wholebatch_20260926a/requests/b32_r1.json")["output_ids"]
    matrix.save(work / "configuration.json", dict(command=command, strategy=strategy,
        batch=32, prompt_tokens=len(tokens), output_tokens=96, graph_timing=True,
        eager_scope="shape metadata only; do not substitute eager durations for graph",
        runtime_files_unchanged=True, profiler_record_shapes=True))
    cases = []
    process = None

    def status(phase, **extra):
        value = dict(phase=phase, time=time.time(), **extra)
        matrix.save(work / "status.json", value)
        print(json.dumps(value), flush=True)

    def run(mode, label, outputs, profile=False):
        case = work / mode
        case.mkdir(exist_ok=True)
        wave_path = case / "waves" / (label + ".jsonl")
        matrix.save(wave_control, dict(batch=32, size=32, log=str(wave_path)))
        before = preemption_snapshot()
        # run_case expects the active profiler's staging directory below its case.
        staging = case / "trace_staging"
        if not staging.exists() and not staging.is_symlink():
            staging.symlink_to(work / "trace_staging", target_is_directory=True)
        matrix.run_case(case, label, tokens, 32, outputs, profile=profile, decode=profile)
        after = preemption_snapshot()
        waves, ids = read_waves(wave_path, 32, outputs)
        records = json.loads((case / "requests" / (label + ".json")).read_text())["records"]
        output_ids = [[token for event in r["events"] for token in event["token_ids"]]
                      for r in records]
        reference = [x[:outputs] for x in baseline]
        check = compare_outputs(reference, output_ids)
        if check["mismatches"] or before is None or before != after:
            raise RuntimeError(f"Validation failed: {mode}/{label}: {check}")
        entry = dict(mode=mode, label=label, profile=profile, outputs=outputs,
                     request_ids=sorted(ids), waves=len(waves), output_check=check,
                     preemption_before=before, preemption_after=after)
        cases.append(entry)
        matrix.save(work / "cases.json", cases)
        return entry

    try:
        signal.alarm(1200)
        with (work / "server.log").open("w") as log:
            process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        matrix.save(work / "server_pid.json", dict(pid=process.pid))
        status("starting", server_pid=process.pid)
        while not matrix.alive():
            if process.poll() is not None:
                raise RuntimeError(f"Server exited: {process.returncode}")
            time.sleep(3)
        signal.alarm(900)
        for mode in args.modes:
            drain()
            matrix.save(mode_control, dict(mode=mode))
            status("warmup", mode=mode)
            run(mode, "warmup", 16)
            status("profiling", mode=mode)
            run(mode, "kernel_profile", 96, profile=True)
        run(args.modes[-1], "flush", 16)
    except BaseException as exc:
        matrix.save(work / "FAILED.json", dict(error=repr(exc), traceback=traceback.format_exc()))
        status("failed", error=repr(exc))
        raise
    finally:
        if process is not None:
            matrix.stop_server(process)
        signal.alarm(0)
        print("CLEANUP_COMPLETE", flush=True)

    for rank in range(8):
        path, = (work / "rank_timings").glob(f"rank_{rank}_*.jsonl")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for case in (c for c in cases if c["profile"]):
            chosen = [r for r in rows if not r["prefill"] and set(r["request_ids"]) == set(case["request_ids"])]
            expected = "FULL" if case["mode"] == "graph" else "NONE"
            if len(chosen) != 95 or any(r["n"] != 32 or r["padded_tokens"] != 32 or
                                         r["graph_mode"] != expected for r in chosen):
                raise RuntimeError(f"Wrong execution shape/mode: rank{rank} {case['mode']}")
    status("extracting")
    subprocess.run([str(ROOT / "env/bin/python"), str(ROOT / "extract_kernel_top30.py"),
                    str(work / "graph/timelines/kernel_profile"),
                    str(work / "eager/timelines/kernel_profile"), str(work / "extracted"),
                    "--allow-single-mode"], check=True)
    matrix.save(work / "COMPLETE.json", dict(passed=True, batch=32, profiled_modes=args.modes))
    status("complete")
    archive = work.with_suffix(".tgz")
    with tarfile.open(archive, "w:gz") as tar:
        for name in ("configuration.json", "cases.json", "server.log", "COMPLETE.json",
                     "status.json", "rank_timings", "extracted", "graph/requests", "graph/waves",
                     "eager/requests", "eager/waves", "graph_dags"):
            if (work / name).exists():
                tar.add(work / name, arcname=work.name + "/" + name)
        for name in ("run_pp2_kernel_profile.py", "extract_kernel_top30.py",
                     "kernel_profile_hook/sitecustomize.py", "pp_deferred_fix/manifest.json"):
            tar.add(ROOT / name, arcname=work.name + "/reproduction/" + name)
        if args.critical_path:
            tar.add(work / "graph/timelines", arcname=work.name + "/graph/timelines")
            tar.add(ROOT / "critical_path_hook/sitecustomize.py",
                    arcname=work.name + "/reproduction/critical_path_hook/sitecustomize.py")
    print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size,
        sha256=hashlib.file_digest(archive.open("rb"), "sha256").hexdigest())), flush=True)


if __name__ == "__main__":
    main()
