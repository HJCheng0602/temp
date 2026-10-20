"""Alternate graph/eager within one loaded model, then take one short profile."""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import traceback
import urllib.request

import parallel_matrix as matrix
from largebatch_pp2 import preemption_snapshot
from run_pp2_graph_comparison import compare_outputs, interrupt
from run_pp8_batch_sweep import summarize_requests

ROOT = Path("/home/admin/dsv41-vllm-20260917")


def drain():
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        text = urllib.request.urlopen(matrix.URL + "/metrics", timeout=10).read().decode()
        values = [float(line.rsplit(" ", 1)[-1]) for line in text.splitlines()
                  if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{"))]
        if len(values) >= 2 and not any(values):
            time.sleep(2)
            return
        time.sleep(1)
    raise RuntimeError("Requests did not drain; refusing mode switch")


def numa_snapshot(work, label):
    rows = []
    for path in Path("/proc").glob("[0-9]*/comm"):
        try:
            name = path.read_text().strip()
            if not name.startswith("VLLM::Worker"):
                continue
            pid = path.parent.name
            maps = (path.parent / "numa_maps").read_text().splitlines()
            large = []
            for line in maps:
                nodes = {int(k): int(v) for k, v in re.findall(r"\bN(\d+)=(\d+)", line)}
                if sum(nodes.values()) >= 65536:
                    large.append(dict(mapping=line, pages_by_node=nodes))
            rows.append(dict(pid=int(pid), name=name, large_mappings=large,
                             status=(path.parent / "status").read_text()))
        except (FileNotFoundError, PermissionError):
            continue
    matrix.save(work / f"numa_{label}.json", rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--profile-mode", choices=("graph", "eager"), default="graph")
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--max-connections", type=int, choices=(32, 64, 128))
    parser.add_argument("--reference-requests", type=Path)
    args = parser.parse_args()
    work = args.directory
    work.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGTERM, interrupt)
    port = 8024
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError("Probe port is already occupied")
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                      "--format=csv,noheader"], text=True).strip()
    if active:
        raise RuntimeError(f"GPUs already active: {active}")
    matrix.URL = f"http://127.0.0.1:{port}"
    strategy = dict(name="pp2_tp4_ep4", pp=2, tp=4, dp=1, ep=True, pp_layers=[20, 20])
    command = matrix.command(strategy, work)
    for key, value in {"--port":port, "--max-model-len":4096,
                       "--max-num-seqs":256, "--max-num-batched-tokens":8192,
                       "--gpu-memory-utilization":.85}.items():
        command[command.index(key) + 1] = str(value)
    command.remove("--enforce-eager")
    profiler = json.loads(command[command.index("--profiler-config") + 1])
    profiler["torch_profiler_record_shapes"] = False
    command[command.index("--profiler-config") + 1] = json.dumps(profiler)
    command += ["--engram-config", '{"cpu_offload":true}', "--async-scheduling",
                "--compilation-config", json.dumps(dict(mode=0,
                    cudagraph_mode="FULL_DECODE_ONLY",
                    cudagraph_capture_sizes=[1,2,4,8,16,32,64,128,256]))]
    env = matrix.environment(strategy)
    if args.max_connections is not None:
        env["CUDA_DEVICE_MAX_CONNECTIONS"] = str(args.max_connections)
    for key in list(env):
        if key.startswith("DSV41_WAVE_"):
            env.pop(key)
    control = work / "control.json"
    ack = work / "acks"
    matrix.save(control, dict(epoch=0, mode="graph"))
    env.update(DSV41_PP_REPLAY="0", DSV41_PP_GRAPH_TRIAL="0",
               DSV41_PP_TIMING_DIR=str(work / "rank_timings"),
               DSV41_PP_TIMING_PER_RANK="1", CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
               DSV41_GRAPH_PROBE_CONTROL=str(control), DSV41_GRAPH_PROBE_ACK=str(ack))
    matrix.save(work / "configuration.json", dict(command=command, strategy=strategy,
                scope="same-process diagnostic, standard scheduler, drained mode switches",
                prompt_tokens=2048, batch=256, output_tokens=96,
                profile_mode=args.profile_mode, profiler_steps=8,
                cuda_environment={key: env.get(key) for key in (
                    "CUDA_DEVICE_MAX_CONNECTIONS", "CUDA_DEVICE_MAX_COPY_CONNECTIONS",
                    "NCCL_GRAPH_MIXING_SUPPORT", "NCCL_GRAPH_STREAM_ORDERING")},
                reference_requests=str(args.reference_requests) if args.reference_requests else None))
    process = None
    epoch = 0
    baseline = None
    reference = (summarize_requests(args.reference_requests)["output_ids"]
                 if args.reference_requests else None)

    def status(phase, **extra):
        value = dict(phase=phase, time=time.time(), **extra)
        matrix.save(work / "status.json", value)
        print(json.dumps(value), flush=True)

    def switch(mode):
        nonlocal epoch
        drain()
        epoch += 1
        matrix.save(control, dict(epoch=epoch, mode=mode))

    def verify_ack(mode):
        values = [json.loads((ack / f"epoch_{epoch}_rank_{rank}.json").read_text())
                  for rank in range(8)]
        if any(v["epoch"] != epoch or v["mode"] != mode for v in values):
            raise RuntimeError("Workers disagree about probe epoch/mode")

    try:
        with (work / "server.log").open("w") as log:
            process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        matrix.save(work / "server_pid.json", dict(pid=process.pid))
        status("starting", pid=process.pid)
        deadline = time.monotonic() + 1200
        while not matrix.alive():
            if process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"Server startup failed: {process.poll()}")
            time.sleep(3)
        unit = matrix.api("/tokenize", dict(model="deepseek-v41-flash",
                           prompt="\n\n".join(matrix.CORPUS)))["tokens"]
        tokens = (unit * (2048 // len(unit) + 1))[:2048]
        matrix.REQUEST_SUFFIXES.clear()
        matrix.save(work / "workload.json", dict(prompt_tokens=tokens))
        numa_snapshot(work, "loaded")
        if not args.skip_timing:
            for label, mode in [("graph_a", "graph"), ("eager_a", "eager"),
                                ("graph_b", "graph"), ("eager_b", "eager")]:
                switch(mode)
                case = work / label
                case.mkdir()
                before = preemption_snapshot()
                status("testing", label=label, mode=mode, epoch=epoch)
                result = matrix.run_case(case, "timing", tokens, 256, 96)
                after = preemption_snapshot()
                summary = summarize_requests(case / "requests/timing.json")
                ids = summary.pop("output_ids")
                records = json.loads((case / "requests/timing.json").read_text())["records"]
                summary["common_window"] = dict(start=max(r["events"][8]["t"] for r in records),
                                                end=min(r["events"][-2]["t"] for r in records))
                if before is None or before != after:
                    raise RuntimeError(f"Invalid preemptions: {before} -> {after}")
                verify_ack(mode)
                if baseline is None:
                    baseline = ids
                summary.update(mode=mode, epoch=epoch, preemptions=dict(before=before, after=after),
                               output_vs_first=compare_outputs(baseline, ids))
                if reference is not None:
                    summary["output_vs_reference"] = compare_outputs(reference, ids)
                matrix.save(case / "summary.json", dict(**result, **summary))
                for key in ("output_vs_first", "output_vs_reference"):
                    if key in summary and summary[key]["mismatches"]:
                        raise RuntimeError(f"Output mismatch: {key}")
                status("case_complete", label=label, p50_ms=summary["pooled_itl_p50_ms"],
                       p95_ms=summary["pooled_itl_p95_ms"], min_intervals=summary["min_measured_intervals"],
                       outputs=summary["output_vs_first"])
        switch(args.profile_mode)
        status("profiling", mode=args.profile_mode, epoch=epoch)
        # The compact extractor streams traces after workers exit; avoid full JSON loads.
        matrix.inspect_traces = lambda directory: [dict(file=p.name, bytes=p.stat().st_size)
                                                   for p in directory.glob("*.json*")]
        matrix.run_case(work, "kernel_profile", tokens, 256, 96, profile=True, decode=True)
        verify_ack(args.profile_mode)
        numa_snapshot(work, "finished")
        status("complete")
        matrix.save(work / "COMPLETE.json", dict(profile_mode=args.profile_mode))
    except BaseException as exc:
        matrix.save(work / "FAILED.json", dict(error=repr(exc), traceback=traceback.format_exc()))
        status("failed", error=repr(exc))
        raise
    finally:
        if process is not None:
            matrix.stop_server(process)
        print("CLEANUP_COMPLETE", flush=True)
    subprocess.run([str(ROOT / "env/bin/python"), str(ROOT / "compact_graph_probe_trace.py"),
                    str(work / "timelines/kernel_profile"), str(work / "compact")], check=True)
    print("COMPACT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
