"""Paired PP2/TP4/EP4 eager and decode-Graph trials with resident batch checks."""

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback

import parallel_matrix as matrix
from largebatch_pp2 import preemption_snapshot
from run_pp8_batch_sweep import ROOT, summarize_requests


def interrupt(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}")


def compare_outputs(expected, actual):
    if len(expected) != len(actual):
        raise RuntimeError("Different request counts in paired outputs")
    mismatches = []
    for index, (left, right) in enumerate(zip(expected, actual)):
        difference = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), None)
        if difference is not None or len(left) != len(right):
            mismatches.append(dict(request=index, first_difference=difference,
                                   eager_tokens=len(left), graph_tokens=len(right)))
    return dict(matched_requests=len(actual) - len(mismatches),
                total_requests=len(actual), mismatches=mismatches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--batches", nargs="+", type=int, default=[32, 128, 256])
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--modes", nargs="+", choices=["eager", "graph"], default=["eager", "graph"])
    args = parser.parse_args()
    directory = args.directory
    directory.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGTERM, interrupt)
    port = 8024
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"Port {port} already in use")
    gpu = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                   "--format=csv,noheader"], text=True).strip()
    if gpu:
        raise RuntimeError(f"GPUs already have active processes: {gpu}")
    matrix.URL = f"http://127.0.0.1:{port}"
    baseline = {}
    baseline_prompt = None
    failed = []

    def status(phase, **extra):
        item = dict(phase=phase, time=time.time(), directory=str(directory), **extra)
        matrix.save(directory / "status.json", item)
        print(json.dumps(item), flush=True)

    for mode in args.modes:
        work = directory / mode
        work.mkdir()
        strategy = dict(name="pp2_tp4_ep4", pp=2, tp=4, dp=1, ep=True, pp_layers=[20, 20])
        command = matrix.command(strategy, work)
        overrides = {"--port": port, "--max-model-len":4096,
                     "--max-num-seqs":max(args.batches), "--max-num-batched-tokens":8192,
                     "--gpu-memory-utilization":.85}
        for option, value in overrides.items():
            command[command.index(option)+1] = str(value)
        command += ["--engram-config", '{"cpu_offload":true}', "--async-scheduling"]
        if mode == "graph":
            command.remove("--enforce-eager")
            command += ["--compilation-config", json.dumps(dict(mode=0,
                cudagraph_mode="FULL_DECODE_ONLY",
                cudagraph_capture_sizes=[1,2,4,8,16,32,64,128,256]))]
        env = matrix.environment(strategy)
        for key in list(env):
            if key.startswith("DSV41_WAVE_"):
                env.pop(key)
        env.update(DSV41_PP_REPLAY="0", DSV41_PP_GRAPH_TRIAL="0",
                   DSV41_PP_TIMING_DIR=str(work / "rank_timings"),
                   DSV41_PP_TIMING_PER_RANK="1", CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7")
        matrix.save(work / "configuration.json", dict(command=command, strategy=strategy,
                    batches=args.batches, prompt_tokens=2048, output_tokens=args.output_tokens,
                    mode=mode, scheduler="standard async continuous scheduling",
                    scope="Paired same-runtime comparison; no PP8 wave scheduler or source replay"))
        process = None
        try:
            with (work / "server.log").open("w") as log:
                process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            matrix.save(work / "server_pid.json", dict(pid=process.pid))
            status("starting", mode=mode, pid=process.pid)
            deadline = time.monotonic() + 1200
            while not matrix.alive():
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited: {process.returncode}")
                if time.monotonic() > deadline:
                    raise TimeoutError("Server startup exceeded 1200 seconds")
                time.sleep(3)
            unit = matrix.api("/tokenize", dict(model="deepseek-v41-flash",
                              prompt="\n\n".join(matrix.CORPUS)))["tokens"]
            tokens = (unit * (2048 // len(unit) + 1))[:2048]
            if baseline_prompt is not None and tokens != baseline_prompt:
                raise RuntimeError("Paired prompt token IDs differ")
            baseline_prompt = tokens
            matrix.REQUEST_SUFFIXES.clear()
            matrix.save(work / "workload.json", dict(prompt_tokens=tokens))
            for batch in args.batches:
                case = work / f"b{batch}"
                case.mkdir()
                before = preemption_snapshot()
                status("testing", mode=mode, batch=batch)
                result = matrix.run_case(case, "timing", tokens, batch, args.output_tokens)
                after = preemption_snapshot()
                summary = summarize_requests(case / "requests/timing.json")
                output_ids = summary.pop("output_ids")
                records = json.loads((case / "requests/timing.json").read_text())["records"]
                summary["common_window"] = dict(
                    start=max(r["events"][8]["t"] for r in records),
                    end=min(r["events"][-2]["t"] for r in records))
                if before is not None and after is not None and after != before:
                    raise RuntimeError(f"Preemption occurred: {before} -> {after}")
                summary["preemptions"] = dict(before=before, after=after)
                summary["distinct_output_sequences"] = len({tuple(x) for x in output_ids})
                if mode == "eager":
                    baseline[batch] = output_ids
                elif batch in baseline:
                    summary["graph_vs_eager"] = compare_outputs(baseline[batch], output_ids)
                matrix.save(case / "summary.json", dict(**result, **summary))
                status("case_complete", mode=mode, batch=batch,
                       itl_p50_ms=summary["pooled_itl_p50_ms"],
                       itl_p95_ms=summary["pooled_itl_p95_ms"],
                       min_intervals=summary["min_measured_intervals"],
                       output_match=summary.get("graph_vs_eager"))
                time.sleep(2)
            matrix.save(work / "COMPLETE.json", dict(mode=mode, batches=args.batches))
        except Exception as exc:
            failed.append(mode)
            matrix.save(work / "FAILED.json", dict(error=repr(exc), traceback=traceback.format_exc()))
            status("mode_failed", mode=mode, error=repr(exc))
        finally:
            if process is not None:
                matrix.stop_server(process)
    status("complete", failed_modes=failed)
    print("CLEANUP_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
