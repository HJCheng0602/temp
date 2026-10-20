"""Fixed B128 decode waves; compare actual group sizes and optional CUDA Graph."""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path("/home/admin/dsv41-vllm-20260917")
sys.path.insert(0, str(ROOT))
import parallel_matrix as matrix


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * q
    a, b = math.floor(position), math.ceil(position)
    return values[a] + (values[b] - values[a]) * (position - a)


def summarize_requests(path):
    records = json.loads(path.read_text())["records"]
    lo = max(row["events"][8]["t"] for row in records)
    hi = min(row["events"][-2]["t"] for row in records)
    medians, p95s, intervals, counts = [], [], [], []
    output_ids = []
    for row in records:
        events = row["events"]
        values = [(b["t"] - a["t"]) * 1000
                  for a, b in zip(events[8:], events[9:])
                  if a["t"] >= lo and b["t"] <= hi
                  and len(a["token_ids"]) == len(b["token_ids"]) == 1]
        if len(values) < 16:
            raise RuntimeError("Insufficient common steady intervals")
        medians.append(statistics.median(values))
        p95s.append(percentile(values, .95))
        intervals.extend(values)
        counts.append(len(values))
        output_ids.append([token for event in events for token in event["token_ids"]])
    return dict(per_request_median_p50_ms=statistics.median(medians),
                per_request_median_p95_ms=percentile(medians, .95),
                per_request_p95_p50_ms=statistics.median(p95s),
                pooled_itl_p50_ms=statistics.median(intervals),
                pooled_itl_p95_ms=percentile(intervals, .95),
                min_measured_intervals=min(counts),
                launch_skew_ms=(max(row["start"] for row in records)
                                - min(row["start"] for row in records))*1000,
                output_ids=output_ids)


def validate_dispatches(path, size, output_tokens):
    waves = [json.loads(line) for line in path.read_text().splitlines()]
    if len(waves) != output_tokens - 1:
        raise RuntimeError(f"Expected {output_tokens - 1} waves, got {len(waves)}")
    expected_ids = None
    for ordinal, wave in enumerate(waves, 2):
        dispatches = wave["dispatches"]
        if wave["output_token_ordinal"] != ordinal:
            raise RuntimeError("Decode wave ordinals are not consecutive")
        if [d["group_idx"] for d in dispatches] != list(range(128 // size)):
            raise RuntimeError("Actual decode group count differs from requested size")
        ids = []
        for dispatch in dispatches:
            scheduled = dispatch["num_scheduled_tokens"]
            if (len(scheduled) != size or any(n != 1 for n in scheduled.values())
                    or set(scheduled) != set(dispatch["request_ids"])):
                raise RuntimeError("Actual decode token count differs from requested size")
            ids.extend(scheduled)
        if len(ids) != 128 or len(set(ids)) != 128:
            raise RuntimeError("Wave does not cover exactly 128 requests")
        if expected_ids is None:
            expected_ids = set(ids)
        elif set(ids) != expected_ids:
            raise RuntimeError("Requests changed between waves")
    return dict(actual_decode_size=size, validated_waves=len(waves))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", choices=["eager", "graph"], default=["eager", "graph"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--output-tokens", type=int, default=96)
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    directory = args.directory or ROOT / "pp8_batch_experiments" / time.strftime("%Y%m%d_%H%M%S")
    directory.mkdir(parents=True, exist_ok=True)
    matrix.URL = "http://127.0.0.1:8028"
    if matrix.alive():
        raise RuntimeError("Port 8028 is already occupied")
    gpu = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if gpu:
        raise RuntimeError(f"GPU processes already exist: {gpu}")
    control = directory / "control.json"
    baseline = {}

    def status(phase, **extra):
        value = dict(phase=phase, time=time.time(), directory=str(directory), **extra)
        matrix.save(directory / "status.json", value)
        print(json.dumps(value), flush=True)

    for mode in args.modes:
        work = directory / mode
        work.mkdir(exist_ok=True)
        (work / "trace_staging").mkdir(exist_ok=True)
        strategy = dict(name=f"pp8_wave_{mode}", tp=1, pp=8, dp=1, ep=False, pp_layers=[5]*8)
        command = matrix.command(strategy, work)
        for option, value in {"--port":8028, "--max-model-len":4096,
                              "--max-num-seqs":128, "--max-num-batched-tokens":2048,
                              "--gpu-memory-utilization":.85}.items():
            command[command.index(option)+1] = str(value)
        command += ["--engram-config", '{"cpu_offload":true}',
                    "--scheduler-cls", "pp8_wave_scheduler.WaveScheduler",
                    "--async-scheduling"]
        if mode == "graph":
            command.remove("--enforce-eager")
            command += ["--compilation-config", json.dumps(dict(mode=0,
                cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=[16,32,64,128]))]
        env = matrix.environment(strategy)
        env.update(DSV41_PP_REPLAY="1", DSV41_PP_GRAPH_TRIAL="1" if mode == "graph" else "0",
                   DSV41_WAVE_BATCH="128", DSV41_WAVE_SIZE=str(args.sizes[0]),
                   DSV41_WAVE_LOG=str(work/"initial_scheduler.jsonl"),
                   DSV41_WAVE_CONTROL=str(control), DSV41_PP_TIMING_DIR=str(work/"stage_timings"),
                   CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
                   PYTHONPATH=str(ROOT)+os.pathsep+env.get("PYTHONPATH", ""))
        matrix.save(work / "configuration.json", dict(command=command, strategy=strategy,
                    total_requests=128, sizes=args.sizes, output_tokens=args.output_tokens,
                    mode=mode, scope="Synchronized decode waves after all requests finish prefill"))
        process = None
        try:
            matrix.save(control, dict(size=args.sizes[0], batch=128,
                                      log=str(work / f"x{args.sizes[0]}_scheduler.jsonl")))
            with (work / "server.log").open("w") as log:
                process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            matrix.save(work / "server_pid.json", dict(pid=process.pid))
            status("starting", mode=mode, pid=process.pid)
            deadline = time.monotonic()+1500
            while not matrix.alive():
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited with code {process.returncode}")
                if time.monotonic()>deadline:
                    raise TimeoutError("Startup timeout")
                time.sleep(3)
            corpus = "\n\n".join(matrix.CORPUS)
            unit = matrix.api("/tokenize", dict(model="deepseek-v41-flash", prompt=corpus))["tokens"]
            tokens = (unit*(2048//len(unit)+1))[:2048]
            matrix.REQUEST_SUFFIXES.clear()
            matrix.save(work / "workload.json", dict(prompt_tokens=tokens))
            for size in args.sizes:
                case = work / f"x{size}"
                case.mkdir(exist_ok=True)
                matrix.save(control, dict(size=size, batch=128, log=str(work/f"x{size}_scheduler.jsonl")))
                status("testing", mode=mode, size=size)
                result = matrix.run_case(case, "timing", tokens, 128, args.output_tokens)
                result.update(validate_dispatches(work / f"x{size}_scheduler.jsonl",
                                                  size, args.output_tokens))
                summary = summarize_requests(case / "requests/timing.json")
                ids = summary.pop("output_ids")
                if mode == "eager":
                    baseline[size] = ids
                elif size in baseline:
                    expected = baseline[size]
                    mismatches = []
                    for index, (left, right) in enumerate(zip(expected, ids)):
                        difference = next((j for j, (a,b) in enumerate(zip(left,right)) if a!=b), None)
                        if difference is not None or len(left)!=len(right):
                            mismatches.append(dict(request=index, first_difference=difference))
                    summary["graph_vs_same_size_eager"] = dict(matched_requests=128-len(mismatches),
                        total_requests=128, mismatches=mismatches)
                summary["distinct_output_sequences"] = len({tuple(row) for row in ids})
                matrix.save(case / "summary.json", dict(**result, **summary))
                status("case_complete", mode=mode, size=size,
                       itl_p50_ms=summary["pooled_itl_p50_ms"],
                       distinct_outputs=summary["distinct_output_sequences"])
                time.sleep(2)
            matrix.save(work / "COMPLETE.json", dict(mode=mode, sizes=args.sizes))
        except BaseException as exc:
            matrix.save(work / "FAILED.json", dict(error=repr(exc), traceback=traceback.format_exc()))
            status("mode_failed", mode=mode, error=repr(exc))
        finally:
            if process is not None:
                matrix.stop_server(process)
    status("complete")
    print("CLEANUP_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
