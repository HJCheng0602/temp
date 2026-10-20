"""Validate the clean Graph launcher, pause/resume, abort, and recovery."""

import argparse
import json
import signal
import socket
import subprocess
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import parallel_matrix as matrix
from largebatch_pp2 import preemption_snapshot
from run_pp2_graph_comparison import compare_outputs, interrupt
from run_pp8_batch_sweep import summarize_requests


ROOT = matrix.ROOT
WORK = ROOT / "pp2_graph_experiments/clean_deferred_20260926a"


def lifecycle_case(tokens, mode):
    records = [dict(events=[]) for _ in range(8)]
    pool = ThreadPoolExecutor(max_workers=8)
    barrier = threading.Barrier(8)
    jobs = [pool.submit(matrix.stream_one, tokens, 96, barrier, records, i)
            for i in range(8)]
    try:
        deadline = time.monotonic() + 90
        while not all(len(r["events"]) >= 4 for r in records):
            if any(j.done() for j in jobs) or time.monotonic() > deadline:
                raise RuntimeError("Requests did not reach the lifecycle test boundary")
            time.sleep(.01)
        started = time.monotonic()
        pause = matrix.api(f"/pause?mode={mode}&clear_cache=false", {}, timeout=30)
        pause_seconds = time.monotonic() - started
        paused = matrix.api("/is_paused", timeout=10)
        assert paused["is_paused"], paused
        time.sleep(.1)
        counts = [len(r["events"]) for r in records]
        time.sleep(.2)
        assert counts == [len(r["events"]) for r in records], "Output advanced while paused"
        resume = matrix.api("/resume", {}, timeout=30)
        errors = []
        for job in jobs:
            try:
                job.result(timeout=60)
            except Exception as exc:
                if mode != "abort" or not str(exc).startswith("Expected 96 streamed token IDs, got "):
                    raise
                errors.append(repr(exc))
        if mode == "abort":
            assert len(errors) == 8, "Abort must terminate all eight active requests"
        matrix.save(WORK / "requests" / f"pause_{mode}.json", dict(records=records))
        result = dict(mode=mode, pause=pause, resume=resume,
                      pause_seconds=pause_seconds, events_at_pause=counts, errors=errors)
        if mode == "keep":
            summary = summarize_requests(WORK / "requests/pause_keep.json")
            reference = summarize_requests(WORK / "requests/timing.json")["output_ids"][:8]
            result["outputs"] = compare_outputs(reference, summary["output_ids"])
            assert not result["outputs"]["mismatches"], result
        matrix.save(WORK / f"lifecycle_{mode}.json", result)
        return result
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def main():
    global WORK
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=WORK)
    args = parser.parse_args()
    WORK = args.directory
    WORK.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGALRM, interrupt)
    matrix.URL = "http://127.0.0.1:8024"
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", 8024)) == 0:
            raise RuntimeError("Port occupied")
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                      "--format=csv,noheader"], text=True).strip()
    if active:
        raise RuntimeError(f"GPUs active: {active}")

    def status(phase, **extra):
        value = dict(phase=phase, time=time.time(), **extra)
        matrix.save(WORK / "status.json", value)
        print(json.dumps(value), flush=True)

    process = None
    try:
        with (WORK / "server.log").open("w") as log:
            process = subprocess.Popen(
                [str(ROOT / "env/bin/python"), str(ROOT / "serve_pp2_graph.py"),
                 "--profile-dir", str(WORK / "trace_staging"),
                 "--timing-dir", str(WORK / "rank_timings"),
                 "--validate-lifecycle"],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        status("starting", server_pid=process.pid)
        deadline = time.monotonic() + 1200
        while not matrix.alive():
            if process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Clean launcher failed to start")
            time.sleep(3)
        signal.alarm(300)
        tokens = json.loads((ROOT / "pp2_graph_experiments/probe_deferred_20260926a/workload.json").read_text())["prompt_tokens"]
        matrix.REQUEST_SUFFIXES.clear()
        before = preemption_snapshot()
        status("testing_clean_graph")
        matrix.run_case(WORK, "timing", tokens, 256, 96)
        summary = summarize_requests(WORK / "requests/timing.json")
        reference = summarize_requests(ROOT / "pp2_graph_experiments/probe_graph_20260926a/graph_a/requests/timing.json")["output_ids"]
        summary["output_vs_original"] = compare_outputs(reference, summary.pop("output_ids"))
        assert not summary["output_vs_original"]["mismatches"]
        assert before is not None and before == preemption_snapshot()
        summary["preemptions"] = before
        matrix.save(WORK / "summary.json", summary)
        status("clean_graph_complete", p50_ms=summary["pooled_itl_p50_ms"])
        status("testing_clean_graph_repeat")
        matrix.run_case(WORK, "timing_repeat", tokens, 256, 96)
        repeat = summarize_requests(WORK / "requests/timing_repeat.json")
        repeat["output_vs_original"] = compare_outputs(reference, repeat.pop("output_ids"))
        assert not repeat["output_vs_original"]["mismatches"]
        assert before == preemption_snapshot()
        matrix.save(WORK / "summary_repeat.json", repeat)
        status("clean_graph_repeat_complete", p50_ms=repeat["pooled_itl_p50_ms"])
        matrix.inspect_traces = lambda directory: [dict(file=p.name, bytes=p.stat().st_size)
                                                    for p in directory.glob("*.json*")]
        status("profiling")
        matrix.run_case(WORK, "kernel_profile", tokens, 256, 96, profile=True, decode=True)
        for mode in ("keep", "abort"):
            status("lifecycle", mode=mode)
            lifecycle_case(tokens, mode)
        status("recovery")
        matrix.run_case(WORK, "recovery", tokens, 8, 96)
        recovery = summarize_requests(WORK / "requests/recovery.json")
        recovery["output_vs_original"] = compare_outputs(reference[:8], recovery.pop("output_ids"))
        assert not recovery["output_vs_original"]["mismatches"]
        matrix.save(WORK / "recovery.json", recovery)
        status("complete")
        matrix.save(WORK / "COMPLETE.json", dict(passed=True))
    except BaseException as exc:
        matrix.save(WORK / "FAILED.json", dict(error=repr(exc), traceback=traceback.format_exc()))
        status("failed", error=repr(exc))
        raise
    finally:
        signal.alarm(0)
        if process is not None:
            matrix.stop_server(process)
        print("CLEANUP_COMPLETE", flush=True)
    subprocess.run([str(ROOT / "env/bin/python"), str(ROOT / "compact_graph_probe_trace.py"),
                    str(WORK / "timelines/kernel_profile"), str(WORK / "compact")], check=True)
    print("COMPACT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
