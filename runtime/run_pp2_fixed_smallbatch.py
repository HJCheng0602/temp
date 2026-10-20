"""Measure small resident batches with the verified deferred-receive Graph fix."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import tarfile
import time
import traceback

import parallel_matrix as matrix
from largebatch_pp2 import preemption_snapshot
from run_pp2_graph_comparison import compare_outputs, interrupt
from run_pp8_batch_sweep import summarize_requests


ROOT = matrix.ROOT


def audit_dispatches(work, cases):
    ranks = {}
    for rank in range(8):
        path, = (work / "rank_timings").glob(f"rank_{rank}_*.jsonl")
        ranks[rank] = [json.loads(line) for line in path.read_text().splitlines()]
    for case in cases:
        rank_summaries = {}
        lo, hi = case["common_window"].values()
        for rank, rows in ranks.items():
            selected = [r for r in rows if lo <= r["cpu_start"] <= hi]
            if not selected or any(r["prefill"] or r["graph_mode"] != "FULL"
                                   for r in selected):
                raise RuntimeError(f"Invalid Graph dispatch: {case['label']} rank {rank}")
            groups = defaultdict(list)
            shapes = defaultdict(list)
            for row in selected:
                groups[tuple(sorted(row["request_ids"]))].append(row)
                shapes[(row["n"], row["padded_tokens"])].append(row["gpu_ms"])
            ids = {request for group in groups for request in group}
            if len(ids) != case["batch"]:
                raise RuntimeError(f"Incomplete resident batch: {case['label']} rank {rank}")
            rank_summaries[str(rank)] = dict(
                full_graph_records=len(selected),
                shapes=[dict(tokens=n, padded_tokens=p, records=len(values),
                             median_gpu_ms=statistics.median(values))
                        for (n, p), values in sorted(shapes.items())],
                groups=[dict(requests=len(ids), records=len(rows),
                             cpu_submit_period_median_ms=statistics.median(
                                 [(b["cpu_start"] - a["cpu_start"]) * 1000
                                  for a, b in zip(rows, rows[1:])])
                             if len(rows) > 1 else None)
                        for ids, rows in groups.items()])
        case["dispatch"] = rank_summaries
        matrix.save(work / "summaries" / (case["label"] + ".json"), case)
    matrix.save(work / "summary.json", cases)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    work = args.directory
    work.mkdir(parents=True, exist_ok=False)
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
    batches = [1, 2, 4, 8, 16, 32]
    workload = ROOT / "pp2_graph_experiments/probe_deferred_20260926a/workload.json"
    tokens = json.loads(workload.read_text())["prompt_tokens"]
    baseline_path = ROOT / "pp2_graph_experiments/probe_graph_20260926a/graph_a/requests/timing.json"
    baseline = summarize_requests(baseline_path)["output_ids"]
    matrix.REQUEST_SUFFIXES.clear()
    matrix.save(work / "configuration.json", dict(
        batches=batches, repeats=3, prompt_tokens=len(tokens), output_tokens=96,
        pp=2, tp=4, ep=4, pp_layers=[20, 20], profiled=False,
        launcher=str(ROOT / "serve_pp2_graph.py"),
        capture_sizes=[1, 2, 4, 8, 16, 32, 64, 96, 120, 128, 136, 144, 160, 192, 224, 256],
        scope="Fixed patched runtime, standard asynchronous scheduling; common resident decode window"))
    matrix.save(work / "workload.json", dict(prompt_tokens=tokens))
    cases = []

    def status(phase, **extra):
        value = dict(phase=phase, time=time.time(), **extra)
        matrix.save(work / "status.json", value)
        print(json.dumps(value), flush=True)

    process = None
    try:
        signal.alarm(1200)
        with (work / "server.log").open("w") as log:
            process = subprocess.Popen(
                [str(ROOT / "env/bin/python"), str(ROOT / "serve_pp2_graph.py"),
                 "--timing-dir", str(work / "rank_timings")],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        matrix.save(work / "server_pid.json", dict(pid=process.pid))
        status("starting", server_pid=process.pid)
        while not matrix.alive():
            if process.poll() is not None:
                raise RuntimeError(f"Server exited: {process.returncode}")
            time.sleep(3)
        signal.alarm(300)
        for batch in batches:
            status("warmup", batch=batch)
            matrix.run_case(work, f"warmup_b{batch}", tokens, batch, 16)
        for repeat in range(1, 4):
            order = batches if repeat % 2 else list(reversed(batches))
            for batch in order:
                label = f"b{batch}_r{repeat}"
                status("testing", label=label)
                before = preemption_snapshot()
                result = matrix.run_case(work, label, tokens, batch, 96)
                after = preemption_snapshot()
                if before is None or before != after:
                    raise RuntimeError(f"Preemption changed: {before} -> {after}")
                path = work / "requests" / (label + ".json")
                summary = summarize_requests(path)
                outputs = compare_outputs(baseline[:batch], summary.pop("output_ids"))
                if outputs["mismatches"]:
                    raise RuntimeError(f"Output mismatch: {label}: {outputs}")
                if summary["min_measured_intervals"] < 32 or not result["steady_batch_verified"]:
                    raise RuntimeError(f"Insufficient common decode window: {label}")
                records = json.loads(path.read_text())["records"]
                summary.update(label=label, batch=batch, repeat=repeat,
                               output_vs_original=outputs,
                               preemptions=dict(before=before, after=after),
                               common_window=dict(
                                   start=max(r["events"][8]["t"] for r in records),
                                   end=min(r["events"][-2]["t"] for r in records)))
                cases.append(summary)
                matrix.save(work / "summary.json", cases)
                status("case_complete", label=label,
                       p50_ms=summary["pooled_itl_p50_ms"],
                       p95_ms=summary["pooled_itl_p95_ms"],
                       min_intervals=summary["min_measured_intervals"])
        # Flush asynchronously recorded CUDA events for the last measured case.
        matrix.run_case(work, "flush", tokens, 1, 16)
    except BaseException as exc:
        matrix.save(work / "FAILED.json", dict(error=repr(exc), traceback=traceback.format_exc()))
        status("failed", error=repr(exc))
        raise
    finally:
        if process is not None:
            matrix.stop_server(process)
        signal.alarm(0)
        print("CLEANUP_COMPLETE", flush=True)
    audit_dispatches(work, cases)
    matrix.save(work / "COMPLETE.json", dict(passed=True, cases=len(cases)))
    status("complete", cases=len(cases))
    archive = work.with_suffix(".tgz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(work, arcname=work.name)
        for name in ("run_pp2_fixed_smallbatch.py", "serve_pp2_graph.py", "pp_deferred_fix/manifest.json"):
            tar.add(ROOT / name, arcname="reproduction/" + name)
    print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size,
                          sha256=hashlib.file_digest(archive.open("rb"), "sha256").hexdigest())), flush=True)


if __name__ == "__main__":
    main()
