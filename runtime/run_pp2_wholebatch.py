"""Validate and measure one unsplit PP2 decode batch per token step."""

import argparse
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
from run_pp8_batch_sweep import percentile, summarize_requests


ROOT = matrix.ROOT


def distribution(values):
    return dict(p50_ms=statistics.median(values), p95_ms=percentile(values, .95),
                samples=len(values))


def read_waves(path, batch, output_tokens):
    waves = [json.loads(line) for line in path.read_text().splitlines()]
    if len(waves) != output_tokens - 1:
        raise RuntimeError(f"Expected {output_tokens - 1} whole-batch steps: {path}")
    request_set = None
    for ordinal, wave in enumerate(waves, 2):
        if wave["output_token_ordinal"] != ordinal or len(wave["dispatches"]) != 1:
            raise RuntimeError("Nonconsecutive or split decode step")
        dispatch, = wave["dispatches"]
        tokens = dispatch["num_scheduled_tokens"]
        if len(tokens) != batch or any(n != 1 for n in tokens.values()):
            raise RuntimeError("Actual decode batch differs from requested batch")
        if set(tokens) != set(dispatch["request_ids"]):
            raise RuntimeError("Inconsistent request set")
        if request_set is None:
            request_set = set(tokens)
        elif request_set != set(tokens):
            raise RuntimeError("Requests changed during the decode run")
    return waves, request_set


def audit_rank_paths(work, cases):
    ranks = {}
    for rank in range(8):
        path, = (work / "rank_timings").glob(f"rank_{rank}_*.jsonl")
        ranks[rank] = [json.loads(line) for line in path.read_text().splitlines()]
    for case in cases:
        waves, ids = read_waves(work / "waves" / (case["label"] + ".jsonl"), case["batch"], 96)
        by_rank = {}
        for rank, rows in ranks.items():
            selected = [r for r in rows if not r["prefill"] and set(r["request_ids"]) == ids]
            if len(selected) != len(waves):
                raise RuntimeError(f"Missing/extra stage steps: {case['label']} rank {rank}: {len(selected)}")
            if any(r["n"] != case["batch"] or r["requests"] != case["batch"]
                   or r["padded_tokens"] != case["batch"] or r["graph_mode"] != "FULL"
                   for r in selected):
                raise RuntimeError(f"Wrong stage batch or Graph: {case['label']} rank {rank}")
            by_rank[rank] = selected
        measurements = []
        lo, hi = case["common_window"]["start"], case["common_window"]["end"]
        for index, wave in enumerate(waves):
            if not lo <= wave["wave_start"] <= wave["wave_end"] <= hi:
                continue
            rows = [by_rank[rank][index] for rank in range(8)]
            if any(not wave["wave_start"] <= r["cpu_start"] <= wave["wave_end"] for r in rows):
                raise RuntimeError("Stage and scheduler ordinal do not align")
            start = min(r["cpu_start"] for r in rows[:4])
            measurements.append(dict(
                output_token_ordinal=wave["output_token_ordinal"],
                stage0_submit_to_tokens_ready_ms=(wave["wave_end"] - start) * 1000,
                scheduler_step_ms=(wave["wave_end"] - wave["wave_start"]) * 1000,
                stage0_gpu_max_ms=max(r["gpu_ms"] for r in rows[:4]),
                stage1_gpu_max_ms=max(r["gpu_ms"] for r in rows[4:]),
                rank_sequences=[r["sequence"] for r in rows]))
        if len(measurements) < 32:
            raise RuntimeError("Insufficient matched full-batch steps")
        case["path"] = {key: distribution([m[key] for m in measurements]) for key in (
            "stage0_submit_to_tokens_ready_ms", "scheduler_step_ms", "stage0_gpu_max_ms", "stage1_gpu_max_ms")}
        case["dispatch_verification"] = dict(ranks=8, steps_per_rank=len(waves),
                                              actual_tokens=case["batch"], padded_tokens=case["batch"],
                                              graph_mode="FULL", groups_per_step=1)
        matrix.save(work / "matched_steps" / (case["label"] + ".json"), measurements)
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
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if active:
        raise RuntimeError(f"GPUs active: {active}")
    batches = [1, 2, 4, 8, 16, 32]
    tokens = json.loads((ROOT / "pp2_graph_experiments/probe_deferred_20260926a/workload.json").read_text())["prompt_tokens"]
    baseline = summarize_requests(ROOT / "pp2_graph_experiments/probe_graph_20260926a/graph_a/requests/timing.json")["output_ids"]
    matrix.REQUEST_SUFFIXES.clear()
    control = work / "control.json"
    matrix.save(work / "configuration.json", dict(
        batches=batches, repeats=3, prompt_tokens=len(tokens), output_tokens=96,
        pp=2, tp=4, ep=4, pp_layers=[20, 20], full_profiler=False,
        scheduler="pp2_fixed_batch_scheduler.FixedBatchScheduler",
        scheduler_semantics="Prefill barrier; one unsplit cohort; await all sampled outputs before next decode step",
        primary_boundary="Host stage0 forward submission to all sampled tokens available at engine; includes handoff, sampling and output copy",
        other_boundaries=["scheduler submission to tokens ready", "client steady ITL", "per-GPU local forward CUDA events"],
        cross_gpu_cuda_event_subtraction=False, added_cuda_synchronizations=False))
    matrix.save(work / "workload.json", dict(prompt_tokens=tokens))
    cases = []

    def status(phase, **extra):
        value = dict(phase=phase, time=time.time(), **extra)
        matrix.save(work / "status.json", value)
        print(json.dumps(value), flush=True)

    def run(label, batch, output_tokens):
        wave_path = work / "waves" / (label + ".jsonl")
        matrix.save(control, dict(batch=batch, size=batch, log=str(wave_path)))
        result = matrix.run_case(work, label, tokens, batch, output_tokens)
        waves, ids = read_waves(wave_path, batch, output_tokens)
        return result, waves, ids

    process = None
    try:
        signal.alarm(1200)
        with (work / "server.log").open("w") as log:
            process = subprocess.Popen(
                [str(ROOT / "env/bin/python"), str(ROOT / "serve_pp2_graph.py"),
                 "--timing-dir", str(work / "rank_timings"), "--fixed-batch-control", str(control)],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        matrix.save(work / "server_pid.json", dict(pid=process.pid))
        status("starting", server_pid=process.pid)
        while not matrix.alive():
            if process.poll() is not None:
                raise RuntimeError(f"Server exited: {process.returncode}")
            time.sleep(3)
        signal.alarm(360)
        for batch in batches:
            status("warmup", batch=batch)
            run(f"warmup_b{batch}", batch, 16)
        for repeat in range(1, 4):
            for batch in (batches if repeat % 2 else list(reversed(batches))):
                label = f"b{batch}_r{repeat}"
                status("testing", label=label)
                before = preemption_snapshot()
                result, waves, ids = run(label, batch, 96)
                after = preemption_snapshot()
                if before is None or before != after:
                    raise RuntimeError(f"Preemption changed: {before} -> {after}")
                path = work / "requests" / (label + ".json")
                summary = summarize_requests(path)
                outputs = compare_outputs(baseline[:batch], summary.pop("output_ids"))
                if outputs["mismatches"] or summary["min_measured_intervals"] < 32 or not result["steady_batch_verified"]:
                    raise RuntimeError(f"Invalid output or steady batch: {label}")
                records = json.loads(path.read_text())["records"]
                lo = max(r["events"][8]["t"] for r in records)
                hi = min(r["events"][-2]["t"] for r in records)
                chosen = [w for w in waves if lo <= w["wave_start"] <= w["wave_end"] <= hi]
                if len(chosen) < 32:
                    raise RuntimeError("Insufficient common whole-batch steps")
                summary.update(label=label, batch=batch, repeat=repeat, output_vs_original=outputs,
                               preemptions=dict(before=before, after=after), common_window=dict(start=lo, end=hi),
                               whole_batch_step=distribution([(w["wave_end"]-w["wave_start"])*1000 for w in chosen]),
                               whole_batch_output_period=distribution([(b["wave_end"]-a["wave_end"])*1000
                                                                       for a, b in zip(chosen, chosen[1:])]))
                cases.append(summary)
                matrix.save(work / "summary.json", cases)
                status("case_complete", label=label, step_ms=summary["whole_batch_step"]["p50_ms"],
                       client_itl_ms=summary["pooled_itl_p50_ms"], actual_batch=batch)
        run("flush", 1, 16)
    except BaseException as exc:
        matrix.save(work / "FAILED.json", dict(error=repr(exc), traceback=traceback.format_exc()))
        status("failed", error=repr(exc))
        raise
    finally:
        if process is not None:
            matrix.stop_server(process)
        signal.alarm(0)
        print("CLEANUP_COMPLETE", flush=True)
    audit_rank_paths(work, cases)
    matrix.save(work / "COMPLETE.json", dict(passed=True, cases=len(cases)))
    status("complete", cases=len(cases))
    archive = work.with_suffix(".tgz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(work, arcname=work.name)
        for name in ("run_pp2_wholebatch.py", "serve_pp2_graph.py", "pp8_wave_scheduler.py",
                     "pp2_fixed_batch_scheduler.py", "pp8_stage_timer.py", "pp_deferred_fix/manifest.json"):
            tar.add(ROOT / name, arcname="reproduction/" + name)
    print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size,
                          sha256=hashlib.file_digest(archive.open("rb"), "sha256").hexdigest())), flush=True)


if __name__ == "__main__":
    main()
