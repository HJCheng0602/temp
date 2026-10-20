"""Run the fixed-cohort PP2/TP4/EP4 fast-GEMM serving trial."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import time
import traceback

import parallel_matrix as matrix
from largebatch_pp2 import preemption_snapshot
from run_pp2_graph_comparison import interrupt
from run_pp2_wholebatch import audit_rank_paths, read_waves
from run_pp8_batch_sweep import percentile, summarize_requests


ROOT = matrix.ROOT
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
OUTPUT_TOKENS = 96
WARMUP_TOKENS = 16


def _distribution(values: list[float]) -> dict:
    if not values:
        raise RuntimeError("No values available for distribution")
    return dict(p50_ms=statistics.median(values), p95_ms=percentile(values, .95),
                samples=len(values))


def _output_ids(path: Path) -> list[list[int]]:
    records = json.loads(path.read_text())["records"]
    output = []
    for record in records:
        ids = [token for event in record.get("events", []) for token in event.get("token_ids", [])]
        if len(ids) != OUTPUT_TOKENS:
            raise RuntimeError(
                f"Expected {OUTPUT_TOKENS} output tokens, got {len(ids)} in {path}"
            )
        output.append(ids)
    return output


def _prefix_comparison(expected: list[list[int]], actual: list[list[int]]) -> dict:
    if len(expected) != len(actual):
        return dict(matched_requests=0, total_requests=len(actual), mismatches=[],
                    prefix_tokens=0, prefix_agreement=0.0,
                    error="different request counts")
    mismatches = []
    prefix_lengths = []
    for index, (left, right) in enumerate(zip(expected, actual)):
        prefix = 0
        for first, second in zip(left, right):
            if first != second:
                break
            prefix += 1
        prefix_lengths.append(prefix)
        if prefix != min(len(left), len(right)) or len(left) != len(right):
            mismatches.append(dict(request=index, first_difference=prefix,
                                   expected_tokens=len(left), actual_tokens=len(right)))
    total = sum(len(row) for row in actual)
    return dict(matched_requests=len(actual) - len(mismatches),
                total_requests=len(actual), mismatches=mismatches,
                prefix_tokens=sum(prefix_lengths),
                prefix_agreement=(sum(prefix_lengths) / total if total else 1.0))


def _verify_manifest() -> None:
    runtime = Path(importlib.metadata.distribution("vllm").locate_file("vllm"))
    manifest = json.loads((ROOT / "pp_deferred_fix/manifest.json").read_text())
    for relative, hashes in manifest["files"].items():
        path = runtime / relative
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != hashes["after"]:
            raise RuntimeError(f"Runtime differs from verified fix: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "dense", "all"), default="baseline")
    parser.add_argument("--timing-dir", type=Path)
    parser.add_argument("--fixed-batch-control", type=Path)
    parser.add_argument("--baseline", type=Path,
                        help="Optional prior trial directory or timing.json for diagnostics")
    args = parser.parse_args()

    work = args.directory
    work.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGALRM, interrupt)
    matrix.URL = "http://127.0.0.1:8024"
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", 8024)) == 0:
            raise RuntimeError("Port occupied")
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if active:
        raise RuntimeError(f"GPUs active: {active}")
    _verify_manifest()

    workload = ROOT / "pp2_graph_experiments/probe_deferred_20260926a/workload.json"
    tokens = json.loads(workload.read_text())["prompt_tokens"]
    matrix.REQUEST_SUFFIXES.clear()
    timing_dir = args.timing_dir or (work / "rank_timings")
    control = args.fixed_batch_control or (work / "control.json")
    matrix.save(work / "workload.json", dict(prompt_tokens=tokens))
    matrix.save(work / "configuration.json", dict(
        mode=args.mode, batches=list(BATCHES), repeats=3,
        prompt_tokens=len(tokens), output_tokens=OUTPUT_TOKENS,
        warmup_output_tokens=WARMUP_TOKENS, pp=2, tp=4, ep=4,
        pp_layers=[20, 20], fixed_cohort=True, cudagraph_mode="FULL_DECODE_ONLY",
        launcher=str(Path(__file__).with_name("serve_trial.py")),
        manifest=str(ROOT / "pp_deferred_fix/manifest.json"),
        startup_timeout_seconds=1800, test_timeout_seconds=1800,
        output_comparison="diagnostic prefix agreement; precision differences do not fail the trial"))

    old_baselines: dict[int, list[list[int]]] = {}
    if args.baseline:
        baseline_path = args.baseline
        if baseline_path.is_dir():
            request_dir = baseline_path / "requests"
            for batch in BATCHES:
                candidates = sorted(request_dir.glob(f"b{batch}_r*.json"))
                if candidates:
                    old_baselines[batch] = _output_ids(candidates[0])
        elif baseline_path.exists():
            old_baselines = {batch: _output_ids(baseline_path)[:batch] for batch in BATCHES}

    cases = []
    fresh_baselines: dict[int, list[list[int]]] = {}
    process = None

    def status(phase: str, **extra) -> None:
        value = dict(phase=phase, mode=args.mode, time=time.time(), **extra)
        matrix.save(work / "status.json", value)
        print(json.dumps(value), flush=True)

    def run(label: str, batch: int, output_tokens: int):
        wave_path = work / "waves" / (label + ".jsonl")
        matrix.save(control, dict(batch=batch, size=batch, log=str(wave_path)))
        result = matrix.run_case(work, label, tokens, batch, output_tokens)
        read_waves(wave_path, batch, output_tokens)
        return result

    try:
        signal.alarm(1800)
        with (work / "server.log").open("w") as log:
            command = [str(ROOT / "env/bin/python"),
                       str(Path(__file__).with_name("serve_trial.py")),
                       "--mode", args.mode, "--timing-dir", str(timing_dir),
                       "--fixed-batch-control", str(control)]
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        matrix.save(work / "server_pid.json", dict(pid=process.pid, command=command))
        status("starting", server_pid=process.pid)
        deadline = time.monotonic() + 1800
        while not matrix.alive():
            if process.poll() is not None:
                raise RuntimeError(f"Server exited: {process.returncode}")
            if time.monotonic() >= deadline:
                raise TimeoutError("Server startup timed out after 1800 seconds")
            time.sleep(3)

        signal.alarm(1800)
        for batch in BATCHES:
            status("warmup", batch=batch)
            run(f"warmup_b{batch}", batch, WARMUP_TOKENS)
        for repeat in range(1, 4):
            order = BATCHES if repeat % 2 else tuple(reversed(BATCHES))
            for batch in order:
                label = f"b{batch}_r{repeat}"
                status("testing", label=label)
                before = preemption_snapshot()
                result = run(label, batch, OUTPUT_TOKENS)
                after = preemption_snapshot()
                if before is None or before != after:
                    raise RuntimeError(f"Preemption changed: {before} -> {after}")
                request_path = work / "requests" / (label + ".json")
                actual = _output_ids(request_path)
                if len(actual) != batch or not result.get("steady_batch_verified"):
                    raise RuntimeError(f"Invalid output batch or steady window: {label}")
                summary = summarize_requests(request_path)
                summary.pop("output_ids", None)
                if summary["min_measured_intervals"] < 32:
                    raise RuntimeError(f"Insufficient common decode window: {label}")
                records = json.loads(request_path.read_text())["records"]
                common_window = dict(
                    start=max(record["events"][8]["t"] for record in records),
                    end=min(record["events"][-2]["t"] for record in records),
                )
                waves, _ = read_waves(
                    work / "waves" / (label + ".jsonl"), batch, OUTPUT_TOKENS
                )
                chosen = [
                    wave for wave in waves
                    if common_window["start"] <= wave["wave_start"]
                    <= wave["wave_end"] <= common_window["end"]
                ]
                if len(chosen) < 32:
                    raise RuntimeError(f"Insufficient common whole-batch steps: {label}")
                reference = fresh_baselines.setdefault(batch, actual)
                checks = [_prefix_comparison(reference, actual)]
                if args.mode != "baseline" and batch in old_baselines:
                    checks.append(_prefix_comparison(old_baselines[batch], actual))
                entry = dict(summary, label=label, batch=batch, repeat=repeat,
                             output_checks=checks,
                             preemptions=dict(before=before, after=after),
                             common_window=common_window,
                             whole_batch_step=_distribution([
                                 (wave["wave_end"] - wave["wave_start"]) * 1000
                                 for wave in chosen
                             ]),
                             whole_batch_output_period=_distribution([
                                 (later["wave_end"] - earlier["wave_end"]) * 1000
                                 for earlier, later in zip(chosen, chosen[1:])
                             ]))
                cases.append(entry)
                matrix.save(work / "summary.json", cases)
                status("case_complete", label=label, batch=batch,
                       pooled_itl_p50_ms=entry["pooled_itl_p50_ms"])
        run("flush", 1, WARMUP_TOKENS)
    except BaseException as exc:
        matrix.save(work / "FAILED.json", dict(error=repr(exc), traceback=traceback.format_exc()))
        status("failed", error=repr(exc))
        raise
    finally:
        if process is not None:
            matrix.stop_server(process)
        signal.alarm(0)
        print("CLEANUP_COMPLETE", flush=True)

    rank_link = work / "rank_timings"
    if timing_dir != rank_link:
        if rank_link.exists() or rank_link.is_symlink():
            rank_link.unlink()
        rank_link.symlink_to(timing_dir, target_is_directory=True)
    audit_rank_paths(work, cases)
    matrix.save(work / "COMPLETE.json", dict(passed=True, mode=args.mode,
                                               cases=len(cases), batches=list(BATCHES)))
    status("complete", cases=len(cases))


if __name__ == "__main__":
    main()
