"""Measure batch-one TTFT for long-context PP2/TP4/EP4 prefill."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import time
import traceback
import urllib.error
import urllib.request

import parallel_matrix as matrix


ROOT = matrix.ROOT
MODEL = matrix.MODEL
PORT = 8025
URL = f"http://127.0.0.1:{PORT}"
LENGTHS = (2048, 4096, 8192, 16384, 32768, 65536, 131072)
OUTPUT_TOKENS = 1


def interrupt(_signum, _frame) -> None:
    raise KeyboardInterrupt


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def api(path: str, payload=None, timeout: int = 1800):
    request = urllib.request.Request(
        URL + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    return json.loads(body) if body else None


def alive() -> bool:
    try:
        api("/health", timeout=2)
        return True
    except (OSError, urllib.error.URLError):
        return False


def verify_manifest() -> None:
    runtime = Path(importlib.metadata.distribution("vllm").locate_file("vllm"))
    manifest_path = ROOT / "pp_deferred_fix/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for relative, hashes in manifest["files"].items():
        path = runtime / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != hashes["after"]:
            raise RuntimeError(f"Runtime differs from verified PP fix: {relative}")


def build_command(work: Path, chunk_tokens: int) -> list[str]:
    strategy = dict(name="pp2_tp4_ep4", pp=2, tp=4, dp=1, ep=True,
                    pp_layers=[20, 20])
    command = matrix.command(strategy, work)
    profiler_index = command.index("--profiler-config")
    del command[profiler_index:profiler_index + 2]
    overrides = {
        "--port": PORT,
        "--max-model-len": max(LENGTHS) + OUTPUT_TOKENS,
        "--max-num-seqs": 1,
        "--max-num-batched-tokens": chunk_tokens,
        "--gpu-memory-utilization": 0.92,
    }
    for option, value in overrides.items():
        command[command.index(option) + 1] = str(value)
    return command


def build_environment(timing_dir: Path) -> dict[str, str]:
    strategy = dict(name="pp2_tp4_ep4", pp=2, tp=4, dp=1, ep=True,
                    pp_layers=[20, 20])
    environment = matrix.environment(strategy)
    environment.update(
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        DSV41_PP_TIMING_DIR=str(timing_dir),
        DSV41_PP_TIMING_PER_RANK="1",
        PYTHONPATH=os.pathsep.join((str(ROOT), environment.get("PYTHONPATH", ""))),
    )
    for key in list(environment):
        if key.startswith(("DSV41_FAST_", "DSV41_WAVE_", "DSV41_GRAPH_PROBE_")):
            environment.pop(key)
    return environment


def stream_one(tokens: list[int], label: str) -> dict:
    payload = dict(
        model="deepseek-v41-flash",
        prompt=tokens,
        temperature=0,
        max_tokens=OUTPUT_TOKENS,
        ignore_eos=True,
        stream=True,
        return_token_ids=True,
        stream_options=dict(include_usage=True),
    )
    request = urllib.request.Request(
        URL + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    record = dict(label=label, prompt_tokens=len(tokens), events=[])
    record["start"] = time.perf_counter()
    with urllib.request.urlopen(request, timeout=1800) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            raw = line[6:].strip()
            if raw == b"[DONE]":
                break
            chunk = json.loads(raw)
            if chunk.get("error"):
                raise RuntimeError(chunk["error"])
            if chunk.get("id"):
                record["request_id"] = chunk["id"]
            if chunk.get("usage"):
                record["usage"] = chunk["usage"]
            for choice in chunk.get("choices", []):
                token_ids = choice.get("token_ids") or []
                if token_ids:
                    record["events"].append(
                        dict(t=time.perf_counter(), token_ids=token_ids)
                    )
    record["end"] = time.perf_counter()
    generated = sum(len(event["token_ids"]) for event in record["events"])
    if generated != OUTPUT_TOKENS:
        raise RuntimeError(f"{label}: expected one output token, got {generated}")
    if record.get("usage", {}).get("prompt_tokens") != len(tokens):
        raise RuntimeError(f"{label}: server did not confirm exact prompt length")
    record["ttft_ms"] = (record["events"][0]["t"] - record["start"]) * 1000
    record["e2e_ms"] = (record["end"] - record["start"]) * 1000
    return record


def read_rank_timings(timing_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(timing_dir.glob("rank_*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                row["file"] = path.name
                rows.append(row)
    return rows


def summarize(records: list[dict], rank_rows: list[dict], chunk_tokens: int) -> dict:
    by_length = {}
    for length in LENGTHS:
        measured = [row for row in records
                    if row["prompt_tokens"] == length and row["label"].startswith("measure_")]
        values = [row["ttft_ms"] for row in measured]
        request_ids = [row["request_id"] for row in measured]

        def belongs(internal_id: str, external_id: str) -> bool:
            return internal_id == external_id or internal_id.startswith(external_id + "-")

        timings = [
            row for row in rank_rows
            if any(belongs(internal_id, external_id)
                   for internal_id in row.get("request_ids", [])
                   for external_id in request_ids)
        ]
        expected_chunks = (length + chunk_tokens - 1) // chunk_tokens
        rank_counts = {}
        gpu_sums = {}
        for rank in range(8):
            per_request_rows = []
            per_request_gpu = []
            for request_id in request_ids:
                selected = [
                    row for row in timings
                    if row.get("global_rank") == rank and row.get("prefill")
                    and any(belongs(internal_id, request_id)
                            for internal_id in row.get("request_ids", []))
                ]
                per_request_rows.append(len(selected))
                per_request_gpu.append(sum(row.get("gpu_ms", 0.0) for row in selected))
            rank_counts[str(rank)] = per_request_rows
            gpu_sums[str(rank)] = per_request_gpu
        by_length[str(length)] = dict(
            repeats=len(values),
            ttft_ms=values,
            median_ttft_ms=statistics.median(values),
            min_ttft_ms=min(values),
            max_ttft_ms=max(values),
            mean_ttft_ms=statistics.mean(values),
            median_us_per_prompt_token=statistics.median(values) * 1000 / length,
            expected_chunks_per_request=expected_chunks,
            matched_prefill_rows_per_request_by_rank=rank_counts,
            prefill_gpu_ms_per_request_by_rank=gpu_sums,
        )
    return dict(lengths=list(LENGTHS), chunk_tokens=chunk_tokens,
                metric="client request start to first streamed token", results=by_length)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--chunk-tokens", default=8192, type=int)
    parser.add_argument("--repeats", default=3, type=int)
    args = parser.parse_args()
    work = args.directory.resolve()
    work.mkdir(parents=True, exist_ok=False)
    timing_dir = work / "rank_timings"
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", PORT)) == 0:
            raise RuntimeError(f"Port {PORT} is already occupied")
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if active:
        raise RuntimeError(f"GPUs already active: {active}")
    verify_manifest()
    command = build_command(work, args.chunk_tokens)
    save(work / "configuration.json", dict(
        strategy="PP2 TP4 EP4", pp_layers=[20, 20], batch=1,
        lengths=list(LENGTHS), output_tokens=OUTPUT_TOKENS,
        chunk_tokens=args.chunk_tokens, repeats=args.repeats,
        warmup="one request per length", command=command,
        metric="client request start to first streamed token",
        prefix_cache=False, cuda_graph=False,
    ))
    process = None
    records = []

    def status(phase: str, **extra) -> None:
        value = dict(phase=phase, time=time.time(), **extra)
        save(work / "status.json", value)
        print(json.dumps(value), flush=True)

    try:
        with (work / "server.log").open("w") as log:
            process = subprocess.Popen(
                command, env=build_environment(timing_dir), stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        save(work / "server_pid.json", dict(pid=process.pid))
        status("loading", pid=process.pid)
        deadline = time.monotonic() + 1800
        while not alive():
            if process.poll() is not None:
                raise RuntimeError(f"Server exited with code {process.returncode}")
            if time.monotonic() > deadline:
                raise TimeoutError("Server startup exceeded 30 minutes")
            time.sleep(3)

        corpus = "\n\n".join(
            f"Passage {index + 1}: {text}" for index, text in enumerate(matrix.CORPUS)
        )
        unit = api("/tokenize", dict(model="deepseek-v41-flash", prompt=corpus))["tokens"]
        all_tokens = (unit * (max(LENGTHS) // len(unit) + 1))[:max(LENGTHS)]
        save(work / "workload.json", dict(
            corpus=corpus, unit_token_ids=unit, exact_lengths=list(LENGTHS),
            construction="Repeat tokenizer-produced corpus token IDs to exact length",
        ))

        for length in LENGTHS:
            label = f"warmup_{length}"
            status("warmup", length=length)
            records.append(stream_one(all_tokens[:length], label))
            save(work / "requests.json", records)

        for repeat in range(1, args.repeats + 1):
            order = LENGTHS if repeat % 2 else tuple(reversed(LENGTHS))
            for length in order:
                label = f"measure_{length}_r{repeat}"
                status("measure", label=label, length=length, repeat=repeat)
                records.append(stream_one(all_tokens[:length], label))
                save(work / "requests.json", records)

        status("flush")
        records.append(stream_one(all_tokens[:2048], "flush"))
        save(work / "requests.json", records)
        time.sleep(1)
    except BaseException as error:
        save(work / "FAILED.json", dict(error=repr(error), traceback=traceback.format_exc()))
        status("failed", error=repr(error))
        raise
    finally:
        if process is not None:
            matrix.stop_server(process)

    rank_rows = read_rank_timings(timing_dir)
    summary = summarize(records, rank_rows, args.chunk_tokens)
    save(work / "summary.json", summary)
    save(work / "COMPLETE.json", dict(passed=True, cases=len(LENGTHS) * args.repeats,
                                       warmups=len(LENGTHS), lengths=list(LENGTHS)))
    status("complete", cases=len(LENGTHS) * args.repeats)


if __name__ == "__main__":
    main()
