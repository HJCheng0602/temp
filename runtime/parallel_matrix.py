"""Run real vLLM configurations, unprofiled measurements, and CUDA timelines.

Runs only its own child servers. Resume by skipping completed strategies.
All timing records retain per-token client timestamps and actual token counts.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import statistics
import subprocess
import tarfile
import threading
import time
import traceback
import urllib.error
import urllib.request

ROOT = Path("/home/admin/dsv41-vllm-20260917")
BASE = Path("/home/admin/dsv41-existing-ops-20260917/env")
OUTPUT = ROOT / "parallel_matrix"
URL = "http://127.0.0.1:8000"
MODEL = "/mnt/fuse/models/DeepSeek-V4.1-Flash-local-engram"
REQUEST_SUFFIXES = []
CORPUS = [
    "A compiler parses source code into a syntax tree, checks types, and lowers expressions into an intermediate representation. Register allocation and instruction scheduling depend on the target architecture.",
    "A distributed database partitions records across servers. Replication improves availability, while consensus determines which updates become committed. Transaction isolation defines what concurrent readers can observe.",
    "A laboratory team measures enzyme activity at several temperatures. Controls are repeated, calibration errors are recorded, and conclusions distinguish correlation from causation.",
    "An observatory records spectra from distant stars. Spectral lines reveal chemical composition, while changes in wavelength can indicate relative motion. Instrument resolution limits interpretation.",
    "A warehouse schedules deliveries using capacity constraints and arrival windows. The planner compares total travel distance, late deliveries, and idle vehicles before choosing a route.",
    "A numerical simulation solves a system of differential equations on a grid. Boundary conditions, time step selection, discretization error, and convergence tests determine the reliability of the result.",
    "An image processing pipeline corrects sensor noise and measures edges. A convolution applies the same filter at every spatial position. Resampling changes resolution but cannot recover missing information.",
    "An operating system maps virtual addresses to physical pages. Page faults, filesystem caches, memory pressure, and storage latency can all influence application startup time.",
    "A network transmits packets through switches with finite buffers. Congestion control balances throughput and latency. End-to-end measurements include queueing and application processing overhead.",
    "A software team reviews a change to an API. Tests cover invalid inputs, concurrency, and backward compatibility. Logs retain request identifiers so failures can be traced across services.",
    "A matrix multiplication computes C[i,j] as the sum of A[i,k] times B[k,j]. Tiling improves reuse, but useful tile sizes depend on dimensions, precision, alignment, and available memory.",
    "An engineer compares inference architectures using measured timelines. Independent operations may overlap, so adding individual kernel durations does not necessarily reproduce wall-clock latency.",
]
STRATEGIES = [
    dict(name="tp8_etp8", tp=8, dp=1, pp=1, ep=False),
    dict(name="tp8_ep8", tp=8, dp=1, pp=1, ep=True),
    dict(name="tp4_dp2_ep8", tp=4, dp=2, pp=1, ep=True),
    dict(name="tp2_dp4_ep8", tp=2, dp=4, pp=1, ep=True),
    dict(name="tp1_dp8_ep8", tp=1, dp=8, pp=1, ep=True),
    # Repeating Kineto start/stop in the same high-memory PP workers can crash
    # while destroying the pilot MemoryTraceLogger. Record the 32K trace as the
    # single profiler session for PP strategies.
    dict(name="tp4_pp2_ep4", tp=4, dp=1, pp=2, ep=True, profiler_pilot=False),
    # KV sources at 2/8/14/20 and candidate source at 20 must stay with consumers.
    dict(name="tp2_pp4_ep2", tp=2, dp=1, pp=4, ep=True,
         pp_layers=[8, 6, 6, 20], profiler_pilot=False,
         preflight_rejection=dict(
             status="unsupported_capacity",
             reason=(
                 "The only stage-local partition supported by the current V4.1 "
                 "KV/index sharing implementation is [8,6,6,20]. Its final "
                 "TP2/EP2 stage requires about 70.87 GiB of parameters per GPU "
                 "on a device with 71.12 GiB CUDA-visible capacity, leaving no "
                 "space for KV cache, workspaces, or activations. A real launch "
                 "failed while requesting a 540 MiB MXFP4 scale-conversion "
                 "temporary with only about 276 MiB free."
             ),
             supported_partition=[8, 6, 6, 20],
             final_stage_parameter_gib=70.87,
             cuda_visible_capacity_gib=71.12,
             observed_free_mib=276,
             failed_allocation_mib=540,
         )),
]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def status(**kwargs):
    value = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), **kwargs)
    save(OUTPUT / "status.json", value)
    print(json.dumps(value), flush=True)


def api(path, payload=None, timeout=600):
    req = urllib.request.Request(
        URL + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read()
    return json.loads(body) if body else None


def alive():
    try:
        api("/health", timeout=2)
        return True
    except (OSError, urllib.error.URLError):
        return False


def environment(strategy=None):
    env = os.environ.copy()
    cuda = ROOT / "env/lib/python3.12/site-packages/nvidia/cu13"
    env.update(
        CUDA_HOME=str(cuda),
        LD_LIBRARY_PATH=f"{BASE}/lib:{cuda}/lib:" + env.get("LD_LIBRARY_PATH", ""),
        LIBRARY_PATH=f"{cuda}/lib:" + env.get("LIBRARY_PATH", ""),
        PATH=f"{ROOT}/env/bin:{cuda}/bin:{BASE}/bin:" + env.get("PATH", ""),
        CXX=f"{BASE}/bin/x86_64-conda-linux-gnu-c++",
        CC=f"{BASE}/bin/x86_64-conda-linux-gnu-cc",
        NVCC_PREPEND_FLAGS=f"-ccbin={BASE}/bin/x86_64-conda-linux-gnu-c++",
        MAX_JOBS="8", FLASHINFER_CUDA_ARCH_LIST="12.0a", OMP_NUM_THREADS="1",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
    )
    env.pop("VLLM_PP_LAYER_PARTITION", None)
    if strategy and strategy.get("pp_layers"):
        layers = strategy["pp_layers"]
        if len(layers) != strategy["pp"] or sum(layers) != 40 or min(layers) <= 0:
            raise ValueError("Invalid V4.1 Flash pipeline partition")
        env["VLLM_PP_LAYER_PARTITION"] = ",".join(map(str, layers))
    return env


def command(strategy, directory):
    profiler = dict(
        profiler="torch", torch_profiler_dir=str(directory / "trace_staging"),
        torch_profiler_with_stack=False, torch_profiler_record_shapes=True,
        torch_profiler_use_gzip=True, torch_profiler_dump_cuda_time_total=False,
        detailed_trace_annotation=True, ignore_frontend=True,
    )
    cmd = [str(ROOT / "env/bin/vllm"), "serve", MODEL,
           "--served-model-name", "deepseek-v41-flash", "--host", "127.0.0.1",
           "--port", "8000", "--tensor-parallel-size", str(strategy["tp"]),
           "--pipeline-parallel-size", str(strategy["pp"]),
           "--data-parallel-size", str(strategy["dp"]),
           "--data-parallel-size-local", str(strategy["dp"]),
           "--distributed-executor-backend", "mp", "--block-size", "128",
           "--max-model-len", "33792", "--max-num-seqs", "32",
           "--max-num-batched-tokens", "8192", "--gpu-memory-utilization", "0.92",
           "--enforce-eager", "--disable-custom-all-reduce", "--language-model-only",
           "--tokenizer-mode", "deepseek_v41", "--no-enable-prefix-caching",
           "--profiler-config", json.dumps(profiler)]
    if strategy["ep"]:
        cmd += ["--enable-expert-parallel", "--all2all-backend", "allgather_reducescatter"]
    if strategy["dp"] > 1:
        cmd += ["--engram-config", json.dumps(dict(cpu_offload=True, embedding_across_dp=True))]
    return cmd


def stream_one(tokens, output_tokens, barrier, records, index, dp_rank=None):
    payload = dict(model="deepseek-v41-flash", prompt=tokens, temperature=0,
                   max_tokens=output_tokens, ignore_eos=True, stream=True,
                   return_token_ids=True, stream_options=dict(include_usage=True))
    headers = {"Content-Type": "application/json"}
    if dp_rank is not None:
        headers["X-data-parallel-rank"] = str(dp_rank)
    req = urllib.request.Request(URL + "/v1/completions", data=json.dumps(payload).encode(),
                                 headers=headers)
    barrier.wait()
    rec = records[index]
    rec["requested_dp_rank"] = dp_rank
    rec["start"] = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=1800) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                raw = line[6:].strip()
                if raw == b"[DONE]":
                    break
                chunk = json.loads(raw)
                if chunk.get("error"):
                    raise RuntimeError(chunk["error"])
                if chunk.get("usage"):
                    rec["usage"] = chunk["usage"]
                for choice in chunk.get("choices", []):
                    ids = choice.get("token_ids") or []
                    if ids:
                        rec["events"].append(dict(t=time.perf_counter(), token_ids=ids))
                    if choice.get("finish_reason"):
                        rec["finish_reason"] = choice["finish_reason"]
        rec["end"] = time.perf_counter()
        count = sum(len(e["token_ids"]) for e in rec["events"])
        if count != output_tokens:
            raise RuntimeError(f"Expected {output_tokens} streamed token IDs, got {count}")
    except BaseException as exc:
        rec["error"] = repr(exc)
        raise


def summarize(records):
    first = [r["events"][0]["t"] for r in records]
    last = [r["events"][-1]["t"] for r in records]
    # Only intervals with every request resident and generating count as B=N.
    common_start = max(first)
    common_end = min(last)
    itls = []
    per_request = []
    for rec in records:
        events = rec["events"]
        intervals = [b["t"] - a["t"] for a, b in zip(events[8:], events[9:])
                     if a["t"] >= common_start and b["t"] <= common_end
                     and len(b["token_ids"]) == 1]
        itls.extend(intervals)
        per_request.append(len(intervals))
    return dict(
        batch=len(records), ttft_ms=[(t-r["start"])*1000 for t,r in zip(first,records)],
        e2e_seconds=max(r["end"] for r in records)-min(r["start"] for r in records),
        common_decode_window_seconds=max(0, common_end-common_start),
        steady_intervals_per_request=per_request,
        steady_batch_verified=min(per_request) >= 16,
        median_client_itl_ms=statistics.median(itls)*1000 if itls else None,
        mean_client_itl_ms=statistics.mean(itls)*1000 if itls else None,
        scope="Client streamed-token intervals, unprofiled; excludes first eight intervals per request and non-overlapping batch portions.",
    )


def inspect_traces(directory):
    output = []
    for path in sorted(directory.glob("*.pt.trace.json.gz")):
        with gzip.open(path, "rt") as handle:
            trace = json.load(handle)
        events = trace.get("traceEvents", [])
        kernels = [e for e in events if e.get("cat") == "kernel"]
        if not kernels:
            raise RuntimeError(f"No CUDA kernel events in {path}")
        communication = [e for e in kernels if "nccl" in e.get("name", "").lower()]
        output.append(dict(file=path.name, bytes=path.stat().st_size,
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           cuda_kernels=len(kernels), nccl_kernels=len(communication),
                           devices=sorted({str(e.get("args", {}).get("device")) for e in kernels})))
        del trace, events, kernels, communication
    if len(output) != 8:
        raise RuntimeError(f"Expected eight GPU rank traces, found {len(output)} in {directory}")
    save(directory / "trace_manifest.json", output)
    return output


def run_case(directory, label, tokens, batch, output_tokens, profile=False, decode=False,
             dp_rank=None, dp_size=None):
    records = [dict(events=[]) for _ in range(batch)]
    barrier = threading.Barrier(batch)
    profiling = False
    staging = directory / "trace_staging"
    traces = directory / "timelines" / label
    try:
        if profile and not decode:
            api("/start_profile", {}, timeout=120)
            profiling = True
        with ThreadPoolExecutor(max_workers=batch) as pool:
            prompts = [tokens[:-len(REQUEST_SUFFIXES[i])] + REQUEST_SUFFIXES[i]
                       if REQUEST_SUFFIXES else tokens for i in range(batch)]
            jobs = [pool.submit(stream_one, prompt, output_tokens, barrier, records, i,
                                i % dp_size if dp_size else dp_rank)
                    for i, prompt in enumerate(prompts)]
            if profile and decode:
                deadline = time.monotonic() + 1800
                while not all(len(r["events"]) >= 8 for r in records):
                    if any(j.done() for j in jobs):
                        for j in jobs:
                            if j.done():
                                j.result()
                        raise RuntimeError("A request finished before stable decode profiling began")
                    if time.monotonic() > deadline:
                        raise TimeoutError("Waiting for all requests to reach decode")
                    time.sleep(0.05)
                api("/start_profile", {}, timeout=120)
                profiling = True
                counts = [len(r["events"]) for r in records]
                while not all(len(r["events"]) >= n + 8 for r, n in zip(records, counts)):
                    if any(j.done() for j in jobs):
                        break
                    time.sleep(0.02)
                api("/stop_profile", {}, timeout=600)
                profiling = False
            for job in jobs:
                job.result()
    finally:
        if profiling:
            api("/stop_profile", {}, timeout=600)
        save(directory / "requests" / (label + ".json"), dict(
            profiled=profile, prompt_tokens=len(tokens), batch=batch,
            requested_dp_rank=dp_rank, requested_dp_size=dp_size,
            requested_output_tokens=output_tokens, records=records))
    if profile:
        traces.mkdir(parents=True, exist_ok=True)
        for path in staging.iterdir():
            if path.is_file():
                path.replace(traces / path.name)
        result = dict(profiled=True, traces=inspect_traces(traces))
    else:
        result = summarize(records)
        for rec in records:
            if rec.get("usage", {}).get("prompt_tokens") != len(tokens):
                raise RuntimeError("Server prompt token count differs from the requested exact length")
    save(directory / "metrics" / (label + ".json"), result)
    return result


def stop_server(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)
    # A failed parent can leave its own workers behind in the same process group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    time.sleep(5)


def pack(directory):
    archive = OUTPUT / (directory.name + ".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(directory, arcname=directory.name)
    save(archive.with_suffix(".manifest.json"), dict(
        file=archive.name, bytes=archive.stat().st_size,
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest()))


def run_strategy(strategy, phase="all"):
    name = strategy["name"]
    directory = OUTPUT / name
    rejection = strategy.get("preflight_rejection")
    if rejection:
        directory.mkdir(parents=True, exist_ok=True)
        save(directory / "configuration.json", dict(
            strategy=strategy, phase=phase, checkpoint=MODEL,
            scope="Full-model capacity preflight; no timing result",
        ))
        save(directory / "UNSUPPORTED.json", dict(
            strategy=name, phase=phase, **rejection,
        ))
        pack(directory)
        status(strategy=name, phase="unsupported", reason=rejection["reason"])
        return
    if (directory / "COMPLETE.json").exists():
        if not (OUTPUT / (name + ".tar.manifest.json")).exists():
            pack(directory)
        return
    if directory.exists():
        interrupted = OUTPUT / "interrupted"
        interrupted.mkdir(exist_ok=True)
        directory.rename(interrupted / f"{name}.{time.time_ns()}")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "trace_staging").mkdir(exist_ok=True)
    evidence = directory / "reproduction"
    evidence.mkdir(exist_ok=True)
    for source in (Path(__file__), ROOT / "pip-freeze-serving.txt",
                   ROOT / "patch_sm120_pages.py", ROOT / "check_sm120_pages.py"):
        if source.is_file():
            shutil.copy2(source, evidence / source.name)
    for label, probe in (
        ("gpu", ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total", "--format=csv"]),
        ("topology", ["nvidia-smi", "topo", "-m"]),
    ):
        result = subprocess.run(probe, text=True, capture_output=True, timeout=30)
        save(evidence / (label + ".json"), dict(command=probe, returncode=result.returncode,
                                               stdout=result.stdout, stderr=result.stderr))
    if alive():
        raise RuntimeError("Port 8000 already has a live server; refusing to replace it")
    cmd = command(strategy, directory)
    single_profile_pp = strategy["pp"] > 1 and phase == "all"
    limitations = [
        "No CP implementation assumed",
        "Fixed synthetic mixed-topic workload, not production traffic",
        "Timing and profiler requests are separate",
        "DP uses Engram sharding across DP",
    ]
    if single_profile_pp:
        limitations.append(
            "vLLM PP workers support one Kineto session per server lifetime; "
            "the full decode run profiles only the representative 32K/B32 cell. "
            "The separate prefill run contains the PP 32K prefill timeline."
        )
    save(directory / "configuration.json", dict(strategy=strategy, command=cmd, phase=phase,
         checkpoint=MODEL, scope="Eager-mode full-model inference with CPU Engram offload",
         limitations=limitations))
    logpath = directory / "server.log"
    results = {}
    process = None
    try:
        with logpath.open("w") as log:
            process = subprocess.Popen(cmd, env=environment(strategy), stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        save(directory / "server_pid.json", dict(pid=process.pid))
        status(strategy=name, phase="loading", server_pid=process.pid)
        deadline = time.monotonic() + 1500
        while not alive():
            if process.poll() is not None:
                raise RuntimeError(f"Server exited with code {process.returncode}; see server.log")
            if time.monotonic() > deadline:
                raise TimeoutError("Server did not become healthy in 25 minutes")
            time.sleep(5)
        capacities = [int(v.replace(",", "")) for v in re.findall(r"GPU KV cache size: ([\d,]+) tokens", logpath.read_text())]
        save(directory / "kv_capacity.json", dict(logged_token_capacities=capacities))
        # Token IDs come from the actual server tokenizer; no special IDs guessed.
        corpus = "\n\n".join(f"Passage {i+1}: {p}" for i,p in enumerate(CORPUS))
        raw = api("/tokenize", dict(model="deepseek-v41-flash", prompt=corpus))
        unit = raw["tokens"]
        all_tokens = (unit * (32768 // len(unit) + 1))[:32768]
        REQUEST_SUFFIXES.clear()
        for i in range(32):
            suffix = f"\nTask {i+1}: Explain the main ideas in passage {i%len(CORPUS)+1} with examples. Write several complete paragraphs.\nAnswer:"
            REQUEST_SUFFIXES.append(api("/tokenize", dict(model="deepseek-v41-flash", prompt=suffix))["tokens"])
        save(directory / "workload.json", dict(corpus=corpus, unit_token_ids=unit,
             request_suffix_token_ids=REQUEST_SUFFIXES, prefix_cache=False,
             construction="Repeat corpus token IDs to exact length, replace the final suffix with a request-specific task of equal total token length"))
        assert len(all_tokens) >= 32768
        status(strategy=name, phase="warmup")
        run_case(directory, "warmup_2k", all_tokens[:2048], 1, 32)
        if strategy["dp"] > 1:
            for rank in range(strategy["dp"]):
                label = f"warmup_32k_dp{rank}"
                status(strategy=name, phase=label)
                run_case(directory, label, all_tokens, 1, 1, dp_rank=rank)
        if strategy.get("profiler_pilot", True):
            status(strategy=name, phase="profiler_pilot")
            results["profiler_pilot"] = run_case(
                directory, "profiler_pilot", all_tokens[:512], 1, 1,
                profile=True,
            )
        for repeat in range(2):
            label = f"prefill_32k_timing_{repeat}"
            status(strategy=name, phase=label)
            results[label] = run_case(directory, label, all_tokens[:32768], 1, 1, dp_rank=0)
        if single_profile_pp:
            results["prefill_32k_timeline"] = dict(
                status="recorded_in_separate_prefill_run",
                reason="PP workers support one Kineto session per server lifetime",
                expected_archive="parallel_matrix_prefill/tp4_pp2_ep4.tar.gz",
            )
            save(directory / "metrics" / "prefill_32k_timeline.json",
                 results["prefill_32k_timeline"])
        else:
            status(strategy=name, phase="prefill_32k_timeline")
            results["prefill_32k_timeline"] = run_case(
                directory, "prefill_32k", all_tokens[:32768], 1, 1,
                profile=True, dp_rank=0,
            )
        if phase == "prefill":
            save(directory / "COMPLETE.json", dict(strategy=name, phase=phase, results=results))
            status(strategy=name, phase="prefill_complete")
            return
        for context in (2048, 32768):
            for batch in (1, 8, 32):
                label = f"decode_ctx{context}_b{batch}"
                # A necessary capacity bound: do not mislabel queued concurrency as resident B.
                local_requests = (batch + strategy["dp"] - 1) // strategy["dp"]
                # Early requests decode while later requests are still prefilling.
                # Leave at least 64 generation steps after the estimated prefill waves.
                output_tokens = max(128, (local_requests * context + 8191) // 8192 + 64)
                if capacities and local_requests * (context + output_tokens) > min(capacities):
                    results[label] = dict(status="insufficient_logged_kv_capacity_for_resident_batch",
                                         requests_per_dp_rank=local_requests, required_tokens=local_requests*(context+output_tokens),
                                         capacity_tokens=min(capacities))
                    save(directory / "metrics" / (label + "_capacity.json"), results[label])
                    status(strategy=name, phase=label, skipped="resident KV capacity")
                    continue
                status(strategy=name, phase=label + "_timing")
                results[label] = run_case(directory, label + "_timing", all_tokens[:context], batch, output_tokens, dp_size=strategy["dp"])
                if not results[label]["steady_batch_verified"]:
                    raise RuntimeError(f"Insufficient common decode interval for {label}; inspect request records")
                if single_profile_pp and (context, batch) != (32768, 32):
                    results[label + "_timeline"] = dict(
                        status="represented_by_decode_ctx32768_b32_timeline",
                        reason="PP workers support one Kineto session per server lifetime",
                    )
                    save(directory / "metrics" / (label + "_timeline.json"),
                         results[label + "_timeline"])
                    continue
                status(strategy=name, phase=label + "_timeline")
                results[label + "_timeline"] = run_case(
                    directory, label + "_timeline", all_tokens[:context], batch,
                    output_tokens, profile=True, decode=True,
                    dp_size=strategy["dp"],
                )
        save(directory / "COMPLETE.json", dict(strategy=name, results=results))
        status(strategy=name, phase="complete")
    except Exception:
        error = traceback.format_exc()
        save(directory / "FAILED.json", dict(strategy=name, error=error, partial_results=results))
        status(strategy=name, phase="failed", error=error)
    finally:
        if process is not None:
            stop_server(process)
        pack(directory)


def main():
    global OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--phase", choices=("all", "prefill"), default="all")
    args = parser.parse_args()
    if args.phase == "prefill":
        OUTPUT = ROOT / "parallel_matrix_prefill"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected = [s for s in STRATEGIES if not args.only or s["name"] in args.only]
    for strategy in selected:
        run_strategy(strategy, phase=args.phase)
    status(phase="matrix_finished", strategies=[s["name"] for s in selected])


if __name__ == "__main__":
    main()
