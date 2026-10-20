"""Large resident-batch sweep for the validated PP2/TP4/EP4 deployment.

This driver deliberately keeps the existing ``parallel_matrix`` implementation
as the execution primitive.  It changes only the requested batch/context set,
uses one eager server per sweep, and records capacity skips explicitly.
Use ``--dry-plan`` on a machine without GPUs to inspect the planned cases.
"""

import argparse
import gzip
import json
import math
from pathlib import Path
import re
import subprocess
import time
import traceback
import urllib.request

import parallel_matrix as matrix


DEFAULT_BATCHES = (16, 32, 64, 128, 256)
DEFAULT_CONTEXTS = (2048, 32768)
STRATEGY = next(s for s in matrix.STRATEGIES if s["name"] == "tp4_pp2_ep4")


def parse_ints(value):
    values = tuple(int(item) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def output_tokens(context, batch):
    """Leave enough generation time for all requests to enter the batch.

    The admission estimate follows the existing driver's prefill-wave model,
    with a larger fixed tail because 192 tokens was short for some 32K cases.
    """
    waves = math.ceil(batch * context / 8192)
    return max(256, waves + 256)


def make_command(output_dir, max_batch, max_model_len):
    command = matrix.command(STRATEGY, output_dir)
    index = command.index("--max-num-seqs") + 1
    command[index] = str(max_batch)
    index = command.index("--max-model-len") + 1
    command[index] = str(max_model_len)
    return command


def capacity_from_log(log_path):
    values = [int(value.replace(",", "")) for value in re.findall(
        r"GPU KV cache size: ([\d,]+) tokens", log_path.read_text()
    )]
    return values


def case_plan(context, batch, capacities, safety_ratio, capacity_mode):
    generated = output_tokens(context, batch)
    required = batch * (context + generated)
    raw_capacity = min(capacities) if capacities else None
    usable = math.floor(raw_capacity * safety_ratio) if raw_capacity else None
    reason = None
    if (capacity_mode == "conservative" and usable is not None
            and required > usable):
        reason = "insufficient_conservative_resident_kv_capacity"
    return {
        "context_tokens": context,
        "requested_batch": batch,
        "requested_output_tokens": generated,
        "required_resident_tokens_upper_bound": required,
        "logged_gpu_kv_capacity_tokens": raw_capacity,
        "conservative_usable_capacity_tokens": usable,
        "capacity_safety_ratio": safety_ratio,
        "capacity_mode": capacity_mode,
        "status": "capacity_skip" if reason else "planned",
        **({"reason": reason} if reason else {}),
    }


def preemption_snapshot():
    """Read the server's cumulative Prometheus preemption counter."""
    try:
        request = urllib.request.Request(matrix.URL + "/metrics")
        with urllib.request.urlopen(request, timeout=5) as response:
            text = response.read().decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return None
    total = 0.0
    found = False
    pattern = re.compile(
        r"(?:^|\s)(?:[A-Za-z0-9_:]+)?preemptions?_total"
        r"(?:\{[^}]*\})?\s+([0-9.eE+-]+)", re.IGNORECASE
    )
    for line in text.splitlines():
        if line.startswith("#") or "preempt" not in line.lower():
            continue
        match = pattern.search(line)
        if match:
            total += float(match.group(1))
            found = True
    return total if found else None


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def trace_microbatch_histogram(trace_dir):
    """Extract explicit batch fields if the profiler emitted any.

    Kernel timing alone cannot identify scheduler microbatches, so an absent
    field is represented as ``None`` rather than inferred from request count.
    """
    histogram = {}
    keys = {"batch_size", "batchsize", "num_seqs", "microbatch_size"}
    for path in sorted(Path(trace_dir).glob("*.pt.trace.json.gz")):
        with gzip.open(path, "rt") as handle:
            trace = json.load(handle)
        for event in trace.get("traceEvents", []):
            args = event.get("args", {})
            for key, value in args.items():
                if key.lower() in keys and isinstance(value, int) and value > 0:
                    histogram[str(value)] = histogram.get(str(value), 0) + 1
    return histogram or None


def installed_versions(env_root):
    python = str(Path(env_root) / "env/bin/python")
    script = (
        "import importlib.metadata as m; "
        "installed={d.metadata['Name'].lower(): d.version for d in m.distributions()}; "
        "names=('vllm','torch','flashinfer-python','nvidia-cuda-nvcc',"
        "'nvidia-cuda-cccl','nvidia-nvvm','nvidia-cuda-crt'); "
        "print({n: installed.get(n.lower()) for n in names})"
    )
    result = subprocess.run([python, "-c", script], text=True,
                            capture_output=True, check=False)
    return {"returncode": result.returncode, "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip()}


def set_module_roots(args):
    matrix.ROOT = Path(args.env_root)
    matrix.BASE = Path(args.base_root) / "env"
    matrix.MODEL = args.model_path
    matrix.OUTPUT = Path(args.output_root)


def run(args):
    set_module_roots(args)
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    max_batch = max(args.batches)
    max_context = max(args.contexts)
    max_output = max(output_tokens(context, batch)
                     for context in args.contexts for batch in args.batches)
    max_model_len = max_context + max_output
    command = make_command(output, max_batch, max_model_len)
    config = {
        "driver": "largebatch_pp2.py",
        "strategy": STRATEGY,
        "contexts": list(args.contexts),
        "batches": list(args.batches),
        "max_num_seqs": max_batch,
        "max_num_batched_tokens": 8192,
        "max_model_len": max_model_len,
        "capacity_mode": args.capacity_mode,
        "profile_context": args.profile_context,
        "capacity_safety_ratio": args.capacity_safety_ratio,
        "command": command,
        "model": args.model_path,
        "root": str(matrix.ROOT),
        "base": str(matrix.BASE),
        "mode": "eager",
        "scope": "PP2 TP4 EP4, CPU Engram offload, full-model inference",
        "versions": {
            "vllm_commit": "8538017f4b1d01b543345f5dbe1993cb66c80fa9",
            "vllm_wheel": "0.3.1.dev7+g8538017f4",
            "torch": "2.13.0",
            "flashinfer": "0.6.18.post1",
            "cuda_toolchain": "13.0.88",
            "cuda_cccl": "13.0.85",
        },
        "microbatch_note": (
            "requested_batch is HTTP request concurrency; actual scheduler microbatch "
            "sizes are recorded only when present in profiler trace/log output"
        ),
    }
    save(output / "configuration.json", config)
    if args.dry_plan:
        plans = [case_plan(context, batch, [], args.capacity_safety_ratio,
                           args.capacity_mode)
                 for context in args.contexts for batch in args.batches]
        save(output / "plan.json", {"configuration": config, "cases": plans})
        print(json.dumps({"status": "dry_plan", "cases": plans}, indent=2))
        return 0

    if matrix.alive():
        raise RuntimeError("port 8000 already has a live server")
    server_dir = output / "server"
    server_dir.mkdir(parents=True, exist_ok=True)
    log_path = server_dir / "server.log"
    process = None
    results = []
    profile_candidates = {}
    try:
        with log_path.open("w") as log:
            process = subprocess.Popen(
                command, env=matrix.environment(STRATEGY), stdout=log,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        deadline = time.monotonic() + args.startup_timeout
        while not matrix.alive():
            if process.poll() is not None:
                raise RuntimeError(f"server exited with code {process.returncode}")
            if time.monotonic() > deadline:
                raise TimeoutError("server did not become healthy")
            time.sleep(5)
        capacity_deadline = time.monotonic() + 120
        capacities = capacity_from_log(log_path)
        while not capacities and time.monotonic() < capacity_deadline:
            time.sleep(2)
            capacities = capacity_from_log(log_path)
        if not capacities and args.capacity_mode == "conservative":
            raise RuntimeError(
                "server became healthy without a parseable GPU KV cache capacity; "
                "refusing to run unknown-capacity cases"
            )
        save(output / "kv_capacity.json", {
            "per_rank_logged_tokens": capacities,
            "minimum_logged_tokens": min(capacities) if capacities else None,
        })
        config["installed_versions"] = installed_versions(args.env_root)
        save(output / "configuration.json", config)

        corpus = "\n\n".join(
            f"Passage {i + 1}: {text}" for i, text in enumerate(matrix.CORPUS)
        )
        unit = matrix.api("/tokenize", {"model": "deepseek-v41-flash", "prompt": corpus})["tokens"]
        all_tokens = (unit * (max_context // len(unit) + 1))[:max_context]
        matrix.REQUEST_SUFFIXES.clear()
        for index in range(max_batch):
            suffix = (
                f"\nTask {index + 1}: Explain the main ideas in passage "
                f"{index % len(matrix.CORPUS) + 1} with examples. Answer:"
            )
            matrix.REQUEST_SUFFIXES.append(
                matrix.api("/tokenize", {"model": "deepseek-v41-flash", "prompt": suffix})["tokens"]
            )

        # Match the baseline's warmup boundary so compilation and allocator
        # setup are excluded from every measured case.
        matrix.run_case(output, "warmup_2k", all_tokens[:2048], 1, 32)
        if max_context != 2048:
            matrix.run_case(output, "warmup_max_context", all_tokens[:max_context], 1, 1)

        for context in args.contexts:
            for batch in args.batches:
                planned = case_plan(context, batch, capacities,
                                    args.capacity_safety_ratio,
                                    args.capacity_mode)
                label = f"decode_ctx{context}_b{batch}"
                if planned["status"] == "capacity_skip":
                    planned["label"] = label
                    save(output / "metrics" / f"{label}.json", planned)
                    results.append(planned)
                    continue
                try:
                    preemptions_before = preemption_snapshot()
                    result = matrix.run_case(
                        output, label + "_timing", all_tokens[:context], batch,
                        planned["requested_output_tokens"], dp_size=1,
                    )
                    preemptions_after = preemption_snapshot()
                    preemption_delta = (
                        preemptions_after - preemptions_before
                        if preemptions_before is not None and preemptions_after is not None
                        else None
                    )
                    planned.update({
                        "preemptions_before": preemptions_before,
                        "preemptions_after": preemptions_after,
                        "preemptions_delta": preemption_delta,
                        "preemption_status": (
                            "clean" if preemption_delta == 0 else
                            "detected" if preemption_delta is not None else "unknown"
                        ),
                    })
                    if not result.get("steady_batch_verified") or min(
                        result.get("steady_intervals_per_request", [0])
                    ) < 32:
                        raise RuntimeError("fewer than 32 common steady ITLs per request")
                    clean = planned["preemption_status"] == "clean"
                    planned.update({
                        "status": (
                            "timing_complete" if clean else
                            "timing_preempted" if planned["preemption_status"] == "detected"
                            else "timing_preemption_unknown"
                        ),
                        "label": label, "timing": result,
                    })
                    if clean:
                        profile_candidates[context] = (batch, planned)
                except Exception as exc:
                    planned.update({"status": "timing_failed", "label": label,
                                    "resident_verified_batch": None,
                                    "error": repr(exc), "traceback": traceback.format_exc()})
                save(output / "metrics" / f"{label}.json", planned)
                results.append(planned)

        # PP workers support one profiler session per server lifetime. Choose
        # the largest verified batch, breaking ties by context length, and
        # record explicit skips for the other context rather than reopening
        # Kineto in the same PP workers.
        candidates = profile_candidates
        if args.profile_context is not None:
            candidates = {
                context: value for context, value in profile_candidates.items()
                if context == args.profile_context
            }
        profile_target = (
            max(candidates.items(), key=lambda item: (item[1][0], item[0]))
            if candidates else None
        )
        for context, (batch, planned) in sorted(profile_candidates.items()):
            if profile_target is None or (context, batch) != (
                profile_target[0], profile_target[1][0]
            ):
                save(output / "metrics" / f"decode_ctx{context}_b{batch}_profile.json", {
                    "status": "profile_skipped_single_kineto_session",
                    "context_tokens": context,
                    "requested_batch": batch,
                    "profiled_representative": None if profile_target is None else {
                        "context_tokens": profile_target[0],
                        "requested_batch": profile_target[1][0],
                    },
                })
                continue
            label = f"decode_ctx{context}_b{batch}_profile"
            try:
                profile = matrix.run_case(
                    output, label, all_tokens[:context], batch,
                    planned["requested_output_tokens"], profile=True, decode=True,
                    dp_size=1,
                )
                planned["profile"] = profile
                planned["actual_microbatch_histogram"] = trace_microbatch_histogram(
                    output / "timelines" / label
                )
                save(output / "metrics" / f"{label}.json", planned)
            except Exception as exc:
                save(output / "metrics" / f"{label}.json", {
                    "status": "profile_failed", "context_tokens": context,
                    "requested_batch": batch, "error": repr(exc),
                    "traceback": traceback.format_exc(),
                })
        save(output / "RESULTS.json", {"configuration": config, "cases": results})
        return 0
    except Exception as exc:
        save(output / "FAILED.json", {
            "configuration": config,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "partial_cases": results,
        })
        raise
    finally:
        if process is not None:
            matrix.stop_server(process)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=parse_ints, default=DEFAULT_BATCHES,
                        help="comma-separated total request batches")
    parser.add_argument("--contexts", type=parse_ints, default=DEFAULT_CONTEXTS,
                        help="comma-separated prompt context lengths")
    parser.add_argument("--output-root", default="/home/admin/dsv41-vllm-20260917/largebatch_pp2")
    parser.add_argument("--model-path", default="/mnt/fuse/models/DeepSeek-V4.1-Flash-local-engram")
    parser.add_argument("--env-root", default="/home/admin/dsv41-vllm-20260917")
    parser.add_argument("--base-root", default="/home/admin/dsv41-existing-ops-20260917")
    parser.add_argument("--capacity-safety-ratio", type=float, default=0.90)
    parser.add_argument("--capacity-mode", choices=("conservative", "observed"),
                        default="conservative",
                        help="skip by allocation bound or run and observe preemption")
    parser.add_argument("--profile-context", type=int,
                        help="profile only this context length")
    parser.add_argument("--startup-timeout", type=int, default=1500)
    parser.add_argument("--dry-plan", action="store_true", help="write the plan without starting vLLM")
    args = parser.parse_args()
    if not 0 < args.capacity_safety_ratio <= 1:
        parser.error("--capacity-safety-ratio must be in (0, 1]")
    try:
        return run(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": repr(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
