"""Collect a 2K PP2/TP4/EP4 timing and profile sweep.

Pipeline-parallel workers support one Kineto session per server lifetime.  This
driver therefore starts a fresh server for every requested batch, preserving
the validated eager configuration while making an unprofiled timing request
and a separate short decode profile for the same workload.  Existing complete
batch directories are skipped; incomplete directories are never overwritten.

Use ``--dry-plan`` on a machine without GPUs to inspect the server commands and
case layout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import time
import traceback

import largebatch_pp2 as largebatch
import parallel_matrix as matrix


DEFAULT_BATCHES = (16, 32, 64, 128, 256)
CONTEXT = 2048
WARMUP_CONTEXT = 32768
BASELINE_MAX_NUM_SEQS = 256
BASELINE_MAX_MODEL_LEN = 34048
STRATEGY = largebatch.STRATEGY


def parse_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(
            "expected positive comma-separated integers"
        )
    return values


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def make_command(directory: Path, batch: int, model_len: int) -> list[str]:
    command = matrix.command(STRATEGY, directory)
    for option, value in (("--max-num-seqs", BASELINE_MAX_NUM_SEQS),
                          ("--max-model-len", BASELINE_MAX_MODEL_LEN)):
        index = command.index(option) + 1
        command[index] = str(value)
    return command


def capacity_from_log(path: Path) -> list[int]:
    if not path.is_file():
        return []
    return [
        int(value.replace(",", ""))
        for value in re.findall(r"GPU KV cache size: ([\d,]+) tokens", path.read_text())
    ]


def choose_run_dir(root: Path, batch: int) -> tuple[Path, bool]:
    """Return a new directory, preserving interrupted or foreign artifacts."""
    preferred = root / f"batch{batch}"
    if not preferred.exists():
        return preferred, False
    if (preferred / "COMPLETE.json").is_file():
        return preferred, True
    stamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = root / f"batch{batch}_attempt_{stamp}_{time.time_ns() % 1_000_000:06d}"
    while candidate.exists():
        candidate = root / f"batch{batch}_attempt_{time.time_ns()}"
    return candidate, False


def build_workload(batch: int, max_context: int) -> tuple[list[int], str]:
    corpus = "\n\n".join(
        f"Passage {index + 1}: {text}"
        for index, text in enumerate(matrix.CORPUS)
    )
    unit = matrix.api(
        "/tokenize", {"model": "deepseek-v41-flash", "prompt": corpus}
    )["tokens"]
    tokens = (unit * (max_context // len(unit) + 1))[:max_context]
    matrix.REQUEST_SUFFIXES.clear()
    for index in range(batch):
        suffix = (
            f"\nTask {index + 1}: Explain the main ideas in passage "
            f"{index % len(matrix.CORPUS) + 1} with examples. Answer:"
        )
        matrix.REQUEST_SUFFIXES.append(
            matrix.api(
                "/tokenize", {"model": "deepseek-v41-flash", "prompt": suffix}
            )["tokens"]
        )
    return tokens, corpus


def configuration(args: argparse.Namespace, batch: int, directory: Path) -> dict:
    generated = largebatch.output_tokens(CONTEXT, batch)
    command = make_command(directory, batch, CONTEXT + generated)
    return {
        "driver": "profile_2k_batch_sweep.py",
        "strategy": STRATEGY,
        "context_tokens": CONTEXT,
        "requested_batch": batch,
        "requested_output_tokens": generated,
        "max_num_seqs": BASELINE_MAX_NUM_SEQS,
        "max_num_batched_tokens": 8192,
        "max_model_len": BASELINE_MAX_MODEL_LEN,
        "mode": "eager",
        "scope": "PP2 TP4 EP4, CPU Engram offload, full-model inference",
        "command": command,
        "model": args.model_path,
        "root": str(matrix.ROOT),
        "base": str(matrix.BASE),
        "startup_timeout_seconds": args.startup_timeout,
        "profile_steps": 8,
        "profile_note": (
            "The profile is started after eight decode tokens and stopped after "
            "eight further token intervals; timing and profile are separate requests."
        ),
    }


def run_batch(args: argparse.Namespace, batch: int, root: Path) -> dict:
    directory, already_complete = choose_run_dir(root, batch)
    if already_complete:
        return {
            "batch": batch,
            "directory": str(directory),
            "status": "already_complete",
        }

    directory.mkdir(parents=True, exist_ok=True)
    config = configuration(args, batch, directory)
    save(directory / "configuration.json", config)
    if args.dry_plan:
        return {"batch": batch, "directory": str(directory), "status": "planned"}

    if matrix.alive():
        raise RuntimeError("port 8000 already has a live server")

    log_path = directory / "server.log"
    process = None
    result = {
        "batch": batch,
        "directory": str(directory),
        "status": "running",
    }
    record_event(root, "batch_start", batch=batch, directory=str(directory))
    generated = config["requested_output_tokens"]
    try:
        with log_path.open("w") as log:
            process = subprocess.Popen(
                config["command"],
                env=matrix.environment(STRATEGY),
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        save(directory / "server_pid.json", {"pid": process.pid})
        deadline = time.monotonic() + args.startup_timeout
        while not matrix.alive():
            if process.poll() is not None:
                raise RuntimeError(f"server exited with code {process.returncode}")
            if time.monotonic() > deadline:
                raise TimeoutError("server did not become healthy")
            time.sleep(5)

        capacities = capacity_from_log(log_path)
        capacity_deadline = time.monotonic() + 120
        while not capacities and time.monotonic() < capacity_deadline:
            time.sleep(2)
            capacities = capacity_from_log(log_path)
        save(directory / "kv_capacity.json", {
            "per_rank_logged_tokens": capacities,
            "minimum_logged_tokens": min(capacities) if capacities else None,
        })

        tokens, corpus = build_workload(batch, WARMUP_CONTEXT)
        save(directory / "workload.json", {
            "corpus": corpus,
            "unit_token_ids": tokens,
            "request_suffix_token_ids": matrix.REQUEST_SUFFIXES,
            "prefix_cache": False,
            "construction": (
                "Repeat corpus token IDs to exact length and replace the final "
                "suffix with a request-specific task."
            ),
        })
        matrix.run_case(directory, "warmup_2k", tokens[:CONTEXT], 1, 32)
        matrix.run_case(directory, "warmup_max_context", tokens, 1, 1)

        timing_label = f"decode_ctx{CONTEXT}_b{batch}_timing"
        profile_label = f"decode_ctx{CONTEXT}_b{batch}_profile"
        record_event(root, "timing_start", batch=batch, directory=str(directory))
        preemptions_before = largebatch.preemption_snapshot()
        timing = matrix.run_case(
            directory, timing_label, tokens[:CONTEXT], batch, generated, dp_size=1
        )
        preemptions_after = largebatch.preemption_snapshot()
        preemption_delta = (
            preemptions_after - preemptions_before
            if preemptions_before is not None and preemptions_after is not None
            else None
        )
        preemption_status = (
            "clean" if preemption_delta == 0
            else "detected" if preemption_delta is not None
            else "unknown"
        )
        steady = timing.get("steady_intervals_per_request", [])
        if not timing.get("steady_batch_verified") or not steady or min(steady) < 32:
            raise RuntimeError("fewer than 32 common steady ITLs per request")
        save(directory / "metrics" / f"decode_ctx{CONTEXT}_b{batch}.json", {
            "status": "timing_complete" if preemption_status == "clean" else "timing_preempted",
            "context_tokens": CONTEXT,
            "requested_batch": batch,
            "preemptions_before": preemptions_before,
            "preemptions_after": preemptions_after,
            "preemptions_delta": preemption_delta,
            "preemption_status": preemption_status,
            "timing": timing,
        })
        if preemption_status != "clean":
            raise RuntimeError(f"timing preemption status was {preemption_status}")
        record_event(root, "profile_start", batch=batch, directory=str(directory))
        profile = matrix.run_case(
            directory, profile_label, tokens[:CONTEXT], batch, generated,
            profile=True, decode=True, dp_size=1,
        )
        save(directory / "metrics" / f"{profile_label}.json", {
            "status": "profile_complete",
            "context_tokens": CONTEXT,
            "requested_batch": batch,
            "profile": profile,
        })
        result.update({"status": "complete", "timing": timing, "profile": profile})
        save(directory / "COMPLETE.json", result)
        record_event(root, "batch_complete", batch=batch, directory=str(directory))
        return result
    except Exception as exc:
        result.update({
            "status": "failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        })
        save(directory / "FAILED.json", result)
        raise
    finally:
        if process is not None:
            matrix.stop_server(process)


def run(args: argparse.Namespace) -> int:
    matrix.ROOT = Path(args.env_root)
    matrix.BASE = Path(args.base_root) / "env"
    matrix.MODEL = args.model_path
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    plan = {
        "driver": "profile_2k_batch_sweep.py",
        "batches": list(args.batches),
        "context_tokens": CONTEXT,
        "warmup_context_tokens": WARMUP_CONTEXT,
        "baseline_max_num_seqs": BASELINE_MAX_NUM_SEQS,
        "baseline_max_model_len": BASELINE_MAX_MODEL_LEN,
        "strategy": STRATEGY,
        "sequential_fresh_server": True,
    }
    installed = None
    if not args.dry_plan:
        try:
            installed = largebatch.installed_versions(args.env_root)
        except (OSError, ValueError) as exc:
            installed = {"error": repr(exc)}
    if installed is not None:
        plan["installed_versions"] = installed
    save(root / "SWEEP_CONFIGURATION.json", plan)
    summaries = []
    for batch in args.batches:
        summaries.append(run_batch(args, batch, root))
        save(root / "SWEEP_STATUS.json", {**plan, "batches": summaries})
    save(root / "SWEEP_RESULTS.json", {**plan, "batches": summaries})
    print(json.dumps({"status": "dry_plan" if args.dry_plan else "complete",
                      "batches": summaries}, indent=2))
    return 0


def record_event(root: Path, event: str, **fields: object) -> None:
    path = root / "SWEEP_EVENTS.json"
    value = {"events": []}
    if path.is_file():
        try:
            value = json.loads(path.read_text())
        except json.JSONDecodeError:
            value = {"events": []}
    value.setdefault("events", []).append({
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "event": event,
        **fields,
    })
    save(path, value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=parse_ints, default=DEFAULT_BATCHES)
    parser.add_argument(
        "--output-root",
        default="/home/admin/dsv41-vllm-20260917/largebatch_2k_batch_sweep",
    )
    parser.add_argument("--model-path", default="/mnt/fuse/models/DeepSeek-V4.1-Flash-local-engram")
    parser.add_argument("--env-root", default="/home/admin/dsv41-vllm-20260917")
    parser.add_argument("--base-root", default="/home/admin/dsv41-existing-ops-20260917")
    parser.add_argument("--startup-timeout", type=int, default=1500)
    parser.add_argument("--dry-plan", action="store_true")
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": repr(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
