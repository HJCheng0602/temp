"""TP4 BF16 all-reduce microbenchmark for prefill-sized tensors.

Run with ``torchrun --nproc-per-node=4 benchmark.py`` in the target runtime.
The CUDA events bracket the collective on its compute stream. Barriers,
tensor fills, validation, tuning, preparation, and graph
capture are outside the measured region.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import timedelta
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable

import torch
import torch.distributed as dist


WORLD_SIZE = 4
HIDDEN = 5120
DEFAULT_MS = (128, 4096, 8192)
DEFAULT_REPEATS = 20
DEFAULT_WARMUPS = 5


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return repr(value)


def _ints(value: str) -> tuple[int, ...]:
    values = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if not values or any(x <= 0 for x in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("expected unique positive comma-separated M values")
    return values


def _env_snapshot() -> dict[str, str]:
    return {k: v for k, v in sorted(os.environ.items())
            if k.startswith(("NCCL_", "CUDA_", "TORCH_", "CUBLAS_", "PYTORCH_"))}


def _versions() -> dict[str, str]:
    result = {"torch": torch.__version__, "cuda": str(torch.version.cuda),
              "nccl": str(torch.cuda.nccl.version())}
    for package in ("flashinfer-python", "vllm"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "unavailable"
    return result


def _workspace_public_state(workspace: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"type": type(workspace).__name__}
    names = [name for name in dir(workspace) if not name.startswith("_")]
    result["public_names"] = names
    for name in names:
        if name in {"all_reduce", "tune", "prepare", "rebind", "rebind_stream", "destroy"}:
            continue
        try:
            value = getattr(workspace, name)
            if not callable(value):
                result[name] = _jsonable(value)
        except Exception as exc:
            result[name + "_error"] = repr(exc)
    return result


def _make_workspace(group: Any, max_numel: int, batches: tuple[int, ...], tune: bool,
                    compute_stream: torch.cuda.Stream) -> tuple[Any, dict[str, Any]]:
    from flashinfer.comm import PcieIpcAllReduceWorkspace

    # Keep 8192 in tune_batches even when a caller selects a subset of shapes.
    tune_batches = tuple(sorted(set(batches) | {8192}))
    with torch.cuda.stream(compute_stream):
        workspace = PcieIpcAllReduceWorkspace(
            group=group, max_numel=max_numel, dtype=torch.bfloat16, tune_batches=tune_batches
        )
    actions: dict[str, Any] = {"tune_batches": list(tune_batches), "tune": False, "prepare": [], "rebind_stream": []}
    if tune:
        with torch.cuda.stream(compute_stream):
            actions["tuned_configs"] = _jsonable(workspace.tune(
                [HIDDEN], dtype=torch.bfloat16, tune_group=group))
        actions["tune"] = True
        torch.cuda.synchronize()
        workspace.rebind_stream()
    for m in batches:
        with torch.cuda.stream(compute_stream):
            resolved = workspace.prepare([(m, HIDDEN)], dtype=torch.bfloat16)
        actions["prepare"].append({"shape": [m, HIDDEN], "config": _jsonable(resolved)})
    rebind = getattr(workspace, "rebind_stream", None)
    if callable(rebind):
        # Rebind is optional in the public API. Record availability without
        # guessing a private signature or changing buffers used by timing.
        actions["rebind_stream"] = "public method available; workspace remains on one compute stream"
    else:
        actions["rebind_stream"] = "unavailable"
    actions["public_state"] = _workspace_public_state(workspace)
    return workspace, actions


def _invoke(backend: str, x: torch.Tensor, out: torch.Tensor, workspace: Any,
            communicator: Any, group: Any) -> torch.Tensor:
    if backend == "flashinfer":
        value = workspace.all_reduce(x, out=out)
        return out if value is None else value
    if backend == "pynccl":
        return communicator.all_reduce(x, out_tensor=out)
    out.copy_(x)
    dist.all_reduce(out, group=group)
    return out


def _timed(operation: Callable[[], Any], compute_stream: torch.cuda.Stream,
           group: Any) -> tuple[float, float, Any]:
    torch.cuda.synchronize()
    dist.barrier(group=group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(compute_stream):
        start.record(compute_stream)
        wall_start = time.perf_counter()
        value = operation()
        end.record(compute_stream)
    end.synchronize()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000
    return float(start.elapsed_time(end)), wall_ms, value


def _gather(value: Any, group: Any) -> list[Any]:
    result: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(result, value, group=group)
    return result


def _validation(backend: str, x: torch.Tensor, out: torch.Tensor, workspace: Any,
                communicator: Any, group: Any, compute_stream: torch.cuda.Stream) -> dict[str, Any]:
    rank = dist.get_rank()

    def op(value: torch.Tensor) -> torch.Tensor:
        return _invoke(backend, value, out, workspace, communicator, group)

    with torch.cuda.stream(compute_stream):
        pattern = torch.full_like(x, float(rank + 1))
        result = op(pattern)
    torch.cuda.synchronize()
    expected_integer = float(WORLD_SIZE * (WORLD_SIZE + 1) // 2)
    integer_delta = float((result - expected_integer).abs().max().item())
    integer_exact = bool(torch.equal(result, torch.full_like(result, expected_integer)))

    torch.manual_seed(20261012 + rank)
    with torch.cuda.stream(compute_stream):
        x.copy_(torch.randn_like(x))
        random_result = op(x)
    torch.cuda.synchronize()
    inputs = [torch.empty_like(x) for _ in range(WORLD_SIZE)]
    dist.all_gather(inputs, x, group=group)
    reference = torch.zeros_like(x, dtype=torch.float32)
    for value in inputs:
        reference.add_(value.float())
    outputs = [torch.empty_like(random_result) for _ in range(WORLD_SIZE)]
    dist.all_gather(outputs, random_result, group=group)
    delta = random_result.float() - reference
    rank_delta = max(float((random_result.float() - other.float()).abs().max().item()) for other in outputs)

    previous = random_result.clone()
    torch.cuda.synchronize()
    with torch.cuda.stream(compute_stream):
        x.copy_(torch.randn_like(x) + float(rank + 1))
        changed_result = op(x)
    torch.cuda.synchronize()
    changed_inputs = [torch.empty_like(x) for _ in range(WORLD_SIZE)]
    dist.all_gather(changed_inputs, x, group=group)
    changed_ref = torch.zeros_like(x, dtype=torch.float32)
    for value in changed_inputs:
        changed_ref.add_(value.float())
    changed_delta = changed_result.float() - changed_ref
    return {
        "integer_pattern_max_abs": integer_delta,
        "integer_pattern_exact": integer_exact,
        "random_fp32_reference_max_abs": float(delta.abs().max().item()),
        "random_fp32_reference_rms": float(torch.sqrt(torch.mean(delta.square())).item()),
        "random_relative_l2": float(torch.linalg.vector_norm(delta).item() /
                                    max(float(torch.linalg.vector_norm(reference).item()), 1e-30)),
        "random_rank_consistency_max_abs": rank_delta,
        "changing_input_max_abs": float(changed_delta.abs().max().item()),
        "changing_input_relative_l2": float(torch.linalg.vector_norm(changed_delta).item() /
                                             max(float(torch.linalg.vector_norm(changed_ref).item()), 1e-30)),
        "changing_input_changed_output": bool(not torch.equal(previous, changed_result)),
        "repeated_output_relative_l2": float(torch.linalg.vector_norm(changed_delta).item() /
                                              max(float(torch.linalg.vector_norm(changed_ref).item()), 1e-30)),
        "repeated_output_correct": bool(float(torch.linalg.vector_norm(changed_delta).item()) /
                                         max(float(torch.linalg.vector_norm(changed_ref).item()), 1e-30) <= 0.02),
        "valid": bool(integer_exact and float(torch.linalg.vector_norm(delta).item()) /
                     max(float(torch.linalg.vector_norm(reference).item()), 1e-30) <= 0.02 and
                     rank_delta == 0.0 and
                     bool(not torch.equal(previous, changed_result)) and
                     float(torch.linalg.vector_norm(changed_delta).item()) /
                     max(float(torch.linalg.vector_norm(changed_ref).item()), 1e-30) <= 0.02),
        "rank": rank,
    }


def _run_shape(args: argparse.Namespace, backend: str, m: int, workspace: Any,
               communicator: Any, group: Any, device: torch.device) -> dict[str, Any]:
    x = torch.empty((m, HIDDEN), device=device, dtype=torch.bfloat16)
    out = torch.empty_like(x)
    compute_stream = args.compute_stream
    if backend == "flashinfer" and not workspace.supports(x):
        raise RuntimeError(f"FlashInfer workspace does not support shape {(m, HIDDEN)}")

    def operation() -> torch.Tensor:
        return _invoke(backend, x, out, workspace, communicator, group)

    validation = _validation(backend, x, out, workspace, communicator, group, compute_stream)
    validation_flags = _gather(validation["valid"], args.control_group)
    if not all(validation_flags):
        raise RuntimeError(f"validation failed before timing for M={m}: {validation}")
    for _ in range(args.warmups):
        with torch.cuda.stream(compute_stream):
            x.normal_(); operation()
    torch.cuda.synchronize()
    eager_samples = []
    wall_samples = []
    for _ in range(args.repeats):
        with torch.cuda.stream(compute_stream):
            x.normal_()
        elapsed, wall_ms, _ = _timed(operation, compute_stream, args.control_group)
        eager_samples.append(elapsed); wall_samples.append(wall_ms)

    record: dict[str, Any] = {
        "M": m, "shape": [m, HIDDEN], "backend": backend,
        "payload_bytes": x.numel() * x.element_size(),
        "launch_config": _jsonable(workspace.tuned_launch_config(x)) if workspace else None,
        "eager_event_ms": eager_samples, "eager_wall_ms": wall_samples,
        "eager_event_max_ms": max(eager_samples),
        "eager_wall_max_ms": max(wall_samples),
        "eager_event_median_ms": statistics.median(eager_samples),
        "eager_wall_median_ms": statistics.median(wall_samples),
    }
    if args.graph:
        static_x = torch.empty_like(x)
        static_out = torch.empty_like(x)
        graph_result: Any = None
        graph = torch.cuda.CUDAGraph()

        def graph_operation() -> None:
            nonlocal graph_result
            graph_result = _invoke(backend, static_x, static_out, workspace, communicator, group)

        try:
            static_x.normal_()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph, stream=compute_stream):
                graph_operation()
            with torch.cuda.stream(compute_stream):
                static_x.fill_(float(dist.get_rank() + 2))
                graph.replay()
            torch.cuda.synchronize()
            graph_exact = bool(torch.equal(static_out, torch.full_like(static_out, 14.0)))
            if not all(_gather(graph_exact, args.control_group)):
                raise RuntimeError("graph replay validation failed")
            record["graph_validation_exact"] = graph_exact
            for _ in range(args.warmups):
                with torch.cuda.stream(compute_stream):
                    static_x.normal_(); graph.replay()
            torch.cuda.synchronize()
            graph_event_samples = []
            graph_wall_samples = []
            for _ in range(args.repeats):
                with torch.cuda.stream(compute_stream):
                    static_x.normal_()
                elapsed, wall_ms, _ = _timed(graph.replay, compute_stream, args.control_group)
                graph_event_samples.append(elapsed); graph_wall_samples.append(wall_ms)
            record.update({"graph_event_ms": graph_event_samples, "graph_wall_ms": graph_wall_samples,
                           "graph_event_max_ms": max(graph_event_samples),
                           "graph_wall_max_ms": max(graph_wall_samples),
                           "graph_event_median_ms": statistics.median(graph_event_samples),
                           "graph_wall_median_ms": statistics.median(graph_wall_samples)})
        except Exception as exc:
            record["graph_error"] = repr(exc)

    record["validation"] = validation
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("flashinfer", "pynccl", "torch"), default="pynccl")
    parser.add_argument("--M", type=_ints, default=DEFAULT_MS, help="comma-separated token rows")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--graph", action="store_true", help="also benchmark one CUDA graph replay")
    parser.add_argument("--no-tune", action="store_true", help="skip FlashInfer tuning")
    parser.add_argument("--output", type=Path, default=Path("prefill_allreduce_20261012.json"))
    args = parser.parse_args()
    if args.repeats <= 0 or args.warmups < 0 or any(m > 8192 for m in args.M):
        parser.error("repeats must be positive, warmups nonnegative, and M <= 8192")

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.set_num_threads(1)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    compute_stream = torch.cuda.Stream(device=device)
    args.compute_stream = compute_stream
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    group = dist.group.WORLD
    gloo = dist.new_group(backend="gloo")
    args.control_group = gloo
    output: dict[str, Any] = {"metadata": {"world_size": WORLD_SIZE, "hidden": HIDDEN,
        "dtype": "torch.bfloat16", "backend": args.backend, "M": list(args.M),
        "repeats": args.repeats, "warmups": args.warmups, "graph": args.graph,
        "tune_requested": not args.no_tune, "env": _env_snapshot(), "versions": _versions(),
        "rank": rank, "device": torch.cuda.get_device_name(device),
        "stream_protocol": "one explicit compute stream for workspace and collective; CUDA event on that stream; device-wide sync wall crosscheck",
        "device_properties": _jsonable(torch.cuda.get_device_properties(device))},
        "workspace": None, "results": {}, "failures": []}
    workspace = None
    communicator = None
    try:
        if dist.get_world_size() != WORLD_SIZE:
            raise RuntimeError(f"expected TP4 torchrun world size {WORLD_SIZE}, got {dist.get_world_size()}")
        if args.backend == "flashinfer":
            workspace, actions = _make_workspace(gloo, max((8192,) + tuple(args.M)) * HIDDEN, args.M, not args.no_tune, compute_stream)
            output["workspace"] = actions
        elif args.backend == "pynccl":
            from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
            communicator = PyNcclCommunicator(gloo, device=device)
            if communicator.disabled:
                raise RuntimeError("PyNcclCommunicator unavailable")
            output["workspace"] = {"api": "vllm PyNcclCommunicator.all_reduce(out_tensor=...)"}
        else:
            output["workspace"] = {"api": "torch.distributed.all_reduce(out)"}
        for m in args.M:
            record = _run_shape(args, args.backend, m, workspace, communicator, group, device)
            gathered = _gather(record, gloo)
            if rank == 0:
                aggregate = {"rank_records": _jsonable(gathered)}
                for mode in ("eager", "graph"):
                    for clock in ("event", "wall"):
                        key = f"{mode}_{clock}_ms"
                        if all(key in row for row in gathered):
                            samples = [max(row[key][i] for row in gathered)
                                       for i in range(args.repeats)]
                            aggregate[f"{mode}_{clock}_max_rank_ms"] = samples
                            aggregate[f"{mode}_{clock}_median_ms"] = statistics.median(samples)
                output["results"][str(m)] = aggregate
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(_jsonable(output), indent=2) + "\n")
                print(json.dumps({"M": m, "backend": args.backend,
                                  "eager_ms": aggregate["eager_event_median_ms"]}), flush=True)
    except Exception as exc:
        failure = {"rank": rank, "error": repr(exc), "traceback": traceback.format_exc()}
        output["failures"].append(failure)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        (args.output.parent / f"{args.output.stem}.rank{rank}.FAILED.json").write_text(json.dumps(failure, indent=2) + "\n")
    finally:
        if callable(getattr(workspace, "destroy", None)):
            try:
                workspace.destroy()
            except Exception as exc:
                output["failures"].append({"rank": rank, "error": repr(exc)})
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(_jsonable(output), indent=2) + "\n")
            print(json.dumps({"output": str(args.output), "results": len(output["results"]),
                              "failures": len(output["failures"])}, separators=(",", ":")), flush=True)
        dist.destroy_process_group()
    if output["failures"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
