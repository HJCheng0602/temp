"""Range benchmark for NCCL and FlashInfer PCIe-IPC all-reduce.

Run with ``torchrun --nproc-per-node=4 microbench.py`` in the target runtime.
The script intentionally uses only public calls already exercised by the
single-shape benchmark.  Unsupported IPC shapes are reported as failures.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import timedelta
import json
import os
from pathlib import Path
import statistics
import traceback
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable

import torch
import torch.distributed as dist


WORLD_SIZE = 4
HIDDEN = 5120
DEFAULT_BATCHES = (1, 2, 3, 4, 7, 8, 16, 31, 32, 64, 127, 128, 256, 512, 1024, 2048, 4096)
DEFAULT_GRAPH_CALLS = 32
DEFAULT_REPEATS = 7
MAX_BATCH = 4096


def _ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from exc
    if not values or any(x <= 0 or x > MAX_BATCH for x in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"batches must be unique integers in [1, {MAX_BATCH}]")
    return values


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, torch.Tensor):
        return {"tensor": True, "shape": list(value.shape), "dtype": str(value.dtype),
                "device": str(value.device), "numel": value.numel()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return repr(value)


def _event_ms(fn: Callable[[], Any]) -> tuple[float, Any]:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)), result


def _gather(value: Any, group: Any) -> list[Any]:
    result: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(result, value, group=group)
    return result


def _env_snapshot() -> dict[str, str]:
    return {key: value for key, value in sorted(os.environ.items())
            if key.startswith(("NCCL_", "CUDA_", "TORCH_", "CUBLAS_", "PYTORCH_"))}


def _workspace_introspection(workspace: Any, sample: torch.Tensor | None = None) -> dict[str, Any]:
    """Record public data without invoking guessed/private configuration APIs."""
    result: dict[str, Any] = {"type": type(workspace).__name__}
    public_names = [name for name in dir(workspace) if not name.startswith("_")]
    result["public_names"] = public_names
    for name in public_names:
        if name in {"all_reduce", "prepare", "tune", "supports", "destroy"}:
            continue
        try:
            value = getattr(workspace, name)
            if callable(value):
                continue
            result[name] = _jsonable(value)
        except Exception as exc:
            result[name] = f"<read-error: {exc!r}>"
    # A public config property, when present, is data-only and safe to read.
    if "config" in public_names:
        try:
            result["config"] = _jsonable(getattr(workspace, "config"))
        except Exception as exc:
            result["config_read_error"] = repr(exc)
    if sample is not None:
        for name in ("launch_config", "tuned_launch_config"):
            method = getattr(workspace, name, None)
            if callable(method):
                try:
                    result[name] = _jsonable(method(sample))
                except Exception as exc:
                    result[name + "_error"] = repr(exc)
    return result


def _make_workspace(group: Any, max_numel: int, batches: tuple[int, ...], tune: bool) -> tuple[Any, str]:
    from flashinfer.comm import PcieIpcAllReduceWorkspace

    workspace = PcieIpcAllReduceWorkspace(
        group=group, max_numel=max_numel, dtype=torch.bfloat16, tune_batches=tuple(batches)
    )
    if tune:
        workspace.tune([HIDDEN], dtype=torch.bfloat16, tune_group=group)
    workspace.prepare([(batch, HIDDEN) for batch in tuple(batches) + (1,)], dtype=torch.bfloat16)
    return workspace, "flashinfer.comm.PcieIpcAllReduceWorkspace"


def _ipc_call(workspace: Any, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    value = workspace.all_reduce(x, out=out)
    return out if value is None else value


def _errors(output: torch.Tensor, reference: torch.Tensor, gathered: list[torch.Tensor]) -> dict[str, Any]:
    delta = output.float() - reference
    ref_norm = float(torch.linalg.vector_norm(reference).item())
    max_error = float(delta.abs().max().item())
    rank_max = max(float((output.float() - other.float()).abs().max().item()) for other in gathered)
    return {"max_abs": max_error, "rms": float(torch.sqrt(torch.mean(delta.square())).item()),
            "relative_l2": float(torch.linalg.vector_norm(delta).item() / max(ref_norm, 1e-30)),
            "rank_consistency_max_abs": rank_max}


def _pattern_check(operation: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> dict[str, Any]:
    pattern = torch.full_like(x, float(dist.get_rank() + 1))
    result = operation(pattern)
    expected = torch.full_like(result, float(WORLD_SIZE * (WORLD_SIZE + 1) // 2))
    diff = (result - expected).abs()
    return {"max_abs": float(diff.max().item()), "exact_integer_pattern": bool(torch.equal(result, expected))}


def _profile_graph(graph: torch.cuda.CUDAGraph, calls: int, rank: int) -> dict[str, Any] | None:
    profiler = None
    if rank == 0:
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
        )
        profiler.__enter__()
    try:
        graph.replay()
        torch.cuda.synchronize()
    finally:
        if profiler is not None:
            profiler.__exit__(None, None, None)
    if profiler is None:
        return None
    events = []
    for event in profiler.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            events.append({"name": event.name, "device_ms": float(event.time_range.elapsed_us()) / 1000})
    names = " ".join(item["name"].lower() for item in events)
    return {"graph_replay_calls": calls, "events": events[:200],
            "kernel_name_hints": {token: (token in names) for token in ("nccl", "flashinfer", "pcie", "ipc", "allreduce")}}


def _run_shape(backend: str, batch: int, x: torch.Tensor, group: Any, workspace: Any, nccl: Any,
               graph_calls: int, repeats: int, eager: bool, profile: bool) -> dict[str, Any]:
    rank = dist.get_rank()
    eager_out = torch.empty_like(x)
    if backend == "nccl":
        def operation(value: torch.Tensor) -> torch.Tensor:
            return nccl.all_reduce(value, out_tensor=eager_out)
    else:
        def operation(value: torch.Tensor) -> torch.Tensor:
            return _ipc_call(workspace, value, eager_out)

    eager_times: list[float] = []
    eager_output = operation(x)
    for _ in range(3):
        operation(x)
    if eager:
        for _ in range(repeats):
            dist.barrier(group=group)
            torch.cuda.synchronize()
            elapsed, eager_output = _event_ms(lambda: operation(x))
            eager_times.append(elapsed)

    static_x = x.clone()
    static_out = torch.empty_like(x)
    graph_result = static_out

    def graph_operation() -> None:
        nonlocal graph_result
        graph_result = (nccl.all_reduce(static_x, out_tensor=static_out) if backend == "nccl"
                        else _ipc_call(workspace, static_x, static_out))

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        for _ in range(graph_calls):
            graph_operation()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    graph_times: list[float] = []
    for _ in range(repeats):
        dist.barrier(group=group)
        torch.cuda.synchronize()
        elapsed, _ = _event_ms(graph.replay)
        graph_times.append(elapsed / graph_calls)
    profile_data = _profile_graph(graph, graph_calls, rank) if profile else None

    pattern = _pattern_check(operation, x)
    # Pattern validation uses the shared output buffer; restore the measured
    # eager result before gathering its error metrics.
    eager_output = operation(x)
    torch.cuda.synchronize()
    reference = torch.zeros_like(x, dtype=torch.float32)
    gathered_inputs = [torch.empty_like(x) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered_inputs, x, group=group)
    for value in gathered_inputs:
        reference.add_(value.float())
    gathered_outputs = [torch.empty_like(eager_output) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered_outputs, eager_output, group=group)
    graph_outputs = [torch.empty_like(graph_result) for _ in range(WORLD_SIZE)]
    dist.all_gather(graph_outputs, graph_result, group=group)
    return {"batch": batch, "numel": int(x.numel()), "rank": rank,
            "eager_ms_per_call": eager_times, "graph_ms_per_call": graph_times,
            "eager_errors": _errors(eager_output, reference, gathered_outputs),
            "graph_errors": _errors(graph_result, reference, graph_outputs),
            "pattern": pattern, "profile": profile_data}


def _aggregate(records: list[dict[str, Any]], repeats: int) -> dict[str, Any]:
    graph_by_repeat = [[float(record["graph_ms_per_call"][i]) for record in records]
                       for i in range(repeats)]
    rank_max = [max(values) for values in graph_by_repeat]
    rank_median = [statistics.median(values) for values in graph_by_repeat]
    result = {"graph_ms_per_call_max_by_repeat": rank_max,
              "graph_ms_per_call_median_rank_by_repeat": rank_median,
              "graph_ms_per_call_max_median": statistics.median(rank_max),
              "graph_ms_per_call_median_rank_median": statistics.median(rank_median),
              "ranks": records, "graph_errors": records[0]["graph_errors"],
              "pattern": records[0]["pattern"]}
    eager = [record["eager_ms_per_call"] for record in records if record["eager_ms_per_call"]]
    if eager:
        result["eager_ms_per_call_max_by_repeat"] = [max(row[i] for row in eager) for i in range(repeats)]
        result["eager_ms_per_call_max_median"] = statistics.median(result["eager_ms_per_call_max_by_repeat"])
        result["eager_errors"] = records[0]["eager_errors"]
    profiles = [record["profile"] for record in records if record.get("profile") is not None]
    if profiles:
        result["profile"] = profiles[0]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("allreduce_range.json"))
    parser.add_argument("--backends", nargs="+", choices=("nccl", "ipc"), default=("nccl", "ipc"))
    parser.add_argument("--batches", type=_ints, default=DEFAULT_BATCHES)
    parser.add_argument("--workspace-max-batch", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=1, help="retained for CLI compatibility; graph calls control work")
    parser.add_argument("--graph-calls", type=int, default=DEFAULT_GRAPH_CALLS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--eager", action="store_true", help="also collect one-call eager timings")
    parser.add_argument("--profile-batches", type=_ints, default=(), help="selected batches for graph replay profiling")
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    if args.graph_calls <= 0 or args.repeats <= 0 or args.iterations <= 0:
        parser.error("--iterations, --graph-calls, and --repeats must be positive")
    if args.workspace_max_batch is not None and not 1 <= args.workspace_max_batch <= MAX_BATCH:
        parser.error("--workspace-max-batch must be in [1, 4096]")
    if "ipc" in args.backends and args.workspace_max_batch is not None and args.workspace_max_batch < max(args.batches):
        parser.error("--workspace-max-batch is smaller than the largest requested batch")
    if any(batch not in args.batches for batch in args.profile_batches):
        parser.error("--profile-batches must be a subset of --batches")

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    gloo = dist.new_group(backend="gloo")
    outputs: dict[str, Any] = {"metadata": {
        "world_size": WORLD_SIZE, "hidden": HIDDEN, "dtype": "torch.bfloat16",
        "graph_calls": args.graph_calls, "repeats": args.repeats, "eager": args.eager,
        "batches": args.batches, "workspace_max_batch": args.workspace_max_batch,
        "torchrun_world_size": dist.get_world_size(), "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda, "nccl_version": torch.cuda.nccl.version(),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"), "env": _env_snapshot(),
        "seed": 20261001, "rank": rank,
    }, "results": {}, "failures": []}
    failure = None
    current_backend, current_batch = "setup", None
    workspace = None
    try:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        nccl = PyNcclCommunicator(gloo, device=device)
        if nccl.disabled:
            raise RuntimeError("PyNcclCommunicator unavailable")
        outputs["metadata"]["nccl_api"] = "PyNcclCommunicator.all_reduce(out_tensor=...)"
        try:
            outputs["metadata"]["flashinfer_version"] = version("flashinfer-python")
        except PackageNotFoundError:
            outputs["metadata"]["flashinfer_version"] = "unavailable-to-importlib.metadata"
        if dist.get_world_size() != WORLD_SIZE:
            raise RuntimeError(f"expected torchrun world size {WORLD_SIZE}, got {dist.get_world_size()}")
        if "ipc" in args.backends:
            max_batch = args.workspace_max_batch or max(args.batches)
            workspace, api = _make_workspace(gloo, max_batch * HIDDEN, args.batches, args.tune)
            outputs["metadata"]["ipc_api"] = api
            outputs["metadata"]["ipc_workspace_introspection"] = _workspace_introspection(workspace)
            outputs["metadata"]["ipc_launch_configs"] = {}
        torch.manual_seed(20261001 + rank)
        for index, batch in enumerate(args.batches):
            x = torch.randn((batch, HIDDEN), device=device, dtype=torch.bfloat16)
            if workspace is not None:
                inspected = _workspace_introspection(workspace, x)
                if rank == 0:
                    outputs["metadata"]["ipc_launch_configs"][str(batch)] = {
                        name: value for name, value in inspected.items()
                        if name in ("launch_config", "tuned_launch_config") or name.endswith("_error")
                    }
            order = list(args.backends)
            if index % 2:
                order.reverse()
            for backend in order:
                current_backend, current_batch = backend, batch
                if backend == "ipc" and not workspace.supports(x):
                    raise RuntimeError(f"FlashInfer unsupported shape (no fallback): batch={batch}")
                record = _run_shape(backend, batch, x, dist.group.WORLD, workspace, nccl,
                                    args.graph_calls, args.repeats, args.eager,
                                    batch in args.profile_batches)
                records = _gather(record, gloo)
                if rank == 0:
                    outputs["results"][f"{backend}:b{batch}"] = _aggregate(records, args.repeats)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(_jsonable(outputs), indent=2) + "\n")
            del x
            torch.cuda.empty_cache()
    except Exception as exc:
        failure = {"rank": rank, "backend": current_backend, "batch": current_batch,
                   "error": repr(exc), "traceback": traceback.format_exc()}
        outputs["failures"].append(failure)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        (args.output.parent / f"{args.output.stem}.rank{rank}.FAILED.json").write_text(
            json.dumps(_jsonable(failure), indent=2) + "\n")
    finally:
        if callable(getattr(workspace, "destroy", None)):
            try:
                workspace.destroy()
            except Exception as exc:
                outputs["failures"].append({"rank": rank, "backend": "ipc", "batch": None,
                                            "error": repr(exc), "traceback": traceback.format_exc()})
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(_jsonable(outputs), indent=2) + "\n")
            print(json.dumps({"output": str(args.output), "results": len(outputs["results"]),
                              "failures": len(outputs["failures"])}, separators=(",", ":")), flush=True)
        dist.destroy_process_group()
    if failure is not None:
        raise RuntimeError(f"rank {rank} microbenchmark failed: {failure['error']}")


if __name__ == "__main__":
    main()
