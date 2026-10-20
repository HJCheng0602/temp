"""Compare NCCL and FlashInfer PCIe-IPC all-reduce on one four-GPU NUMA group.

Run with ``torchrun --nproc-per-node=4 microbench.py`` in the target runtime.
The benchmark deliberately keeps the FlashInfer lane strict: an unavailable
API or unsupported shape is recorded as a failure instead of silently falling
back to NCCL.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time
import traceback
from typing import Any, Callable
from importlib.metadata import PackageNotFoundError, version

import torch
import torch.distributed as dist


WORLD_SIZE = 4
HIDDEN = 5120
GRAPH_CALLS = 32
DEFAULT_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def _ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(x) for x in value.split(",") if x.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from exc
    if not values or any(x <= 0 for x in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("batches must be unique positive integers")
    return values


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
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


def _all_gather_objects(value: Any, group: Any) -> list[Any]:
    result: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(result, value, group=group)
    return result


def _flashinfer_workspace(group: Any, max_numel: int, batches: tuple[int, ...], tune: bool) -> tuple[Any, str]:
    """Use the verified modern FlashInfer PCIe IPC API without fallbacks."""
    from flashinfer.comm import PcieIpcAllReduceWorkspace

    workspace = PcieIpcAllReduceWorkspace(
        group=group, max_numel=max_numel, dtype=torch.bfloat16,
        tune_batches=tuple(batches),
    )
    if tune:
        workspace.tune([HIDDEN], dtype=torch.bfloat16, tune_group=group)
    workspace.prepare(
        [(batch, HIDDEN) for batch in tuple(batches) + (1,)],
        dtype=torch.bfloat16,
    )
    return workspace, "flashinfer.comm.PcieIpcAllReduceWorkspace"


def _ipc_call(workspace: Any, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    if out is None:
        raise RuntimeError("FlashInfer PCIe IPC benchmark requires its verified out argument")
    value = workspace.all_reduce(x, out=out)
    return out if value is None else value


def _supports(workspace: Any, x: torch.Tensor) -> bool:
    method = getattr(workspace, "supports", None)
    if method is None:
        raise RuntimeError("FlashInfer workspace has no supports(); refusing implicit fallback")
    return bool(method(x))


def _error_metrics(output: torch.Tensor, reference: torch.Tensor, gathered: list[torch.Tensor]) -> dict[str, Any]:
    delta = output.float() - reference
    ref_norm = float(torch.linalg.vector_norm(reference).item())
    max_error = float(delta.abs().max().item())
    rms = float(torch.sqrt(torch.mean(delta.square())).item())
    relative = float(torch.linalg.vector_norm(delta).item() / max(ref_norm, 1e-30))
    rank_max = 0.0
    for other in gathered:
        rank_max = max(rank_max, float((output.float() - other.float()).abs().max().item()))
    return {"max_abs": max_error, "max": max_error, "rms": rms,
            "relative_l2": relative, "norm_relative": relative,
            "rank_consistency_max_abs": rank_max}


def _pattern_check(op: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> dict[str, Any]:
    pattern = torch.full_like(x, float(dist.get_rank() + 1))
    result = op(pattern)
    expected = torch.full_like(result, float(WORLD_SIZE * (WORLD_SIZE + 1) // 2))
    diff = (result - expected).abs()
    return {"max_abs": float(diff.max().item()), "exact_integer_pattern": bool(torch.equal(result, expected))}


def _run_shape(backend: str, batch: int, x: torch.Tensor, group: Any, workspace: Any, nccl: Any,
               iterations: int, repeats: int) -> dict[str, Any]:
    rank = dist.get_rank()
    eager_out = torch.empty_like(x)
    operation: Callable[[torch.Tensor], torch.Tensor]
    if backend == "nccl":
        def operation(value: torch.Tensor) -> torch.Tensor:
            return nccl.all_reduce(value, out_tensor=eager_out)
    else:
        def operation(value: torch.Tensor) -> torch.Tensor:
            return _ipc_call(workspace, value, eager_out)

    for _ in range(3):
        operation(x)
    torch.cuda.synchronize()
    eager_times: list[float] = []
    eager_output: torch.Tensor | None = None
    for _ in range(repeats):
        dist.barrier(group=group)
        torch.cuda.synchronize()
        def eager_batch() -> torch.Tensor:
            nonlocal eager_output
            for _ in range(iterations):
                eager_output = operation(x)
            return eager_output
        elapsed, eager_output = _event_ms(eager_batch)
        eager_times.append(elapsed / iterations)
    assert eager_output is not None

    static_x = x.clone()
    static_out = torch.empty_like(x)
    graph_result = static_out
    if backend == "nccl":
        def graph_operation() -> None:
            nonlocal graph_result
            graph_result = nccl.all_reduce(static_x, out_tensor=static_out)
    else:
        def graph_operation() -> None:
            nonlocal graph_result
            graph_result = _ipc_call(workspace, static_x, static_out)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        for _ in range(GRAPH_CALLS):
            graph_operation()
    graph_times: list[float] = []
    graph_output = graph_result
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    for _ in range(repeats):
        dist.barrier(group=group)
        torch.cuda.synchronize()
        elapsed, _ = _event_ms(graph.replay)
        graph_times.append(elapsed / GRAPH_CALLS)
    reference = torch.zeros_like(x, dtype=torch.float32)
    gathered_inputs = [torch.empty_like(x) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered_inputs, x, group=group)
    for value in gathered_inputs:
        reference.add_(value.float())
    gathered_outputs = [torch.empty_like(eager_output) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered_outputs, eager_output, group=group)
    graph_outputs = [torch.empty_like(graph_output) for _ in range(WORLD_SIZE)]
    dist.all_gather(graph_outputs, graph_output, group=group)
    result = {
        "batch": batch, "numel": int(x.numel()),
        "eager_ms_per_call": eager_times, "graph_ms_per_call": graph_times,
        "eager_errors": _error_metrics(eager_output, reference, gathered_outputs),
        "graph_errors": _error_metrics(graph_output, reference, graph_outputs),
        "pattern": _pattern_check(operation, x), "rank": rank,
    }
    return result


_GLOO: Any = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("allreduce_microbench.json"))
    parser.add_argument("--batches", type=_ints, default=DEFAULT_BATCHES)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    if args.iterations <= 0 or args.repeats <= 0:
        parser.error("--iterations and --repeats must be positive")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    global _GLOO
    _GLOO = dist.new_group(backend="gloo")
    outputs: dict[str, Any] = {"metadata": {
        "world_size": WORLD_SIZE, "hidden": HIDDEN, "dtype": "torch.bfloat16",
        "graph_calls": GRAPH_CALLS, "torchrun_world_size": dist.get_world_size(),
        "torch_version": torch.__version__, "flashinfer_version": None,
        "seed": 20260930, "rank": rank,
    }, "results": {}, "failures": []}
    failure: dict[str, Any] | None = None
    current_backend = "setup"
    current_batch: int | None = None
    try:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        nccl = PyNcclCommunicator(_GLOO, device=device)
        if nccl.disabled:
            raise RuntimeError("PyNcclCommunicator unavailable")
        outputs["metadata"]["nccl_api"] = "PyNcclCommunicator.all_reduce(out_tensor=static_out)"
        outputs["metadata"]["nccl_version"] = torch.cuda.nccl.version()
        outputs["metadata"]["cuda_visible_devices"] = os.getenv("CUDA_VISIBLE_DEVICES")
        try:
            outputs["metadata"]["flashinfer_version"] = version("flashinfer-python")
        except PackageNotFoundError:
            outputs["metadata"]["flashinfer_version"] = "unavailable-to-importlib.metadata"
        if dist.get_world_size() != WORLD_SIZE:
            raise RuntimeError(f"expected torchrun world size {WORLD_SIZE}, got {dist.get_world_size()}")
        torch.manual_seed(20260930 + rank)
        inputs = {b: torch.randn((b, HIDDEN), device=device, dtype=torch.bfloat16) for b in args.batches}
        for backend in ("nccl", "flashinfer_pcie_ipc"):
            current_backend = backend
            workspace = None
            api_name = None
            if backend != "nccl":
                workspace, api_name = _flashinfer_workspace(
                    _GLOO, max(max(args.batches), 1) * HIDDEN, args.batches, args.tune)
                unsupported = [b for b, value in inputs.items() if not _supports(workspace, value)]
                if unsupported:
                    raise RuntimeError(f"FlashInfer unsupported shapes (no fallback): batches={unsupported}")
            outputs["metadata"][backend + "_api"] = api_name
            for batch, x in inputs.items():
                current_batch = batch
                key = f"{backend}:b{batch}"
                local = _run_shape(backend, batch, x, dist.group.WORLD, workspace, nccl,
                                   args.iterations, args.repeats)
                ranks = _all_gather_objects(local, _GLOO)
                if rank == 0:
                    outputs["results"][key] = {
                        "eager_ms_per_call_max": [max(r["eager_ms_per_call"][i] for r in ranks)
                                                   for i in range(args.repeats)],
                        "graph_ms_per_call_max": [max(r["graph_ms_per_call"][i] for r in ranks)
                                                   for i in range(args.repeats)],
                        "eager_errors": ranks[0]["eager_errors"],
                        "graph_errors": ranks[0]["graph_errors"],
                        "pattern": ranks[0]["pattern"], "ranks": ranks,
                    }
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(_jsonable(outputs), indent=2) + "\n")
            if backend == "nccl" and workspace is not None and callable(getattr(workspace, "destroy", None)):
                workspace.destroy()
        # A single small FlashInfer profiler session on rank zero; all ranks
        # still execute the same IPC calls so the protocol remains aligned.
        profile_batch = 8 if 8 in inputs else args.batches[0]
        profile_workspace = workspace
        profile_api = outputs["metadata"].get("flashinfer_pcie_ipc_api")
        if profile_workspace is None:
            raise RuntimeError("FlashInfer workspace was not retained for profiling")
        if not _supports(profile_workspace, inputs[profile_batch]):
            raise RuntimeError(f"profiler shape B={profile_batch} is unsupported by FlashInfer")
        dist.barrier(group=_GLOO)
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                      torch.profiler.ProfilerActivity.CUDA],
                                          record_shapes=False) if rank == 0 else None
        if profiler is not None:
            profiler.__enter__()
        profile_out = torch.empty_like(inputs[profile_batch])
        try:
            for _ in range(4):
                _ipc_call(profile_workspace, inputs[profile_batch], profile_out)
        finally:
            if profiler is not None:
                profiler.__exit__(None, None, None)
                events = [{"name": event.name, "device_ms": float(event.time_range.elapsed_us()) / 1000}
                          for event in profiler.events()
                          if event.device_type == torch.autograd.DeviceType.CUDA]
                names = " ".join(str(item["name"]).lower() for item in events)
                outputs["profiler"] = {
                    "backend": "flashinfer_pcie_ipc", "api": profile_api,
                    "batch": profile_batch, "rank": 0, "events": events[:100],
                    "verify_flashinfer_kernel_name": any(
                        token in names for token in ("flashinfer", "pcie", "ipc", "allreduce")),
                }
        if callable(getattr(profile_workspace, "destroy", None)):
            profile_workspace.destroy()
    except Exception as exc:
        failure = {"rank": rank, "backend": current_backend, "batch": current_batch,
                   "error": repr(exc), "traceback": traceback.format_exc()}
        outputs["failures"].append(failure)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        (args.output.parent / f"{args.output.stem}.rank{rank}.FAILED.json").write_text(
            json.dumps(_jsonable(failure), indent=2) + "\n")
    finally:
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
