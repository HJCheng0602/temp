"""CUDA graph replay timing with explicit hot and displaced-cache modes."""

import statistics
import json
from pathlib import Path


def measure_graph(invoke, samples=5, replays=10, warmup=10):
    import torch

    for _ in range(warmup):
        invoke()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = invoke()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    hot, cold, outputs = [], [], []
    for _ in range(samples):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        hot.append(start.elapsed_time(end) * 1000 / replays)
        outputs.append(output.clone())
    # Exclude displacement itself from the event interval. This bounds the
    # hot-weight bias of replaying only a few experts on a large-L2 device.
    displacement = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=output.device)
    displacement.fill_(1)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    for _ in range(samples):
        for start, end in zip(starts, ends):
            displacement.add_(1)
            start.record()
            graph.replay()
            end.record()
        ends[-1].synchronize()
        cold.append(statistics.mean(start.elapsed_time(end) * 1000
                                    for start, end in zip(starts, ends)))
        outputs.append(output.clone())
    return output.clone(), hot, cold, outputs


def profile_graph(invoke, path, replays=3):
    import torch
    for _ in range(5):
        invoke()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke()
    displacement = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    displacement.fill_(1)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(replays):
            displacement.add_(1)
            graph.replay()
        torch.cuda.synchronize()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(path))
    trace = json.loads(path.read_text())
    kernels = sorted([e for e in trace["traceEvents"]
                      if e.get("cat") == "kernel" and "marlin_moe_wna16::Marlin" in e.get("name", "")],
                     key=lambda e: e["ts"])
    if len(kernels) != 2 * replays:
        raise RuntimeError(f"Expected two Marlin GEMMs per replay, found {len(kernels)} kernels")
    return {"replays": replays, "trace": str(path),
            "gate_up_us": statistics.mean(e["dur"] for e in kernels[::2]),
            "down_us": statistics.mean(e["dur"] for e in kernels[1::2]),
            "kernel_names": sorted({e["name"] for e in kernels})}
