"""Launch the verified PP2/TP4/EP4 decode-Graph configuration."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import parallel_matrix as matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8024)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--timing-dir", type=Path)
    parser.add_argument("--validate-lifecycle", action="store_true")
    args = parser.parse_args()
    root = Path(importlib.metadata.distribution("vllm").locate_file("vllm"))
    manifest = json.loads((matrix.ROOT / "pp_deferred_fix/manifest.json").read_text())
    for relative, hashes in manifest["files"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != hashes["after"]:
            raise RuntimeError(f"Runtime differs from verified fix: {relative}")
    strategy = dict(name="pp2_tp4_ep4", pp=2, tp=4, dp=1, ep=True, pp_layers=[20, 20])
    command = matrix.command(strategy, matrix.ROOT)
    command.remove("--enforce-eager")
    profiler_index = command.index("--profiler-config")
    del command[profiler_index:profiler_index + 2]
    if args.profile_dir:
        command += ["--profiler-config", json.dumps(dict(
            profiler="torch", torch_profiler_dir=str(args.profile_dir),
            torch_profiler_with_stack=False, torch_profiler_record_shapes=False,
            torch_profiler_use_gzip=True, torch_profiler_dump_cuda_time_total=False,
            detailed_trace_annotation=True, ignore_frontend=True))]
    for key, value in {"--port": args.port, "--max-model-len": 4096,
                       "--max-num-seqs": 256, "--max-num-batched-tokens": 8192,
                       "--gpu-memory-utilization": .85}.items():
        command[command.index(key) + 1] = str(value)
    command += ["--engram-config", '{"cpu_offload":true}', "--async-scheduling",
                "--compilation-config", json.dumps(dict(
                    mode=0, cudagraph_mode="FULL_DECODE_ONLY",
                    cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32, 64, 96,
                                             120, 128, 136, 144, 160, 192, 224, 256]))]
    env = matrix.environment(strategy)
    for key in list(env):
        if key.startswith(("DSV41_WAVE_", "DSV41_PP_", "DSV41_GRAPH_PROBE_")):
            env.pop(key)
    for key in ("CUDA_DEVICE_MAX_CONNECTIONS", "CUDA_DEVICE_MAX_COPY_CONNECTIONS",
                "NCCL_GRAPH_MIXING_SUPPORT", "NCCL_GRAPH_STREAM_ORDERING"):
        env.pop(key, None)
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    if args.timing_dir:
        env["DSV41_PP_TIMING_DIR"] = str(args.timing_dir)
        env["DSV41_PP_TIMING_PER_RANK"] = "1"
    if args.validate_lifecycle:
        env["VLLM_SERVER_DEV_MODE"] = "1"
    print(json.dumps(dict(strategy=strategy, command=command)), flush=True)
    os.execve(command[0], command, env)


if __name__ == "__main__":
    main()
