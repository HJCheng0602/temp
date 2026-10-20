"""Capture two warmed 32K requests using the completed AllReduce A/B setup."""
import argparse
import hashlib
import importlib.util
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from run import worker_entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path("/home/admin/dsv41-vllm-20260917"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tune-cache", type=Path, default=Path(
        "/home/admin/prefill_ar_model_20261013/results/pcie_ipc_all_reduce_ws4.json"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.root))
    spec = importlib.util.spec_from_file_location("ced_run", args.root / "ced_prefill/run.py")
    ced = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ced)
    ced.worker_entry = worker_entry
    import parallel_matrix
    os.environ.update(parallel_matrix.environment())
    for name in list(os.environ):
        if name.startswith("NCCL_"):
            del os.environ[name]
    os.environ.update(NCCL_DEBUG="WARN", MAX_JOBS="4", OMP_NUM_THREADS="1",
                      OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC="0",
                      VLLM_ALLREDUCE_USE_FLASHINFER="0",
                      VLLM_ALLREDUCE_USE_SYMM_MEM="0")
    cache = args.output / "pcie_ipc_all_reduce_ws4.json"
    shutil.copy2(args.tune_cache, cache)
    handoff = Path("/dev/shm/prefill_ar_timeline_20261013")
    workload = args.root / "prefill_ttft_sweep/results/baseline_8k_20260927a/workload.json"
    unit = ced.load_workload(workload)
    settings = {"model": ced.DEFAULT_MODEL, "output": str(args.output),
                "max_len": 131072, "chunk": 8192, "decode_offload_gb": 12.0,
                "ced_dir": str(args.root / "ced_prefill"),
                "report_dir": str(args.root / "prefill_report_20260929"),
                "handoff_dir": str(handoff)}
    ced.save(args.output / "configuration.json", {
        **settings, "length": 32768, "batch": 1,
        "scope": "CED P4+D4; prefill TP AllReduce switch; decode remains NCCL",
        "capture": "Nsight CUDA+NVTX; no per-AllReduce event instrumentation",
        "workload_sha256": hashlib.sha256(workload.read_bytes()).hexdigest(),
        "source_sha256": {name: hashlib.sha256(
            Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("trace.py", "adapter.py", "run.py")}})
    ctx = mp.get_context("spawn")
    prefill = decode = None
    profiles = []

    def status(phase, **values):
        ced.save(args.output / "status.json", {"phase": phase, "time": time.time(), **values})
        print(phase, values, flush=True)

    def case(backend, name):
        prefill.command("configure_allreduce", {"backend": backend,
                        "cache_path": str(cache), "profile": False})
        prepare = {"request_id": name, "length": 32768, "accuracy": False}
        prefill.command("prepare_case", prepare)
        decode.command("prepare_case", prepare)
        start = time.time_ns()
        result = ced.run_case(prefill, decode, unit, 32768, 8192, handoff, name, "pd")
        result.update(backend=backend, request_start_time_ns=start,
                      request_end_time_ns=time.time_ns())
        result["communication"] = prefill.command("allreduce_snapshot")
        for reply in result["communication"]:
            calls = reply["result"]
            assert calls["calls"] == 164, calls
            assert calls["flashinfer_calls"] == (164 if backend == "flashinfer" else 0), calls
            assert not calls["events"], "Event instrumentation would perturb the trace"
        ced.remove_handoff_files(handoff)
        return result

    try:
        status("loading_prefill")
        prefill = ced.Group(ctx, settings, "prefill", range(4), 29870, 1800)
        ced.save(args.output / "pids.json", {"prefill": prefill.pids()})
        prefill.ready()
        ced.save(args.output / "runtime_versions.json", prefill.command("runtime_versions"))
        status("loading_decode")
        decode = ced.Group(ctx, settings, "decode", range(4, 8), 29880, 1800)
        ced.save(args.output / "pids.json", {"prefill": prefill.pids(), "decode": decode.pids()})
        decode.ready()
        for backend in ("nccl", "flashinfer"):
            status("warming", backend=backend)
            case(backend, f"warm_{backend}_32768")
        status("profiling")
        prefill.command("profiler_start")
        decode.command("profiler_start")
        for backend in ("nccl", "flashinfer"):
            result = case(backend, f"profile_{backend}_32768")
            profiles.append(result)
            ced.save(args.output / "profiles.json", profiles)
            status("profiled", backend=backend, ttft_ms=result["ttft_ms"])
        decode.command("profiler_stop")
        prefill.command("profiler_stop")
        ced.save(args.output / "COMPLETE.json", {"profiled_requests": len(profiles)})
        status("complete")
    except BaseException:
        ced.save(args.output / "FAILED.json", {"error": traceback.format_exc()})
        raise
    finally:
        for group in (decode, prefill):
            if group is not None:
                try:
                    group.stop()
                except Exception:
                    pass
                group.terminate()
        ced.remove_handoff_files(handoff)


if __name__ == "__main__":
    main()
