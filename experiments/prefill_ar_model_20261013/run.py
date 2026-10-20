"""Alternate NCCL and FlashInfer in one loaded P4/D4 model pair."""
import argparse
import hashlib
import importlib.util
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
import traceback


def worker_entry(connection, settings, role, rank, gpu_ids, port):
    sys.path.insert(0, settings["ced_dir"])
    sys.path.insert(0, settings["report_dir"])
    from adapter import entry
    entry(connection, settings, role, rank, gpu_ids, port)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path("/home/admin/dsv41-vllm-20260917"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", default="32768,2048,4096,8192,16384,65536,131072")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tune-cache", type=Path, default=Path(
        "/home/admin/prefill_allreduce_20261013/run1/autotune/pcie_ipc_all_reduce_ws4.json"))
    args = parser.parse_args()
    sizes = [int(x) for x in args.sizes.split(",")]
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
    handoff = Path("/dev/shm/prefill_ar_integration_20261013")
    workload = args.root / "prefill_ttft_sweep/results/baseline_8k_20260927a/workload.json"
    unit = ced.load_workload(workload)
    settings = {"model": ced.DEFAULT_MODEL, "output": str(args.output),
                "max_len": max(sizes), "chunk": 8192, "decode_offload_gb": 12.0,
                "ced_dir": str(args.root / "ced_prefill"),
                "report_dir": str(args.root / "prefill_report_20260929"),
                "handoff_dir": str(handoff)}
    ced.save(args.output / "configuration.json", {**settings, "sizes": sizes,
             "repeats": args.repeats, "scope": "prefill TP group only; decode stays NCCL",
             "workload_sha256": hashlib.sha256(workload.read_bytes()).hexdigest(),
             "environment": {k: v for k, v in os.environ.items()
                             if k.startswith(("NCCL_", "VLLM_", "FLASHINFER_", "PYTHONPATH"))}})
    ctx = mp.get_context("spawn")
    prefill = decode = None
    records = []

    def status(phase, **values):
        ced.save(args.output / "status.json", {"phase": phase, "time": time.time(), **values})
        print(phase, values, flush=True)

    def case(backend, length, name, accuracy=False, profile=False):
        prefill.command("configure_allreduce", {"backend": backend,
                        "cache_path": str(cache), "profile": profile})
        prepare = {"request_id": name, "length": length, "accuracy": accuracy}
        prefill.command("prepare_case", prepare)
        decode.command("prepare_case", prepare)
        result = ced.run_case(prefill, decode, unit, length, 8192, handoff, name, "pd")
        result["backend"] = backend
        result["communication"] = prefill.command("allreduce_snapshot")
        for reply in result["communication"]:
            calls = reply["result"]
            assert calls["calls"] > 0, "No TP AllReduce calls observed"
            expected = calls["calls"] if backend == "flashinfer" else 0
            assert calls["flashinfer_calls"] == expected, calls
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
        status("warming_32768")
        for backend in ("nccl", "flashinfer"):
            case(backend, 32768, f"warm_{backend}")
        accuracy = {}
        for backend in ("nccl", "flashinfer"):
            status("accuracy", backend=backend)
            case(backend, 32768, "accuracy_32768", accuracy=True)
            accuracy[backend] = decode.command("record_ar_accuracy", {"name": backend})
        accuracy["comparisons"] = decode.command("compare_ar_accuracy")
        ced.save(args.output / "accuracy.json", accuracy)
        for length in sizes:
            for backend in ("nccl", "flashinfer"):
                case(backend, length, f"warm_{length}_{backend}")
            for repeat in range(args.repeats):
                order = ("nccl", "flashinfer") if repeat % 2 == 0 else ("flashinfer", "nccl")
                for backend in order:
                    result = case(backend, length, f"timing_{length}_{repeat}_{backend}")
                    result["repeat"] = repeat
                    records.append(result)
                    ced.save(args.output / "records.json", records)
                    status("measured", length=length, repeat=repeat, backend=backend,
                           ttft_ms=result["ttft_ms"], prefill_ms=result["prefill_ms"])
        profiles = []
        for backend in ("nccl", "flashinfer"):
            status("operator_events", backend=backend)
            profiles.append(case(backend, 32768, f"events_{backend}", profile=True))
        ced.save(args.output / "operator_profiles.json", profiles)
        summary = []
        for length in sorted(sizes):
            row = {"length": length}
            for backend in ("nccl", "flashinfer"):
                subset = [r for r in records if r["length"] == length and r["backend"] == backend]
                row[backend] = {key: statistics.median(r[key] for r in subset)
                                for key in ("ttft_ms", "prefill_ms", "export_ms",
                                            "restore_ms", "replay_and_sample_ms")}
                row[backend]["ttft_samples_ms"] = [r["ttft_ms"] for r in subset]
            row["ttft_reduction_percent"] = 100 * (1 - row["flashinfer"]["ttft_ms"] / row["nccl"]["ttft_ms"])
            row["prefill_reduction_percent"] = 100 * (1 - row["flashinfer"]["prefill_ms"] / row["nccl"]["prefill_ms"])
            summary.append(row)
        ced.save(args.output / "summary.json", summary)
        ced.save(args.output / "COMPLETE.json", {"timed_requests": len(records)})
        status("complete", timed_requests=len(records))
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
