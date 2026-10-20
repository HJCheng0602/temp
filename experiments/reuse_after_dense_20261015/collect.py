"""Paired NCCL encoder-only profiles after the ordinary projection fix.

Only the four prefill workers are loaded. No weights or serving source is
changed; the installed exact-shape policy is switched between requests.
"""
import argparse
import hashlib
import importlib.util
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time
import traceback


def worker_entry(connection, settings, role, rank, gpu_ids, port):
    for name in ("ced_dir", "report_dir", "ar_dir", "fix_dir"):
        sys.path.insert(0, settings[name])
    from dense_adapter import entry
    entry(connection, settings, role, rank, gpu_ids, port)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path("/home/admin/dsv41-vllm-20260917")
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("ced_run", root / "ced_prefill/run.py")
    ced = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ced)
    ced.worker_entry = worker_entry
    import parallel_matrix
    os.environ.update(parallel_matrix.environment())
    for key in list(os.environ):
        if key.startswith("NCCL_"):
            del os.environ[key]
    os.environ.update(NCCL_DEBUG="WARN", MAX_JOBS="2", OMP_NUM_THREADS="1",
                      OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC="0",
                      VLLM_ALLREDUCE_USE_FLASHINFER="0", VLLM_ALLREDUCE_USE_SYMM_MEM="0",
                      DSV41_PREFILL_DENSE="1")
    os.environ["DSV41_PREFILL_DENSE_POLICY"] = str(root / "ced_prefill/prefill_dense_policy.json")
    settings = {"model": ced.DEFAULT_MODEL, "output": str(args.output),
                "max_len": 131072, "chunk": 8192, "decode_offload_gb": 12.0,
                "ced_dir": str(root / "ced_prefill"),
                "report_dir": str(root / "prefill_report_20260929"),
                "ar_dir": "/home/admin/prefill_ar_model_20261013",
                "fix_dir": "/home/admin/prefill_dense_fix_20261015"}
    workload = root / "prefill_ttft_sweep/results/baseline_8k_20260927a/workload.json"
    unit = ced.load_workload(workload)
    prompt = (unit * (32768 // len(unit) + 1))[:32768]
    cache = "/home/admin/prefill_ar_model_20261013/results/pcie_ipc_all_reduce_ws4.json"
    ced.save(args.output / "configuration.json", {
        **settings, "length": 32768, "batch": 1, "collective": "nccl",
        "scope": "P4 only; encoder layers 0..19 and layer20 cache head; no decode or KV handoff",
        "workload_sha256": hashlib.sha256(workload.read_bytes()).hexdigest(),
        "source_sha256": {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in
            (Path(__file__), root / "ced_prefill/prefill_dense.py",
             root / "ced_prefill/prefill_dense_policy.json", root / "ced_prefill/worker.py")}})
    group = None
    records, profiles = [], []

    def status(phase, **values):
        ced.save(args.output / "status.json", {"phase": phase, "time": time.time(), **values})
        print(phase, values, flush=True)

    def case(mode, request_id, verify=False):
        group.command("configure_dense", {"enabled": mode == "repaired", "profile": False,
                                           "verify": verify})
        group.command("configure_allreduce", {"backend": "nccl", "cache_path": cache,
                                               "profile": False})
        group.command("prepare_case", {"request_id": request_id, "length": 32768,
                                       "accuracy": False})
        started = time.perf_counter()
        workers = group.command("prefill", {"request_id": request_id, "tokens": prompt,
                                             "chunk": 8192})
        result = {"mode": mode, "request_id": request_id,
                  "prefill_wall_ms": (time.perf_counter() - started) * 1000,
                  "workers": workers, "projections": group.command("projection_snapshot"),
                  "communication": group.command("allreduce_snapshot")}
        for reply in result["communication"]:
            assert reply["result"]["calls"] == 164
            assert reply["result"]["flashinfer_calls"] == 0
        for reply in result["projections"]:
            s = reply["result"]
            assert s["calls"] == 400 and s["selected_calls"] == (400 if mode == "repaired" else 0)
            if verify:
                assert sum(x["equal_calls"] for x in s["verification"]) == 400
                assert sum(x["finite_calls"] for x in s["verification"]) == 400
        return result

    def interrupted(signum, frame):
        raise SystemExit(f"Interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        status("loading_prefill")
        group = ced.Group(mp.get_context("spawn"), settings, "prefill", range(4), 29950, 1800)
        ced.save(args.output / "pids.json", {"prefill": group.pids()})
        group.ready()
        ced.save(args.output / "runtime_versions.json", group.command("runtime_versions"))
        for mode in ("default", "repaired"):
            status("warming", mode=mode)
            warm = case(mode, f"warm_{mode}", verify=mode == "repaired")
            if mode == "repaired":
                ced.save(args.output / "projection_verification.json", warm["projections"])
        for repeat in range(3):
            for mode in (("default", "repaired") if repeat % 2 == 0 else ("repaired", "default")):
                record = case(mode, f"timing_{repeat}_{mode}")
                record["repeat"] = repeat
                records.append(record)
                ced.save(args.output / "records.json", records)
                status("timed", mode=mode, repeat=repeat, prefill_ms=record["prefill_wall_ms"])
        status("profiling")
        group.command("profiler_start")
        for mode in ("default", "repaired"):
            record = case(mode, f"profile_{mode}_32768")
            profiles.append(record)
            ced.save(args.output / "profiles.json", profiles)
            status("profiled", mode=mode, prefill_ms=record["prefill_wall_ms"])
        group.command("profiler_stop")
        ced.save(args.output / "COMPLETE.json", {"profiled_requests": 2, "timed_requests": 6,
                                               "projection_equal_calls": 1600})
        status("complete")
    except BaseException:
        ced.save(args.output / "FAILED.json", {"error": traceback.format_exc()})
        raise
    finally:
        if group is not None:
            try:
                group.stop()
            except Exception:
                pass
            group.terminate()


if __name__ == "__main__":
    main()
