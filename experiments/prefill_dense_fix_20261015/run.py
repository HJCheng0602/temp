"""Paired 32K TTFT comparison; only ordinary prefill projections differ."""
import argparse
import hashlib
import importlib.util
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import signal
import statistics
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
    p.add_argument("--root", type=Path, default=Path("/home/admin/dsv41-vllm-20260917"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--collective", choices=("flashinfer", "nccl"), default="flashinfer")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.root))
    spec = importlib.util.spec_from_file_location("ced_run", args.root / "ced_prefill/run.py")
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
    fix_dir = Path(__file__).resolve().parent
    policy = args.root / "ced_prefill/prefill_dense_policy.json"
    os.environ["DSV41_PREFILL_DENSE_POLICY"] = str(policy)
    cache = args.output / "pcie_ipc_all_reduce_ws4.json"
    shutil.copy2("/home/admin/prefill_ar_model_20261013/results/pcie_ipc_all_reduce_ws4.json", cache)
    handoff = Path("/dev/shm/prefill_dense_fix_20261015")
    workload = args.root / "prefill_ttft_sweep/results/baseline_8k_20260927a/workload.json"
    unit = ced.load_workload(workload)
    settings = {"model": ced.DEFAULT_MODEL, "output": str(args.output),
                "max_len": 131072, "chunk": 8192, "decode_offload_gb": 12.0,
                "ced_dir": str(args.root / "ced_prefill"),
                "report_dir": str(args.root / "prefill_report_20260929"),
                "ar_dir": "/home/admin/prefill_ar_model_20261013",
                "fix_dir": str(fix_dir), "handoff_dir": str(handoff)}
    code = [Path(__file__), fix_dir / "dense_adapter.py", policy,
            args.root / "ced_prefill/prefill_dense.py", args.root / "ced_prefill/worker.py"]
    ced.save(args.output / "configuration.json", {**settings, "length": 32768, "batch": 1,
             "repeats": args.repeats, "allreduce": f"prefill {args.collective}; decode NCCL",
             "scope": "same loaded P4+D4 pair; only exact-shape prefill MXFP8 fallback tactics change",
             "workload_sha256": hashlib.sha256(workload.read_bytes()).hexdigest(),
             "source_sha256": {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in code}})
    ctx = mp.get_context("spawn")
    prefill = decode = None
    records = []

    def status(phase, **values):
        ced.save(args.output / "status.json", {"phase": phase, "time": time.time(), **values})
        print(phase, values, flush=True)

    def case(mode, name, accuracy=False, profile=False, verify=False):
        prefill.command("configure_dense", {"enabled": mode == "repaired", "profile": profile,
                                             "verify": verify})
        prepare = {"request_id": name, "length": 32768, "accuracy": accuracy}
        prefill.command("prepare_case", prepare)
        decode.command("prepare_case", prepare)
        result = ced.run_case(prefill, decode, unit, 32768, 8192, handoff, name, "pd")
        result["projection_mode"] = mode
        result["projections"] = prefill.command("projection_snapshot")
        for reply in result["projections"]:
            stats = reply["result"]
            assert stats["calls"] > 0, "No ordinary projection calls intercepted"
            assert stats["selected_calls"] == (stats["calls"] if mode == "repaired" else 0), stats
        ced.remove_handoff_files(handoff)
        return result

    def interrupted(signum, frame):
        raise SystemExit(f"Interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        status("loading_prefill")
        prefill = ced.Group(ctx, settings, "prefill", range(4), 29910, 1800)
        ced.save(args.output / "pids.json", {"prefill": prefill.pids()})
        prefill.ready()
        ced.save(args.output / "runtime_versions.json", prefill.command("runtime_versions"))
        status("loading_decode")
        decode = ced.Group(ctx, settings, "decode", range(4, 8), 29920, 1800)
        ced.save(args.output / "pids.json", {"prefill": prefill.pids(), "decode": decode.pids()})
        decode.ready()
        prefill.command("configure_allreduce", {"backend": args.collective,
                        "cache_path": str(cache), "profile": False})
        for mode in ("default", "repaired"):
            status("warming", mode=mode)
            warm = case(mode, f"warm_{mode}", verify=mode == "repaired")
            if mode == "repaired":
                ced.save(args.output / "real_projection_verification.json", warm["projections"])
                verified = all(row['calls'] == row['equal_calls'] == row['finite_calls']
                               for rank in warm['projections']
                               for row in rank['result']['verification'])
        accuracy = {"flashinfer": {}, "nccl": {}}
        fi_cases = (("default", "default"), ("default_repeat", "default"),
                    ("repaired", "repaired")) if args.collective == "flashinfer" else ()
        for name, mode in fi_cases:
            status("accuracy_flashinfer", name=name)
            case(mode, "accuracy_32768", accuracy=True)
            accuracy['flashinfer'][name] = decode.command("record_dense_accuracy", {"name": name})
        if fi_cases:
            accuracy['flashinfer']['default_repeat_comparisons'] = decode.command(
                "compare_dense_accuracy", {"left_name": "default", "right_name": "default_repeat"})
            accuracy['flashinfer']['projection_comparisons'] = decode.command("compare_dense_accuracy")
        prefill.command("configure_allreduce", {"backend": "nccl", "cache_path": str(cache)})
        for mode in ("default", "repaired"):
            status("accuracy_nccl", mode=mode)
            case(mode, "accuracy_nccl_32768", accuracy=True)
            accuracy['nccl'][mode] = decode.command("record_dense_accuracy", {"name": 'nccl_' + mode})
        accuracy['nccl']['comparisons'] = decode.command("compare_dense_accuracy",
            {"left_name": "nccl_default", "right_name": "nccl_repaired"})
        ced.save(args.output / "accuracy.json", accuracy)
        assert verified, "At least one real projection output changed"
        assert all(r['result']['passed'] for r in accuracy['nccl']['comparisons']), \
            "Projection accuracy with NCCL check failed"
        prefill.command("configure_allreduce", {"backend": args.collective, "cache_path": str(cache)})
        for repeat in range(args.repeats):
            modes = ("default", "repaired") if repeat % 2 == 0 else ("repaired", "default")
            for mode in modes:
                result = case(mode, f"timing_{repeat}_{mode}")
                result["repeat"] = repeat
                records.append(result)
                ced.save(args.output / "records.json", records)
                status("measured", mode=mode, repeat=repeat, ttft_ms=result["ttft_ms"],
                       prefill_ms=result["prefill_ms"])
        profiles = []
        for mode in ("default", "repaired"):
            status("projection_events", mode=mode)
            profiles.append(case(mode, f"events_{mode}", profile=True))
        ced.save(args.output / "projection_profiles.json", profiles)
        summary = {"length": 32768, "batch": 1, "chunk": 8192, "repeats": args.repeats,
                   "prefill_collective": args.collective}
        for mode in ("default", "repaired"):
            subset = [r for r in records if r["projection_mode"] == mode]
            summary[mode] = {key: statistics.median(r[key] for r in subset)
                             for key in ("ttft_ms", "prefill_ms", "export_ms", "restore_ms",
                                         "replay_and_sample_ms")}
            summary[mode]["ttft_samples_ms"] = [r["ttft_ms"] for r in subset]
            summary[mode]["prefill_samples_ms"] = [r["prefill_ms"] for r in subset]
        summary["ttft_reduction_percent"] = 100 * (1-summary["repaired"]["ttft_ms"]/summary["default"]["ttft_ms"])
        summary["prefill_reduction_percent"] = 100 * (1-summary["repaired"]["prefill_ms"]/summary["default"]["prefill_ms"])
        summary["paired_ttft_savings_ms"] = [
            next(r["ttft_ms"] for r in records if r["repeat"] == i and r["projection_mode"] == "default")
            - next(r["ttft_ms"] for r in records if r["repeat"] == i and r["projection_mode"] == "repaired")
            for i in range(args.repeats)]
        ced.save(args.output / "summary.json", summary)
        ced.save(args.output / "COMPLETE.json", {"timed_requests": len(records),
            "real_projection_outputs_equal": verified,
            "nccl_projection_accuracy_passed": True,
            "flashinfer_projection_accuracy_passed": all(r['result']['passed']
                for r in accuracy['flashinfer'].get('projection_comparisons', [])) if fi_cases else None,
            "flashinfer_default_repeat_accuracy_passed": all(r['result']['passed']
                for r in accuracy['flashinfer'].get('default_repeat_comparisons', [])) if fi_cases else None})
        status("complete", **summary)
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
