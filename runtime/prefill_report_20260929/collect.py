"""Load one P4/D4 pair for independent accuracy, timings and Nsight capture."""
import argparse
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback


def entry(connection, settings, role, rank, gpu_ids, port):
    sys.path.insert(0, settings["ced_dir"])
    from report_worker import worker_entry
    worker_entry(connection, settings, role, rank, gpu_ids, port)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/admin/dsv41-vllm-20260917"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", default="2048,4096,8192,16384,32768,65536,131072")
    parser.add_argument("--accuracy-sizes", default="2048,4096,8192,16384,32768,65536,131072")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--accuracy-only", action="store_true")
    parser.add_argument("--chunk", type=int, default=8192)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root / "ced_prefill"))
    sys.path.insert(0, str(args.root))
    import run as ced
    import parallel_matrix
    os.environ.update(parallel_matrix.environment())
    if args.diagnostics:
        os.environ["PREFILL_REPORT_HEAD_DIAG"] = "1"
    ced.worker_entry = entry
    sizes = [int(x) for x in args.sizes.split(",")]
    accuracy_sizes = [int(x) for x in args.accuracy_sizes.split(",") if x]
    workload = args.root / "prefill_ttft_sweep/results/baseline_8k_20260927a/workload.json"
    unit = ced.load_workload(workload)
    args.output.mkdir(parents=True, exist_ok=False)
    settings = {"model": ced.DEFAULT_MODEL, "output": str(args.output),
                "max_len": max(sizes + accuracy_sizes), "chunk": args.chunk,
                "decode_offload_gb": 12.0, "ced_dir": str(args.root / "ced_prefill")}
    handoff = Path("/dev/shm/prefill_report_20260929")
    settings["handoff_dir"] = str(handoff)
    ced.save(args.output / "configuration.json", {**settings, "sizes": sizes,
             "accuracy_sizes": accuracy_sizes, "repeats": args.repeats,
             "workload_sha256": hashlib.sha256(workload.read_bytes()).hexdigest(),
             "profile": args.profile})
    ctx = mp.get_context("spawn")
    prefill = decode = None
    records = []
    accuracy = []
    profiles = []

    def status(phase, **kw):
        ced.save(args.output / "status.json", {"phase": phase, "time": time.time(), **kw})
        print(phase, kw, flush=True)

    def case(length, name, accuracy_mode=False):
        prepare = {"request_id": name, "length": length, "accuracy": accuracy_mode}
        prefill.command("prepare_case", prepare)
        decode.command("prepare_case", prepare)
        start_ns = time.time_ns()
        value = ced.run_case(prefill, decode, unit, length, args.chunk, handoff, name, "pd")
        value["request_start_time_ns"] = start_ns
        value["request_end_time_ns"] = time.time_ns()
        return value

    try:
        status("loading_prefill")
        prefill = ced.Group(ctx, settings, "prefill", range(4), 29850, 3600)
        ced.save(args.output / "pids.json", {"prefill": prefill.pids()})
        prefill.ready()
        status("loading_decode")
        decode = ced.Group(ctx, settings, "decode", range(4, 8), 29860, 3600)
        ced.save(args.output / "pids.json", {"prefill": prefill.pids(), "decode": decode.pids()})
        decode.ready()
        status("warming")
        case(min(sizes), "shape_initial")
        for length in accuracy_sizes:
            name = f"accuracy_{length}"
            status("accuracy", length=length)
            value = case(length, name, True)
            value["ced"] = decode.command("record_transferred")
            prompt = (unit * (length // len(unit) + 1))[:length]
            for mode in ("native_bounded", "canonical_full", "original_full"):
                status("accuracy_reference", length=length, mode=mode)
                value[mode] = decode.command("full_reference", {
                    "name": mode, "mode": mode, "tokens": prompt, "chunk": args.chunk})
            value["comparisons"] = decode.command("compare_accuracy")
            accuracy.append(value)
            ced.save(args.output / "accuracy.json", accuracy)
            ced.remove_handoff_files(handoff)
        for length in ([] if args.accuracy_only else sizes):
            status("timing", length=length)
            case(length, f"shape_{length}")
            for repeat in range(args.repeats):
                records.append(case(length, f"timing_{length}_{repeat}"))
                ced.save(args.output / "records.json", records)
                ced.remove_handoff_files(handoff)
        if args.profile:
            status("profiling")
            prefill.command("profiler_start")
            decode.command("profiler_start")
            for length in sizes:
                profiles.append(case(length, f"profile_{length}"))
                ced.save(args.output / "profiles.json", profiles)
                ced.remove_handoff_files(handoff)
            decode.command("profiler_stop")
            prefill.command("profiler_stop")
        status("complete", records=len(records), accuracy=len(accuracy), profiles=len(profiles))
        ced.save(args.output / "COMPLETE.json", {"records": len(records), "accuracy": len(accuracy)})
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
