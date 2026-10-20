"""Run bounded, sequential TP4 communication experiments in an isolated directory."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--pythonpath", default="")
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", default="default,p2p,ring_ll,ring_simple,flashinfer")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--nccl-debug", choices=("INFO", "WARN"), default="INFO")
    parser.add_argument("--tune-cache-dir", type=Path)
    parser.add_argument("--no-tune", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    definitions = {
        "default": ("pynccl", {}),
        "p2p": ("pynccl", {"NCCL_P2P_LEVEL": "PHB"}),
        "ring_ll": ("pynccl", {"NCCL_P2P_LEVEL": "PHB", "NCCL_ALGO": "allreduce:Ring", "NCCL_PROTO": "allreduce:LL"}),
        "ring_simple": ("pynccl", {"NCCL_P2P_LEVEL": "PHB", "NCCL_ALGO": "allreduce:Ring", "NCCL_PROTO": "allreduce:Simple"}),
        "flashinfer": ("flashinfer", {}),
    }
    for name in args.cases.split(","):
        backend, overrides = definitions[name]
        result_path = args.output / f"{name}.json"
        if result_path.exists():
            raise FileExistsError(f"refusing to overwrite {result_path}")
        env = {k: v for k, v in os.environ.items() if not k.startswith("NCCL_")}
        env.update(CUDA_VISIBLE_DEVICES=args.devices, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   MAX_JOBS="4", FLASHINFER_AUTOTUNE_DIR=str(args.tune_cache_dir or args.output / "autotune"),
                   NCCL_DEBUG=args.nccl_debug, NCCL_DEBUG_SUBSYS="INIT,ENV,GRAPH,TUNING,P2P,SHM",
                   NCCL_DEBUG_FILE=str(args.output / f"{name}.%h.%p.nccl.log"))
        if args.pythonpath:
            env["PYTHONPATH"] = args.pythonpath
        env.update(overrides)
        command = [args.python, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=4",
                   str(Path(__file__).with_name("benchmark.py")), "--backend", backend,
                   "--M", "128,4096,8192", "--repeats", "21", "--warmups", "8",
                   "--output", str(result_path)]
        if args.graph:
            command.append("--graph")
        if args.no_tune:
            command.append("--no-tune")
        record = {"case": name, "command": command, "started": time.time(),
                  "env": {k: v for k, v in env.items()
                          if k.startswith(("NCCL_", "CUDA_", "FLASHINFER_", "OMP_", "MKL_", "MAX_JOBS", "PYTHONPATH"))}}
        launch_path = args.output / f"{name}.launch.json"
        with (args.output / f"{name}.log").open("w") as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            record["pid"] = child.pid
            launch_path.write_text(json.dumps(record, indent=2) + "\n")
            (args.output / "active.json").write_text(json.dumps(record, indent=2) + "\n")
            try:
                record["exit_code"] = child.wait(timeout=args.timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                record["exit_code"] = "timeout_or_interrupt"
            record["finished"] = time.time()
            launch_path.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({"case": name, "exit_code": record["exit_code"],
                          "seconds": record["finished"] - record["started"]}), flush=True)
        if record["exit_code"] != 0:
            (args.output / "active.json").write_text(json.dumps({"state": "failed", "case": name}))
            raise RuntimeError(f"{name} failed; inspect its log")
    (args.output / "active.json").write_text(json.dumps({"state": "completed"}))


if __name__ == "__main__":
    main()
