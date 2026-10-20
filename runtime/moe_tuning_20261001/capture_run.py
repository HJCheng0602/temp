"""Run one fresh B64 decode request and capture selected Marlin MoE inputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

DEPLOYMENT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(DEPLOYMENT))
import parallel_matrix as matrix
from decode_report_20260929 import collect
from run_pp2_graph_comparison import interrupt
from allreduce_trial_20260930.run_decode_ab import _backend_environment, _assert_host_available


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--delegate-site", type=Path, default=None)
    args = parser.parse_args()
    _assert_host_available()
    collect._manifest_check()
    work = args.directory
    work.mkdir(parents=True, exist_ok=False)
    capture_dir = work / "moe_inputs"
    arm = work / "capture.arm"
    workload = matrix.ROOT / "pp2_graph_experiments/probe_deferred_20260926a/workload.json"
    tokens = json.loads(workload.read_text())["prompt_tokens"]
    command = collect._command(work, "all")
    index = command.index("--compilation-config")
    del command[index:index + 2]
    command += ["--enforce-eager"]
    env = _backend_environment(work, "flashinfer_pcie_ipc", "marlin")
    env.update(
        DSV41_TUNING_CAPTURE="1",
        DSV41_MOE_CAPTURE_DIR=str(capture_dir),
        DSV41_MOE_CAPTURE_ARM=str(arm),
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
        PYTHONPATH=os.pathsep.join((str(Path(__file__).with_name("capture_startup")),
                                    str(Path(__file__).parent),
                                    str(matrix.ROOT / "allreduce_trial_20260930/deps"),
                                    env.get("PYTHONPATH", ""))),
    )
    delegate_site = args.delegate_site
    if delegate_site is None:
        candidate = DEPLOYMENT / "decode_report_20260929" / "startup" / "sitecustomize.py"
        if candidate.is_file():
            delegate_site = candidate
    if delegate_site is not None:
        env["DSV41_CAPTURE_DELEGATE_SITE"] = str(delegate_site)
    matrix.URL = "http://127.0.0.1:8024"
    matrix.REQUEST_SUFFIXES.clear()
    matrix.save(work / "wave_control.json", dict(batch=1, size=1,
                log=str(work / "waves" / "initial.jsonl")))
    process = None
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGALRM, interrupt)
    try:
        signal.alarm(args.startup_timeout + 600)
        (work / "configuration.json").write_text(json.dumps({
            "schema": "dsv41_moe_capture_v1", "batch": 64,
            "prompt_tokens": len(tokens), "output_tokens": 16,
            "pp": 2, "tp": 4, "ep": 4, "enforce_eager": True,
            "selected_layers": [0, 5, 10, 19, 20, 25, 30, 39],
            "command": command,
        }, indent=2) + "\n")
        with (work / "server.log").open("w") as log:
            process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        deadline = time.monotonic() + args.startup_timeout
        while not matrix.alive():
            if process.poll() is not None:
                raise RuntimeError(f"server exited with {process.returncode}")
            if time.monotonic() >= deadline:
                raise TimeoutError("server startup timeout")
            time.sleep(3)
        arm.touch()
        matrix.save(work / "wave_control.json", dict(batch=64, size=64,
                    log=str(work / "waves" / "capture.jsonl")))
        matrix.run_case(work, "capture_b64", tokens, 64, 16)
        captures = sorted(capture_dir.glob("*.pt"))
        if not captures:
            raise RuntimeError("No real B64 decode inputs were captured")
        (work / "COMPLETE.json").write_text(json.dumps({
            "schema": "dsv41_moe_capture_v1", "capture_dir": str(capture_dir),
            "captures": [p.name for p in captures], "performance_measurement": False
        }, indent=2) + "\n")
    except BaseException as exc:
        (work / "FAILED.json").write_text(json.dumps({
            "error": repr(exc), "traceback": traceback.format_exc()
        }, indent=2) + "\n")
        raise
    finally:
        arm.unlink(missing_ok=True)
        signal.alarm(0)
        if process is not None:
            matrix.stop_server(process)


if __name__ == "__main__":
    main()
