"""Capture real B64 operands without treating capture latency as performance."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import parallel_matrix as matrix
from allreduce_trial_20260930.run_decode_ab import _assert_host_available, _backend_environment
from decode_report_20260929 import collect
from run_pp2_graph_comparison import interrupt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    _assert_host_available()
    collect._manifest_check()
    work = args.directory
    work.mkdir(parents=True, exist_ok=False)
    arm = work / "capture.arm"
    source = ROOT / "pp2_graph_experiments/probe_deferred_20260926a/workload.json"
    tokens = json.loads(source.read_text())["prompt_tokens"]
    command = collect._command(work, "all")
    i = command.index("--compilation-config")
    del command[i:i + 2]
    command += ["--enforce-eager"]
    env = _backend_environment(work, "flashinfer_pcie_ipc", "marlin")
    env.update(DSV41_OTHER_CAPTURE_ARM=str(arm),
               DSV41_OTHER_CAPTURE_DIR=str(work / "inputs"),
               DSV41_MARLIN_TILE_CONFIG='{"batch64":32}',
               DSV41_OTHER_DELEGATE_SITE=str(ROOT / "decode_report_20260929/startup/sitecustomize.py"),
               PYTHONPATH=os.pathsep.join((str(HERE / "capture_startup"), str(HERE),
                                         str(ROOT / "moe_tuning_20261001"),
                                         str(ROOT / "allreduce_trial_20260930/deps"),
                                         env.get("PYTHONPATH", ""))))
    matrix.URL = "http://127.0.0.1:8024"
    matrix.REQUEST_SUFFIXES.clear()
    matrix.save(work / "configuration.json", dict(batch=64, prompt_tokens=len(tokens),
                output_tokens=16, command=command, performance_measurement=False))
    matrix.save(work / "wave_control.json", dict(batch=1, size=1, log=str(work / "initial.jsonl")))
    process = None
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGALRM, interrupt)
    try:
        signal.alarm(2400)
        with (work / "server.log").open("w") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        matrix.save(work / "status.json", dict(phase="starting", server_pid=process.pid))
        while not matrix.alive():
            if process.poll() is not None:
                raise RuntimeError(f"capture server exited: {process.returncode}")
            time.sleep(3)
        arm.touch()
        matrix.save(work / "wave_control.json", dict(batch=64, size=64,
                    log=str(work / "waves/capture_b64.jsonl")))
        matrix.save(work / "status.json", dict(phase="capturing", server_pid=process.pid))
        matrix.run_case(work, "capture_b64", tokens, 64, 16)
        files = sorted((work / "inputs").glob("*.pt"))
        if not files:
            raise RuntimeError("No real B64 inputs captured")
        matrix.save(work / "COMPLETE.json", dict(files=[p.name for p in files],
                    performance_measurement=False))
        matrix.save(work / "status.json", dict(phase="complete", captures=len(files)))
    except BaseException as exc:
        matrix.save(work / "FAILED.json", dict(error=repr(exc), traceback=traceback.format_exc()))
        raise
    finally:
        arm.unlink(missing_ok=True)
        signal.alarm(0)
        if process is not None:
            matrix.stop_server(process)


if __name__ == "__main__":
    main()
