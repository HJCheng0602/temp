"""Run baseline and tuned B64 services serially, preserving all evidence."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import parallel_matrix as matrix


def main():
    records = []
    for label, policy in (("service_baseline", None),
                          ("service_m32", {"batch64": 32})):
        command = [str(matrix.ROOT / "env/bin/python3.12"),
                   str(HERE / "service_ab.py"), "--directory", str(HERE / label)]
        if policy:
            command += ["--tile-config", json.dumps(policy)]
        record = dict(label=label, command=command, started=time.time())
        print(json.dumps(record), flush=True)
        with (HERE / f"{label}_driver.log").open("w") as log:
            process = subprocess.Popen(command, env=matrix.environment(),
                                       cwd=matrix.ROOT, stdout=log,
                                       stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL,
                                       start_new_session=True)
            try:
                record["exit_code"] = process.wait(timeout=3600)
            except BaseException:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        record["finished"] = time.time()
        records.append(record)
        (HERE / "service_pair_runs.json").write_text(json.dumps(records, indent=2))
        if record["exit_code"]:
            raise RuntimeError(f"{label} failed; inspect its saved log")
        deadline = time.monotonic() + 60
        while subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                text=True).strip():
            if time.monotonic() >= deadline:
                raise RuntimeError("GPU workers did not exit after service cleanup")
            time.sleep(2)


if __name__ == "__main__":
    main()
