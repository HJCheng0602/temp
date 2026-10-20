"""Bounded FlashInfer tile comparison on the selected B64 capture."""

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
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True).strip()
    if active:
        raise RuntimeError(f"GPUs occupied: {active}")
    output = HERE / "fi_tiles"
    output.mkdir(exist_ok=False)
    env = matrix.environment()
    env.update(CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MAX_JOBS="2",
               PYTHONPATH=os.pathsep.join((str(matrix.ROOT / "allreduce_trial_20260930/deps"),
                                          str(matrix.ROOT / "fast_gemm_trial"), str(HERE))))
    records = []
    for tile in ((128, 128, 128, 128), (64, 128, 64, 128), (128, 64, 128, 64)):
        label = "_".join(map(str, tile))
        command = [str(matrix.ROOT / "env/bin/python3.12"), str(HERE / "flashinfer_probe.py"),
                   "--capture", str(HERE / "capture_b64/moe_inputs/rank07_layer20_call0.pt"),
                   "--checkpoint", matrix.MODEL, "--batch", "64", "--output",
                   str(output / f"{label}.json"), "--force-tile", *map(str, tile)]
        record = dict(tile=tile, started=time.time(), command=command)
        print(json.dumps(record), flush=True)
        with (output / f"{label}.log").open("w") as log:
            process = subprocess.Popen(command, env=env, cwd=matrix.ROOT,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
            try:
                record["exit_code"] = process.wait(timeout=600)
            except BaseException:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        record["finished"] = time.time()
        records.append(record)
        (output / "runs.json").write_text(json.dumps(records, indent=2))
        if record["exit_code"]:
            raise RuntimeError(f"FlashInfer tile {tile} failed; inspect the log")
    (output / "COMPLETE.json").write_text(json.dumps(dict(runs=len(records))))


if __name__ == "__main__":
    main()
