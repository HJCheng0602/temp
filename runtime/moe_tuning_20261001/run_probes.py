"""Sequential, bounded GPU probes using fresh decode captures."""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[20])
    parser.add_argument("--calls", type=int, nargs="+", default=[0])
    parser.add_argument("--batches", type=int, nargs="+", default=[64])
    parser.add_argument("--backends", nargs="+", choices=["marlin", "flashinfer"], default=["marlin"])
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                      "--format=csv,noheader"], text=True).strip()
    if active:
        raise RuntimeError(f"GPU processes already running: {active}")
    import torch
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    groups = {}
    for path in sorted(args.captures.glob("*.pt")):
        item = torch.load(path, map_location="cpu", weights_only=True)
        layer, ordinal = item["layer"], item["ordinal"]
        if layer not in args.layers or ordinal not in args.calls:
            continue
        if list(item["x"].shape) != [64, 5120]:
            raise ValueError(f"Not a real B64 capture: {path}")
        ids, emap = item["topk_ids"].long(), item["expert_map"]
        selected = ids[(ids >= 0) & (emap[ids.clamp_min(0)] >= 0)]
        unique, counts = selected.unique(return_counts=True)
        entry = dict(path=str(path), layer=layer, ordinal=ordinal, rank=item["rank"],
                     local_assignments=selected.numel(),
                     expert_rows=dict(zip(unique.tolist(), counts.tolist())))
        groups.setdefault((layer, ordinal), []).append(entry)
    chosen = [max(rows, key=lambda x: x["local_assignments"])
              for _, rows in sorted(groups.items())]
    if not chosen or any(not x["local_assignments"] for x in chosen):
        raise ValueError("No nonzero local expert captures")
    manifest = dict(selection="most local routed assignments per layer and call",
                    candidates=[x for rows in groups.values() for x in rows], chosen=chosen)
    (args.output / "captures.json").write_text(json.dumps(manifest, indent=2))
    env = matrix.environment()
    for key in list(env):
        if key.startswith("DSV41_"):
            env.pop(key)
    env.update(CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MAX_JOBS="2",
               PYTHONPATH=os.pathsep.join((str(matrix.ROOT / "allreduce_trial_20260930/deps"),
                                          str(matrix.ROOT / "fast_gemm_trial"), str(HERE))))
    completed = []
    for capture in chosen:
        for backend in args.backends:
            label = f"l{capture['layer']}_r{capture['rank']}_c{capture['ordinal']}_{backend}"
            command = [str(matrix.ROOT / "env/bin/python3.12"),
                       str(HERE / ("marlin_tile_sweep.py" if backend == "marlin" else "flashinfer_probe.py")),
                       "--capture", capture["path"], "--checkpoint", matrix.MODEL,
                       "--output", str(args.output / f"{label}.json")]
            if backend == "marlin":
                command += ["--batches", *map(str, args.batches)]
            else:
                command += ["--batch", str(args.batches[-1])]
            record = dict(label=label, command=command, started=time.time())
            print(json.dumps(record), flush=True)
            with (args.output / f"{label}.log").open("w") as log:
                process = subprocess.Popen(command, env=env, cwd=matrix.ROOT,
                                           stdout=log, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, start_new_session=True)
                try:
                    record["exit_code"] = process.wait(timeout=args.timeout)
                except BaseException:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise
            record["finished"] = time.time()
            completed.append(record)
            (args.output / "runs.json").write_text(json.dumps(completed, indent=2))
            if record["exit_code"]:
                raise RuntimeError(f"{label} failed; inspect its log before continuing")
    (args.output / "COMPLETE.json").write_text(json.dumps(dict(runs=len(completed))))


if __name__ == "__main__":
    main()
