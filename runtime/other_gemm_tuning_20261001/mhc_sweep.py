#!/usr/bin/env python3
"""Sequential launcher for the narrow real-input mHC split-K sweep.

This launcher performs no CUDA work itself.  Each child invokes ``mhc_probe``
for one captured shape and split factor, so it remains suitable for a single
GPU host and leaves the probe's semantic and timing JSON as the evidence.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DSV41_MHC_SWEEP_SCHEMA = "dsv41_mhc_sweep_v1"
DSV41_MHC_SPLITS = (1, 2, 4, 8, 16)


def dsv41_mhc_sweep_capture_shape(path: Path) -> str:
    """Use the hook filename to label a capture without loading GPU tensors."""
    name = path.name
    if "20480_" in name or "20480x" in name:
        return "full"
    if "5120_" in name or "5120x" in name:
        return "first"
    return "unknown"


def dsv41_mhc_sweep_main(args: argparse.Namespace) -> dict[str, Any]:
    captures = sorted(args.capture_dir.glob(args.pattern))
    if not captures:
        raise FileNotFoundError(
            f"no captures matching {args.pattern!r} under {args.capture_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    probe = args.probe.resolve()
    records: list[dict[str, Any]] = []
    for capture in captures:
        shape = dsv41_mhc_sweep_capture_shape(capture)
        if shape == "unknown":
            continue
        for split_k in DSV41_MHC_SPLITS:
            output = args.output_dir / f"{shape}_split{split_k}.json"
            command = [
                sys.executable,
                str(probe),
                "--input-capture",
                str(capture.resolve()),
                "--candidate",
                "tilelang",
                "deep_gemm",
                "--split-k",
                str(split_k),
                "--tile-n",
                str(args.tile_n),
                "--n-thr",
                str(args.n_thr),
                "--warmup",
                str(args.warmup),
                "--samples",
                str(args.samples),
                "--replays",
                str(args.replays),
                "--displacement-mib",
                str(args.displacement_mib),
                "--output",
                str(output),
            ]
            completed = subprocess.run(command, text=True, capture_output=True, timeout=600)
            record: dict[str, Any] = {
                "capture": str(capture),
                "shape": shape,
                "split_k": split_k,
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout[-2000:],
                "stderr": completed.stderr[-4000:],
                "output": str(output),
            }
            if output.is_file():
                try:
                    child = json.loads(output.read_text(encoding="utf-8"))
                    record["source_kind"] = child.get("source_kind")
                    record["candidates"] = child.get("cases", {}).get(shape, {}).get("candidates", {})
                except (OSError, json.JSONDecodeError) as exc:
                    record["output_error"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
    result = {
        "schema": DSV41_MHC_SWEEP_SCHEMA,
        "probe": str(probe),
        "capture_dir": str(args.capture_dir.resolve()),
        "variants": {"split_k": list(DSV41_MHC_SPLITS), "tile_n": args.tile_n, "n_thr": args.n_thr},
        "execution": {
            "sequential": True,
            "gpu_execution": "delegated to child probe; this launcher does not allocate CUDA tensors",
        },
        "records": records,
    }
    (args.output_dir / "sweep.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def dsv41_mhc_sweep_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="mhc_rank*.pt")
    parser.add_argument("--probe", type=Path, default=Path(__file__).with_name("mhc_probe.py"))
    parser.add_argument("--output-dir", type=Path, default=Path("mhc_sweep_results"))
    parser.add_argument("--tile-n", type=int, choices=(4, 8, 12), default=12)
    parser.add_argument("--n-thr", type=int, choices=(256, 512), default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--displacement-mib", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    dsv41_mhc_sweep_main(dsv41_mhc_sweep_args())
