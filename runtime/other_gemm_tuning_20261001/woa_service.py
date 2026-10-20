"""Run the established PP2 IPC A/B service with the opt-in B64 wo_a hook."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

DEPLOYMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOYMENT))
from allreduce_trial_20260930 import run_decode_ab as driver


def _tile(value: str) -> int:
    result = int(value)
    if result not in (8, 16, 32, 48, 64):
        raise argparse.ArgumentTypeError("tile must be one of 8,16,32,48,64")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--tile", type=_tile, default=None)
    parser.add_argument("--tile-config", type=json.loads, default=None)
    parser.add_argument("--backend", choices=driver.BACKENDS,
                        default="flashinfer_pcie_ipc")
    parser.add_argument("--flashinfer-root", type=Path, default=None)
    parser.add_argument("--with-profile", action="store_true")
    parser.add_argument("--woa-mode", choices=("cublaslt", "mm_full_accum"),
                        default=os.environ.get("DSV41_WOA_MODE", "mm_full_accum"))
    args, rest = parser.parse_known_args()
    if args.tile is not None and args.tile_config is not None:
        parser.error("choose --tile or --tile-config")
    if args.tile_config is not None and not isinstance(args.tile_config, dict):
        parser.error("--tile-config must be a JSON object")
    tile_config = (dict(default=args.tile) if args.tile is not None else args.tile_config)
    if tile_config is not None:
        tile_config = {str(k): _tile(str(v)) for k, v in tile_config.items()}

    os.environ["DSV41_WOA_MODE"] = args.woa_mode
    os.environ.setdefault("DSV41_WOA_EXTENSION", str(
        DEPLOYMENT / "fast_gemm_trial" / "dsv4_wo_a_cublaslt.so"))
    original_env = driver._backend_environment
    flashinfer_root = args.flashinfer_root
    if flashinfer_root is None:
        candidate = DEPLOYMENT / "allreduce_trial_20260930" / "deps"
        if candidate.is_dir():
            flashinfer_root = candidate

    def woa_environment(work, backend, moe_backend="native",
                        marlin_canonical_order=False):
        env = original_env(work, backend, moe_backend, marlin_canonical_order)
        if tile_config is not None:
            (work / "tile_config.json").write_text(
                json.dumps(tile_config, indent=2, sort_keys=True) + "\n")
        env.update({
            "DSV41_WOA_ENABLE": "1",
            "DSV41_WOA_MODE": args.woa_mode,
            "DSV41_WOA_EXTENSION": os.environ["DSV41_WOA_EXTENSION"],
            "DSV41_WOA_DELEGATE_SITE": str(
                DEPLOYMENT / "decode_report_20260929" / "startup" / "sitecustomize.py"),
        })
        # This must be first so Python executes woa_startup/sitecustomize.py.
        env["PYTHONPATH"] = os.pathsep.join((str(Path(__file__).with_name("woa_startup")),
                                               str(Path(__file__).parent),
                                               str(DEPLOYMENT / "moe_tuning_20261001"),
                                               env.get("PYTHONPATH", "")))
        if tile_config is not None and moe_backend == "marlin":
            env["DSV41_MARLIN_TILE_CONFIG"] = json.dumps(tile_config, sort_keys=True)
        return env

    driver._backend_environment = woa_environment
    forwarded = ["--directory", str(args.directory), "--backend", args.backend,
                 "--moe-backend", "marlin", "--batches", "64", "--repeats", "3",
                 "--profile-batches", "64"]
    if not args.with_profile:
        forwarded.append("--skip-profile")
    if flashinfer_root is not None:
        forwarded += ["--flashinfer-root", str(flashinfer_root)]
    sys.argv = [sys.argv[0], *forwarded, *rest]
    driver.main()


if __name__ == "__main__":
    main()
