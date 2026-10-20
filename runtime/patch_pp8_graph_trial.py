"""Enable a narrowly scoped PP replay CUDA-Graph trial in installed vLLM.

Run this inside the vLLM environment after the source-replay patch. The trial
is enabled only with ``DSV41_PP_GRAPH_TRIAL=1`` and a FULL graph mode.
"""

import ast
import difflib
import hashlib
import importlib.metadata
import json
from pathlib import Path


VLLM_ROOT = Path(importlib.metadata.distribution("vllm").locate_file("vllm"))
TARGET = VLLM_ROOT / "models/deepseek_v41/nvidia/pp_replay.py"
BACKUP = VLLM_ROOT / "patches/pp8-graph-trial"
ENABLED_BY = "DSV41_PP_GRAPH_TRIAL=1"
ALLOWED_MODES = ("FULL_DECODE_ONLY", "FULL")

BEFORE = '''    if not vllm_config.model_config.enforce_eager:
        raise ValueError("Source replay currently requires enforce_eager")
'''
AFTER = '''    if not vllm_config.model_config.enforce_eager:
        graph_trial = os.environ.get("DSV41_PP_GRAPH_TRIAL") == "1"
        graph_mode = getattr(vllm_config.compilation_config, "cudagraph_mode", None)
        graph_mode_name = getattr(graph_mode, "name", str(graph_mode))
        if not graph_trial or graph_mode_name not in {"FULL_DECODE_ONLY", "FULL"}:
            raise ValueError(
                "Source replay requires enforce_eager, or an explicit graph "
                "trial with FULL_DECODE_ONLY/FULL cudagraph mode"
            )
'''


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    if not TARGET.exists():
        raise FileNotFoundError(
            f"Installed replay helper not found: {TARGET}. "
            "Install patch_pp8_replay.py first."
        )
    BACKUP.mkdir(parents=True, exist_ok=True)
    before_path = BACKUP / "pp_replay.before.py"
    original = (
        before_path.read_text() if before_path.exists() else TARGET.read_text()
    )
    modified = original.replace(BEFORE, AFTER, 1)
    if modified == original:
        if AFTER not in original or original.count(AFTER) != 1:
            raise RuntimeError("Expected one exact graph-trial patch site")
        modified = original
    elif original.count(BEFORE) != 1:
        raise RuntimeError("Expected one exact enforce_eager guard")
    ast.parse(original)
    ast.parse(modified)
    current = TARGET.read_text()
    if current not in (original, modified):
        raise RuntimeError("Installed helper has other edits; refusing to overwrite")
    if not before_path.exists():
        before_path.write_text(original)
    TARGET.write_text(modified)
    (BACKUP / "pp_replay.diff").write_text(
        "".join(
            difflib.unified_diff(
                original.splitlines(True),
                modified.splitlines(True),
                fromfile="pp_replay.before.py",
                tofile=str(TARGET),
            )
        )
    )
    result = {
        "target": str(TARGET),
        "enabled_by": ENABLED_BY,
        "allowed_cudagraph_modes": list(ALLOWED_MODES),
        "original_sha256": sha256(original.encode()),
        "patched_sha256": sha256(modified.encode()),
        "backup": str(before_path),
        "launch_config": {
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [16, 32, 64, 128],
        },
    }
    (BACKUP / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
