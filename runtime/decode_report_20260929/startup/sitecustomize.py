"""Compose fast-GEMM and critical-path startup hooks for one worker process."""

from __future__ import annotations

import runpy
from pathlib import Path

_here = Path(__file__).resolve()
_root = _here.parents[2]
_fast = _root / "fast_gemm_trial" / "sitecustomize.py"
_critical = _root / "critical_path_hook" / "sitecustomize.py"

if _fast.is_file():
    runpy.run_path(str(_fast), run_name="<fast_gemm_trial.sitecustomize>")
if _critical.is_file():
    runpy.run_path(str(_critical), run_name="<critical_path_hook.sitecustomize>")
