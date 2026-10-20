"""Preserve the existing startup hook chain, then install the capture hook."""

from __future__ import annotations

import os
import runpy
from pathlib import Path

delegate = os.environ.get("DSV41_CAPTURE_DELEGATE_SITE")
if delegate:
    path = Path(delegate)
    if path.is_file() and path.resolve() != Path(__file__).resolve():
        runpy.run_path(str(path), run_name="<delegated.sitecustomize>")

if os.environ.get("DSV41_TUNING_CAPTURE") == "1":
    from capture_hook import install_import_hook

    install_import_hook()
