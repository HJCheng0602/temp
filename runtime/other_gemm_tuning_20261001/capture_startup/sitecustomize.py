"""Keep the baseline startup chain, then install operand capture."""

import os
import runpy
from pathlib import Path

delegate = Path(os.environ["DSV41_OTHER_DELEGATE_SITE"])
runpy.run_path(str(delegate), run_name="<baseline.sitecustomize>")
from capture_hook import install

install()
if os.environ.get("DSV41_MARLIN_TILE_CONFIG"):
    from tune_hook import install_import_hook

    install_import_hook()
