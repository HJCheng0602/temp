"""Compose the established startup chain, then install the opt-in wo_a hook."""

from __future__ import annotations

import os
import runpy
from pathlib import Path

delegate = os.environ.get("DSV41_WOA_DELEGATE_SITE")
if delegate and Path(delegate).is_file():
    runpy.run_path(delegate, run_name="<delegated.woa_sitecustomize>")

if os.environ.get("DSV41_WOA_ENABLE") == "1":
    from woa_hook import install
    install()

if os.environ.get("DSV41_MARLIN_TILE_CONFIG"):
    from tune_hook import install_import_hook
    install_import_hook()
