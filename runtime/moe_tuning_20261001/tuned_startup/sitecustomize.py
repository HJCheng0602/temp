"""Compose the verified startup chain and the optional Marlin tile hook."""

from __future__ import annotations

import os
import runpy
from pathlib import Path

delegate = os.environ.get("DSV41_TUNED_DELEGATE_SITE")
if delegate:
    path = Path(delegate)
    if path.is_file() and path.resolve() != Path(__file__).resolve():
        runpy.run_path(str(path), run_name="<delegated.sitecustomize>")

if os.environ.get("DSV41_MARLIN_TILE") or os.environ.get("DSV41_MARLIN_TILE_CONFIG"):
    from tune_hook import install_import_hook

    install_import_hook()
