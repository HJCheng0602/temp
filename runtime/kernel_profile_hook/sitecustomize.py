"""Opt-in process-local eager dispatch for a separate shape-only profile."""

import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys


class ProfileFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != "vllm.v1.worker.gpu.dp_utils":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        original_exec = spec.loader.exec_module

        def execute(module):
            original_exec(module)
            dispatch = module.dispatch_cg_and_sync_dp

            def wrapped(*args, **kwargs):
                control = json.loads(Path(os.environ["DSV41_KERNEL_PROFILE_MODE"]).read_text())
                if control["mode"] == "eager":
                    kwargs["need_eager"] = True
                return dispatch(*args, **kwargs)

            module.dispatch_cg_and_sync_dp = wrapped

        spec.loader.exec_module = execute
        return spec


if os.environ.get("DSV41_KERNEL_PROFILE_MODE"):
    sys.meta_path.insert(0, ProfileFinder())
