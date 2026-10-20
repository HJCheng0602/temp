"""Install optional stage timing without synchronizing the forward path."""

import ast
import difflib
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil


def main():
    root = Path(importlib.metadata.distribution("vllm").locate_file("vllm"))
    target = root / "v1/worker/gpu/model_runner.py"
    backup = root / "patches/pp8-stage-timer"
    backup.mkdir(parents=True, exist_ok=True)
    before = backup / "model_runner.before.py"
    original = before.read_text() if before.exists() else target.read_text()
    edited = original
    replacements = [
        ("from vllm.logger import init_logger\n",
         "from vllm.logger import init_logger\nfrom . import pp8_stage_timer\n"),
        ("        # Run model.\n",
         "        _pp8_ticket = None if dummy_run else pp8_stage_timer.start(input_batch, batch_desc)\n"
         "        # Run model.\n"),
        ("        self.kv_connector.finish_forward()\n",
         "        pp8_stage_timer.stop(_pp8_ticket, model_output)\n"
         "        self.kv_connector.finish_forward()\n"),
    ]
    for old, new in replacements:
        if edited.count(old) != 1:
            raise RuntimeError(f"Unexpected timing patch site: {old!r}")
        edited = edited.replace(old, new, 1)
    ast.parse(edited)
    if target.read_text() not in (original, edited):
        raise RuntimeError("Installed runner differs from expected versions")
    if not before.exists():
        before.write_text(original)
    target.write_text(edited)
    shutil.copy2(Path(__file__).with_name("pp8_stage_timer.py"),
                 target.with_name("pp8_stage_timer.py"))
    (backup / "model_runner.diff").write_text("".join(difflib.unified_diff(
        original.splitlines(True), edited.splitlines(True))))
    manifest = dict(target=str(target), enabled_by="DSV41_PP_TIMING_DIR",
                    original_sha256=hashlib.sha256(original.encode()).hexdigest(),
                    patched_sha256=hashlib.sha256(edited.encode()).hexdigest())
    (backup / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
