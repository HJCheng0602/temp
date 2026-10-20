"""Install or restore the opt-in same-process graph dispatch probe."""

import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--backup-name", default="pp2-graph-probe")
    args = parser.parse_args()
    if Path(args.backup_name).name != args.backup_name:
        raise ValueError("Backup name must be a single directory name")
    root = Path(importlib.metadata.distribution("vllm").locate_file("vllm"))
    target = root / "v1/worker/gpu/model_runner.py"
    backup = root / "patches" / args.backup_name
    backup.mkdir(parents=True, exist_ok=True)
    before = backup / "model_runner.before.py"
    original = before.read_text() if before.exists() else target.read_text()
    modified = original
    replacements = [
        ("from vllm.logger import init_logger\n",
         "from vllm.logger import init_logger\nfrom . import pp2_graph_probe\n"),
        ("            need_eager=is_profile or skip_compiled,\n",
         "            need_eager=is_profile or skip_compiled or "
         "(not dummy_run and pp2_graph_probe.force_eager()),\n"),
    ]
    for old, new in replacements:
        if modified.count(old) != 1:
            raise RuntimeError(f"Expected one patch site: {old!r}")
        modified = modified.replace(old, new, 1)
    ast.parse(modified)
    if target.read_text() not in (original, modified):
        raise RuntimeError("Runner has unrelated changes; refusing to overwrite")
    if args.restore:
        target.write_text(original)
        print("PROBE_RESTORED", flush=True)
        return
    if not before.exists():
        before.write_text(original)
    target.write_text(modified)
    shutil.copy2(Path(__file__).with_name("pp2_graph_probe.py"),
                 target.with_name("pp2_graph_probe.py"))
    manifest = dict(original_sha256=hashlib.sha256(original.encode()).hexdigest(),
                    patched_sha256=hashlib.sha256(modified.encode()).hexdigest())
    (backup / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
