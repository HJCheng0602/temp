"""Install the reviewed PP receive backport against exact recorded source hashes."""

import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).parent
    root = Path(importlib.metadata.distribution("vllm").locate_file("vllm"))
    backup = root / "patches/pp-deferred-receive-20260926"
    manifest = json.loads((source / "manifest.json").read_text())
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
    ).strip()
    if active:
        raise RuntimeError("Stop GPU jobs before changing worker code")
    candidates = {}
    for relative, hashes in manifest["files"].items():
        target = root / relative
        expected = hashes["after" if args.restore else "before"]
        if sha(target.read_bytes()) != expected:
            raise RuntimeError(f"Unexpected source hash: {relative}")
        candidate = (backup if args.restore else source) / relative
        data = candidate.read_bytes()
        if sha(data) != hashes["before" if args.restore else "after"]:
            raise RuntimeError(f"Unexpected replacement hash: {relative}")
        ast.parse(data, filename=relative)
        candidates[relative] = data
    if args.check:
        print("CHECK_OK", len(candidates), flush=True)
        return
    if not args.restore:
        if backup.exists():
            raise RuntimeError("Backup exists; refusing to overwrite it")
        for relative in candidates:
            original = backup / relative
            original.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / relative, original)
        shutil.copy2(source / "manifest.json", backup / "manifest.json")
    written = []
    originals = {relative: (root / relative).read_bytes() for relative in candidates}
    try:
        for relative, data in candidates.items():
            target = root / relative
            temporary = target.with_suffix(".py.deferred-tmp")
            temporary.write_bytes(data)
            shutil.copymode(target, temporary)
            temporary.replace(target)
            written.append(relative)
    except BaseException:
        for relative in written:
            (root / relative).write_bytes(originals[relative])
        raise
    print("RESTORED" if args.restore else "INSTALLED", len(written), flush=True)


if __name__ == "__main__":
    main()
