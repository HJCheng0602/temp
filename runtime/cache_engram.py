"""Stage only the two Engram shards locally; link the remaining mounted files."""
import json
from pathlib import Path
import shutil
import time

source = Path("/mnt/fuse/deepseek-ai/DeepSeek-V4.1-Flash")
destination = Path("/mnt/fuse/models/DeepSeek-V4.1-Flash-local-engram")
engram = {f"model-{i:05d}-of-00048.safetensors" for i in (47, 48)}
destination.mkdir(parents=True, exist_ok=True)
needed = sum((source / n).stat().st_size for n in engram
             if not (destination / n).exists())
if shutil.disk_usage(destination).free < needed + 8 * 2**30:
    raise RuntimeError("Not enough local disk space for Engram staging")
for item in source.iterdir():
    if item.name in engram or not item.is_file():
        continue
    target = destination / item.name
    if not target.exists():
        target.symlink_to(item)
for name in sorted(engram):
    src = source / name
    dst = destination / name
    if dst.exists():
        if dst.is_symlink() or dst.stat().st_size != src.stat().st_size:
            raise RuntimeError(f"Unexpected existing destination: {dst}")
        continue
    temporary = destination / (name + ".partial")
    started = time.monotonic()
    with src.open("rb") as inp, temporary.open("wb") as out:
        shutil.copyfileobj(inp, out, length=16 * 2**20)
    if temporary.stat().st_size != src.stat().st_size:
        raise RuntimeError(f"Copy size mismatch: {name}")
    temporary.rename(dst)
    print(json.dumps({"file": name, "bytes": dst.stat().st_size,
                      "seconds": time.monotonic() - started}), flush=True)
print("ENGRAM_LOCAL_CACHE_READY", str(destination), flush=True)
