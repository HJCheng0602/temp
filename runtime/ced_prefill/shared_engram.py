"""Share immutable CPU-offloaded Engram tables between P/D processes.

The normal vLLM Engram DP sharing path only joins one node-local DP group.
CED's prefill and decode groups are independent TP groups, so this module
uses one memfd-backed mapping per TP rank and a small JSON rendezvous file.
The writer keeps the memfd open for the lifetime of its process; consumers
open the writer's ``/proc/<pid>/fd/<fd>`` path and map the same pages.

Linux/CUDA only.  Importing this module does not import torch or numpy.
"""

from __future__ import annotations

import json
import mmap
import os
from pathlib import Path
import tempfile
import time
import weakref


_INSTALL_MARKER = "_ced_shared_engram_install"
_DEFAULT_TIMEOUT = 900.0


def _create_memfd() -> int:
    if hasattr(os, "memfd_create"):
        return os.memfd_create("vllm_ced_engram", os.MFD_CLOEXEC)
    # Some Python builds target glibc predating its memfd_create wrapper.
    import ctypes
    import platform

    numbers = {"x86_64": 319, "aarch64": 279}
    machine = platform.machine()
    if platform.system() != "Linux" or machine not in numbers:
        raise RuntimeError(f"No memfd_create fallback for {platform.system()} {machine}")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    fd = libc.syscall(ctypes.c_long(numbers[machine]),
                      ctypes.c_char_p(b"vllm_ced_engram"), ctypes.c_uint(1))
    if fd < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return int(fd)


def _torch_numpy():
    import numpy as np
    import torch

    return torch, np


def _unregister(mapping: mmap.mmap, pointer: int) -> None:
    torch, _ = _torch_numpy()
    result = torch.cuda.cudart().cudaHostUnregister(pointer)
    if getattr(result, "value", result) != 0:
        raise RuntimeError(f"cudaHostUnregister failed: {result}")
    # The numpy owner still retains the mmap while this finalizer runs.  The
    # owning SharedPinnedBuffer closes it after releasing its tensor views.


class SharedPinnedBuffer:
    """A registered shared mapping containing one Engram table and scales."""

    def __init__(
        self,
        *,
        fd: int,
        num_bytes: int,
        num_rows: int,
        dim: int,
        block_size: int,
        owner_fd: bool,
    ) -> None:
        torch, np = _torch_numpy()
        if num_bytes <= 0 or num_rows <= 0 or dim <= 0:
            raise ValueError("shared Engram dimensions must be positive")
        if dim % block_size:
            raise ValueError("Engram dim must be divisible by block size")
        expected = num_rows * dim + num_rows * (dim // block_size)
        if num_bytes != expected:
            raise ValueError(
                f"shared Engram byte mismatch: descriptor={num_bytes}, expected={expected}"
            )
        stat_size = os.fstat(fd).st_size
        if stat_size != num_bytes:
            raise ValueError(
                f"shared Engram fd size {stat_size} differs from descriptor {num_bytes}"
            )
        self.fd = fd
        self._owner_fd = owner_fd
        self.num_bytes = num_bytes
        self.num_rows = num_rows
        self.dim = dim
        self.block_size = block_size
        self._mapping = mmap.mmap(
            fd, num_bytes, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE
        )
        self._owner = np.frombuffer(self._mapping, dtype=np.uint8)
        pointer = self._owner.ctypes.data
        result = torch.cuda.cudart().cudaHostRegister(pointer, num_bytes, 0)
        if getattr(result, "value", result) != 0:
            self._owner = None
            self._mapping.close()
            raise RuntimeError(f"cudaHostRegister failed: {result}")
        self._finalizer = weakref.finalize(
            self._owner, _unregister, self._mapping, pointer
        )
        self._finalizer.atexit = False
        self._bytes = torch.from_numpy(self._owner)
        if not self._bytes.is_pinned():
            self.close(close_fd=False)
            raise RuntimeError("shared Engram mapping was not recognized as pinned")
        weight_bytes = num_rows * dim
        self.weight = self._bytes[:weight_bytes].view(torch.float8_e4m3fn).view(
            num_rows, dim
        )
        self.weight_scale_inv = self._bytes[weight_bytes:].view(torch.uint8).view(
            num_rows, dim // block_size
        )
        self._loaded_callbacks = 0

    @classmethod
    def create(cls, num_rows: int, dim: int, block_size: int) -> "SharedPinnedBuffer":
        num_bytes = num_rows * dim + num_rows * (dim // block_size)
        fd = _create_memfd()
        try:
            os.ftruncate(fd, num_bytes)
            return cls(
                fd=fd,
                num_bytes=num_bytes,
                num_rows=num_rows,
                dim=dim,
                block_size=block_size,
                owner_fd=True,
            )
        except Exception:
            os.close(fd)
            raise

    @classmethod
    def open(
        cls,
        record: dict[str, object],
        *,
        num_rows: int,
        dim: int,
        block_size: int,
        vocab_start_idx: int,
    ) -> "SharedPinnedBuffer":
        expected = {
            "num_rows": num_rows,
            "dim": dim,
            "block_size": block_size,
            "vocab_start_idx": vocab_start_idx,
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise ValueError(
                    f"shared Engram descriptor {key}={record.get(key)!r}, "
                    f"expected {value!r}"
                )
        if record.get("writer_pid", 0) <= 0 or record.get("writer_fd", -1) < 0:
            raise ValueError("invalid shared Engram writer identity")
        proc_path = str(record["proc_path"])
        try:
            fd = os.open(proc_path, os.O_RDWR)
        except OSError as exc:
            raise RuntimeError(f"cannot open live writer memfd {proc_path}: {exc}") from exc
        try:
            return cls(
                fd=fd,
                num_bytes=int(record["num_bytes"]),
                num_rows=num_rows,
                dim=dim,
                block_size=block_size,
                owner_fd=False,
            )
        except Exception:
            os.close(fd)
            raise

    def close(self, *, close_fd: bool = True) -> None:
        if getattr(self, "fd", -1) < 0:
            return
        if getattr(self, "_finalizer", None) is not None:
            self._finalizer()
            self._finalizer = None
        # Drop every Tensor view before closing the mmap.  Otherwise the
        # NumPy exporter keeps the mapping live and mmap.close raises
        # BufferError.
        self.weight = None
        self.weight_scale_inv = None
        self._bytes = None
        self._owner = None
        self._mapping.close()
        if close_fd:
            os.close(self.fd)
        self.fd = -1

    def writer_descriptor(
        self, *, num_rows: int, dim: int, block_size: int, vocab_start_idx: int
    ) -> dict[str, object]:
        if not self._owner_fd:
            raise RuntimeError("only the memfd owner can publish a descriptor")
        return {
            "num_rows": num_rows,
            "dim": dim,
            "block_size": block_size,
            "vocab_start_idx": vocab_start_idx,
            "num_bytes": self.num_bytes,
            "writer_pid": os.getpid(),
            "writer_fd": self.fd,
            "proc_path": f"/proc/{os.getpid()}/fd/{self.fd}",
            "ready": False,
        }


def _descriptor_path(outputroot: Path, rank: int) -> Path:
    return outputroot / f"ced_shared_engram_tp{rank}.json"


def _read_json(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid shared Engram descriptor {path}: {exc}") from exc


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _publish_table(path: Path, key: str, record: dict[str, object]) -> None:
    state: dict[str, object]
    if path.exists():
        state = _read_json(path)
    else:
        state = {"version": 1, "tables": {}}
    tables = state.setdefault("tables", {})
    assert isinstance(tables, dict)
    tables[key] = record
    _write_json_atomic(path, state)


def _set_ready(path: Path, key: str) -> None:
    state = _read_json(path)
    tables = state.get("tables")
    if not isinstance(tables, dict) or key not in tables:
        raise RuntimeError(f"missing shared Engram descriptor table {key}")
    record = dict(tables[key])
    record["ready"] = True
    tables[key] = record
    _write_json_atomic(path, state)


def _wait_table(path: Path, key: str, timeout: float) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    last_error = "descriptor not published"
    while time.monotonic() < deadline:
        try:
            state = _read_json(path)
            tables = state.get("tables")
            if isinstance(tables, dict) and key in tables:
                record = tables[key]
                if isinstance(record, dict) and record.get("ready"):
                    return record
                last_error = "table is published but not ready"
        except RuntimeError as exc:
            last_error = str(exc)
        time.sleep(0.05)
    raise TimeoutError(f"timed out waiting for Engram table {key}: {last_error}")


def install(role: str, rank: int, outputroot: str | os.PathLike[str]) -> None:
    """Patch NVIDIA Engram allocation for an independent P/D TP group.

    Call this before constructing the vLLM ``Worker``.  ``rank`` must be the
    TP rank shared by the P and D workers, and both roles must use the same
    ``outputroot``.  This patch is intentionally limited to CPU-offloaded,
    non-DP Engram instances.
    """
    if role not in ("prefill", "decode", "P", "D"):
        raise ValueError(f"unsupported shared Engram role {role!r}")
    role = {"P": "prefill", "D": "decode"}.get(role, role)
    if rank < 0:
        raise ValueError("Engram TP rank must be non-negative")
    outputroot_path = Path(outputroot)
    descriptor_path = _descriptor_path(outputroot_path, rank)
    timeout = float(os.environ.get("CED_SHARED_ENGRAM_TIMEOUT", _DEFAULT_TIMEOUT))

    from vllm.models.deepseek_v41.nvidia import engram as engram_module

    cls = engram_module.ParallelEngramEmbedding
    if getattr(cls, _INSTALL_MARKER, None) is not None:
        return
    if role == "prefill":
        outputroot_path.mkdir(parents=True, exist_ok=True)
        descriptor_path.unlink(missing_ok=True)
    counter = {"value": 0}

    def allocate(self):
        if not self.cpu_offload:
            raise RuntimeError("shared Engram patch requires cpu_offload=True")
        if self.dp_shared_memory or self.dp_size != 1:
            raise RuntimeError(
                "shared P/D Engram patch requires dp_shared_memory=False and dp_size=1"
            )
        key = (
            f"slot{counter['value']}-rows{self.part_num_embeddings}-dim{self.dim}"
            f"-start{self.vocab_start_idx}"
        )
        counter["value"] += 1
        if role == "prefill":
            buffer = SharedPinnedBuffer.create(
                self.part_num_embeddings, self.dim, self.block_size
            )
            record = buffer.writer_descriptor(
                num_rows=self.part_num_embeddings,
                dim=self.dim,
                block_size=self.block_size,
                vocab_start_idx=self.vocab_start_idx,
            )
            _publish_table(descriptor_path, key, record)
            self._ced_shared_engram_buffer = buffer
            loaded = {"count": 0}

            def writer_loader(param, loaded_weight):
                engram_module._engram_head_shard_weight_loader(param, loaded_weight)
                loaded["count"] += 1
                if loaded["count"] >= 2:
                    _set_ready(descriptor_path, key)

            self._weight_loader = writer_loader
            return buffer.weight, buffer.weight_scale_inv

        record = _wait_table(descriptor_path, key, timeout)
        buffer = SharedPinnedBuffer.open(
            record,
            num_rows=self.part_num_embeddings,
            dim=self.dim,
            block_size=self.block_size,
            vocab_start_idx=self.vocab_start_idx,
        )
        self._ced_shared_engram_buffer = buffer
        self._weight_loader = lambda param, loaded_weight: None
        return buffer.weight, buffer.weight_scale_inv

    setattr(cls, _INSTALL_MARKER, (role, rank, str(outputroot_path)))
    cls._allocate_weights = allocate


def _self_test_reader(
    descriptor_path: str, key: str, expected: int, write_value: int
) -> None:
    # Keep the source process on GPU 0 and exercise a separate CUDA context
    # in the reader.  The environment must be set before importing torch.
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    state = _wait_table(Path(descriptor_path), key, 30.0)
    record = state
    buffer = SharedPinnedBuffer.open(
        record,
        num_rows=int(record["num_rows"]),
        dim=int(record["dim"]),
        block_size=int(record["block_size"]),
        vocab_start_idx=int(record["vocab_start_idx"]),
    )
    torch, _ = _torch_numpy()
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    accelerator_view = get_accelerator_view_from_cpu_tensor(buffer._bytes)
    observed = accelerator_view[:1].clone().cpu()
    if int(observed.item()) != expected:
        raise AssertionError(f"shared Engram readback {int(observed.item())} != {expected}")
    accelerator_view[1] = write_value
    torch.cuda.synchronize()
    del accelerator_view, observed
    buffer.close()


def self_test() -> None:
    """Spawn a small writer/reader mapping test; requires CUDA UVA."""
    import multiprocessing as mp
    import shutil

    root = Path(tempfile.mkdtemp(prefix="ced_shared_engram_test_"))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    buffer = None
    process = None
    try:
        buffer = SharedPinnedBuffer.create(64, 128, 32)
        buffer._bytes[0] = 1
        path = root / "descriptor.json"
        key = "self-test"
        _publish_table(path, key, buffer.writer_descriptor(
            num_rows=64, dim=128, block_size=32, vocab_start_idx=0
        ) | {"ready": True})
        context = mp.get_context("spawn")
        process = context.Process(
            target=_self_test_reader, args=(str(path), key, 1, 77)
        )
        process.start()
        process.join(60)
        if process.is_alive():
            process.terminate()
            process.join(10)
            raise TimeoutError("shared Engram self-test reader did not exit")
        if process.exitcode != 0:
            raise RuntimeError(
                f"shared Engram self-test reader exited {process.exitcode}"
            )
        if int(buffer._bytes[1].item()) != 77:
            raise AssertionError("shared Engram GPU write was not visible to source")
    finally:
        if process is not None and process.is_alive():
            process.terminate()
            process.join(10)
        if buffer is not None:
            buffer.close()
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("only --self-test is supported")
    self_test()
    print("shared Engram self-test: ok")
