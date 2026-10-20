"""Opt-in CUDA Graph topology capture for critical-path attribution.

Enable with ``DSV41_CRITICAL_PATH_DIR=/path/to/output``.  The hook is
process-local and writes one small JSON file per captured graph.
"""

from __future__ import annotations

import functools
import ctypes
import glob
import json
import os
from pathlib import Path
import runpy
import threading
import time
import traceback


_OUT = os.environ.get("DSV41_CRITICAL_PATH_DIR")


def _chain_existing_hook() -> None:
    """Keep the sibling eager-profile sitecustomize active when present."""
    base = Path(__file__).resolve().parent.parent / "kernel_profile_hook" / "sitecustomize.py"
    if base.is_file():
        runpy.run_path(str(base), run_name="<kernel_profile_hook.sitecustomize>")


if not _OUT:
    _chain_existing_hook()
else:
    _chain_existing_hook()
    _root = Path(_OUT).expanduser().resolve()
    _root.mkdir(parents=True, exist_ok=True)
    _lock = threading.Lock()
    _seen: set[int] = set()

    def _identity() -> dict[str, object]:
        keys = (
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "PP_RANK",
            "TP_RANK",
            "EP_RANK",
            "CUDA_VISIBLE_DEVICES",
        )
        result: dict[str, object] = {"pid": os.getpid(), "time_ns": time.time_ns(), **{k.lower(): os.environ[k] for k in keys if k in os.environ}}
        try:
            import torch

            result["cuda_device"] = torch.cuda.current_device()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                result["torch_rank"] = torch.distributed.get_rank()
        except Exception:
            pass
        return result

    def _write_atomic(path: Path, value: object) -> None:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
        tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
        tmp.write_bytes(payload)
        os.replace(tmp, path)

    def _dump_error(graph_key: int, exc: BaseException) -> None:
        error = {"kind": "critical_path_graph_error", "graph_key": graph_key, **_identity(), "error": repr(exc), "traceback": traceback.format_exc()}
        with _lock:
            _write_atomic(_root / f"graph_error_pid{os.getpid()}_{graph_key}.json", error)

    def _load_cuda_library(soname: str, package_glob: str):
        candidates = []
        try:
            maps = Path("/proc/self/maps").read_text().splitlines()
            candidates.extend(line.rsplit(" ", 1)[-1] for line in maps if soname in line)
        except OSError:
            pass
        candidates.extend(glob.glob(package_glob))
        candidates.append(soname)
        for candidate in dict.fromkeys(candidates):
            try:
                return ctypes.CDLL(candidate)
            except OSError:
                continue
        raise OSError(f"unable to load {soname} from the active CUDA runtime")

    def _fallback_graph_data(graph: object) -> dict[str, object]:
        """Enumerate topology without cudaGraphNodeGetToolsId (CUDA 13.0)."""
        import torch
        from cuda.bindings import runtime

        package_root = Path(torch.__file__).resolve().parent.parent
        cupti = _load_cuda_library("libcupti.so.13", str(package_root / "nvidia/cu13/lib/libcupti.so*"))
        graph_handle = ctypes.c_void_p(int(graph.raw_cuda_graph()))
        exec_handle = ctypes.c_void_p(int(graph.raw_cuda_graph_exec()))
        cupti.cuptiGetGraphId.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        cupti.cuptiGetGraphId.restype = ctypes.c_int
        cupti.cuptiGetGraphExecId.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        cupti.cuptiGetGraphExecId.restype = ctypes.c_int
        cupti.cuptiGetGraphNodeId.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
        cupti.cuptiGetGraphNodeId.restype = ctypes.c_int

        def checked(result):
            status, *values = result
            if int(status):
                raise RuntimeError(f"CUDA graph enumeration failed: {status}")
            return values

        raw = runtime.cudaGraph_t(init_value=graph_handle.value)
        _, count = checked(runtime.cudaGraphGetNodes(raw))
        nodes, count = checked(runtime.cudaGraphGetNodes(raw, numNodes=count))
        nodes = nodes[:count]
        graph_id = ctypes.c_uint32(0)
        exec_id = ctypes.c_uint32(0)
        if cupti.cuptiGetGraphId(graph_handle, ctypes.byref(graph_id)) != 0:
            raise RuntimeError("cuptiGetGraphId failed")
        if cupti.cuptiGetGraphExecId(exec_handle, ctypes.byref(exec_id)) != 0:
            raise RuntimeError("cuptiGetGraphExecId failed")
        ids: list[int] = []
        for node in nodes:
            node_id = ctypes.c_uint64(0)
            if cupti.cuptiGetGraphNodeId(ctypes.c_void_p(int(node)), ctypes.byref(node_id)) != 0:
                raise RuntimeError("cuptiGetGraphNodeId failed")
            ids.append(int(node_id.value))
        index_by_handle = {int(node): i for i, node in enumerate(nodes)}
        result_nodes = []
        for i, node in enumerate(nodes):
            node_type, = checked(runtime.cudaGraphNodeGetType(node))
            result_nodes.append({"index": i, "node_id": ids[i], "graph_id": int(exec_id.value), "source_graph_id": int(graph_id.value), "node_type": int(node_type), "dependencies": [], "dependents": []})
        *_, edge_count = checked(runtime.cudaGraphGetEdges(raw))
        edges = []
        if edge_count:
            source, destination, edge_data, edge_count = checked(runtime.cudaGraphGetEdges(raw, numEdges=edge_count))
            for i in range(edge_count):
                src, dst = index_by_handle[int(source[i])], index_by_handle[int(destination[i])]
                result_nodes[src]["dependents"].append(dst)
                result_nodes[dst]["dependencies"].append(src)
                info = {field: int(getattr(edge_data[i], field)) for field in ("from_port", "to_port", "type") if hasattr(edge_data[i], field)}
                edges.append({"source": src, "destination": dst, **info})
        return {"exec_graph_id": int(exec_id.value), "graph_id": int(graph_id.value), "nodes": result_nodes, "edges": edges, "id_source": "CUPTI"}

    def _dump_graph(graph: object) -> bool:
        key = id(graph)
        if key in _seen:
            return True
        try:
            # get_graph_data is available in the target PyTorch runtime and
            # includes CUPTI-compatible graph/node IDs plus dependency edges.
            instantiate = getattr(graph, "instantiate", None)
            if instantiate is not None:
                instantiate()
            get_data = getattr(graph, "get_graph_data", None)
            try:
                data = get_data() if get_data is not None else _fallback_graph_data(graph)
            except Exception:
                data = _fallback_graph_data(graph)
            record = {"kind": "critical_path_graph", "graph_key": key, **_identity(), "graph": data}
            with _lock:
                _write_atomic(_root / f"graph_pid{os.getpid()}_{key}.json", record)
                _seen.add(key)
            return True
        except BaseException as exc:
            _dump_error(key, exc)
            raise

    try:
        import torch

        _CUDAGraph = torch.cuda.CUDAGraph
        _original_new = _CUDAGraph.__new__
        _original_init = _CUDAGraph.__init__
        _original_capture_end = getattr(_CUDAGraph, "capture_end", None)
        _original_replay = getattr(_CUDAGraph, "replay", None)

        @functools.wraps(_original_new)
        def _new(cls, *args, **kwargs):
            if args:
                args = (True, *args[1:])
                kwargs.pop("keep_graph", None)
            else:
                kwargs["keep_graph"] = True
            return _original_new(cls, *args, **kwargs)

        _CUDAGraph.__new__ = staticmethod(_new)

        @functools.wraps(_original_init)
        def _init(self, *args, **kwargs):
            if args:
                args = (True, *args[1:])
                kwargs.pop("keep_graph", None)
            else:
                kwargs["keep_graph"] = True
            _original_init(self, *args, **kwargs)
            setattr(self, "_dsv41_critical_path_graph", True)
            setattr(self, "_dsv41_critical_path_dumped", False)

        _CUDAGraph.__init__ = _init

        if _original_capture_end is not None:
            @functools.wraps(_original_capture_end)
            def _capture_end(self, *args, **kwargs):
                result = _original_capture_end(self, *args, **kwargs)
                if getattr(self, "_dsv41_critical_path_graph", False):
                    if _dump_graph(self):
                        self._dsv41_critical_path_dumped = True
                return result

            _CUDAGraph.capture_end = _capture_end

        if _original_replay is not None:
            @functools.wraps(_original_replay)
            def _replay(self, *args, **kwargs):
                result = _original_replay(self, *args, **kwargs)
                if getattr(self, "_dsv41_critical_path_graph", False) and not getattr(self, "_dsv41_critical_path_dumped", False):
                    if _dump_graph(self):
                        self._dsv41_critical_path_dumped = True
                return result

            _CUDAGraph.replay = _replay

        _profile = torch.profiler.profile

        @functools.wraps(_profile)
        def _profile_with_sync_events(*args, **kwargs):
            if kwargs.get("experimental_config") is None:
                config_type = getattr(torch.profiler, "_ExperimentalConfig", None)
                if config_type is None:
                    config_type = torch._C._profiler._ExperimentalConfig
                kwargs["experimental_config"] = config_type(enable_cuda_sync_events=True)
            return _profile(*args, **kwargs)

        torch.profiler.profile = _profile_with_sync_events
    except BaseException as exc:
        _write_atomic(_root / f"hook_error_pid{os.getpid()}.json", {"kind": "critical_path_hook_error", **_identity(), "error": repr(exc), "traceback": traceback.format_exc()})
