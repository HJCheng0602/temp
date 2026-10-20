"""Opt-in eager dispatch for drained, same-process CUDA Graph comparisons."""

import json
import os
from pathlib import Path

_stamp = None
_control = None


def force_eager():
    global _stamp, _control
    filename = os.environ.get("DSV41_GRAPH_PROBE_CONTROL")
    if not filename:
        return False
    path = Path(filename)
    stamp = path.stat().st_mtime_ns
    if stamp != _stamp:
        value = json.loads(path.read_text())
        if value.get("mode") not in ("eager", "graph"):
            raise ValueError(f"Invalid graph probe control: {value}")
        _control, _stamp = value, stamp
        import torch.distributed as dist

        root = Path(os.environ["DSV41_GRAPH_PROBE_ACK"])
        root.mkdir(parents=True, exist_ok=True)
        rank = dist.get_rank()
        target = root / f"epoch_{value['epoch']}_rank_{rank}.json"
        target.write_text(json.dumps(dict(**value, rank=rank, pid=os.getpid())))
    return _control["mode"] == "eager"
