"""Process-local AllReduce switch for the existing CED prefill runner."""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import statistics

from report_worker import ReportRunner


class ComparisonRunner(ReportRunner):
    def runtime_versions(self):
        import torch
        import flashinfer
        import vllm
        return {"torch": torch.__version__, "cuda": torch.version.cuda,
                "nccl": torch.cuda.nccl.version(), "flashinfer": flashinfer.__version__,
                "flashinfer_path": flashinfer.__file__, "vllm": vllm.__version__,
                "vllm_path": vllm.__file__, "gpu": torch.cuda.get_device_name(),
                "capability": torch.cuda.get_device_capability()}

    def configure_allreduce(self, backend, cache_path=None, profile=False):
        import torch
        from vllm.distributed.parallel_state import get_tp_group
        from vllm.distributed.device_communicators.flashinfer_pcie_ipc_all_reduce import (
            FlashInferPcieIpcAllReduce,
        )

        assert self.role == "prefill"
        assert backend in ("nccl", "flashinfer")
        group = get_tp_group()
        communicator = group.device_communicator
        torch.cuda.synchronize()
        if not hasattr(self, "ar_comm"):
            assert communicator.fi_pcie_ipc_ar_comm is None
            ar = FlashInferPcieIpcAllReduce(
                group=group.device_group, tune_group=group.cpu_group,
                device=torch.device("cuda", self.rank),
            )
            ar.setup(hidden_dim=5120, dtype=torch.bfloat16,
                     capture_sizes=(128, 2048, 4096, 8192),
                     tune_cache=Path(cache_path))
            assert ar.initialized, "FlashInfer workspace was not initialized"
            self.ar_comm = ar
            communicator.fi_pcie_ipc_ar_comm = ar
            original_dispatch = communicator.all_reduce
            original_flashinfer = ar.all_reduce

            def flashinfer_call(tensor):
                self.ar_hits[tuple(tensor.shape)] += 1
                return original_flashinfer(tensor)

            def dispatch(tensor):
                shape = tuple(tensor.shape)
                self.ar_calls[shape] += 1
                if not self.ar_profile:
                    return original_dispatch(tensor)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                result = original_dispatch(tensor)
                end.record()
                self.ar_events.append((shape, start, end))
                return result

            ar.all_reduce = flashinfer_call
            communicator.all_reduce = dispatch
        self.ar_comm.disabled = backend == "nccl"
        self.ar_comm.workspace.rebind_stream()
        self.ar_backend = backend
        self.ar_calls = Counter()
        self.ar_hits = Counter()
        self.ar_events = []
        self.ar_profile = profile
        return {"backend": backend, "rank": self.rank, "initialized": True}

    def allreduce_snapshot(self):
        import torch
        torch.cuda.synchronize()
        events = [{"shape": list(shape), "elapsed_ms": start.elapsed_time(end)}
                  for shape, start, end in self.ar_events]
        shapes = []
        for shape, count in sorted(self.ar_calls.items()):
            probe = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
            config = self.ar_comm.workspace.tuned_launch_config(probe)
            shapes.append({"shape": list(shape), "calls": count,
                           "flashinfer_calls": self.ar_hits[shape],
                           "payload_bytes": probe.numel() * probe.element_size(),
                           "flashinfer_config": asdict(config) if config else None})
            del probe
        return {"backend": self.ar_backend, "rank": self.rank, "shapes": shapes,
                "calls": sum(self.ar_calls.values()),
                "flashinfer_calls": sum(self.ar_hits.values()), "events": events,
                "event_sum_ms": sum(e["elapsed_ms"] for e in events),
                "event_median_ms": statistics.median(e["elapsed_ms"] for e in events)
                if events else None}

    def record_ar_accuracy(self, name):
        return self.save_path(name, self.last_replay_sample, continuation_steps=3)

    def compare_ar_accuracy(self):
        import torch
        from accuracy_checks import tensor_metrics, kl_divergence
        left = self.accuracy_paths["nccl"]
        right = self.accuracy_paths["flashinfer"]
        rows = []
        same_history = True
        for a, b in zip([left] + left["continuations"],
                        [right] + right["continuations"]):
            same_token = a["sample"]["token_ids"] == b["sample"]["token_ids"]
            rows.append({"same_history": same_history, "token_equal": same_token,
                         "nccl_token_ids": a["sample"]["token_ids"],
                         "flashinfer_token_ids": b["sample"]["token_ids"],
                         "finite": bool(torch.isfinite(a["logits"]).all()
                                        and torch.isfinite(b["logits"]).all()),
                         "logits": tensor_metrics(a["logits"], b["logits"])
                         if same_history else None,
                         "kl": kl_divergence(a["logits"], b["logits"])
                         if same_history else None})
            same_history = same_history and same_token
        result = {"rank": self.rank, "steps": rows,
                  "all_tokens_equal": all(r["token_equal"] for r in rows),
                  "all_finite": all(r["finite"] for r in rows)}
        self.accuracy_paths.clear()
        return result


def entry(connection, settings, role, rank, gpu_ids, port):
    import sys
    sys.path.insert(0, settings["ced_dir"])
    sys.path.insert(0, settings["report_dir"])
    import report_worker
    report_worker.ReportRunner = ComparisonRunner
    report_worker.worker_entry(connection, settings, role, rank, gpu_ids, port)
