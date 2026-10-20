"""Live CED measurement adapter, isolated from historical experiments."""
from contextlib import contextmanager
from pathlib import Path
import time

import worker

ORIGINAL_SPARSE = None


class ReportRunner(worker.RequestRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capture_logits = False
        self.profile_active = False
        self.accuracy_paths = {}
        self.phase_records = []
        self.diagnostics = __import__("os").environ.get("PREFILL_REPORT_HEAD_DIAG") == "1"
        self.diagnostic_mode = "ced"
        if self.diagnostics:
            from head_diagnostics import install
            self.diagnostic_handles = install(self)

    def step(self, *args, **kwargs):
        model = self.runner.model
        name = "compute_logits_local" if self.runner.batch_sharder is not None else "compute_logits"
        original = getattr(model, name)

        def capture(*a, **kw):
            value = original(*a, **kw)
            self.last_logits = value.detach().float().cpu()
            return value

        if self.capture_logits and kwargs.get("sample"):
            setattr(model, name, capture)
        try:
            return super().step(*args, **kwargs)
        finally:
            setattr(model, name, original)

    def prepare_case(self, request_id, length, accuracy=False):
        import torch
        self.capture_logits = accuracy
        self.case_name = request_id
        self.case_length = length
        self.diagnostic_mode = "ced"
        self.last_logits = None
        if accuracy:
            self.finish()
            seen = set()
            for name in self.specs:
                cache = self.modules[name].kv_cache
                key = cache.untyped_storage().data_ptr()
                if key in seen:
                    continue
                seen.add(key)
                # Initialize registered storage so unwritten cache padding is
                # deterministic in byte comparisons, outside timed requests.
                storage = torch.empty(0, dtype=torch.uint8, device=cache.device)
                storage.set_(cache.untyped_storage(), 0,
                             (cache.untyped_storage().nbytes(),), (1,))
                storage.zero_()
            torch.cuda.synchronize()
        return {"prepared": request_id}

    @contextmanager
    def phase(self, name):
        import torch
        label = f"report|{self.case_name}|L={self.case_length}|{self.role}|rank={self.rank}|{name}"
        start = time.time_ns()
        torch.cuda.nvtx.range_push(label)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
            if self.profile_active:
                self.phase_records.append({"name": label, "start_time_ns": start,
                                           "end_time_ns": time.time_ns()})

    def prefill(self, *args, **kwargs):
        with self.phase("prefill"):
            return super().prefill(*args, **kwargs)

    def export(self, *args, **kwargs):
        with self.phase("export"):
            return super().export(*args, **kwargs)

    def restore(self, *args, **kwargs):
        with self.phase("restore"):
            result = super().restore(*args, **kwargs)
        if self.capture_logits:
            self.transferred_cache = self.snapshot_source(label="transferred")
        return result

    def replay(self, *args, **kwargs):
        with self.phase("replay"):
            return super().replay(trace_enabled=False)

    def snapshot_source(self, label=None):
        import torch
        from accuracy_checks import cache_snapshot
        snapshots = {}
        for name, spec in self.specs.items():
            if ".layers.20." not in name or ".swa_cache" in name:
                continue
            cache = self.modules[name].kv_cache
            group = self.layer_groups[name]
            block_ids = self.blocks[group]
            count = (len(self.tokens) + spec.block_size - 1) // spec.block_size
            ids = torch.tensor(block_ids[:count], device=cache.device, dtype=torch.long)
            snapshots[name] = cache_snapshot(cache, ids, label=name)
            if self.diagnostics and self.rank == 0:
                filename = f"{self.case_name}_{label or self.diagnostic_mode}_{name}.pages.pt"
                torch.save(cache.index_select(0, ids).detach().cpu(), self.output / filename)
        return snapshots

    def save_path(self, name, sample, continuation_steps=3):
        import torch
        logits = self.last_logits.clone()
        cache = self.snapshot_source()
        continuations = []
        for index in range(continuation_steps):
            value = self.step(1, len(self.tokens) + index, sample=True)
            continuations.append({"sample": value, "logits": self.last_logits.clone()})
        value = {"logits": logits, "sample": sample, "continuations": continuations,
                 "cache_snapshots": cache}
        self.accuracy_paths[name] = value
        if self.rank == 0:
            torch.save(value, self.output / f"{self.case_name}_{name}.pt")
        return {"sample": sample, "finite": bool(torch.isfinite(logits).all()),
                "continuation": [x["sample"] for x in continuations],
                "cache_snapshots": cache}

    def record_transferred(self, name="ced", continuation_steps=3):
        result = self.save_path(name, self.last_replay_sample, continuation_steps)
        result["transferred_source_cache"] = self.transferred_cache
        return result

    def full_reference(self, name, mode, tokens, chunk, continuation_steps=3):
        from native_reference import native_reference
        self.prepare_case(self.case_name, len(tokens), accuracy=True)
        self.diagnostic_mode = mode
        new = self.begin(f"accuracy_{name}", tokens)
        self.backbone._ced_encoder_only = False
        final = None
        for start in range(0, len(tokens), chunk):
            count = min(chunk, len(tokens) - start)
            with native_reference(self, mode, prompt_length=len(tokens),
                                  original_sparse=ORIGINAL_SPARSE):
                final = self.step(count, start, new=new if start == 0 else None,
                                  sample=start + count == len(tokens))
        if mode == "original_full":
            with native_reference(self, mode, prompt_length=len(tokens),
                                  original_sparse=ORIGINAL_SPARSE):
                return self.save_path(name, final, continuation_steps)
        return self.save_path(name, final, continuation_steps)

    def compare_accuracy(self):
        import torch
        from accuracy_checks import compare_paths, tensor_metrics, kl_divergence
        result = compare_paths(self.accuracy_paths)
        result["finite"] = {name: bool(torch.isfinite(path["logits"]).all())
                            for name, path in self.accuracy_paths.items()}
        result["continuation"] = {}
        for name, path in self.accuracy_paths.items():
            if name == "ced":
                continue
            left = self.accuracy_paths["ced"]["continuations"]
            right = path["continuations"]
            rows = []
            same_history = (self.accuracy_paths["ced"]["sample"]["token_ids"]
                            == path["sample"]["token_ids"])
            for a, b in zip(left, right):
                rows.append({"same_history": same_history,
                             "sample_equal": a["sample"]["token_ids"] == b["sample"]["token_ids"],
                             "logits": tensor_metrics(a["logits"], b["logits"]) if same_history else None,
                             "kl": kl_divergence(a["logits"], b["logits"]) if same_history else None})
                same_history = same_history and rows[-1]["sample_equal"]
            result["continuation"][name] = rows
        worker.save(self.output / f"{self.case_name}_accuracy.json", result)
        self.accuracy_paths.clear()
        return result

    def profiler_start(self):
        import torch
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        self.profile_active = True
        return {"pid": __import__("os").getpid(), "time_ns": time.time_ns()}

    def profiler_stop(self):
        import torch
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        self.profile_active = False
        worker.save(self.output / "phase_timestamps.json", self.phase_records)
        return {"stopped": True}


def worker_entry(connection, settings, role, rank, gpu_ids, port):
    import os
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu_ids, OMP_NUM_THREADS="1",
                      OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    import runtime
    install = runtime.install

    def install_and_retain(selected_role):
        global ORIGINAL_SPARSE
        from vllm.models.deepseek_v41.nvidia import flashinfer_sparse
        ORIGINAL_SPARSE = flashinfer_sparse.flashinfer_trtllm_batch_decode_sparse_mla_dsv4
        install(selected_role)

    runtime.install = install_and_retain
    worker.RequestRunner = ReportRunner
    worker.worker_main(connection, settings, role, rank, gpu_ids, port)
