"""B=1 CED worker using the installed vLLM worker and metadata builders."""

from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import time
import traceback


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def dedicated_cache_config(specs, max_len):
    """Use disjoint page regions per layer for one explicitly managed request."""
    from vllm.v1.kv_cache_interface import (
        CircularBufferSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor,
    )

    blocks = 1 + max(math.ceil(max_len / spec.block_size)
                     for spec in specs.values() if not isinstance(spec, CircularBufferSpec))
    groups = []
    tensors = []
    offset = 0
    for name, spec in specs.items():
        matching = next((group for group in groups if group.kv_cache_spec == spec), None)
        if matching is None:
            groups.append(KVCacheGroupSpec(layer_names=[name], kv_cache_spec=spec))
        else:
            matching.layer_names.append(name)
        page = spec.page_size_bytes
        offset = math.ceil(offset / 512) * 512
        tensors.append(KVCacheTensor(size=0, layers=[name], layer_stride=page * blocks,
                                     block_stride=page, offset=offset))
        offset += page * blocks
    for tensor in tensors:
        tensor.size = offset
    return KVCacheConfig(num_blocks=blocks, kv_cache_tensors=tensors,
                         kv_cache_groups=groups)


class RequestRunner:
    def __init__(self, worker, config, output, rank, role):
        import runtime

        self.worker = worker
        self.runner = worker.model_runner
        self.config = config
        self.output = Path(output)
        self.rank = rank
        self.role = role
        self.backbone = runtime.text_backbone(self.runner.model)
        self.specs = self.runner.get_kv_cache_spec()
        self.groups = self.runner.kv_cache_config.kv_cache_groups
        self.layer_groups = {name: index for index, group in enumerate(self.groups)
                             for name in group.layer_names}
        self.modules = config.compilation_config.static_forward_context
        self.active = None
        self.last_logits = None

    def block_ids(self, length):
        from vllm.v1.kv_cache_interface import CircularBufferSpec

        return tuple([1] if isinstance(group.kv_cache_spec, CircularBufferSpec)
                     else list(range(1, math.ceil(length / group.kv_cache_spec.block_size) + 1))
                     for group in self.groups)

    def begin(self, request_id, tokens, computed=0):
        from vllm import SamplingParams
        from vllm.v1.core.sched.output import NewRequestData

        if self.active is not None:
            self.finish()
        self.active = request_id
        self.tokens = tokens
        self.token_hash = hashlib.sha256(
            json.dumps(tokens, separators=(",", ":")).encode()
        ).hexdigest()
        self.blocks = self.block_ids(len(tokens) + 8)
        self.backbone._ced_tail = None
        self.backbone._ced_audit = []
        self.backbone._ced_replay = None
        self.backbone._ced_head_check = None
        return NewRequestData(
            req_id=request_id, prompt_token_ids=tokens, prefill_token_ids=tokens,
            mm_features=[], sampling_params=SamplingParams(temperature=0, max_tokens=8),
            pooling_params=None, block_ids=self.blocks, num_computed_tokens=computed,
            lora_request=None,
        )

    def step(self, count, computed, *, new=None, sample=False):
        import torch
        from vllm.v1.core.sched.output import SchedulerOutput, CachedRequestData

        schedule = SchedulerOutput.make_empty()
        schedule.num_scheduled_tokens = {self.active: count}
        schedule.total_num_scheduled_tokens = count
        schedule.num_common_prefix_blocks = [0] * len(self.groups)
        if new is not None:
            schedule.scheduled_new_reqs = [new]
        else:
            cached = CachedRequestData.make_empty()
            cached.req_ids = [self.active]
            cached.new_block_ids = [None]
            cached.num_computed_tokens = [computed]
            cached.num_output_tokens = [max(0, computed - len(self.tokens) + 1)]
            schedule.scheduled_cached_reqs = cached
        self.worker.execute_model(schedule)
        if sample:
            model = self.runner.model
            logits_method = ("compute_logits_local" if self.runner.batch_sharder is not None
                             else "compute_logits")
            original_logits = getattr(model, logits_method)
            capture = self.active.startswith(("warmup", "reference"))

            def capture_logits(*args, **kwargs):
                logits = original_logits(*args, **kwargs)
                self.last_logits = logits.detach().float().cpu()
                return logits

            if capture:
                setattr(model, logits_method, capture_logits)
            try:
                result = self.worker.sample_tokens(None)
            finally:
                if capture:
                    setattr(model, logits_method, original_logits)
            if hasattr(result, "get_output"):
                result = result.get_output()
            torch.cuda.synchronize()
            return dict(token_ids=result.sampled_token_ids,
                        request_ids=result.req_ids)
        state = self.runner.execute_model_state
        self.runner.postprocess_num_computed_tokens(state.input_batch)
        self.runner.model_state.postprocess_state(state.input_batch.idx_mapping, 0)
        self.runner.kv_connector.post_forward(state.finished_req_ids)
        self.runner.execute_model_state = None
        torch.cuda.synchronize()
        return None

    def prefill(self, request_id, tokens, chunk):
        from contextlib import nullcontext
        from trace_utils import capture_backbone
        from sparse_probe import capture_sparse_calls

        new = self.begin(request_id, tokens)
        started = time.perf_counter()
        chunks = []
        diagnostic = request_id in ("warmup", "reference_encoder")
        context = capture_backbone(self.backbone, max_layers=20) if diagnostic else nullcontext()
        probe = (capture_sparse_calls(self.output, request_id + "_encoder")
                 if diagnostic and self.rank == 0 else nullcontext())
        with context as trace, probe:
            for start in range(0, len(tokens), chunk):
                count = min(chunk, len(tokens) - start)
                before = time.perf_counter()
                self.step(count, start, new=new if start == 0 else None)
                chunks.append(dict(start=start, rows=count,
                                   elapsed_ms=(time.perf_counter() - before) * 1000))
        self.encoder_trace = trace.records if diagnostic else None
        if diagnostic:
            trace.save_raw(self.output / f"{request_id}_encoder_trace.pt")
        return dict(prefill_ms=(time.perf_counter() - started) * 1000,
                    chunks=chunks, audit=self.backbone._ced_audit)

    def export(self, path):
        import torch
        from vllm.v1.kv_cache_interface import CircularBufferSpec, SlidingWindowSpec

        started = time.perf_counter()
        length = len(self.tokens)
        caches = {}
        for name, spec in self.specs.items():
            layer = int(name.split(".layers.")[1].split(".")[0])
            if layer == 20 and ".swa_cache" in name:
                continue
            cache = self.modules[name].kv_cache
            if not isinstance(cache, torch.Tensor):
                raise TypeError(f"Unsupported cache container {name}: {type(cache)}")
            if isinstance(spec, CircularBufferSpec):
                logical = [0]
            elif isinstance(spec, SlidingWindowSpec):
                logical = list(range(max(0, length - 128) // spec.block_size,
                                     math.ceil(length / spec.block_size)))
            else:
                logical = list(range(math.ceil(length / spec.block_size)))
            source_ids = [self.blocks[self.layer_groups[name]][i] for i in logical]
            ids = torch.tensor(source_ids, device=cache.device, dtype=torch.long)
            caches[name] = dict(logical_blocks=logical, block_size=spec.block_size,
                                data=cache.index_select(0, ids).cpu())
        tail = {key: value.cpu() for key, value in self.backbone._ced_tail.items()}
        # Engram's slot-keyed rolling state is outside the paged KV registry.
        engram = {}
        if self.backbone.engram_hash is not None:
            for name, value in vars(self.backbone.engram_hash).items():
                if isinstance(value, torch.Tensor) and value.device.type == "cuda":
                    if value.numel() < 100_000_000:
                        engram[name] = value.cpu()
        payload = dict(version=1, request_id=self.active, prompt_tokens=length,
                       token_hash=self.token_hash,
                       tail=tail, caches=caches, engram=engram)
        if self.active == "warmup":
            payload["encoder_trace"] = self.encoder_trace
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        return dict(export_ms=(time.perf_counter() - started) * 1000,
                    bytes=path.stat().st_size, cache_count=len(caches), tail_rows=len(tail["positions"]))

    def restore(self, path, request_id, tokens):
        import torch

        started = time.perf_counter()
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["request_id"] != request_id or payload["prompt_tokens"] != len(tokens):
            raise RuntimeError("P/D handoff request mismatch")
        token_hash = hashlib.sha256(
            json.dumps(tokens, separators=(",", ":")).encode()
        ).hexdigest()
        if payload["token_hash"] != token_hash:
            raise RuntimeError("P/D handoff prompt content mismatch")
        count = len(payload["tail"]["positions"])
        self.last_handoff_path = path
        validate = request_id.startswith("warmup")
        self.restore_checks = {}
        if validate:
            self.handoff_tail_cpu = payload["tail"]
            self.handoff_encoder_trace = payload.get("encoder_trace")
        self.pending_new = self.begin(request_id, tokens, len(tokens) - count)
        for name, item in payload["caches"].items():
            if name not in self.specs or item["block_size"] != self.specs[name].block_size:
                raise RuntimeError(f"Incompatible destination cache: {name}")
            cache = self.modules[name].kv_cache
            ids = [self.blocks[self.layer_groups[name]][i] for i in item["logical_blocks"]]
            device_ids = torch.tensor(ids, device=cache.device, dtype=torch.long)
            cache.index_copy_(0, device_ids, item["data"].to(cache.device))
            if validate:
                self.restore_checks[name] = bool(torch.equal(
                    cache.index_select(0, device_ids).cpu(), item["data"]
                ))
                if not self.restore_checks[name]:
                    raise RuntimeError(f"Cache transfer changed bytes: {name}")
        if payload["engram"]:
            for name, value in payload["engram"].items():
                destination = getattr(self.backbone.engram_hash, name)
                if destination.shape != value.shape:
                    raise RuntimeError(f"Engram state shape mismatch: {name}")
                destination.copy_(value)
        self.backbone._ced_replay = {key: value.to(self.worker.device)
                                     for key, value in payload["tail"].items()}
        torch.cuda.synchronize()
        return dict(import_ms=(time.perf_counter() - started) * 1000, replay_rows=count)

    def replay(self, trace_enabled=True):
        from contextlib import nullcontext
        from trace_utils import capture_backbone
        from sparse_probe import capture_sparse_calls

        count = self.backbone._ced_replay["hidden"].shape[0]
        self.backbone._ced_validate = self.active.startswith(("warmup", "reference"))
        started = time.perf_counter()
        diagnostic = trace_enabled and self.active in ("warmup", "reference_replay")
        context = capture_backbone(self.backbone) if diagnostic else nullcontext()
        serial = getattr(self, "trace_serial", 0)
        probe = (capture_sparse_calls(self.output, f"{self.active}_replay_{serial}")
                 if diagnostic and self.rank == 0 else nullcontext())
        with context as trace, probe:
            result = self.step(count, len(self.tokens) - count, new=self.pending_new, sample=True)
        self.replay_trace = trace.records if diagnostic else None
        if diagnostic:
            trace.save_raw(self.output / f"replay_trace_{serial}.pt")
            self.trace_serial = serial + 1
        self.last_replay_sample = result
        self.backbone._ced_replay = None
        self.backbone._ced_validate = False
        return dict(replay_and_sample_ms=(time.perf_counter() - started) * 1000,
                    result=result, audit=self.backbone._ced_audit,
                    head_check=self.backbone._ced_head_check)

    def reference(self, chunk):
        """Compare transferred-state replay with same-semantics local execution."""
        import torch
        from trace_utils import compare_traces

        if self.last_logits is None:
            raise RuntimeError("Run a sampled warmup before the reference comparison")
        transferred_logits = self.last_logits.clone()
        transferred_sample = self.last_replay_sample["token_ids"]
        transferred_trace = self.replay_trace
        transferred_encoder_trace = self.handoff_encoder_trace
        transferred_decode = self.decode(3)
        tokens = list(self.tokens)
        self.restore(self.last_handoff_path, self.active, tokens)
        repeated_replay = self.replay()
        repeated_logits = self.last_logits.clone()
        repeat_difference = transferred_logits - repeated_logits
        replay_trace_difference = compare_traces(transferred_trace, self.replay_trace)
        save(self.output / "repeat_trace_difference.json", replay_trace_difference)
        repeated_allclose = bool(torch.allclose(
            transferred_logits, repeated_logits, rtol=0.01, atol=0.02
        ))
        restore_checks = dict(self.restore_checks)
        transferred_tail = self.handoff_tail_cpu
        self.restore(self.last_handoff_path, self.active, tokens)
        untraced_replay = self.replay(trace_enabled=False)
        untraced_logits = self.last_logits.clone()
        untraced_difference = transferred_logits - untraced_logits
        untraced_allclose = bool(torch.allclose(
            transferred_logits, untraced_logits, rtol=0.01, atol=0.02
        ))
        self.backbone._ced_encoder_only = True
        try:
            encoder = self.prefill("reference_encoder", tokens, chunk)
        finally:
            self.backbone._ced_encoder_only = False
        tail = self.backbone._ced_tail
        encoder_trace_difference = compare_traces(transferred_encoder_trace, self.encoder_trace)
        save(self.output / "encoder_trace_difference.json", encoder_trace_difference)
        boundary_difference = tail["hidden"].float().cpu() - transferred_tail["hidden"].float()
        boundary_check = dict(
            max_abs=float(boundary_difference.abs().max()),
            rmse=float(boundary_difference.square().mean().sqrt()),
            relative_l2=float(boundary_difference.norm()
                              / transferred_tail["hidden"].float().norm().clamp_min(1e-12)),
        )
        self.pending_new = self.begin("reference_replay", tokens,
                                      len(tokens) - len(tail["positions"]))
        self.backbone._ced_replay = tail
        replay_result = self.replay()
        local_logits = self.last_logits
        if not torch.isfinite(transferred_logits).all() or not torch.isfinite(local_logits).all():
            raise RuntimeError("Non-finite logits in the bounded-replay comparison")
        difference = transferred_logits - local_logits
        result = dict(
            encoder=encoder, replay=replay_result,
            max_abs=float(difference.abs().max()),
            rmse=float(difference.square().mean().sqrt()),
            argmax_equal=bool(torch.equal(transferred_logits.argmax(-1), local_logits.argmax(-1))),
            same_semantics=True, tolerance=dict(rtol=0.01, atol=0.02),
            allclose=bool(torch.allclose(transferred_logits, local_logits, rtol=0.01, atol=0.02)),
            restored_cache_bytes_equal=restore_checks,
            same_handoff_repeat=dict(
                allclose=repeated_allclose, max_abs=float(repeat_difference.abs().max()),
                rmse=float(repeat_difference.square().mean().sqrt()),
                sample=repeated_replay["result"],
                sample_equal=repeated_replay["result"]["token_ids"] == transferred_sample,
            ),
            encoder_boundary_difference=boundary_check,
            logits_shape=list(transferred_logits.shape),
            repeat_trace_first_mismatch=replay_trace_difference["first_mismatch"],
            encoder_trace_first_mismatch=encoder_trace_difference["first_mismatch"],
            untraced_repeat=dict(allclose=untraced_allclose,
                                 max_abs=float(untraced_difference.abs().max()),
                                 rmse=float(untraced_difference.square().mean().sqrt()),
                                 sample=untraced_replay["result"]),
        )
        local_decode = self.decode(3)
        result["transferred_decode"] = transferred_decode
        result["local_decode"] = local_decode
        result["continuation_equal"] = (
            [step["result"]["token_ids"] for step in transferred_decode]
            == [step["result"]["token_ids"] for step in local_decode]
        )
        new = self.begin("reference_full", tokens)
        full_result = None
        for start in range(0, len(tokens), chunk):
            count = min(chunk, len(tokens) - start)
            full_result = self.step(count, start, new=new if start == 0 else None,
                                    sample=start + count == len(tokens))
        full_logits = self.last_logits
        if not torch.isfinite(full_logits).all():
            raise RuntimeError("Non-finite logits in the full-forward comparison")
        full_difference = local_logits - full_logits
        result["approximation_to_full"] = dict(
            max_abs=float(full_difference.abs().max()),
            rmse=float(full_difference.square().mean().sqrt()),
            argmax_equal=bool(torch.equal(local_logits.argmax(-1), full_logits.argmax(-1))),
            sampled_token_equal=full_result["token_ids"] == replay_result["result"]["token_ids"],
            full_sample=full_result, bounded_sample=replay_result["result"],
            exact_equivalence_required=False,
        )
        # Cross-node encoder differences are diagnostic; transport and replay
        # must still reproduce the same transferred state with tracing disabled.
        result["acceptance_basis"] = "byte-exact caches, repeat replay, and matching greedy continuation"
        result["passed"] = (bool(restore_checks) and all(restore_checks.values())
                            and repeated_allclose and untraced_allclose and result["argmax_equal"]
                            and result["same_handoff_repeat"]["sample_equal"]
                            and result["continuation_equal"])
        save(self.output / "reference.json", result)
        if not result["passed"]:
            raise RuntimeError(f"Bounded-replay reference differs; see {self.output / 'reference.json'}")
        return result

    def decode(self, steps):
        results = []
        for step in range(steps):
            started = time.perf_counter()
            result = self.step(1, len(self.tokens) + step, sample=True)
            results.append(dict(elapsed_ms=(time.perf_counter() - started) * 1000,
                                result=result))
        return results

    def finish(self):
        from vllm.v1.core.sched.output import SchedulerOutput

        if self.active is not None:
            schedule = SchedulerOutput.make_empty()
            schedule.finished_req_ids = {self.active}
            self.worker.execute_model(schedule)
        self.active = None
        self.backbone._ced_tail = None
        self.backbone._ced_replay = None


def worker_main(connection, settings, role, rank, gpu_ids, port):
    output = Path(settings["output"]) / f"{role}_rank{rank}"
    output.mkdir(parents=True, exist_ok=True)
    log = open(output / "worker.log", "a", buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu_ids, OMP_NUM_THREADS="1",
                      OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      VLLM_USE_V2_MODEL_RUNNER="1")
    allocator = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(filter(None, (
        allocator, "pinned_max_cached_size_mb:1024,pinned_max_round_threshold_mb:1024",
    )))
    worker = None
    try:
        import torch
        import runtime
        import shared_engram
        from vllm.config import set_current_vllm_config
        from vllm.engine.arg_utils import EngineArgs
        from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
        from vllm.v1.worker.gpu_worker import Worker

        runtime.install(role)
        shared_engram.install(role, rank, settings["output"])
        arguments = EngineArgs(
            model=settings["model"], tensor_parallel_size=4,
            pipeline_parallel_size=1, enable_expert_parallel=True,
            all2all_backend="allgather_reducescatter", enforce_eager=True,
            disable_custom_all_reduce=True, language_model_only=True,
            tokenizer_mode="deepseek_v41", enable_prefix_caching=False,
            max_model_len=settings["max_len"] + 8, max_num_seqs=1,
            max_num_batched_tokens=settings["chunk"], block_size=128,
            gpu_memory_utilization=0.95, distributed_executor_backend="mp",
            enable_dbo=False,
            cpu_offload_gb=settings["decode_offload_gb"] if role == "decode" else 0,
            engram_config=dict(cpu_offload=True),
        )
        config = arguments.create_engine_config()
        with set_current_vllm_config(config):
            worker = Worker(config, local_rank=rank, rank=rank,
                            distributed_init_method=f"tcp://127.0.0.1:{port}",
                            is_driver_worker=rank == 0)
            worker.init_device()
            worker.load_model()
            linear_backends = {}
            for name, module in worker.model_runner.model.named_modules():
                method = getattr(module, "quant_method", None)
                if method is not None and (name.endswith("fused_wqa_wkv") or name.endswith("wo_b")):
                    linear_backends[name] = dict(
                        method=type(method).__name__,
                        kernels={key: type(value).__name__ for key, value in vars(method).items()
                                 if "kernel" in key},
                    )
            save(output / "linear_backends.json", linear_backends)
            import gc
            import ctypes

            gc.collect()
            torch.cuda.synchronize()
            torch._C._host_emptyCache()
            ctypes.CDLL(None).malloc_trim(0)
            save(output / "host_memory.json", dict(
                allocator=os.environ["PYTORCH_CUDA_ALLOC_CONF"],
                pinned=torch.cuda.memory.host_memory_stats(),
                process_status=Path("/proc/self/status").read_text(),
                smaps_rollup=Path("/proc/self/smaps_rollup").read_text(),
            ))
            model_manifest = runtime.manifest(worker.model_runner.model)
            if role == "prefill":
                import re
                from cache_head import SOURCE20_PARAMETER_ALLOWLIST

                allowed = {"attn_norm.weight"} | {
                    "attn." + name for name in SOURCE20_PARAMETER_ALLOWLIST
                }
                for parameter in model_manifest["parameters"]:
                    match = re.search(r"(?:^|\.)layers\.(\d+)\.(.*)", parameter["name"])
                    if match and int(match[1]) >= 20:
                        if int(match[1]) != 20 or match[2] not in allowed:
                            raise RuntimeError(f"Unexpected P decoder parameter: {parameter['name']}")
            save(output / "manifest.json", model_manifest)
            spec = worker.model_runner.get_kv_cache_spec()
            layout = resolve_kv_cache_layout(
                config, [worker.get_supported_kv_cache_layouts()], spec.values()
            )
            cache_config = dedicated_cache_config(spec, settings["max_len"] + 8)
            cache_config.kv_cache_layout = layout.name
            worker.initialize_from_config(cache_config)
            torch.cuda.synchronize()
            if role == "prefill":
                from prefill_dense import install as install_prefill_dense
                install_prefill_dense()
            runner = RequestRunner(worker, config, output, rank, role)
            connection.send(dict(status="ready", role=role, rank=rank,
                                 parameter_gpu_bytes=model_manifest["gpu_bytes"],
                                 gpu_allocated_bytes=torch.cuda.memory_allocated(),
                                 gpu_reserved_bytes=torch.cuda.memory_reserved(),
                                 cache_bytes=cache_config.kv_cache_tensors[0].size,
                                 cache_groups=len(cache_config.kv_cache_groups)))
            while True:
                command = connection.recv()
                if command["op"] == "stop":
                    runner.finish()
                    connection.send(dict(status="stopped"))
                    break
                with torch.inference_mode():
                    result = getattr(runner, command["op"])(**command.get("args", {}))
                connection.send(dict(status="ok", result=result))
    except BaseException:
        error = traceback.format_exc()
        print(error, flush=True)
        save(output / "error.json", dict(error=error))
        try:
            connection.send(dict(status="error", error=error))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        connection.close()
