"""Opt-in logical-index ordering and bounded graph-safe decode diagnostics."""

import functools
from dataclasses import asdict
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import sys

BUFFERS = {}
MOE_BUFFERS = {}


def patch_moe(module):
    cls = module.Mxfp4MoEMethod
    original = cls.apply

    @functools.wraps(original)
    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        import torch
        import torch.distributed as dist
        prefix = layer.layer_name
        if '.layers.20.' in prefix and dist.get_rank() == 4 and x.shape[0] <= 256:
            sources = dict(x=x[:min(2, x.shape[0])],
                           topk_ids=topk_ids[:min(2, x.shape[0])],
                           topk_weights=topk_weights[:min(2, x.shape[0])],
                           expert_map=layer.expert_map)
            key = x.shape[0]
            if key not in MOE_BUFFERS:
                experts = self.moe_kernel.fused_experts
                MOE_BUFFERS[key] = dict(tensors={name: torch.empty_like(value)
                                                for name, value in sources.items()},
                                       metadata=dict(layer=20, prefix=prefix,
                                           apply_router_weight_on_input=layer.apply_router_weight_on_input,
                                           activation=str(layer.activation),
                                           swiglu_limit=layer.swiglu_limit,
                                           activation_config=asdict(experts.activation_config),
                                           expert_backend=type(experts).__name__,
                                           gemm1_clamp_limit=getattr(experts, 'gemm1_clamp_limit', None),
                                           capture_source='Mxfp4MoEMethod.apply before expert computation'))
            for name, value in sources.items():
                MOE_BUFFERS[key]['tensors'][name].copy_(value)
        return original(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input)

    cls.apply = apply
    print('DSV41_MOE_INPUT_CAPTURE_INSTALLED', flush=True)


def patch_indexer_topk(module):
    cls = module.SparseIndexerTopk
    original = cls.resolve_backend

    @functools.wraps(original)
    def resolve(self, logits, topk_tokens, num_rows):
        result = original(self, logits, topk_tokens, num_rows)
        key = (result, topk_tokens, num_rows, tuple(logits.shape))
        seen = getattr(self, '_dsv41_reported_backends', set())
        if key not in seen:
            print('DSV41_INDEXER_TOPK_BACKEND ' + json.dumps(key), flush=True)
            seen.add(key)
            self._dsv41_reported_backends = seen
        return result

    cls.resolve_backend = resolve


def canonical_indices(indices):
    import torch
    sentinel = torch.iinfo(indices.dtype).max
    ordered = torch.where(indices < 0, sentinel, indices).sort(dim=-1).values
    return torch.where(ordered == sentinel, -1, ordered)


def patch_sparse(module):
    if getattr(module, '_dsv41_sparse_order', False):
        return
    canonical = os.environ.get('DSV41_SPARSE_ORDER') == 'canonical'
    original_map = module.compute_global_topk_indices_and_lens

    @functools.wraps(original_map)
    def map_indices(topk_indices, *args, **kwargs):
        # Sort logical positions before page translation. Physical page IDs
        # can change between requests even when their token histories match.
        if canonical:
            topk_indices = canonical_indices(topk_indices)
        return original_map(topk_indices, *args, **kwargs)

    module.compute_global_topk_indices_and_lens = map_indices
    cls = module.DeepseekV4FlashInferSM120Attention
    original_decode = cls._forward_decode

    @functools.wraps(original_decode)
    def decode(self, q, kv_cache, swa_metadata, attn_metadata, swa_only, output):
        import torch
        import torch.distributed as dist
        active = (os.environ.get('DSV41_SPARSE_AUDIT') == '1'
                  and '.layers.20.' in self.prefix and dist.get_rank() == 4
                  and not swa_only)
        if active:
            count = swa_metadata.num_decode_tokens
            rows = min(4, count)
            key = (count, self.prefix)
            sources = dict(indices=self.topk_indices_buffer[:rows], query=q[:rows])
            if key not in BUFFERS:
                BUFFERS[key] = {name: torch.empty_like(value) for name, value in sources.items()}
                BUFFERS[key]['output'] = torch.empty_like(output[:rows])
            for name, value in sources.items():
                BUFFERS[key][name].copy_(value)
        result = original_decode(self, q, kv_cache, swa_metadata, attn_metadata, swa_only, output)
        if active:
            BUFFERS[key]['output'].copy_(output[:rows])
        return result

    cls._forward_decode = decode
    original_prefill = cls._forward_prefill

    @functools.wraps(original_prefill)
    def prefill(self, q, compressed_k_cache, swa_k_cache, output, attn_metadata, swa_metadata):
        import torch
        import torch.distributed as dist
        from prefill_order_probe import _wave_label
        label = _wave_label() if os.environ.get('DSV41_PREFILL_PROBE') == '1' else None
        active = (label is not None and dist.get_rank() in (0, 4)
                  and any(f'.layers.{layer}.' in self.prefix for layer in (0, 1, 2, 20))
                  and not torch.cuda.is_current_stream_capturing())
        payload = None
        if active:
            payload = dict(label=label, rank=dist.get_rank(), prefix=self.prefix,
                           rows=q.shape[0])
            if self.topk_indices_buffer is not None:
                row = swa_metadata.num_decode_tokens + swa_metadata.num_prefill_tokens - 1
                payload['indices'] = self.topk_indices_buffer[row].cpu().tolist()
            payload['query_sha256'] = hashlib.sha256(
                q[-4:].contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        result = original_prefill(self, q, compressed_k_cache, swa_k_cache, output,
                                  attn_metadata, swa_metadata)
        if payload is not None:
            payload['output_sha256'] = hashlib.sha256(
                output[-4:].contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
            path = Path(os.environ['DSV41_PRECISION_DIR']).parent / f'prefill_attention_rank{dist.get_rank()}.jsonl'
            with path.open('a') as stream:
                stream.write(json.dumps(payload) + '\n')
        return result

    cls._forward_prefill = prefill
    module._dsv41_sparse_order = True
    print('DSV41_SPARSE_ORDER_INSTALLED ' + json.dumps(dict(canonical=canonical)), flush=True)


def record(batch, requests):
    """Read graph-populated buffers after the model, outside graph capture."""
    if batch.has_prefill or not requests:
        return
    import torch
    count = len(batch.idx_mapping_np)
    matching = [(key, values) for key, values in BUFFERS.items() if key[0] == count]
    positions = batch.positions[batch.logits_indices].cpu().tolist()
    if count == 1 and count in MOE_BUFFERS:
        meta = requests.get(int(batch.idx_mapping_np[0]))
        if meta is not None and meta['group'] == 'capture' and int(positions[0]) == meta['prompt_len']:
            directory = Path(os.environ['DSV41_PRECISION_DIR']).parent / 'moe_inputs'
            directory.mkdir(exist_ok=True)
            name = hashlib.sha256(meta['key'].encode()).hexdigest()[:16]
            path = directory / (name + '.pt')
            if not path.exists():
                item = MOE_BUFFERS[count]
                payload = {name: value.detach().cpu() for name, value in item['tensors'].items()}
                payload.update(item['metadata'], request_key=meta['key'], position=positions[0])
                torch.save(payload, path)
    for (_, prefix), values in matching:
        cpu = {name: tensor.detach().cpu() for name, tensor in values.items()}
        for row in range(min(4, count)):
            meta = requests.get(int(batch.idx_mapping_np[row]))
            if meta is None:
                continue
            ordinal = int(positions[row]) - meta['prompt_len'] + 2
            if ordinal not in set(range(2, 18)) | {32, 64, 96, 128, 160}:
                continue
            payload = dict(key=meta['key'], group=meta['group'], ordinal=ordinal,
                           position=positions[row], prefix=prefix,
                           indices=cpu['indices'][row].tolist())
            for name in ('query', 'output'):
                array = cpu[name][row].contiguous().view(torch.uint8).numpy()
                payload[name + '_sha256'] = hashlib.sha256(array.tobytes()).hexdigest()
            path = Path(os.environ['DSV41_PRECISION_DIR']).parent / 'sparse_order_audit.jsonl'
            with path.open('a') as stream:
                stream.write(json.dumps(payload) + '\n')


def install_import_hook():
    targets = {'vllm.models.deepseek_v41.nvidia.flashinfer_sparse': patch_sparse,
               'vllm.model_executor.layers.indexer_topk': patch_indexer_topk}
    if os.environ.get('DSV41_MOE_CAPTURE') == '1':
        targets['vllm.model_executor.layers.quantization.mxfp4'] = patch_moe

    class Finder:
        def find_spec(self, fullname, path=None, target_module=None):
            if fullname not in targets:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return None
            original_loader = spec.loader

            class Loader:
                def create_module(self, module_spec):
                    return original_loader.create_module(module_spec) if hasattr(original_loader, 'create_module') else None

                def exec_module(self, module):
                    original_loader.exec_module(module)
                    targets[fullname](module)

            spec.loader = Loader()
            return spec

    sys.meta_path.insert(0, Finder())
    for target, patch in targets.items():
        if target in sys.modules:
            patch(sys.modules[target])
