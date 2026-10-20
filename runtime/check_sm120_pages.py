"""Validate real vLLM cache specs without allocating model weights."""

from types import SimpleNamespace

import torch

from vllm.config import CacheConfig
from vllm.models.deepseek_v41.attention import (
    DeepseekV4Attention,
    DeepseekV4IndexerCache,
)
from vllm.models.deepseek_v41.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferSM120Attention,
)
from vllm.v1.attention.backends.mla.indexer import DeepseekV41IndexerBackend
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache


cache = CacheConfig(block_size=128, cache_dtype="fp8_ds_mla")
config = SimpleNamespace(cache_config=cache)
attention = DeepseekV4FlashInferSM120Attention
assert attention.swa_page_tokens == 64
for ratio in (1, 2):
    layer = SimpleNamespace(
        is_kv_source=True,
        compressed_page_states=attention.compressed_page_states,
        compress_ratio=ratio,
        kv_cache_dtype="fp8_ds_mla",
        head_dim=512,
        kv_cache_torch_dtype=torch.uint8,
        kv_page_alignment=576,
        compressed_bytes_per_token=584,
    )
    spec = DeepseekV4Attention.get_kv_cache_spec(layer, config)
    assert spec.num_states == 64
    assert spec.block_size in attention.backend_cls.get_supported_kernel_block_sizes()
    indexer = SimpleNamespace(
        cache_config=cache, compress_ratio=ratio, head_dim=132,
        dtype=torch.uint8, sparse_logits=False,
    )
    index_spec = DeepseekV4IndexerCache.get_kv_cache_spec(indexer, config)
    assert index_spec.num_states == 64
    assert index_spec.block_size in DeepseekV41IndexerBackend.get_supported_kernel_block_sizes()
    print("ratio", ratio, "KV tokens/states", spec.block_size, spec.num_states,
          "index tokens/states", index_spec.block_size, index_spec.num_states)

swa = SimpleNamespace(
    cache_config=cache, block_size=attention.swa_page_tokens, head_dim=512,
    dtype=torch.uint8, window_size=128, packed_bytes_per_token=584,
    packed_page_alignment=576,
)
swa_spec = DeepseekV4SWACache.get_kv_cache_spec(swa, config)
assert swa_spec.num_states == 64
assert swa_spec.page_size_bytes % 576 == 0
print("SWA states", swa_spec.num_states, "page bytes", swa_spec.page_size_bytes)
print("SM120_CACHE_SPECS_OK")
