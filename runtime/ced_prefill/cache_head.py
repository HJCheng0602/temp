"""Small DeepSeek V4.1 layer-20 global-cache producer for CED prefill.

The CED prefill path needs the ratio-1 source compressor and its index-K
projection, but does not need layer-20 query/output/FFN weights.  This module
keeps those four checkpoint leaves under the same names used by the full
attention module and leaves cache allocation/metadata ownership to the caller.
"""

from __future__ import annotations

import torch
from torch import nn


SOURCE20_PARAMETER_ALLOWLIST = (
    "compressor.fused_wkv_wgate.weight",
    "compressor.norm.weight",
    "indexer.wk.weight",
    "indexer.k_norm.weight",
)


class _RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        return (
            x * torch.rsqrt(variance + self.eps).to(x.dtype)
        ) * self.weight.to(x.dtype)


class _Compressor(nn.Module):
    def __init__(self, hidden_size: int, head_dim: int, eps: float) -> None:
        super().__init__()
        # Ratio 1 has no gate or open-group state (see DeepseekCompressor).
        self.fused_wkv_wgate = nn.Linear(hidden_size, head_dim, bias=False)
        self.norm = _RMSNorm(head_dim, eps)


class _Indexer(nn.Module):
    def __init__(self, head_dim: int, index_head_dim: int, eps: float) -> None:
        super().__init__()
        self.wk = nn.Linear(head_dim, index_head_dim, bias=False)
        self.k_norm = _RMSNorm(index_head_dim, eps)


class Source20CacheHead(nn.Module):
    """Produce layer-20 ratio-1 compressed KV and indexer-K records.

    ``x20`` must already be the normalized layer-20 attention input (the
    result of layer-20 mHC pre plus attention RMSNorm).  The two cache tensors
    and slot mappings use the canonical vLLM paged layouts.  This class does
    not allocate KV caches and does not depend on forward-context metadata,
    which makes the P/D transfer boundary explicit.
    """

    compress_ratio = 1

    def __init__(
        self,
        hidden_size: int,
        head_dim: int = 512,
        index_head_dim: int = 128,
        rms_norm_eps: float = 1e-6,
        compressor: nn.Module | None = None,
    ) -> None:
        super().__init__()
        # A caller with a live VllmConfig may inject the authoritative
        # DeepseekCompressor here.  The fallback has the same ratio-1 leaves
        # but avoids constructing a forward-context/cache registration during
        # manifest-only model setup.
        self.compressor = (
            _Compressor(hidden_size, head_dim, rms_norm_eps)
            if compressor is None
            else compressor
        )
        self.indexer = _Indexer(head_dim, index_head_dim, rms_norm_eps)

    @staticmethod
    def parameter_allowlist() -> tuple[str, ...]:
        """Return the only checkpoint leaves this head is allowed to load."""
        return SOURCE20_PARAMETER_ALLOWLIST

    def forward(
        self,
        x20: torch.Tensor,
        positions: torch.Tensor,
        *,
        compressed_kv_cache: torch.Tensor,
        compressed_slot_mapping: torch.Tensor,
        index_k_cache: torch.Tensor,
        index_slot_mapping: torch.Tensor,
        main_cos_sin_cache: torch.Tensor,
        index_cos_sin_cache: torch.Tensor | None = None,
        use_fp4_index_cache: bool = False,
        fp8_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Write both source caches and return the unrotated latent.

        The returned latent is useful for diagnostics or a caller that needs
        to verify the source projection.  Cache writes are delegated to the
        same vLLM kernels used by ``DeepseekCompressor`` and the indexer.
        """
        if x20.ndim != 2 or positions.ndim != 1:
            raise ValueError("x20 must be [tokens, hidden] and positions [tokens]")
        if x20.shape[0] != positions.shape[0]:
            raise ValueError("x20 and positions must have the same token count")
        if positions.dtype != torch.int64:
            raise TypeError("positions must be torch.int64")

        kv_score = torch.mm(
            x20,
            self.compressor.fused_wkv_wgate.weight.t(),
            out_dtype=torch.float32,
        )
        latent = torch.empty(
            (x20.shape[0], 512), dtype=torch.bfloat16, device=x20.device
        )

        # These imports are intentionally lazy: importing this module must not
        # initialize CUDA kernels or require a live vLLM forward context.
        from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import (
            fused_save_compress_norm,
            rope_quant_insert,
        )
        from vllm.models.deepseek_v41.common.ops.indexer_k_store import (
            indexer_k_norm_rope_store,
        )

        norm_eps = getattr(self.compressor.norm, "variance_epsilon", None)
        if norm_eps is None:
            norm_eps = self.compressor.norm.eps
        fused_save_compress_norm(
            kv_score,
            positions,
            None,
            compressed_slot_mapping,
            None,
            None,
            self.compressor.norm.weight,
            norm_eps,
            self.compress_ratio,
            latent,
        )
        rope_quant_insert(
            latent,
            positions,
            main_cos_sin_cache,
            compressed_kv_cache,
            compressed_slot_mapping,
            self.compress_ratio,
            fp8_scale=fp8_scale,
        )

        index_k_pre = self.indexer.wk(latent)
        # ReplicatedLinear/MergedColumnParallelLinear return (output, bias),
        # while the manifest-only fallback uses nn.Linear and returns a tensor.
        if isinstance(index_k_pre, tuple):
            index_k_pre = index_k_pre[0]
        index_k_pre = index_k_pre.contiguous()
        index_cos_sin_cache = (
            main_cos_sin_cache
            if index_cos_sin_cache is None
            else index_cos_sin_cache
        )
        index_norm_eps = getattr(self.indexer.k_norm, "variance_epsilon", None)
        if index_norm_eps is None:
            index_norm_eps = self.indexer.k_norm.eps
        indexer_k_norm_rope_store(
            index_k_pre,
            positions,
            index_cos_sin_cache,
            self.indexer.k_norm.weight,
            index_norm_eps,
            index_k_cache,
            index_slot_mapping,
            self.compress_ratio,
            use_fp4_index_cache,
        )
        return latent

    def load_weights(self, weights: dict[str, torch.Tensor]) -> None:
        """Load an already-filtered local state dictionary.

        The model-level loader should filter checkpoint names before calling
        this method; rejecting extras here prevents accidentally retaining a
        full layer-20 attention or MoE module in the prefill process.
        """
        unexpected = set(weights) - set(SOURCE20_PARAMETER_ALLOWLIST)
        missing = set(SOURCE20_PARAMETER_ALLOWLIST) - set(weights)
        if unexpected or missing:
            raise ValueError(
                f"source-20 head manifest mismatch: unexpected={sorted(unexpected)}, "
                f"missing={sorted(missing)}"
            )
        with torch.no_grad():
            for name, value in weights.items():
                target = self
                parts = name.split(".")
                for part in parts[:-1]:
                    target = getattr(target, part)
                parameter = getattr(target, parts[-1])
                parameter.copy_(value)
