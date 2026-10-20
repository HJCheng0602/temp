"""Opt-in model adaptations for the isolated CED experiment."""

from __future__ import annotations

import re
from itertools import islice

import torch
from torch import nn

from cache_head import SOURCE20_PARAMETER_ALLOWLIST, Source20CacheHead


def install(role: str) -> None:
    from vllm.models.deepseek_v41.nvidia import model as m
    from vllm.models.deepseek_v41.nvidia import flashinfer_sparse as sparse

    if role not in ("prefill", "decode", "reference"):
        raise ValueError(role)
    original_sparse = sparse.flashinfer_trtllm_batch_decode_sparse_mla_dsv4

    def canonical_sparse(*args, **kwargs):
        indices = kwargs.get("extra_sparse_indices")
        if indices is not None:
            # Parallel top-k emits an unstable order. Canonicalize the selected
            # set so BF16 attention reductions are reproducible across P and D.
            sentinel = torch.iinfo(indices.dtype).max
            ordered = torch.where(indices < 0, sentinel, indices).sort(dim=-1).values
            kwargs["extra_sparse_indices"] = torch.where(ordered == sentinel, -1, ordered)
        return original_sparse(*args, **kwargs)

    sparse.flashinfer_trtllm_batch_decode_sparse_mla_dsv4 = canonical_sparse
    original_init = m.DeepseekV4Model.__init__
    original_forward = m.DeepseekV4Model.forward
    original_load = m.DeepseekV4Model.load_weights
    original_collapse = m.hc_collapse_triton
    active = {}

    if role == "prefill":
        original_make_layers = m.make_layers

        def make_layers(num_layers, layer_fn, prefix):
            def factory(prefix):
                layer_id = m.extract_layer_index(prefix)
                if layer_id < 20:
                    return layer_fn(prefix)
                if layer_id == 20:
                    return SourceCacheBlock(active["config"], prefix, active["building"])
                return m.PPMissingLayer()

            start, end, layers = original_make_layers(num_layers, factory, prefix)
            if start != 0 or end != 40:
                raise ValueError("CED prefill requires PP1 with the original 40-layer config")
            return 0, 20, layers

        m.make_layers = make_layers
        m.ParallelLMHead = lambda *args, **kwargs: m.PPMissingLayer()

    def init(self, *, vllm_config, prefix=""):
        active["config"] = vllm_config
        active["building"] = self
        original_init(self, vllm_config=vllm_config, prefix=prefix)
        self._ced_role = role
        self._ced_tail = None
        self._ced_replay = None
        self._ced_positions = None
        self._ced_audit = []
        self._ced_encoder_only = False
        self._ced_validate = False
        self._ced_head_check = None
        if self.use_sequence_parallel:
            raise NotImplementedError("CED experiment currently requires non-SP TP4/EP4")

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, lookback_token_ids=None):
        if self._ced_replay is not None:
            return replay(self, positions)
        self._ced_positions = positions
        self._ced_input_ids = input_ids
        active["model"] = self
        end_layer = self.end_layer
        if self._ced_encoder_only:
            self.end_layer = 20
        try:
            return original_forward(self, input_ids, positions, intermediate_tensors,
                                    inputs_embeds, lookback_token_ids)
        finally:
            self.end_layer = end_layer
            active.pop("model", None)

    def collapse(hidden, pre_mix):
        collapsed = original_collapse(hidden, pre_mix)
        current = active.get("model")
        if current is not None and (role == "prefill" or current._ced_encoder_only):
            positions = current._ced_positions
            if positions is None or hidden.shape[0] != positions.shape[0]:
                raise RuntimeError("Missing encoder boundary positions")
            SourceCacheBlock.produce(current.layers[20], collapsed, positions)
            count = min(128, hidden.shape[0])
            tail = dict(hidden=hidden[-count:].clone(), pre_mix=pre_mix[-count:].clone(),
                        positions=positions[-count:].clone(),
                        input_ids=current._ced_input_ids[-count:].clone())
            if count < 128 and current._ced_tail is not None:
                previous = current._ced_tail
                tail = {key: torch.cat((previous[key], value), dim=0)[-128:]
                        for key, value in tail.items()}
            current._ced_tail = tail
            current._ced_audit.append(dict(kind="encoder", rows=hidden.shape[0],
                                           first_layer=0, end_layer=20))
        return collapsed

    def load_weights(self, weights):
        if role == "prefill":
            def allowed(name):
                match = re.search(r"(?:^|\.)layers\.(\d+)\.(.*)", name)
                if match is None:
                    return True
                index, leaf = int(match[1]), match[2]
                if index < 20:
                    return True
                if index > 20:
                    return False
                if leaf == "attn_norm.weight":
                    return True
                leaf = leaf.removeprefix("attn.")
                leaf = leaf.replace("compressor.wkv.", "compressor.fused_wkv_wgate.")
                return leaf in SOURCE20_PARAMETER_ALLOWLIST

            weights = ((name, value) for name, value in weights if allowed(name))
        return original_load(self, weights)

    m.DeepseekV4Model.__init__ = init
    m.DeepseekV4Model.forward = forward
    m.DeepseekV4Model.load_weights = load_weights
    m.hc_collapse_triton = collapse


class SourceCacheBlock(nn.Module):
    """Register canonical cache owners, retaining only the source projection head."""

    def __init__(self, config, prefix, parent):
        super().__init__()
        from vllm.models.deepseek_v41.nvidia import model as m

        text = config.model_config.hf_config
        self.attn_norm = m.RMSNorm(text.hidden_size, text.rms_norm_eps)
        self.attn = m._select_dsv4_attn_cls(config)(
            config, prefix=prefix + ".attn",
            topk_indices_buffer=parent.topk_indices_buffer,
            candidate_block_buffer=parent.candidate_block_buffer,
            aux_stream_list=None,
        )
        # Remove unused linears as modules, so the generic weight finalizer
        # cannot try to repack query/output weights that were never loaded.
        for name in ("fused_wqa_wkv", "q_norm", "wq_b", "kv_norm", "wo_a", "wo_b"):
            delattr(self.attn, name)
        for name in ("wq_b", "weights_proj"):
            delattr(self.attn.indexer, name)
        for name, _ in list(self.attn.named_parameters()):
            if name not in SOURCE20_PARAMETER_ALLOWLIST:
                owner, _, leaf = name.rpartition(".")
                module = self.attn.get_submodule(owner) if owner else self.attn
                delattr(module, leaf)
        names = set(dict(self.attn.named_parameters()))
        if names != set(SOURCE20_PARAMETER_ALLOWLIST):
            raise RuntimeError(f"Unexpected source head parameters: {sorted(names)}")

    def produce(self, collapsed, positions):
        from vllm.forward_context import get_forward_context

        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            raise RuntimeError("Cache production requires real request metadata")
        x = self.attn_norm(collapsed)
        attn = self.attn
        source_meta = metadata[attn.prefix]
        index_meta = metadata[attn.indexer.k_cache.prefix]
        return Source20CacheHead.forward(
            attn, x, positions,
            compressed_kv_cache=attn._compressed_kv_cache(),
            compressed_slot_mapping=source_meta.slot_mapping,
            index_k_cache=attn.indexer.k_cache.kv_cache,
            index_slot_mapping=index_meta.slot_mapping,
            main_cos_sin_cache=attn.rotary_emb.cos_sin_cache,
            index_cos_sin_cache=attn.indexer_rotary_emb.cos_sin_cache,
            use_fp4_index_cache=attn.indexer.use_fp4_kv,
        )


def replay(model, positions):
    from vllm.forward_context import get_forward_context
    from vllm.models.deepseek_v41.nvidia import model as m

    payload = model._ced_replay
    count = payload["hidden"].shape[0]
    if count > 128 or positions.shape[0] != count:
        raise RuntimeError("Decoder replay must contain exactly the transferred tail")
    if not torch.equal(positions, payload["positions"]):
        raise RuntimeError("Replay positions changed during the P/D handoff")
    metadata = get_forward_context().attn_metadata
    seen = set()
    for layer in islice(model.layers, 20, 40):
        swa = metadata[layer.attn.swa_cache_layer.prefix]
        if id(swa) in seen:
            continue
        seen.add(id(swa))
        indices = swa.prefill_swa_indices
        lengths = swa.prefill_swa_lens
        if indices is None or lengths is None or swa.num_prefill_tokens != count:
            raise RuntimeError("Expected a pure bounded-replay prefill batch")
        width = indices.shape[-1]
        rows = torch.arange(count, device=positions.device)
        cols = torch.arange(width, device=positions.device)
        source = swa.slot_mapping[:count].to(indices.dtype)
        selected = source[cols.clamp(max=count - 1)]
        mask = (cols[None, :] <= rows[:, None]) & (cols[None, :] < count)
        indices.copy_(torch.where(mask, selected[None, :], -1).reshape_as(indices))
        lengths.copy_((rows + 1).to(lengths.dtype))

    hidden = payload["hidden"]
    pre_mix = payload["pre_mix"]
    input_ids = payload["input_ids"]
    residual = post_mix = res_mix = None
    for index, layer in enumerate(islice(model.layers, 20, 40), start=20):
        hook = None
        observed = {}
        if model._ced_validate and index == 20:
            expected_x = layer.attn_norm(m.hc_collapse_triton(hidden, pre_mix))
            caches = (layer.attn.kv_cache, layer.attn.indexer.k_cache.kv_cache)
            slots = (metadata[layer.attn.prefix].slot_mapping[:count],
                     metadata[layer.attn.indexer.k_cache.prefix].slot_mapping[:count])
            before = [cache[slot // cache.shape[1], slot % cache.shape[1]].clone()
                      for cache, slot in zip(caches, slots)]

            def observe_input(module, args):
                observed["x"] = args[1].detach().clone()

            hook = layer.attn.register_forward_pre_hook(observe_input)
        try:
            hidden, residual, post_mix, res_mix, pre_mix, _ = layer(
                hidden, positions, input_ids, pre_mix, post_mix, res_mix, residual,
                None, None,
            )
        finally:
            if hook is not None:
                hook.remove()
        if model._ced_validate and index == 20:
            actual_x = observed["x"]
            difference = actual_x.float() - expected_x.float()
            after = [cache[slot // cache.shape[1], slot % cache.shape[1]]
                     for cache, slot in zip(caches, slots)]
            model._ced_head_check = dict(
                normalized_input_max_abs=float(difference.abs().max()),
                normalized_input_rmse=float(difference.square().mean().sqrt()),
                source_tail_equal_fraction=[float((old == new).float().mean())
                                            for old, new in zip(before, after)],
                rows=count,
            )
    hidden = m.mhc_post_tilelang(hidden, residual, post_mix, res_mix)
    hidden = m.hc_collapse_triton(hidden, pre_mix)
    model._ced_audit.append(dict(kind="decoder_replay", rows=count,
                                first_layer=20, end_layer=40))
    return model.norm(hidden)


def text_backbone(model):
    language = getattr(model, "language_model", model)
    return language.model


def manifest(model):
    rows = [dict(name=name, shape=list(value.shape), dtype=str(value.dtype),
                 device=str(value.device), bytes=value.numel() * value.element_size())
            for name, value in model.named_parameters()]
    return dict(parameters=rows,
                gpu_bytes=sum(row["bytes"] for row in rows if row["device"].startswith("cuda")),
                cpu_bytes=sum(row["bytes"] for row in rows if row["device"] == "cpu"))
