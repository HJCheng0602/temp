"""Experimental, eager TP1 pipeline support through source-attention replay.

Checkpoint semantics are unchanged. Source attention inputs cross PP boundaries;
receiver-local attention replicas maintain ordinary paged cache state. Full
attention replay adds work and must be identified separately in measurements.
"""

import os
import re


def crossing_sources(config, boundary):
    sources = set()
    for layer in range(boundary, config.num_hidden_layers):
        if not config.compress_ratios[layer]:
            continue
        kv = max(x for x in config.kv_source_layer_ids if x <= layer)
        index = max(x for x in config.index_source_layer_ids if x <= layer)
        needed = [kv, index]
        candidate = config.candidate_source_layer_id
        if layer in config.index_source_layer_ids and 0 <= candidate < layer:
            needed.append(candidate)
        sources.update(x for x in needed if x < boundary)
    return sorted(sources)


def key(source):
    return f"pp_source_input_{source}"


def prepare(model, vllm_config, prefix, attn_cls, aux_streams):
    model._pp_replay = None
    if os.environ.get("DSV41_PP_REPLAY") != "1":
        return
    import torch.nn as nn
    from vllm.distributed import get_pp_group
    from vllm.distributed.utils import get_pp_indices

    parallel = vllm_config.parallel_config
    if parallel.tensor_parallel_size != 1 or parallel.data_parallel_size != 1:
        raise ValueError("Source replay currently requires TP1 and DP1")
    if not vllm_config.model_config.enforce_eager:
        graph_trial = os.environ.get("DSV41_PP_GRAPH_TRIAL") == "1"
        graph_mode = getattr(vllm_config.compilation_config, "cudagraph_mode", None)
        graph_mode_name = getattr(graph_mode, "name", str(graph_mode))
        if not graph_trial or graph_mode_name not in {"FULL_DECODE_ONLY", "FULL"}:
            raise ValueError(
                "Source replay requires enforce_eager, or an explicit graph "
                "trial with FULL_DECODE_ONLY/FULL cudagraph mode"
            )
    if vllm_config.speculative_config is not None or model.use_sequence_parallel:
        raise ValueError("Source replay does not support speculative/SP execution")
    start, end = get_pp_indices(
        model.config.num_hidden_layers,
        get_pp_group().rank_in_group,
        parallel.pipeline_parallel_size,
    )
    incoming = crossing_sources(model.config, start)
    outgoing = crossing_sources(model.config, end)
    holders = {}
    # Register real source cache modules before constructing their consumers.
    for source in incoming:
        holder = nn.Module()
        holder.attn = attn_cls(
            vllm_config,
            prefix=f"{prefix}.layers.{source}.attn",
            topk_indices_buffer=model.topk_indices_buffer,
            aux_stream_list=aux_streams,
            candidate_block_buffer=model.candidate_block_buffer,
        )
        holders[source] = holder
    model._pp_replay = dict(
        start=start, end=end, incoming=incoming, outgoing=outgoing,
        holders=holders, captured={}, incoming_tensors={},
    )


def attach(model):
    state = model._pp_replay
    if state is None:
        return
    assert (model.start_layer, model.end_layer) == (state["start"], state["end"])
    for source, holder in state["holders"].items():
        # Canonical module paths preserve the normal attention weight loaders.
        model.layers[source] = holder
    for source in state["outgoing"]:
        if model.start_layer <= source < model.end_layer:
            def capture(module, args, source=source):
                tensor = args[1]
                if tensor.ndim != 2 or tensor.shape[1] != model.config.hidden_size:
                    raise RuntimeError("Unexpected source-attention input shape")
                state["captured"][source] = tensor.clone()
            model.layers[source].attn.register_forward_pre_hook(capture)


def empty_inputs(model, batch_size, dtype, device):
    state = model._pp_replay
    if state is None:
        return {}
    import torch
    return {
        key(source): torch.zeros(
            (batch_size, model.config.hidden_size), dtype=dtype, device=device,
        )
        for source in state["incoming"]
    }


def receive(model, positions, intermediate_tensors):
    state = model._pp_replay
    if state is None:
        return
    from torch.profiler import record_function
    from vllm.forward_context import get_forward_context
    metadata = get_forward_context().attn_metadata
    state["captured"].clear()
    state["incoming_tensors"] = {}
    for source in state["incoming"]:
        attention = state["holders"][source].attn
        if isinstance(metadata, dict) and metadata:
            caches = [attention.swa_cache_layer.kv_cache,
                      attention._compressed_kv_cache()]
            if attention.indexer is not None:
                caches.append(attention.indexer.k_cache.kv_cache)
            if any(cache.numel() == 0 for cache in caches):
                raise RuntimeError(f"Unbound replay cache for source {source}")
        tensor = intermediate_tensors[key(source)]
        if tensor.shape != (positions.shape[0], model.config.hidden_size):
            raise RuntimeError(f"Misaligned PP replay input for source {source}")
        state["incoming_tensors"][source] = tensor
        with record_function(f"pp_source_replay_{source}"):
            # Standard attention joins its auxiliary streams before returning.
            # The output is discarded; only local cache/index state is used.
            attention(positions, tensor, None)


def send(model):
    state = model._pp_replay
    if state is None:
        return {}
    result = {}
    for source in state["outgoing"]:
        if source in state["captured"]:
            value = state["captured"][source]
        else:
            value = state["incoming_tensors"][source]
        result[key(source)] = value
    return result


def skip_weight(model, name):
    state = model._pp_replay
    if state is None:
        return False
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
    return bool(
        match and int(match.group(1)) in state["holders"]
        and ".attn." not in name
    )


def finalize(model):
    state = model._pp_replay
    if state is not None:
        for holder in state["holders"].values():
            callback = getattr(holder.attn, "finalize_loaded_weights", None)
            if callback is not None:
                callback()
