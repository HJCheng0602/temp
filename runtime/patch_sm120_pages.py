"""Patch vLLM 8538017f4 for V4.1 on SM120 and pipeline parallelism.

Run with the isolated vLLM Python. Originals and unified diffs are retained.
This changes cache allocation/read behavior, carries token IDs between PP stages,
and scopes synchronized FlashInfer autotuning to each pipeline stage's TP group.
"""

import ast
import difflib
import importlib.util
from pathlib import Path


ROOT = Path(importlib.util.find_spec("vllm").origin).parent
BACKUP = Path("/home/admin/dsv41-vllm-20260917/patches/sm120-pages")
EDITS = {
    "models/deepseek_v41/nvidia/model.py": [
        (
            "                \"hidden_states\": torch.zeros(\n"
            "                    (batch_size, self.hc_mult, self.config.hidden_size),\n"
            "                    dtype=dtype,\n"
            "                    device=device,\n"
            "                ),\n",
            "                \"hidden_states\": torch.zeros(\n"
            "                    (batch_size, self.hc_mult, self.config.hidden_size),\n"
            "                    dtype=dtype,\n"
            "                    device=device,\n"
            "                ),\n"
            "                # V4 MoE routing needs token ids on every PP stage.\n"
            "                \"input_ids\": torch.zeros(\n"
            "                    (batch_size,), dtype=torch.int32, device=device\n"
            "                ),\n",
        ),
        (
            "        else:\n"
            "            assert intermediate_tensors is not None\n"
            "            hidden_states = intermediate_tensors[\"hidden_states\"]\n\n"
            "        if self.use_mega_moe:\n",
            "        else:\n"
            "            assert intermediate_tensors is not None\n"
            "            hidden_states = intermediate_tensors[\"hidden_states\"]\n"
            "            # The PP dummy/profile path does not provide input_ids to\n"
            "            # non-first ranks through the model call. Carry them with\n"
            "            # the stage output so vision/hash MoE routing remains valid.\n"
            "            input_ids = intermediate_tensors[\"input_ids\"]\n\n"
            "        pp_input_ids = input_ids\n"
            "        if self.use_mega_moe:\n",
        ),
        (
            "            return IntermediateTensors(\n"
            "                {\"hidden_states\": hidden_states, \"pre_mix\": pre_mix}\n"
            "            )\n",
            "            return IntermediateTensors(\n"
            "                {\n"
            "                    \"hidden_states\": hidden_states,\n"
            "                    \"pre_mix\": pre_mix,\n"
            "                    \"input_ids\": pp_input_ids,\n"
            "                }\n"
            "            )\n",
        ),
    ],
    "models/deepseek_v41/attention.py": [
        (
            "    swa_backend_cls: ClassVar[type[AttentionBackend] | None] = None\n",
            "    swa_backend_cls: ClassVar[type[AttentionBackend] | None] = None\n"
            "    swa_page_tokens: ClassVar[int] = 32\n"
            "    compressed_page_states: ClassVar[int | None] = None\n",
        ),
        ("            block_size=32,\n", "            block_size=self.swa_page_tokens,\n"),
        (
            "            block_size=vllm_config.cache_config.block_size,\n",
            "            block_size=(\n"
            "                self.compressed_page_states * self.compress_ratio\n"
            "                if self.compressed_page_states is not None\n"
            "                else vllm_config.cache_config.block_size\n"
            "            ),\n",
        ),
        (
            "            block_size=self.cache_config.block_size,\n",
            "            block_size=(\n"
            "                64 * self.compress_ratio\n"
            "                if current_platform.is_device_capability_family(120)\n"
            "                else self.cache_config.block_size\n"
            "            ),\n",
        ),
    ],
    "models/deepseek_v41/nvidia/flashinfer_sparse.py": [
        (
            "class DeepseekV4FlashInferSM120Attention(DeepseekV4Attention):\n",
            "class DeepseekV41SM120SparseBackend(DeepseekV4FlashInferMLASparseBackend):\n"
            "    @staticmethod\n"
            "    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:\n"
            "        # 64 compressed states: ratio 1 -> 64, ratio 2 -> 128 tokens.\n"
            "        return [128, 64]\n\n\n"
            "class DeepseekV4FlashInferSM120Attention(DeepseekV4Attention):\n",
        ),
        (
            '    """DeepSeek V4 sparse MLA attention through FlashInfer\'s SM120 kernels."""\n\n'
            "    backend_cls = DeepseekV4FlashInferMLASparseBackend\n",
            '    """DeepSeek V4 sparse MLA attention through FlashInfer\'s SM120 kernels."""\n\n'
            "    swa_page_tokens = 64\n"
            "    compressed_page_states = 64\n"
            "    backend_cls = DeepseekV41SM120SparseBackend\n",
        ),
        (
            "        if kv_cache.dim() == 4:\n"
            "            return kv_cache\n"
            "        return kv_cache.unsqueeze(-2)\n",
            "        # Block 0 is vLLM's reserved null block. SM120 maps masked -1\n"
            "        # indices there; NaNs left by dummy/warmup writes can survive\n"
            "        # zero attention weights in a partial tile's PV product.\n"
            "        # Clear only this unused page, before every real cache read.\n"
            "        kv_cache[0].zero_()\n"
            "        if kv_cache.dim() == 4:\n"
            "            return kv_cache\n"
            "        return kv_cache.unsqueeze(-2)\n",
        ),
    ],
    "v1/attention/backends/mla/indexer.py": [
        (
            "        return [64 if current_platform.is_device_capability_family(90) else 128]\n",
            "        if current_platform.is_device_capability_family(120):\n"
            "            return [128, 64]\n"
            "        return [64 if current_platform.is_device_capability_family(90) else 128]\n",
        ),
    ],
    "v1/worker/utils.py": [
        (
            "    for tensor in kv_cache_config.kv_cache_tensors:\n"
            "        layer_name = tensor.layers[0]\n"
            "        group_id, group = next(\n"
            "            (group_id, group)\n"
            "            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)\n"
            "            if layer_name in group.layer_names\n"
            "        )\n",
            "    for tensor in kv_cache_config.kv_cache_tensors:\n"
            "        # PP workers receive the global tensor layout but only the\n"
            "        # groups for their local layers. The tensor's first layer may\n"
            "        # belong to another stage, so select a local representative.\n"
            "        local_match = next(\n"
            "            (\n"
            "                (group_id, group, layer_name)\n"
            "                for layer_name in tensor.layers\n"
            "                for group_id, group in enumerate(\n"
            "                    kv_cache_config.kv_cache_groups\n"
            "                )\n"
            "                if layer_name in group.layer_names\n"
            "            ),\n"
            "            None,\n"
            "        )\n"
            "        if local_match is None:\n"
            "            continue\n"
            "        group_id, group, layer_name = local_match\n",
        ),
    ],
    "model_executor/warmup/kernel_warmup.py": [
        (
            "    from vllm.distributed.parallel_state import get_world_group\n\n"
            "    world = get_world_group()\n",
            "    from vllm.distributed.parallel_state import get_tp_group\n\n"
            "    # Pipeline stages execute different layers and GEMM shapes.\n"
            "    # Synchronizing FlashInfer tactics over the global world makes\n"
            "    # their collective sequences diverge and deadlock. A TP group\n"
            "    # contains ranks executing the same stage and shape sequence.\n"
            "    world = get_tp_group()\n",
        ),
    ],
}


def main():
    pending = []
    for relative, replacements in EDITS.items():
        target = ROOT / relative
        backup = BACKUP / relative
        original = backup.read_text() if backup.exists() else target.read_text()
        modified = original
        recognized = {original}
        for before, after in replacements:
            if modified.count(before) != 1:
                raise RuntimeError(f"Source mismatch in {relative}: {before!r}")
            modified = modified.replace(before, after, 1)
            recognized.add(modified)
        ast.parse(modified)
        if target.read_text() not in recognized:
            raise RuntimeError(f"Unrecognized local edits in {target}")
        pending.append((target, backup, original, modified))
    for target, backup, original, modified in pending:
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            backup.write_text(original)
        target.write_text(modified)
        backup.with_suffix(".diff").write_text("".join(difflib.unified_diff(
            original.splitlines(True), modified.splitlines(True),
            fromfile=str(backup), tofile=str(target),
        )))
        print("PATCHED", target)


if __name__ == "__main__":
    main()
