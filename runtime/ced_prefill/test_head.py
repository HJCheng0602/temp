"""Small CUDA smoke test for the layer-20 source cache head.

This intentionally constructs only the head and random weights.  It does not
load a checkpoint or initialize a vLLM model.  Run it explicitly in the
cached runtime environment, for example::

    python deployment/ced_prefill/test_head.py --gpu 0

The script prints one JSON object on success or failure so a harness can use
it without parsing Triton output.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback


def _configure_environment(gpu: str) -> None:
    """Install the same CUDA/Python environment used by parallel_matrix."""
    deployment = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(deployment))
    from parallel_matrix import environment

    env = environment()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ.update(env)

def _run(lengths: tuple[int, ...]) -> list[dict[str, object]]:
    import torch
    from torch import nn

    from vllm.model_executor.layers.layernorm import RMSNorm

    from cache_head import Source20CacheHead

    torch.set_num_threads(1)
    torch.manual_seed(1)

    class TupleLinear(nn.Module):
        """ReplicatedLinear-compatible projection for the small test head."""

        def __init__(self, input_size: int, output_size: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(
                torch.empty(output_size, input_size, dtype=torch.bfloat16)
            )
            nn.init.normal_(self.weight, mean=0.0, std=0.02)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
            return torch.mm(x, self.weight.t()), None

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this test does not fall back to CPU")
    device = torch.device("cuda")
    hidden_size = 5120
    head_dim = 512
    index_dim = 128
    eps = 1e-6
    head = Source20CacheHead(
        hidden_size=hidden_size,
        head_dim=head_dim,
        index_head_dim=index_dim,
        rms_norm_eps=eps,
    ).to(device=device, dtype=torch.bfloat16)

    # Use the actual vLLM RMSNorm implementation so the variance_epsilon path
    # used by the runtime is exercised, including its parameter dtype.
    head.compressor.norm = RMSNorm(head_dim, eps).to(
        device=device, dtype=torch.bfloat16
    )
    head.indexer.k_norm = RMSNorm(index_dim, eps).to(
        device=device, dtype=torch.bfloat16
    )
    head.indexer.wk = TupleLinear(head_dim, index_dim).to(device=device)
    head.eval()
    head.requires_grad_(False)

    results: list[dict[str, object]] = []
    for length in lengths:
        blocks = (length + 127) // 128
        x20 = torch.randn(
            (length, hidden_size), device=device, dtype=torch.bfloat16
        )
        positions = torch.arange(length, device=device, dtype=torch.int64)
        slots = positions.clone()
        # GPT-J cache layout is [cos(32), sin(32)]. Identity RoPE keeps the
        # numeric latent reference simple while still executing the kernel.
        cos_sin = torch.cat(
            (
                torch.ones((length, 32), device=device, dtype=torch.float32),
                torch.zeros((length, 32), device=device, dtype=torch.float32),
            ),
            dim=1,
        ).contiguous()
        main_cache = torch.zeros(
            (blocks, 128, 528), device=device, dtype=torch.uint8
        )
        index_cache = torch.zeros(
            (blocks, 128, 132), device=device, dtype=torch.uint8
        )

        latent = head(
            x20,
            positions,
            compressed_kv_cache=main_cache,
            compressed_slot_mapping=slots,
            index_k_cache=index_cache,
            index_slot_mapping=slots,
            main_cos_sin_cache=cos_sin,
            index_cos_sin_cache=cos_sin,
            use_fp4_index_cache=False,
        )
        torch.cuda.synchronize()

        with torch.no_grad():
            projected = torch.mm(
                x20,
                head.compressor.fused_wkv_wgate.weight.t(),
                out_dtype=torch.float32,
            )
            variance = projected.square().mean(dim=-1, keepdim=True)
            expected = (
                projected
                * torch.rsqrt(variance + eps)
                * head.compressor.norm.weight.float()
            ).to(torch.bfloat16)
        max_latent_error = float(
            (latent.float() - expected.float()).abs().max().item()
        )
        # Packed pages store all values first and then all scale records;
        # their apparent last dimension is not a per-token record stride.
        main_values = main_cache.flatten(1)[:, :128 * 512].reshape(blocks, 128, 512)
        index_values = index_cache.flatten(1)[:, :128 * 128].reshape(blocks, 128, 128)
        main_written = int(main_values.any(dim=-1).sum().item())
        index_written = int(index_values.any(dim=-1).sum().item())
        if not torch.isfinite(latent.float()).all():
            raise AssertionError(f"non-finite latent at length {length}")
        if max_latent_error > 0.02:
            raise AssertionError(
                f"latent RMSNorm mismatch at length {length}: "
                f"max error {max_latent_error}"
            )
        if main_written != length or index_written != length:
            raise AssertionError(
                f"cache write count at length {length}: "
                f"main={main_written}, index={index_written}"
            )
        results.append(
            {
                "tokens": length,
                "latent_max_abs_error": max_latent_error,
                "main_rows_written": main_written,
                "index_rows_written": index_written,
                "main_record_bytes": 528,
                "index_record_bytes": 132,
            }
        )
        del x20, latent, main_cache, index_cache
        torch.cuda.empty_cache()
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=os.environ.get("CED_TEST_GPU", "0"))
    args = parser.parse_args()
    try:
        _configure_environment(args.gpu)
        if os.environ.get("CED_HEAD_TEST_BOOTSTRAPPED") != "1":
            os.environ["CED_HEAD_TEST_BOOTSTRAPPED"] = "1"
            os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)
        from vllm.config import VllmConfig, set_current_vllm_config

        with set_current_vllm_config(VllmConfig()):
            results = _run((1, 128, 2048))
        print(json.dumps({"status": "ok", "tests": results}, sort_keys=True))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {"status": "error", "error": f"{type(exc).__name__}: {exc}"},
                sort_keys=True,
            )
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
