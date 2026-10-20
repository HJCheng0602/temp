import json
from types import SimpleNamespace

import torch

from compare_marlin_candidate import (
    HIDDEN, INTERMEDIATE, LOCAL_EXPERTS, GLOBAL_EXPERTS, TOPK, LUT,
    _packed_weights, _routes, _reference,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import _pack_deepgemm_mxfp4_scales
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    deepgemm_moe_permute, deepgemm_unpermute_and_reduce,
)
from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import DeepGemmFP4Experts
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.utils.deep_gemm import m_grouped_fp8_fp4_gemm_nt_contiguous, mk_alignment_scope


def rel(a, b):
    return float(((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item())


def dequant(q, s, block=128):
    if s.dtype == torch.int32:
        s = ((s.long()[..., None] >> (torch.arange(4, device=s.device) * 8)) & 255).flatten(-2)
        s = torch.exp2(s.float() - 127)
    elif s.dtype == torch.uint8:
        s = torch.exp2(s.float() - 127)
    return q.float() * s[..., :q.shape[-1] // block].repeat_interleave(block, -1)


torch.set_num_threads(1)
torch.manual_seed(1729)
device = torch.device("cuda:0")
batch = 8
x = torch.randn(batch, HIDDEN, device=device, dtype=torch.bfloat16) * .1
ids = _routes(None, batch, 1729, device)
weights = torch.full((batch, TOPK), 1 / TOPK, device=device)
emap = torch.arange(GLOBAL_EXPERTS, device=device, dtype=torch.int32)
emap[LOCAL_EXPERTS:] = -1
w1, w2, s1, s2 = _packed_weights(1746, device)
s1, s2 = _pack_deepgemm_mxfp4_scales(w1, w2, s1, s2)
aq, sf = per_token_group_quant_fp8(x, 128, column_major_scales=False, use_ue8m0=True)
aqd = dequant(aq, sf)
pa, ps, experts, inv, align = deepgemm_moe_permute(aq, sf, ids, LOCAL_EXPERTS, emap, None)
samples = [(t, j, int(e), int(inv[t, j])) for t, row in enumerate(ids.tolist())
           for j, e in enumerate(row) if e < LOCAL_EXPERTS][:8]
lut = torch.tensor(LUT, device=device)


def unpack(w, e):
    return torch.stack((lut[(w[e] & 15).long()], lut[(w[e] >> 4).long()]), -1).flatten(1) / 128


report = dict(batch=batch, input_quant_l2=rel(aqd, x),
              input_scale=dict(dtype=str(sf.dtype), shape=list(sf.shape), stride=list(sf.stride())),
              weight_scale=dict(dtype=str(s1.dtype), shape=list(s1.shape), stride=list(s1.stride())),
              m_sum=pa.shape[0], alignment=align, samples=[])
out1 = torch.empty((pa.shape[0], 2 * INTERMEDIATE), device=device, dtype=torch.bfloat16)
out2 = torch.empty((pa.shape[0], HIDDEN), device=device, dtype=torch.bfloat16)
obj = SimpleNamespace(_ACT_BLOCK_K=128, gemm1_clamp_limit=None,
                      adjust_N_for_activation=lambda n, activation: n // 2)
with mk_alignment_scope(align):
    m_grouped_fp8_fp4_gemm_nt_contiguous((pa, ps), (w1.view(torch.int8), s1), out1, experts,
                                       recipe_a=(1, 128), recipe_b=(1, 32))
    aq2, sf2 = DeepGemmFP4Experts._act_mul_quant(
        obj, out1, torch.empty((pa.shape[0], INTERMEDIATE), device=device, dtype=torch.float8_e4m3fn),
        MoEActivation.SILU)
    m_grouped_fp8_fp4_gemm_nt_contiguous((aq2, sf2), (w2.view(torch.int8), s2), out2, experts,
                                       recipe_a=(1, 128), recipe_b=(1, 32))
torch.cuda.synchronize()
for t, j, e, row in samples:
    ref1 = aqd[t] @ unpack(w1, e).T
    q2d = dequant(aq2[row], sf2[row])
    gate, up = out1[row].float().chunk(2)
    ref2 = q2d @ unpack(w2, e).T
    report["samples"].append(dict(token=t, expert=e, permuted_row=row,
                                  permute_l2=rel(dequant(pa[row], ps[row]), aqd[t]),
                                  fc1_l2=rel(out1[row], ref1),
                                  silu_quant_l2=rel(q2d, torch.nn.functional.silu(gate) * up),
                                  fc2_l2=rel(out2[row], ref2)))
out = torch.empty_like(x)
deepgemm_unpermute_and_reduce(out2, ids, weights, inv, emap, out)
torch.cuda.synchronize()
manual = torch.zeros_like(x, dtype=torch.float32)
for t, row in enumerate(ids.tolist()):
    for j, e in enumerate(row):
        if e < LOCAL_EXPERTS:
            manual[t] += out2[int(inv[t, j])].float() / TOPK
report["unpermute_l2"] = rel(out, manual)
report["final_l2"] = rel(out, _reference(x, ids, w1, w2))
print(json.dumps(report, indent=2), flush=True)
