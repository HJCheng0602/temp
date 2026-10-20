"""Use measured SM120 MXFP8 tactics for exact prefill shapes.

The existing FlashInfer runner owns argument validation, scale handling,
allocation and execution. Only its fallback tactic is changed. No online
benchmarking or collective is introduced into model execution.
"""
from collections import Counter
import json
import os
from pathlib import Path

_STATE = None


class ProjectionPolicy:
    def __init__(self, policy, torch):
        self.torch = torch
        self.policy = {tuple(row["shape"]): int(row["tactic"])
                       for row in policy["entries"]}
        if not self.policy or any(t not in range(6) for t in self.policy.values()):
            raise ValueError("Invalid prefill projection policy")
        self.enabled = os.environ.get("DSV41_PREFILL_DENSE", "1") != "0"
        self.profile = False
        self.calls = Counter()
        self.events = []
        self.verify = False
        self.verification = {}

    def configure(self, enabled, profile=False, verify=False):
        self.torch.cuda.synchronize()
        self.enabled = bool(enabled)
        self.profile = bool(profile)
        self.calls.clear()
        self.events.clear()
        self.verify = bool(verify)
        if verify:
            self.verification.clear()

    def wrap(self, runner):
        original = runner.forward

        def forward(inputs, tactic=-1, do_preparation=False, **kwargs):
            a, b, _, _, dtype, _, _ = inputs
            key = (int(a.shape[0]), int(b.shape[1]), int(a.shape[1]))
            selected = tactic
            eligible = (key in self.policy and dtype == self.torch.bfloat16
                        and a.dtype == self.torch.float8_e4m3fn
                        and b.dtype == self.torch.float8_e4m3fn
                        and not do_preparation)
            if eligible:
                if self.enabled and tactic == -1:
                    selected = self.policy[key]
                self.calls[(key, selected)] += 1
            if self.verify and eligible and self.enabled:
                baseline = original(inputs, -1, do_preparation, **kwargs).clone()
                result = original(inputs, selected, do_preparation, **kwargs)
                equal = self.torch.equal(baseline, result)
                finite = bool(self.torch.isfinite(result).all())
                stats = self.verification.setdefault(key, dict(calls=0, equal_calls=0,
                    finite_calls=0, max_abs=0.0, max_relative_l2=0.0))
                stats['calls'] += 1
                stats['equal_calls'] += int(equal)
                stats['finite_calls'] += int(finite)
                if not equal:
                    delta = baseline.float() - result.float()
                    stats['max_abs'] = max(stats['max_abs'], float(delta.abs().max()))
                    stats['max_relative_l2'] = max(stats['max_relative_l2'],
                        float(delta.norm() / baseline.float().norm().clamp_min(1e-12)))
                return result
            if self.profile and eligible:
                start = self.torch.cuda.Event(enable_timing=True)
                end = self.torch.cuda.Event(enable_timing=True)
                start.record()
                result = original(inputs, selected, do_preparation, **kwargs)
                end.record()
                self.events.append((key, selected, start, end))
                return result
            return original(inputs, selected, do_preparation, **kwargs)

        runner.forward = forward
        return runner

    def snapshot(self):
        self.torch.cuda.synchronize()
        elapsed = Counter()
        for key, tactic, start, end in self.events:
            elapsed[(key, tactic)] += start.elapsed_time(end)
        return {"enabled": self.enabled, "profile": self.profile,
                "verification": [{"shape": list(key), **value}
                                 for key, value in sorted(self.verification.items())],
                "calls": sum(self.calls.values()),
                "selected_calls": sum(v for (k, t), v in self.calls.items() if t >= 0),
                "shapes": [{"shape": list(k), "tactic": t, "calls": v,
                            "event_sum_ms": elapsed[(k, t)] if self.profile else None}
                           for (k, t), v in sorted(self.calls.items())]}


def install(policy_path=None):
    global _STATE
    if _STATE is not None:
        return _STATE
    import torch
    import flashinfer
    from flashinfer.gemm import gemm_base

    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This projection policy was validated only on SM120")
    path = Path(policy_path or os.environ.get(
        "DSV41_PREFILL_DENSE_POLICY", str(Path(__file__).with_name("prefill_dense_policy.json"))))
    policy = json.loads(path.read_text())
    if policy["flashinfer_version"] != flashinfer.__version__:
        raise RuntimeError("FlashInfer version differs from the measured projection policy")
    module = gemm_base.get_gemm_sm120_module_cutlass_mxfp8()
    raw = gemm_base._load_gemm_sm120_mxfp8_module()
    if int(raw.mxfp8_gemm_tactic_num()) != 6:
        raise RuntimeError("SM120 tactic enumeration changed; remeasure the policy")
    state = ProjectionPolicy(policy, torch)
    factory = module.cutlass_mxfp8_gemm_runner
    module.cutlass_mxfp8_gemm_runner = lambda: state.wrap(factory())
    _STATE = state
    print("DSV41_PREFILL_DENSE", {"policy": str(path), "shapes": len(state.policy),
                                  "enabled": state.enabled}, flush=True)
    return state
