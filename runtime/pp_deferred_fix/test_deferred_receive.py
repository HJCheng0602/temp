"""CPU-only checks for the deferred PP receive backport.

The runtime module is loaded with small dependency stubs so these tests do not
initialize a process group, CUDA, or the full vLLM package.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
import unittest
from collections import deque
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).parent
PP_UTILS_PATH = ROOT / "v1/worker/gpu/pp_utils.py"
RUNNER_PATH = ROOT / "v1/worker/gpu/model_runner.py"


def _stub(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    sys.modules[name] = module
    return module


def _load_pp_utils():
    _stub("vllm")
    envs = _stub("vllm.envs")
    envs.VLLM_XPU_PP_MICROBATCH = True
    distributed = _stub("vllm.distributed")
    parallel_state = _stub("vllm.distributed.parallel_state")
    parallel_state.get_pp_group = lambda: None
    distributed.parallel_state = parallel_state
    platforms = _stub("vllm.platforms")
    platforms.current_platform = types.SimpleNamespace(is_xpu=lambda: False, is_cuda=lambda: True)
    sequence = _stub("vllm.sequence")
    sequence.IntermediateTensors = object
    utils = _stub("vllm.utils")
    torch_utils = _stub("vllm.utils.torch_utils")
    torch_utils.async_tensor_h2d = lambda value, device: torch.as_tensor(value, device=device)
    utils.torch_utils = torch_utils
    v1 = _stub("vllm.v1")
    worker = _stub("vllm.v1.worker")
    gpu = _stub("vllm.v1.worker.gpu")
    input_batch = _stub("vllm.v1.worker.gpu.input_batch")
    input_batch.InputBatch = object
    v1.worker = worker
    worker.gpu = gpu
    gpu.input_batch = input_batch
    spec = importlib.util.spec_from_file_location("deferred_pp_utils", PP_UTILS_PATH)
    assert spec
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # The deployment runtime is Python 3.10+, but keep this test runnable on
    # older host utility Python versions where ``Event | None`` is evaluated.
    source = "from __future__ import annotations\n" + PP_UTILS_PATH.read_text()
    exec(compile(source, str(PP_UTILS_PATH), "exec"), module.__dict__)
    return module


pp_utils = _load_pp_utils()


class DeferredReceiveTests(unittest.TestCase):
    def test_fifo_delay_and_flush_are_idempotent(self):
        handler = object.__new__(pp_utils.PPHandler)
        slots = [types.SimpleNamespace(launched=False) for _ in range(3)]
        handler.queue = deque([None, slots[0], slots[1], slots[2]])
        handler.recv_launch_delay = 3
        handler.pending_post_model_receive = None
        handler.is_last_rank = False
        launched = []

        def launch(slot):
            if not slot.launched:
                slot.launched = True
                launched.append(slot)

        handler._launch_receive = launch
        handler._advance_receive_queue()
        handler._advance_receive_queue()
        handler.launch_post_model_receive()
        handler.flush_pending_collectives()
        handler.flush_pending_collectives()
        self.assertEqual(launched, slots)

    def test_all_aborted_consume_still_posts_receive(self):
        slot = types.SimpleNamespace(
            launched=False,
            event=None,
            sampled_tokens=torch.zeros((1, 2), dtype=torch.int64),
            combined=torch.zeros((2, 4), dtype=torch.int32),
            num_sampled=torch.zeros(4, dtype=torch.int32),
            num_rejected=torch.zeros(4, dtype=torch.int32),
            idx_mapping=torch.zeros(1, dtype=torch.int64),
            idx_mapping_np=np.array([0]),
            need_sampled_mask=np.array([True]),
            gen_at_receive_np=np.array([0]),
            draft_tokens=None,
        )
        handler = object.__new__(pp_utils.PPHandler)
        handler.queue = deque([slot])
        handler.recv_launch_delay = 0
        handler.pending_post_model_receive = None
        handler.is_last_rank = False
        handler.req_idx_gen_np = np.array([1], dtype=np.int32)
        calls = []
        def launch(value):
            value.launched = True
            calls.append(value)

        handler._launch_receive = launch
        self.assertIsNone(handler.get_prev_sampled_outputs())
        self.assertEqual(calls, [slot])
        self.assertTrue(slot.launched)

    def test_combined_buffer_alignment(self):
        for count in range(1, 9):
            combined = pp_utils._alloc_combined(count, torch.device("cpu"))
            sampled, rejected = combined.unbind(dim=0)
            self.assertGreaterEqual(combined.shape[1], count)
            self.assertEqual(sampled.data_ptr() % 16, 0)
            self.assertEqual(rejected.data_ptr() % 16, 0)

    def test_runner_hooks_and_warmup_contract(self):
        source = RUNNER_PATH.read_text()
        tree = ast.parse(source)
        methods = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertIn("warmup_pp_decode_update", methods)
        self.assertIn("launch_post_model_receive", source)
        self.assertIn("flush_pending_collectives", source)
        warmup = ast.get_source_segment(source, methods["warmup_pp_decode_update"])
        assert warmup is not None
        self.assertIn("dtype=torch.int64", warmup)
        self.assertIn("dtype=torch.int32", warmup)
        self.assertIn("device=self.device", warmup)
        shutdown = ast.get_source_segment(source, methods["shutdown"])
        assert shutdown is not None
        self.assertLess(
            shutdown.index("self.pp_handler.flush_pending_collectives()"),
            shutdown.index("torch.accelerator.synchronize()"),
        )


if __name__ == "__main__":
    unittest.main()
