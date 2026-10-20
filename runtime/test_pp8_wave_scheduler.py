import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


class _Output:
    def __init__(self, ids):
        self.num_scheduled_tokens = ids
        self.preempted_req_ids = set()


class _Request:
    def __init__(self, request_id, prompt=10, output=1):
        self.request_id = request_id
        self.num_prompt_tokens = prompt
        self.num_computed_tokens = prompt
        self._output_token_ids = [0] * output
        self.next_decode_eligible_step = 0
        self.num_output_placeholders = 0


def _load_scheduler(pp_size=8, fixed=False):
    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.init_logger = lambda _: types.SimpleNamespace(info=lambda *a, **k: None)
    output_mod = types.ModuleType("vllm.v1.core.sched.output")
    output_mod.SchedulerOutput = _Output

    class FakeAsyncScheduler:
        def __init__(self, *args, **kwargs):
            self.parallel_config = types.SimpleNamespace(pipeline_parallel_size=pp_size)
            self.pp_size = pp_size
            self.current_step = 0
            self.requests = {}
            self.running = []
            self.next_outputs = []
            self.seen_gates = []
            self.clear_on_update = False
            self.finish_at = None

        def schedule(self, throttle_prefills=False):
            self.current_step += 1
            self.seen_gates.append({r.request_id: r.next_decode_eligible_step for r in self.running})
            return self.next_outputs.pop(0) if self.next_outputs else _Output({})

        def update_from_output(self, scheduler_output, model_runner_output):
            if model_runner_output.sampled_token_ids is not None:
                for request_id, index in model_runner_output.req_id_to_index.items():
                    request = self.requests.get(request_id)
                    if request is None:
                        continue
                    request._output_token_ids.extend(
                        model_runner_output.sampled_token_ids[index]
                    )
                    request.next_decode_eligible_step = self.current_step + self.pp_size
                    if self.finish_at is not None and len(request._output_token_ids) >= self.finish_at:
                        self.requests.pop(request_id, None)
                        self.running = [r for r in self.running if r.request_id != request_id]
            if self.clear_on_update:
                self.requests.clear()
            return "parent-output"

    async_mod = types.ModuleType("vllm.v1.core.sched.async_scheduler")
    async_mod.AsyncScheduler = FakeAsyncScheduler
    vllm = types.ModuleType("vllm")
    vllm_v1 = types.ModuleType("vllm.v1")
    vllm_core = types.ModuleType("vllm.v1.core")
    vllm_sched = types.ModuleType("vllm.v1.core.sched")
    modules = {
        "vllm": vllm,
        "vllm.logger": logger_mod,
        "vllm.v1": vllm_v1,
        "vllm.v1.core": vllm_core,
        "vllm.v1.core.sched": vllm_sched,
        "vllm.v1.core.sched.output": output_mod,
        "vllm.v1.core.sched.async_scheduler": async_mod,
    }
    old = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    path = Path(__file__).with_name("pp8_wave_scheduler.py")
    spec = importlib.util.spec_from_file_location("pp8_wave_scheduler_tested", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if fixed:
        previous_base = sys.modules.get("pp8_wave_scheduler")
        sys.modules["pp8_wave_scheduler"] = module
        fixed_spec = importlib.util.spec_from_file_location(
            "pp2_fixed_batch_scheduler_tested", path.with_name("pp2_fixed_batch_scheduler.py"))
        fixed_module = importlib.util.module_from_spec(fixed_spec)
        fixed_spec.loader.exec_module(fixed_module)
        if previous_base is None:
            sys.modules.pop("pp8_wave_scheduler")
        else:
            sys.modules["pp8_wave_scheduler"] = previous_base
    for name, previous in old.items():
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    return fixed_module.FixedBatchScheduler if fixed else module.WaveScheduler


class WaveSchedulerStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scheduler_cls = _load_scheduler()

    def setUp(self):
        os.environ["DSV41_WAVE_BATCH"] = "4"
        os.environ["DSV41_WAVE_SIZE"] = "2"
        os.environ.pop("DSV41_WAVE_CONTROL", None)
        os.environ.pop("DSV41_WAVE_LOG", None)
        self.scheduler = self.scheduler_cls()
        self.scheduler.requests = {
            str(i): _Request(str(i)) for i in range(4)
        }
        self.scheduler.running = list(self.scheduler.requests.values())
        self.scheduler.next_outputs = [
            _Output({"0": 1, "1": 1}),
            _Output({"2": 1, "3": 1}),
            _Output({}),
        ]

    def test_wave_waits_for_all_cohorts_and_all_outputs(self):
        self.scheduler.schedule()
        self.scheduler.update_from_output(
            _Output({}), types.SimpleNamespace(
                sampled_token_ids=[[11], [12]],
                req_id_to_index={"0": 0, "1": 1},
            )
        )
        self.scheduler.schedule()
        self.assertTrue(self.scheduler._awaiting_wave_outputs)
        self.scheduler.schedule()
        self.assertEqual(self.scheduler._wave_empty_count, 1)
        self.assertTrue(self.scheduler._awaiting_wave_outputs)

        self.scheduler.update_from_output(
            _Output({}), types.SimpleNamespace(
                sampled_token_ids=[[13], [14]],
                req_id_to_index={"2": 0, "3": 1},
            )
        )
        self.assertFalse(self.scheduler._awaiting_wave_outputs)
        self.assertEqual(self.scheduler._group_index, 0)
        self.assertTrue(self.scheduler._prefill_barrier_complete)

        self.scheduler.next_outputs.extend([
            _Output({"0": 1, "1": 1}),
            _Output({"2": 1, "3": 1}),
        ])
        self.scheduler.schedule()
        self.scheduler.schedule()
        self.assertTrue(self.scheduler._awaiting_wave_outputs)

    def test_three_waves_respect_pp_cadence_and_reset_after_final_output(self):
        self.scheduler.next_outputs = [
            _Output({"0": 1, "1": 1}),
            _Output({"2": 1, "3": 1}),
        ]
        self.scheduler.finish_at = 4

        def update(ids):
            self.scheduler.update_from_output(
                _Output({}), types.SimpleNamespace(
                    sampled_token_ids=[[10] for _ in ids],
                    req_id_to_index={request_id: i for i, request_id in enumerate(ids)},
                )
            )

        self.scheduler.schedule()
        update(["0", "1"])
        self.scheduler.schedule()
        update(["2", "3"])

        for wave in range(2):
            # Both cohorts are held until the parent cadence permits the first
            # cohort again; those calls must remain empty PP advancement steps.
            self.scheduler.next_outputs.extend(
                [_Output({})] * 7
                + [_Output({"0": 1, "1": 1}), _Output({"2": 1, "3": 1})]
            )
            dispatches_before = len(self.scheduler._wave_dispatches)
            while len(self.scheduler._wave_dispatches) == dispatches_before:
                self.scheduler.schedule()
            update(["0", "1"])
            self.scheduler.schedule()
            update(["2", "3"])
            if wave == 0:
                self.assertTrue(self.scheduler._prefill_barrier_complete)

        self.assertFalse(self.scheduler._armed)
        self.assertEqual(self.scheduler.requests, {})

    def test_prefill_gate_blocks_decode_before_arm(self):
        scheduler = self.scheduler_cls()
        request = _Request("0")
        request.num_computed_tokens = request.num_prompt_tokens
        scheduler.requests = {"0": request}
        scheduler.running = [request]
        scheduler.schedule()
        self.assertGreater(scheduler.seen_gates[-1]["0"], scheduler.current_step)
        self.assertEqual(request.next_decode_eligible_step, 0)

    def test_multiple_sampled_tokens_fail_closed(self):
        self.scheduler.schedule()
        self.scheduler.schedule()
        with self.assertRaisesRegex(RuntimeError, "more than one"):
            self.scheduler.update_from_output(
                _Output({}), types.SimpleNamespace(
                    sampled_token_ids=[[1, 2]], req_id_to_index={"0": 0}
                )
            )

    def test_final_output_resets_after_parent_removes_requests(self):
        self.scheduler.schedule()
        self.scheduler.schedule()
        self.scheduler.clear_on_update = True
        self.scheduler.update_from_output(
            _Output({}), types.SimpleNamespace(
                sampled_token_ids=[[1], [2], [3], [4]],
                req_id_to_index={str(i): i for i in range(4)},
            )
        )
        self.assertFalse(self.scheduler._armed)

    def test_idle_schedule_does_not_load_stale_control_before_next_workload(self):
        with tempfile.TemporaryDirectory() as directory:
            control = Path(directory) / "control.json"
            control.write_text(json.dumps({"batch": 4, "size": 4}))
            os.environ["DSV41_WAVE_CONTROL"] = str(control)
            self.scheduler.next_outputs = []
            self.scheduler.requests = {}
            self.scheduler.running = []
            self.scheduler._reset_for_next_workload()

            # This idle call is between workloads; the old control must not be
            # consumed before the runner writes the next workload's control.
            self.scheduler.schedule()
            self.assertEqual(self.scheduler._wave_size, 2)
            self.assertFalse(self.scheduler._control_loaded_for_workload)

            self.scheduler.requests = {
                str(i): _Request(str(i)) for i in range(4)
            }
            self.scheduler.running = list(self.scheduler.requests.values())
            self.scheduler.schedule()
            self.assertEqual(self.scheduler._wave_size, 4)
            self.assertTrue(self.scheduler._control_loaded_for_workload)


class FixedBatchSchedulerTests(unittest.TestCase):
    def setUp(self):
        self.scheduler_cls = _load_scheduler(pp_size=2, fixed=True)
        os.environ["DSV41_WAVE_BATCH"] = "8"
        os.environ["DSV41_WAVE_SIZE"] = "8"
        os.environ.pop("DSV41_WAVE_CONTROL", None)
        os.environ.pop("DSV41_WAVE_LOG", None)

    def test_rejects_split_control(self):
        os.environ["DSV41_WAVE_SIZE"] = "4"
        with self.assertRaisesRegex(ValueError, "unsplit"):
            self.scheduler_cls()

    def test_rejects_parent_dispatching_only_part_of_batch(self):
        scheduler = self.scheduler_cls()
        scheduler.requests = {str(i): _Request(str(i)) for i in range(8)}
        scheduler.running = list(scheduler.requests.values())
        scheduler.next_outputs = [_Output({str(i): 1 for i in range(2)})]
        with self.assertRaisesRegex(RuntimeError, "does not equal cohort"):
            scheduler.schedule()

    def test_all_eight_samples_required_before_next_whole_batch(self):
        scheduler = self.scheduler_cls()
        scheduler.requests = {str(i): _Request(str(i)) for i in range(8)}
        scheduler.running = list(scheduler.requests.values())
        scheduler.next_outputs = [_Output({str(i): 1 for i in range(8)}), _Output({})]
        scheduler.schedule()
        self.assertEqual(len(scheduler._wave_dispatches), 1)
        scheduler.update_from_output(_Output({}), types.SimpleNamespace(
            sampled_token_ids=[[1]] * 7, req_id_to_index={str(i): i for i in range(7)}))
        scheduler.schedule()
        self.assertTrue(scheduler._awaiting_wave_outputs)
        scheduler.update_from_output(_Output({}), types.SimpleNamespace(
            sampled_token_ids=[[1]], req_id_to_index={"7": 0}))
        self.assertFalse(scheduler._awaiting_wave_outputs)
        self.assertEqual(scheduler._group_index, 0)


if __name__ == "__main__":
    unittest.main()
