"""Deterministic PP8 decode waves for one bounded HTTP workload.

This module is selected with ``--scheduler-cls=pp8_wave_scheduler.WaveScheduler``.
It is intentionally fail-closed: the measurements are invalid if vLLM's
scheduler does not dispatch exactly one fixed-size cohort at a time.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


class WaveScheduler(AsyncScheduler):
    """Gate a single 128-request run into fixed decode cohorts."""

    expected_pp_size = 8

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._wave_total = self._env_int("DSV41_WAVE_BATCH", 128)
        self._wave_size = self._env_int("DSV41_WAVE_SIZE", 16)
        self._wave_log = os.environ.get("DSV41_WAVE_LOG")
        if self._wave_total <= 0 or self._wave_size <= 0:
            raise ValueError("DSV41_WAVE_BATCH and DSV41_WAVE_SIZE must be positive")
        if self.parallel_config.pipeline_parallel_size != self.expected_pp_size:
            raise ValueError(f"WaveScheduler requires PP{self.expected_pp_size}")
        if self._wave_total % self._wave_size:
            raise ValueError("DSV41_WAVE_BATCH must be divisible by DSV41_WAVE_SIZE")

        self._armed = False
        self._request_ids: tuple[str, ...] = ()
        self._cohorts: list[tuple[str, ...]] = []
        self._group_index = 0
        self._wave_number = 0
        self._wave_dispatches: list[dict[str, Any]] = []
        self._wave_start: float | None = None
        self._wave_empty_count = 0
        self._wave_before_outputs: dict[str, int] = {}
        self._wave_received: dict[str, int] = {}
        self._awaiting_wave_outputs = False
        self._prefill_barrier_complete = False
        self._control_loaded_for_workload = False
        self._logger_handle = None

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        value = os.environ.get(name)
        return default if value is None else int(value)

    def _write_log(self, value: dict[str, Any]) -> None:
        if not self._wave_log:
            return
        if self._logger_handle is None:
            path = Path(self._wave_log)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._logger_handle = path.open("a", encoding="utf-8")
        self._logger_handle.write(json.dumps(value, sort_keys=True) + "\n")
        self._logger_handle.flush()

    def _load_control(self) -> None:
        control_path = os.environ.get("DSV41_WAVE_CONTROL")
        if not control_path:
            return
        path = Path(control_path)
        if not path.is_file():
            return
        control = json.loads(path.read_text(encoding="utf-8"))
        size = int(control.get("size", self._wave_size))
        batch = int(control.get("batch", self._wave_total))
        log_path = control.get("log", self._wave_log)
        if batch <= 0 or size <= 0 or batch % size:
            raise ValueError(f"invalid wave control: {control}")
        if batch != self._wave_total or size != self._wave_size:
            self._wave_total, self._wave_size = batch, size
        if log_path != self._wave_log:
            if self._logger_handle is not None:
                self._logger_handle.close()
                self._logger_handle = None
            self._wave_log = log_path

    def _reset_for_next_workload(self) -> None:
        self._armed = False
        self._request_ids = ()
        self._cohorts = []
        self._group_index = 0
        self._wave_dispatches = []
        self._wave_start = None
        self._wave_empty_count = 0
        self._wave_before_outputs = {}
        self._wave_received = {}
        self._awaiting_wave_outputs = False
        self._prefill_barrier_complete = False
        self._control_loaded_for_workload = False

    @staticmethod
    def _output_count(request: Any) -> int:
        return len(request._output_token_ids)

    def _check_request_set(self) -> None:
        current = tuple(sorted(self.requests))
        if not self._armed:
            # The runner may call schedule once while the previous workload is
            # drained and before it writes the next control file. Do not read
            # a stale control file during that idle interval.
            if current and not self._control_loaded_for_workload:
                self._load_control()
                self._control_loaded_for_workload = True
            if len(current) > self._wave_total:
                raise RuntimeError(
                    f"wave workload exceeded {self._wave_total} requests: {current}"
                )
            if len(current) == self._wave_total:
                self._armed = True
                self._request_ids = current
                self._cohorts = [
                    current[i : i + self._wave_size]
                    for i in range(0, self._wave_total, self._wave_size)
                ]
                logger.info("PP%d wave scheduler armed with %d requests", self.expected_pp_size, len(current))
            return
        missing = set(self._request_ids) - set(current)
        if missing and all(self._wave_received.get(request_id, 0) == 1 for request_id in missing):
            return
        if missing:
            raise RuntimeError(
                "a request disappeared before its sampled output was received: "
                f"{sorted(missing)}"
            )
        if self._awaiting_wave_outputs:
            return
        if not current and not self._awaiting_wave_outputs:
            raise RuntimeError("wave workload finished before its final output drain")
        if current != self._request_ids:
            raise RuntimeError(
                "wave request set changed unexpectedly: "
                f"expected={self._request_ids}, actual={current}"
            )

    def _all_first_samples_returned(self) -> bool:
        if self._prefill_barrier_complete:
            return True
        complete = self._armed and all(
            request_id in self.requests
            and self._output_count(self.requests[request_id]) == 1
            for request_id in self._request_ids
        )
        if complete:
            self._prefill_barrier_complete = True
        return complete

    def _finish_wave(self, now: float) -> None:
        if not self._awaiting_wave_outputs:
            return
        deltas = dict(self._wave_received)
        if len(deltas) != self._wave_total or any(delta != 1 for delta in deltas.values()):
            raise RuntimeError(f"expected exactly one sampled token per request: {deltas}")
        self._write_log(
            {
                "kind": "wave",
                "wave": self._wave_number,
                "group_idx": self._group_index,
                "dispatches": self._wave_dispatches,
                "wave_start": self._wave_start,
                "wave_end": now,
                "emptycount": self._wave_empty_count,
                "output_token_ordinal": max(self._wave_before_outputs.values()) + 1,
            }
        )
        self._awaiting_wave_outputs = False
        self._wave_before_outputs = {}
        self._wave_received = {}
        self._wave_dispatches = []
        self._wave_start = None
        self._wave_empty_count = 0
        self._group_index = 0

    def update_from_output(self, scheduler_output: SchedulerOutput, model_runner_output: Any):
        wave_complete_time = None
        sampled = model_runner_output.sampled_token_ids
        if sampled is not None and self._wave_before_outputs:
            index = model_runner_output.req_id_to_index
            for request_id in tuple(self._wave_before_outputs):
                req_index = index.get(request_id)
                if req_index is not None:
                    self._wave_received[request_id] = (
                        self._wave_received.get(request_id, 0)
                        + len(sampled[req_index])
                    )
                    if self._wave_received[request_id] > 1:
                        raise RuntimeError(
                            f"more than one token/request in wave output: {request_id}"
                        )
        if (
            self._awaiting_wave_outputs
            and len(self._wave_received) == self._wave_total
            and all(value == 1 for value in self._wave_received.values())
        ):
            wave_complete_time = time.monotonic()
        output = super().update_from_output(scheduler_output, model_runner_output)
        if wave_complete_time is not None:
            old_request_ids = set(self._request_ids)
            self._finish_wave(wave_complete_time)
            if not old_request_ids.intersection(self.requests):
                self._reset_for_next_workload()
        return output

    def _temporary_gate_ids(self, allowed: set[str]) -> dict[Any, int]:
        saved: dict[Any, int] = {}
        # Do not replace self.running: the parent owns admission, preemption,
        # and V2 PP cadence. Only make non-cohort requests ineligible this call.
        sentinel = self.current_step + max(self.pp_size, 1) + self._wave_total + 1
        for request in self.running:
            if request.request_id not in allowed:
                saved[request] = request.next_decode_eligible_step
                request.next_decode_eligible_step = sentinel
        return saved

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self._check_request_set()
        now = time.monotonic()

        # Until every request has produced its first sample, allow pending
        # prefills to advance but suppress requests that already sampled.
        if not self._armed:
            allowed = {
                request.request_id
                for request in self.running
                if request.num_computed_tokens < request.num_prompt_tokens
            }
            saved = self._temporary_gate_ids(allowed)
            try:
                return super().schedule(throttle_prefills=throttle_prefills)
            finally:
                for request, value in saved.items():
                    request.next_decode_eligible_step = value

        if not self._all_first_samples_returned():
            allowed = {
                request.request_id
                for request in self.running
                if request.num_computed_tokens < request.num_prompt_tokens
            }
        else:
            if self._awaiting_wave_outputs:
                allowed = set()
            else:
                if self._group_index >= len(self._cohorts):
                    raise RuntimeError("invalid wave group index")
                selected = self._cohorts[self._group_index]
                candidate_step = self.current_step + 1
                ready = all(
                    request_id in self.requests
                    and self.requests[request_id].next_decode_eligible_step <= candidate_step
                    and self.requests[request_id].num_output_placeholders == 0
                    for request_id in selected
                )
                allowed = set(selected) if ready else set()

        saved = self._temporary_gate_ids(allowed)
        try:
            output = super().schedule(throttle_prefills=throttle_prefills)
        finally:
            for request, value in saved.items():
                request.next_decode_eligible_step = value

        actual_ids = tuple(sorted(output.num_scheduled_tokens))
        if not self._all_first_samples_returned():
            # Prefill may legitimately be split; no fixed decode wave exists yet.
            return output
        if output.preempted_req_ids:
            raise RuntimeError(f"unexpected preemption: {output.preempted_req_ids}")
        if actual_ids:
            if self._awaiting_wave_outputs:
                raise RuntimeError(
                    f"dispatch occurred before all wave outputs returned: {actual_ids}"
                )
            expected = tuple(sorted(self._cohorts[self._group_index]))
            if actual_ids != expected:
                raise RuntimeError(
                    f"actual dispatch does not equal cohort: expected={expected}, "
                    f"actual={actual_ids}, tokens={output.num_scheduled_tokens}"
                )
            if any(value != 1 for value in output.num_scheduled_tokens.values()):
                raise RuntimeError(
                    f"more than one token/request in wave: {output.num_scheduled_tokens}"
                )
            if self._wave_start is None:
                self._wave_start = now
            self._wave_dispatches.append(
                {"time": now, "group_idx": self._group_index,
                 "request_ids": list(actual_ids),
                 "num_scheduled_tokens": dict(output.num_scheduled_tokens)}
            )
            for request_id in expected:
                if request_id in self._wave_before_outputs:
                    raise RuntimeError(f"request dispatched twice in one wave: {request_id}")
                self._wave_before_outputs[request_id] = self._output_count(
                    self.requests[request_id]
                )
                self._wave_received[request_id] = 0
            self._group_index += 1
            if self._group_index == len(self._cohorts):
                self._awaiting_wave_outputs = True
                self._wave_number += 1
        else:
            self._wave_empty_count += 1
        return output
