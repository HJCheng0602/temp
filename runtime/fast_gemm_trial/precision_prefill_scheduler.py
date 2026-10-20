"""Opt-in admission barrier for repeatable prefill diagnostics.

The normal wave scheduler can start prefilling the first HTTP arrivals before
the complete fixed cohort is present.  This subclass holds the scheduler at a
valid empty step until the cohort is complete, then orders the FCFS waiting
queue once by the request's logical identifier.  Decode wave behavior remains
owned by ``FixedBatchScheduler``.
"""

from __future__ import annotations

from typing import Any

from pp2_fixed_batch_scheduler import FixedBatchScheduler
from vllm.logger import init_logger
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy


logger = init_logger(__name__)


class PrecisionPrefillScheduler(FixedBatchScheduler):
    """Fixed PP2 scheduler with an all-request prefill admission barrier."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.policy != SchedulingPolicy.FCFS:
            raise ValueError("PrecisionPrefillScheduler requires FCFS policy")
        if not isinstance(self.waiting, FCFSRequestQueue):
            raise TypeError("PrecisionPrefillScheduler requires FCFSRequestQueue")

    def _order_waiting_once(self) -> None:
        if not isinstance(self.waiting, FCFSRequestQueue):
            raise TypeError("PrecisionPrefillScheduler requires FCFSRequestQueue")
        ordered = sorted(
            tuple(self.waiting), key=lambda request: request.request_id
        )
        self.waiting.clear()
        self.waiting.extend(ordered)
        logger.info(
            "precision prefill admission barrier released: count=%d request_ids=%s",
            len(ordered),
            [request.request_id for request in ordered],
        )

    def schedule(self, throttle_prefills: bool = False):
        was_armed = self._armed
        self._check_request_set()

        if not self._armed:
            old_budget = self.max_num_scheduled_tokens
            self.max_num_scheduled_tokens = 0
            try:
                output = super().schedule(throttle_prefills=throttle_prefills)
            finally:
                self.max_num_scheduled_tokens = old_budget
            if output.num_scheduled_tokens or output.preempted_req_ids:
                raise RuntimeError(
                    "prefill admission barrier produced a non-empty scheduler output"
                )
            return output

        if not was_armed:
            if self.running or self.skipped_waiting:
                raise RuntimeError(
                    "cohort armed after requests were already admitted or skipped"
                )
            self._order_waiting_once()

        # The parent owns running-list order and PP cadence after admission;
        # only the initial FCFS waiting queue is reordered above.
        return super().schedule(throttle_prefills=throttle_prefills)
