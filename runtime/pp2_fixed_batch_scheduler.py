"""One complete, fixed-size decode batch per PP2 token step."""

from pp8_wave_scheduler import WaveScheduler


class FixedBatchScheduler(WaveScheduler):
    expected_pp_size = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._check_whole_batch()

    def _check_whole_batch(self):
        if self._wave_total != self._wave_size:
            raise ValueError("FixedBatchScheduler requires one unsplit cohort")

    def _load_control(self):
        super()._load_control()
        self._check_whole_batch()
