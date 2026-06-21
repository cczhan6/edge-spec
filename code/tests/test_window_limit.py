from __future__ import annotations

import unittest

from src.simulator import Simulator
from tests.common import accepting_model_runner, small_config


class WindowLimitTest(unittest.TestCase):
    def test_full_drafts_beyond_fixed_window_like_dsi(self) -> None:
        config, _, workload = small_config(num_requests=1, output_len=8)
        config["speculation"]["W_default"] = 2
        config["speculation"]["gamma_candidates"] = [1]
        config["speculation"]["gamma_fixed"] = 1

        result = Simulator(
            config,
            accepting_model_runner(),
            workload,
            "combined_strong_heterogeneous",
            "full",
        ).run()

        self.assertGreater(result.requests[0].max_outstanding_observed, 2)
        self.assertTrue(all(runtime.total_busy_time_ms > 0.0 for runtime in result.devices))

    def test_wo_async_keeps_single_segment_window(self) -> None:
        config, model_runner, workload = small_config(num_requests=2, output_len=24)
        result = Simulator(config, model_runner, workload, "combined_strong_heterogeneous", "wo_async").run()
        self.assertTrue(all(request.max_outstanding_observed <= 1 for request in result.requests))
        self.assertTrue(all(runtime.total_busy_time_ms > 0.0 for runtime in result.devices))


if __name__ == "__main__":
    unittest.main()
