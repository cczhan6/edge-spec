from __future__ import annotations

import unittest

from src.simulator import Simulator
from tests.common import rejecting_model_runner, small_config


class ConservativeRollbackTest(unittest.TestCase):
    def test_conservative_rollback_discards_pending_segments(self) -> None:
        config, _, workload = small_config(num_requests=1, output_len=8)
        model_runner = rejecting_model_runner()
        result = Simulator(config, model_runner, workload, "balanced_drafter", "conservative_rollback").run()
        self.assertTrue(any(segment.status == "discarded" for segment in result.segments))
        self.assertEqual(result.requests[0].in_flight_segments, [])


if __name__ == "__main__":
    unittest.main()
