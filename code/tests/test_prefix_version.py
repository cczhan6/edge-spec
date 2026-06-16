from __future__ import annotations

import unittest

from src.simulator import Simulator
from tests.common import rejecting_model_runner, small_config


class PrefixVersionTest(unittest.TestCase):
    def test_rejection_marks_old_pending_segment_stale(self) -> None:
        config, _, workload = small_config(num_requests=1, output_len=8)
        model_runner = rejecting_model_runner()
        result = Simulator(config, model_runner, workload, "balanced_drafter", "full").run()
        request = result.requests[0]
        self.assertGreater(request.prefix_version, 0)
        self.assertTrue(any(segment.status == "stale" for segment in result.segments))


if __name__ == "__main__":
    unittest.main()
