from __future__ import annotations

import unittest

from src.simulator import Simulator
from tests.common import small_config


class LaneNoMicrobatchTest(unittest.TestCase):
    def test_lane_trace_always_has_batch_size_one(self) -> None:
        config, model_runner, workload = small_config(num_requests=6, output_len=8)
        config["edge"]["num_lanes"] = 1
        result = Simulator(config, model_runner, workload, "balanced_drafter", "full").run()
        lane_events = [event for event in result.event_trace if event["event"] == "lane_verify"]
        self.assertGreater(len(lane_events), 1)
        self.assertTrue(all(event["batch_size"] == 1 for event in lane_events))
        starts = [event["start_time_ms"] for event in lane_events]
        finishes = [event["finish_time_ms"] for event in lane_events]
        self.assertTrue(all(starts[index] >= finishes[index - 1] for index in range(1, len(starts))))


if __name__ == "__main__":
    unittest.main()
