from __future__ import annotations

import unittest

from src.simulator import Simulator
from tests.common import accepting_model_runner, small_config


class IntraRequestParallelVerifyTest(unittest.TestCase):
    def test_full_verifies_later_position_before_frontier_verify_finishes(self) -> None:
        config, _, workload = small_config(num_requests=1, output_len=6)
        config["edge"]["num_lanes"] = 2
        config["speculation"]["W_default"] = 2
        config["speculation"]["gamma_candidates"] = [1]
        config["speculation"]["gamma_fixed"] = 1

        result = Simulator(
            config,
            accepting_model_runner(),
            workload,
            "balanced_drafter",
            "full",
        ).run()

        lane_events = [
            event
            for event in result.event_trace
            if event["event"] == "lane_verify" and event["request_id"] == 0
        ]
        self.assertGreaterEqual(len(lane_events), 2)
        first, second = lane_events[:2]
        first_segment = result.segments[first["segment_id"]]
        second_segment = result.segments[second["segment_id"]]

        self.assertLess(first_segment.base_pos, second_segment.base_pos)
        self.assertLess(second["start_time_ms"], first["finish_time_ms"])


if __name__ == "__main__":
    unittest.main()
