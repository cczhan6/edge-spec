from __future__ import annotations

import unittest

from src.simulator import Simulator
from tests.common import small_config


class TargetOnlyCapacityTest(unittest.TestCase):
    def test_target_only_requests_share_edge_capacity(self) -> None:
        config, model_runner, workload = small_config(num_requests=2, output_len=4)
        config["edge"]["num_lanes"] = 4
        result = Simulator(config, model_runner, workload, "combined_strong_heterogeneous", "target_only").run()
        requests = result.requests
        self.assertEqual(len(result.lanes), 0)
        self.assertAlmostEqual(requests[1].finish_time_ms - requests[0].finish_time_ms, 50.0)
        self.assertEqual(
            [event["lane_id"] for event in result.event_trace if event["event"] == "target_only_service"],
            [0, 0],
        )
        self.assertGreater(requests[0].target_only_uplink_ms, 0.0)
        self.assertGreater(requests[0].target_only_downlink_ms, 0.0)
        self.assertEqual(requests[0].target_only_uplink_payload_bytes, 160)
        self.assertEqual(requests[0].target_only_downlink_payload_bytes, 144)


if __name__ == "__main__":
    unittest.main()
