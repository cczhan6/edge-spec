from __future__ import annotations

import unittest

from src.communication import network_delay_ms
from src.latency import target_only_latency_ms
from src.simulator import Simulator
from tests.common import small_config


class TargetOnlyCapacityTest(unittest.TestCase):
    def test_target_only_requests_share_edge_capacity(self) -> None:
        config, model_runner, workload = small_config(num_requests=2, output_len=4)
        config["edge"]["num_lanes"] = 4
        result = Simulator(
            config,
            model_runner,
            workload,
            "combined_strong_heterogeneous",
            "target_only",
        ).run()
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

    def test_target_only_ttft_returns_after_first_decode(self) -> None:
        config, model_runner, workload = small_config(num_requests=1, output_len=4)
        config["edge"]["target_only_startup_ms"] = 7

        result = Simulator(
            config,
            model_runner,
            workload,
            "combined_strong_heterogeneous",
            "target_only",
        ).run()
        request = result.requests[0]
        device = result.devices[request.device_id].device
        first_token_payload_bytes = (
            int(config["network"]["packet_header_bytes"])
            + int(config["network"]["packet_token_bytes"])
        )
        first_token_downlink_ms = network_delay_ms(
            int(config["simulation"]["seed"]),
            device,
            "downlink",
            "target-only-first-token:0",
            first_token_payload_bytes,
        )

        expected_ttft_ms = (
            request.target_only_uplink_ms
            + request.target_only_queue_wait_ms
            + target_only_latency_ms(config["edge"], 1)
            + first_token_downlink_ms
        )
        self.assertAlmostEqual(request.ttft_ms, expected_ttft_ms)
        self.assertLess(request.ttft_ms, request.latency_ms)


if __name__ == "__main__":
    unittest.main()
