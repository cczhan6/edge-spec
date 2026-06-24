from __future__ import annotations

import unittest

from src.latency import target_only_latency_ms
from src.methods import get_method_spec
from src.simulator import Simulator
from tests.common import small_config


class TargetOnlyContractTest(unittest.TestCase):
    def test_target_only_method_is_registered(self) -> None:
        config, _, _ = small_config(num_requests=1, output_len=4)

        spec = get_method_spec("target_only", config)

        self.assertEqual(spec.runtime, "target_only")
        self.assertEqual(spec.window_size, 0)
        self.assertEqual(spec.num_lanes, 0)
        self.assertFalse(spec.global_batch)

    def test_target_only_greedy_output_matches_model_runner(self) -> None:
        config, model_runner, workload = small_config(num_requests=2, output_len=5)

        result = Simulator(
            config,
            model_runner,
            workload,
            "combined_strong_heterogeneous",
            "target_only",
        ).run()

        for request in result.requests:
            self.assertEqual(
                request.generated_ids,
                model_runner.target_only(request.prompt_ids, request.output_len),
            )
            self.assertEqual(request.edge_generated_ids, request.generated_ids)

    def test_target_only_has_no_draft_verify_batch_or_network_events(self) -> None:
        config, model_runner, workload = small_config(num_requests=2, output_len=4)

        result = Simulator(
            config,
            model_runner,
            workload,
            "combined_strong_heterogeneous",
            "target_only",
        ).run()

        self.assertEqual(result.segments, [])
        event_names = [event["event"] for event in result.event_trace]
        self.assertEqual(
            event_names,
            [
                "target_only_service",
                "target_only_service",
                "request_finish",
                "request_finish",
            ],
        )
        for event in result.event_trace:
            self.assertNotIn("uplink_ms", event)
            self.assertNotIn("uplink_payload_bytes", event)
            self.assertNotIn("downlink_ms", event)
            self.assertNotIn("downlink_payload_bytes", event)

    def test_target_only_target_resource_serializes_fcfs(self) -> None:
        config, model_runner, workload = small_config(num_requests=3, output_len=4)
        config["edge"]["num_lanes"] = 4

        result = Simulator(
            config,
            model_runner,
            workload,
            "combined_strong_heterogeneous",
            "target_only",
        ).run()

        service_events = [
            event for event in result.event_trace if event["event"] == "target_only_service"
        ]
        self.assertEqual([event["request_id"] for event in service_events], [0, 1, 2])
        self.assertEqual([event["lane_id"] for event in service_events], [0, 0, 0])
        for previous, current in zip(service_events, service_events[1:]):
            self.assertGreaterEqual(current["start_time_ms"], previous["finish_time_ms"])
        expected_service_ms = target_only_latency_ms(config["edge"], 4)
        self.assertEqual(
            [event["compute_ms"] for event in service_events],
            [expected_service_ms, expected_service_ms, expected_service_ms],
        )

    def test_target_only_decode_ready_no_prefill_or_prompt_transfer(self) -> None:
        config, model_runner, workload = small_config(num_requests=1, output_len=4)

        result = Simulator(
            config,
            model_runner,
            workload,
            "combined_strong_heterogeneous",
            "target_only",
        ).run()

        request = result.requests[0]
        self.assertEqual(request.decode_ready_time_ms, request.arrival_time_ms)
        self.assertEqual(request.target_only_downlink_ms, 0.0)
        self.assertEqual(request.target_only_downlink_payload_bytes, 0)

    def test_target_only_no_speculative_counters_incremented(self) -> None:
        config, model_runner, workload = small_config(num_requests=2, output_len=4)

        result = Simulator(
            config,
            model_runner,
            workload,
            "combined_strong_heterogeneous",
            "target_only",
        ).run()

        self.assertTrue(all(request.accepted_tokens == 0 for request in result.requests))
        self.assertTrue(all(request.rejected_count == 0 for request in result.requests))
        self.assertTrue(all(request.rollback_count == 0 for request in result.requests))
        self.assertTrue(all(request.wasted_draft_tokens == 0 for request in result.requests))
        self.assertTrue(all(runtime.generated_draft_tokens == 0 for runtime in result.devices))
        self.assertTrue(all(runtime.accepted_draft_tokens == 0 for runtime in result.devices))


if __name__ == "__main__":
    unittest.main()
