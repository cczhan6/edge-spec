from __future__ import annotations

import unittest

from src.config import validate_config
from src.methods import get_method_spec
from src.simulator import Simulator
from tests.common import accepting_model_runner, rejecting_model_runner, small_config


class ServerOnlyLinearTest(unittest.TestCase):
    def test_server_only_linear_default_batch_size_is_one(self) -> None:
        config, _, _ = small_config(num_requests=1, output_len=4)

        self.assertEqual(config["server_only"]["batch_size"], 1)

    def test_server_only_linear_method_is_registered(self) -> None:
        config, _, _ = small_config(num_requests=1, output_len=4)

        spec = get_method_spec("server_only_linear", config)

        self.assertEqual(spec.runtime, "server_only_specedge")
        self.assertEqual(spec.candidate_strategy, "linear")
        self.assertFalse(spec.global_batch)

    def test_server_only_linear_uses_linear_interfaces_only(self) -> None:
        config, model_runner, workload = small_config(num_requests=1, output_len=8)
        config["speculation"]["gamma_fixed"] = 3

        result = Simulator(
            config,
            model_runner,
            workload,
            "combined_strong_heterogeneous",
            "server_only_linear",
        ).run()

        self.assertTrue(result.segments)
        self.assertTrue(all(segment.tree_strategy == "linear" for segment in result.segments))
        self.assertTrue(all(segment.draft_tree is None for segment in result.segments))
        self.assertTrue(all(segment.target_verify_tree_nodes == 1 for segment in result.segments))

    def test_server_only_linear_has_no_network_or_proactive_events(self) -> None:
        config, model_runner, workload = small_config(num_requests=2, output_len=6)

        result = Simulator(
            config,
            model_runner,
            workload,
            "combined_strong_heterogeneous",
            "server_only_linear",
        ).run()

        self.assertTrue(all(request.target_only_downlink_ms == 0.0 for request in result.requests))
        self.assertTrue(all(request.target_only_downlink_payload_bytes == 0 for request in result.requests))
        self.assertTrue(all(segment.uplink_delay_ms == 0.0 for segment in result.segments))
        self.assertTrue(all(segment.downlink_delay_ms == 0.0 for segment in result.segments))
        self.assertFalse(any(event["event"] == "server_only_response_downlink" for event in result.event_trace))
        self.assertFalse(any(event["event"] == "proactive_draft" for event in result.event_trace))
        for event in result.event_trace:
            self.assertNotIn("uplink_ms", event)
            self.assertNotIn("uplink_payload_bytes", event)
            self.assertNotIn("downlink_ms", event)
            self.assertNotIn("downlink_payload_bytes", event)

    def test_server_only_linear_round_order_is_draft_then_verify(self) -> None:
        config, _, workload = small_config(num_requests=1, output_len=9)
        config["speculation"]["gamma_fixed"] = 3

        result = Simulator(
            config,
            accepting_model_runner(),
            workload,
            "combined_strong_heterogeneous",
            "server_only_linear",
        ).run()

        service_events = [
            event
            for event in result.event_trace
            if event["event"] in {"server_only_draft", "server_only_verify"}
        ]
        self.assertEqual(
            [event["event"] for event in service_events],
            [
                "server_only_draft",
                "server_only_verify",
                "server_only_draft",
                "server_only_verify",
                "server_only_draft",
                "server_only_verify",
            ],
        )
        for draft_event, verify_event in zip(service_events[::2], service_events[1::2]):
            self.assertEqual(draft_event["resource"], "server_draft_gpu")
            self.assertEqual(verify_event["resource"], "server_target_gpu")
            self.assertGreaterEqual(
                verify_event["start_time_ms"],
                draft_event["finish_time_ms"],
            )
        for previous_verify, next_draft in zip(service_events[1::2], service_events[2::2]):
            self.assertGreaterEqual(
                next_draft["start_time_ms"],
                previous_verify["finish_time_ms"],
            )

    def test_server_only_linear_output_equals_target_only(self) -> None:
        config, _, workload = small_config(num_requests=2, output_len=7)

        for model_runner in (accepting_model_runner(), rejecting_model_runner()):
            target = Simulator(
                config,
                model_runner,
                workload,
                "combined_strong_heterogeneous",
                "target_only",
            ).run()
            server_only = Simulator(
                config,
                model_runner,
                workload,
                "combined_strong_heterogeneous",
                "server_only_linear",
            ).run()

            self.assertEqual(
                [request.generated_ids for request in server_only.requests],
                [request.generated_ids for request in target.requests],
            )

    def test_server_only_rejects_unsupported_batch_size(self) -> None:
        config, model_runner, workload = small_config(num_requests=2, output_len=6)
        config["server_only"]["batch_size"] = 2

        with self.assertRaisesRegex(ValueError, "server_only.batch_size > 1"):
            validate_config(config)
        with self.assertRaisesRegex(ValueError, "server_only.batch_size > 1"):
            Simulator(
                config,
                model_runner,
                workload,
                "combined_strong_heterogeneous",
                "server_only_linear",
            )


if __name__ == "__main__":
    unittest.main()
