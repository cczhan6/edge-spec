from __future__ import annotations

import unittest

from src.methods import SUPPORTED_METHODS
from src.simulator import Simulator
from tests.common import small_config


class InitialPromptPrefillTest(unittest.TestCase):
    def test_default_decode_only_path_excludes_prompt_upload_and_prefill(self) -> None:
        for method in SUPPORTED_METHODS:
            with self.subTest(method=method):
                config, model_runner, workload = small_config(
                    num_requests=1,
                    output_len=4,
                )
                self.assertFalse(config["simulation"]["include_prefill"])
                result = Simulator(
                    config,
                    model_runner,
                    workload,
                    "combined_strong_heterogeneous",
                    method,
                ).run()
                request = result.requests[0]

                if method == "target_only":
                    service = next(
                        event
                        for event in result.event_trace
                        if event["event"] == "target_only_service"
                    )
                    self.assertEqual(request.decode_start_time_ms, request.start_time_ms)
                    self.assertEqual(request.target_only_uplink_payload_bytes, 0)
                    self.assertEqual(request.target_only_uplink_ms, 0.0)
                    self.assertEqual(service["target_prefill_ms"], 0.0)
                    continue

                draft_event_name = (
                    "server_only_draft" if method == "server_only" else "draft_compute"
                )
                first_draft = next(
                    event
                    for event in result.event_trace
                    if event["event"] == draft_event_name
                )
                self.assertEqual(first_draft["draft_prefill_ms"], 0.0)

                verify_event = next(
                    event
                    for event in result.event_trace
                    if event["event"]
                    in {"lane_verify", "global_batch_verify", "server_only_verify"}
                )
                self.assertEqual(verify_event["target_prefill_ms"], 0.0)

                if method == "server_only":
                    self.assertEqual(request.target_only_uplink_payload_bytes, 0)
                    self.assertEqual(request.target_only_uplink_ms, 0.0)
                else:
                    segment = result.segments[0]
                    payload_tokens = (
                        segment.retained_tree_nodes or segment.tree_budget_nodes
                        if method == "SpecEdge"
                        else segment.gamma
                    )
                    expected_payload_bytes = (
                        config["network"]["packet_header_bytes"]
                        + payload_tokens * config["network"]["packet_token_bytes"]
                    )
                    self.assertEqual(segment.uplink_payload_bytes, expected_payload_bytes)

    def test_prefill_can_be_enabled_for_legacy_experiments(self) -> None:
        for method in SUPPORTED_METHODS:
            with self.subTest(method=method):
                config, model_runner, workload = small_config(
                    num_requests=1,
                    output_len=4,
                )
                config["simulation"]["include_prefill"] = True
                result = Simulator(
                    config,
                    model_runner,
                    workload,
                    "combined_strong_heterogeneous",
                    method,
                ).run()
                request = result.requests[0]

                if method == "target_only":
                    service = next(
                        event
                        for event in result.event_trace
                        if event["event"] == "target_only_service"
                    )
                    self.assertIsNone(request.decode_start_time_ms)
                    self.assertGreater(request.target_only_uplink_payload_bytes, 0)
                    self.assertGreater(service["target_prefill_ms"], 0.0)
                    continue

                draft_event_name = (
                    "server_only_draft" if method == "server_only" else "draft_compute"
                )
                first_draft = next(
                    event
                    for event in result.event_trace
                    if event["event"] == draft_event_name
                )
                self.assertGreater(first_draft["draft_prefill_ms"], 0.0)

                verify_event = next(
                    event
                    for event in result.event_trace
                    if event["event"]
                    in {"lane_verify", "global_batch_verify", "server_only_verify"}
                )
                self.assertGreater(verify_event["target_prefill_ms"], 0.0)

                if method == "server_only":
                    self.assertGreater(request.target_only_uplink_payload_bytes, 0)
                else:
                    segment = result.segments[0]
                    no_prompt_payload_tokens = (
                        segment.retained_tree_nodes or segment.tree_budget_nodes
                        if method == "SpecEdge"
                        else segment.gamma
                    )
                    no_prompt_payload_bytes = (
                        config["network"]["packet_header_bytes"]
                        + no_prompt_payload_tokens * config["network"]["packet_token_bytes"]
                    )
                    self.assertGreater(segment.uplink_payload_bytes, no_prompt_payload_bytes)


if __name__ == "__main__":
    unittest.main()
