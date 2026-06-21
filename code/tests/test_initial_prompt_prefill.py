from __future__ import annotations

import unittest

from src.methods import SUPPORTED_METHODS
from src.simulator import Simulator
from tests.common import small_config


class InitialPromptPrefillTest(unittest.TestCase):
    def test_all_methods_count_prompt_upload_and_prefill(self) -> None:
        for method in SUPPORTED_METHODS:
            with self.subTest(method=method):
                config, model_runner, workload = small_config(
                    num_requests=1,
                    output_len=4,
                )
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
                    self.assertGreater(result.segments[0].uplink_payload_bytes, 0)
                    self.assertGreater(
                        result.segments[0].uplink_payload_bytes,
                        config["network"]["packet_header_bytes"]
                        + result.segments[0].gamma * config["network"]["packet_token_bytes"],
                    )


if __name__ == "__main__":
    unittest.main()
