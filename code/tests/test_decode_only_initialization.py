from __future__ import annotations

import unittest

from src.methods import SUPPORTED_METHODS
from src.simulator import Simulator
from tests.common import small_config


class DecodeOnlyInitializationTest(unittest.TestCase):
    def test_requests_are_decode_ready_on_arrival(self) -> None:
        for method in SUPPORTED_METHODS:
            with self.subTest(method=method):
                config, model_runner, workload = small_config(
                    num_requests=2,
                    output_len=4,
                )
                result = Simulator(
                    config,
                    model_runner,
                    workload,
                    "combined_strong_heterogeneous",
                    method,
                ).run()

                for request in result.requests:
                    self.assertEqual(
                        request.decode_ready_time_ms,
                        request.arrival_time_ms,
                    )

                for event in result.event_trace:
                    self.assertNotIn("draft_prefill_ms", event)
                    self.assertNotIn("target_prefill_ms", event)

    def test_first_segment_upload_excludes_prompt(self) -> None:
        for method in SUPPORTED_METHODS:
            if method in {"target_only", "server_only", "server_only_linear", "server_only_tree"}:
                continue
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
                segment = result.segments[0]
                expected_bytes = (
                    int(config["network"]["packet_header_bytes"])
                    + segment.draft_payload_tokens
                    * int(config["network"]["packet_token_bytes"])
                )

                self.assertEqual(segment.uplink_payload_tokens, segment.draft_payload_tokens)
                self.assertEqual(segment.uplink_payload_bytes, expected_bytes)


if __name__ == "__main__":
    unittest.main()
