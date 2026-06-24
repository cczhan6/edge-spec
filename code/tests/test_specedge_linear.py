from __future__ import annotations

import unittest

from src.methods import get_method_spec
from src.simulator import Simulator
from tests.common import accepting_model_runner, rejecting_model_runner, small_config


class SpecEdgeLinearTest(unittest.TestCase):
    def test_specedge_linear_method_is_registered(self) -> None:
        config, _, _ = small_config(num_requests=1, output_len=4)

        spec = get_method_spec("specedge_linear", config)

        self.assertEqual(spec.runtime, "specedge")
        self.assertEqual(spec.candidate_strategy, "linear")
        self.assertTrue(spec.global_batch)
        self.assertTrue(spec.batch_timeout)

    def test_specedge_linear_uses_edge_draft_and_network_without_prompt_upload(self) -> None:
        config, model_runner, workload = small_config(num_requests=2, output_len=6)
        config["specedge"]["server_batch_size"] = 2

        result = Simulator(
            config,
            model_runner,
            workload,
            "combined_strong_heterogeneous",
            "specedge_linear",
        ).run()

        self.assertTrue(result.segments)
        self.assertTrue(all(segment.tree_strategy == "linear" for segment in result.segments))
        self.assertTrue(all(segment.draft_tree is None for segment in result.segments))
        for segment in result.segments:
            self.assertEqual(segment.device_id, result.requests[segment.request_id].device_id)
            self.assertEqual(segment.uplink_payload_tokens, segment.draft_payload_tokens)
            self.assertEqual(segment.uplink_payload_tokens, segment.gamma)
            self.assertGreater(segment.uplink_delay_ms, 0.0)
            self.assertGreaterEqual(segment.downlink_delay_ms, 0.0)
            self.assertLess(segment.uplink_payload_tokens, result.requests[segment.request_id].prompt_token_count + segment.gamma)

    def test_specedge_linear_dynamic_batch_takes_ready_requests(self) -> None:
        config, model_runner, workload = small_config(num_requests=2, output_len=4)
        config["speculation"]["gamma_candidates"] = [1]
        config["specedge"]["server_batch_type"] = "dynamic"
        config["specedge"]["server_batch_size"] = 2

        result = Simulator(
            config,
            model_runner,
            workload,
            "combined_strong_heterogeneous",
            "specedge_linear",
        ).run()
        first_batch = next(
            event for event in result.event_trace if event["event"] == "global_batch_verify"
        )

        self.assertEqual(first_batch["batch_size"], 2)
        self.assertEqual(first_batch["batch_type"], "dynamic")
        self.assertEqual(first_batch["tree_strategy"], "linear")

    def test_specedge_linear_static_batch_waits_for_full_batch(self) -> None:
        config, model_runner, workload = small_config(num_requests=2, output_len=4)
        config["speculation"]["gamma_candidates"] = [1]
        config["specedge"]["server_batch_type"] = "static"
        config["specedge"]["server_batch_size"] = 2
        config["specedge"]["server_batch_timeout_ms"] = None

        result = Simulator(
            config,
            model_runner,
            workload,
            "combined_strong_heterogeneous",
            "specedge_linear",
        ).run()
        first_batch = next(
            event for event in result.event_trace if event["event"] == "global_batch_verify"
        )

        self.assertEqual(first_batch["batch_size"], 2)
        self.assertEqual(first_batch["batch_type"], "static")

    def test_specedge_linear_proactive_runs_while_waiting_and_is_not_early_committed(self) -> None:
        config, _, workload = small_config(num_requests=1, output_len=12)
        config["speculation"]["gamma_candidates"] = [1]
        config["specedge"]["server_batch_size"] = 1
        model_runner = accepting_model_runner()

        result = Simulator(
            config,
            model_runner,
            workload,
            "combined_strong_heterogeneous",
            "specedge_linear",
        ).run()

        proactive_events = [
            event for event in result.event_trace if event["event"] == "proactive_draft"
        ]
        self.assertTrue(proactive_events)
        verification_results = [
            event for event in result.event_trace if event["event"] == "verification_result"
        ]
        self.assertLess(
            proactive_events[0]["start_time_ms"],
            verification_results[0]["finish_time_ms"],
        )
        self.assertEqual(
            result.requests[0].generated_ids,
            model_runner.target_only(result.requests[0].prompt_ids, result.requests[0].output_len),
        )

    def test_specedge_linear_alignment_failure_discards_proactive_state(self) -> None:
        config, _, workload = small_config(num_requests=1, output_len=8)
        config["speculation"]["gamma_candidates"] = [1]
        config["specedge"]["server_batch_size"] = 1

        result = Simulator(
            config,
            rejecting_model_runner(),
            workload,
            "combined_strong_heterogeneous",
            "specedge_linear",
        ).run()

        self.assertTrue(any(segment.proactive_wasted_tokens for segment in result.segments))
        self.assertEqual(result.requests[0].proactive_draft_ids, [])

    def test_specedge_linear_output_equals_target_only(self) -> None:
        config, _, workload = small_config(num_requests=2, output_len=7)
        config["specedge"]["server_batch_size"] = 2

        for model_runner in (accepting_model_runner(), rejecting_model_runner()):
            target = Simulator(
                config,
                model_runner,
                workload,
                "combined_strong_heterogeneous",
                "target_only",
            ).run()
            specedge = Simulator(
                config,
                model_runner,
                workload,
                "combined_strong_heterogeneous",
                "specedge_linear",
            ).run()

            self.assertEqual(
                [request.generated_ids for request in specedge.requests],
                [request.generated_ids for request in target.requests],
            )


if __name__ == "__main__":
    unittest.main()
