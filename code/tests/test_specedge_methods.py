from __future__ import annotations

import unittest

from src.entities import Segment
from src.latency import expected_emitted_tokens, verify_latency_ms
from src.methods import get_method_spec
from src.simulator import Simulator
from tests.common import accepting_model_runner, small_config


class SpecEdgeMethodTest(unittest.TestCase):
    def test_specedge_name_maps_to_specedge_runtime(self) -> None:
        config, _, _ = small_config(num_requests=1, output_len=4)
        spec = get_method_spec("SpecEdge", config)
        self.assertEqual(spec.runtime, "specedge")
        self.assertTrue(spec.global_batch)

    def test_specedge_linear_budget_does_not_expand_as_token_tree(self) -> None:
        config, model_runner, workload = small_config(num_requests=2, output_len=8)
        config["sync_batch"]["B_global"] = 2
        config["specedge"]["server_batch_size"] = 2
        result = Simulator(config, model_runner, workload, "balanced_drafter", "SpecEdge").run()
        verified = [segment for segment in result.segments if segment.accepted_count is not None]
        self.assertTrue(verified)
        self.assertTrue(all(segment.tree_budget_nodes == segment.gamma for segment in verified))

    def test_specedge_verify_compute_uses_budget_field(self) -> None:
        config, model_runner, workload = small_config(num_requests=1, output_len=8)
        simulator = Simulator(config, model_runner, workload, "balanced_drafter", "SpecEdge")
        segment = Segment(
            segment_id=0,
            request_id=0,
            device_id=0,
            draft_model="small",
            prefix_version=0,
            base_pos=0,
            scheduled_gamma=4,
            prefix_ids=[1],
            draft_ids=[2, 3, 4, 5],
            create_time_ms=0.0,
            draft_start_time_ms=0.0,
            tree_budget_nodes=2,
        )

        self.assertEqual(
            simulator._verify_latency_for_segments([segment]),
            verify_latency_ms(config["edge"], [2]),
        )
        self.assertEqual(simulator._segment_payload_tokens(segment), 2)

    def test_specedge_dynamic_batch_collects_same_time_validate_requests(self) -> None:
        config, model_runner, workload = small_config(num_requests=2, output_len=4)
        config["speculation"]["gamma_candidates"] = [1]
        config["specedge"]["server_batch_type"] = "dynamic"
        config["specedge"]["server_batch_size"] = 2

        result = Simulator(config, model_runner, workload, "balanced_drafter", "SpecEdge").run()
        first_batch = next(
            event for event in result.event_trace if event["event"] == "global_batch_verify"
        )

        self.assertEqual(first_batch["batch_size"], 2)
        self.assertEqual(first_batch["batch_type"], "dynamic")

    def test_specedge_pipeline_gamma_calibrates_draft_depth(self) -> None:
        config, model_runner, workload = small_config(num_requests=1, output_len=8)
        config["speculation"]["gamma_candidates"] = [1, 2, 4]
        config["device_pools"]["heterogeneous"]["templates"]["low_end"]["draft_token_rate_tok_s"] = 100
        config["device_pools"]["heterogeneous"]["templates"]["low_end"]["uplink_mbps"] = 1000
        config["device_pools"]["heterogeneous"]["templates"]["low_end"]["downlink_mbps"] = 1000
        config["device_pools"]["heterogeneous"]["templates"]["low_end"]["rtt_ms"] = 20
        config["device_pools"]["heterogeneous"]["templates"]["low_end"]["jitter_ms"] = 0
        simulator = Simulator(config, model_runner, workload, "balanced_drafter", "SpecEdge")
        simulator._schedule_request_arrivals()
        device = simulator.devices[0]
        alpha = simulator.acceptance.estimate(0, device.acceptance_prior)
        simulator._last_specedge_verify_ms = simulator._specedge_edge_cycle_ms(
            device,
            2,
            expected_emitted_tokens(alpha, 2),
        )

        gamma = simulator._select_gamma(simulator.requests[0], device, 0.0, 8)

        self.assertEqual(gamma, 2)

    def test_specedge_proactive_hit_reuses_linear_continuation(self) -> None:
        config, _, workload = small_config(num_requests=1, output_len=12)
        config["speculation"]["gamma_candidates"] = [1]
        config["specedge"]["server_batch_size"] = 1
        result = Simulator(config, accepting_model_runner(), workload, "balanced_drafter", "SpecEdge").run()
        self.assertTrue(any(segment.proactive_used for segment in result.segments))
        self.assertTrue(any(segment.proactive_hit for segment in result.segments))
        self.assertTrue(any(segment.proactive_used for segment in result.segments if segment.draft_compute_ms == 0.0))
        self.assertTrue(any(event["event"] == "pipeline_schedule" for event in result.event_trace))

    def test_specedge_proactive_draft_occupies_edge_device(self) -> None:
        config, _, workload = small_config(num_requests=2, output_len=8)
        config["simulation"]["num_devices"] = 1
        config["device_pools"]["heterogeneous"]["templates"]["low_end"]["count"] = 1
        config["device_pools"]["medium_only"]["templates"]["medium"]["count"] = 1
        config["speculation"]["gamma_candidates"] = [1]
        config["specedge"]["server_batch_size"] = 1
        result = Simulator(config, accepting_model_runner(), workload, "balanced_drafter", "SpecEdge").run()
        first_proactive = next(
            event
            for event in result.event_trace
            if event["event"] == "proactive_draft" and event["request_id"] == 0
        )
        second_request_draft = next(
            event
            for event in result.event_trace
            if event["event"] == "draft_compute" and event["request_id"] == 1
        )

        self.assertGreaterEqual(
            second_request_draft["start_time_ms"],
            first_proactive["finish_time_ms"],
        )

    def test_server_only_has_request_network_and_matches_target_only(self) -> None:
        config, model_runner, workload = small_config(num_requests=2, output_len=12)
        config["specedge"]["server_batch_size"] = 1
        spec = get_method_spec("server_only", config)
        self.assertFalse(spec.global_batch)
        target = Simulator(config, model_runner, workload, "balanced_drafter", "target_only").run()
        server_only = Simulator(config, model_runner, workload, "balanced_drafter", "server_only").run()
        self.assertEqual(
            [request.generated_ids for request in server_only.requests],
            [request.generated_ids for request in target.requests],
        )
        for server_request, target_request in zip(server_only.requests, target.requests):
            self.assertGreater(server_request.target_only_uplink_ms, 0.0)
            self.assertGreater(server_request.target_only_downlink_ms, 0.0)
            self.assertEqual(
                server_request.target_only_uplink_payload_bytes,
                target_request.target_only_uplink_payload_bytes,
            )
            self.assertEqual(
                server_request.target_only_downlink_payload_bytes,
                target_request.target_only_downlink_payload_bytes,
            )
            self.assertAlmostEqual(server_request.ttft_ms, server_request.latency_ms)
        self.assertTrue(all(segment.uplink_delay_ms == 0.0 for segment in server_only.segments))
        self.assertTrue(all(segment.downlink_delay_ms == 0.0 for segment in server_only.segments))
        self.assertTrue(any(event["event"] == "server_only_draft" for event in server_only.event_trace))
        self.assertTrue(any(event["event"] == "server_only_verify" for event in server_only.event_trace))
        self.assertFalse(any(event["event"] == "global_batch_verify" for event in server_only.event_trace))
        self.assertEqual(server_only.batch_waiting_time_ms, 0.0)

    def test_server_only_processes_one_request_lifecycle_at_a_time(self) -> None:
        config, _, workload = small_config(num_requests=2, output_len=6)
        config["speculation"]["gamma_candidates"] = [1]
        config["server_only"]["draft_startup_ms"] = 3
        config["server_only"]["draft_token_rate_tok_s"] = 1000
        result = Simulator(config, accepting_model_runner(), workload, "balanced_drafter", "server_only").run()

        service_events = [
            event
            for event in result.event_trace
            if event["event"] in {"server_only_draft", "server_only_verify"}
        ]
        draft_events = [
            event for event in result.event_trace if event["event"] == "server_only_draft"
        ]
        self.assertTrue(all(event["draft_model"] == "server_only:medium" for event in draft_events))
        self.assertEqual([event["scheduled_gamma"] for event in draft_events], [4, 1, 4, 1])
        self.assertEqual([event["compute_ms"] for event in draft_events], [7.0, 4.0, 7.0, 4.0])

        service_events.sort(key=lambda event: event["start_time_ms"])
        for previous, current in zip(service_events, service_events[1:]):
            self.assertGreaterEqual(current["start_time_ms"], previous["finish_time_ms"])

        first_generation_done = max(
            event["finish_time_ms"]
            for event in service_events
            if event["request_id"] == 0
        )
        first_finish = next(
            event
            for event in result.event_trace
            if event["event"] == "request_finish" and event["request_id"] == 0
        )
        second_first_draft = next(
            event
            for event in result.event_trace
            if event["event"] == "server_only_draft" and event["request_id"] == 1
        )
        self.assertGreaterEqual(
            second_first_draft["start_time_ms"],
            first_generation_done,
        )
        self.assertLess(
            second_first_draft["start_time_ms"],
            first_finish["finish_time_ms"],
        )

    def test_server_only_uses_fixed_linear_depth(self) -> None:
        config, _, workload = small_config(num_requests=1, output_len=10)
        config["speculation"]["gamma_candidates"] = [1]
        config["specedge"]["max_beam_len"] = 4

        result = Simulator(config, accepting_model_runner(), workload, "balanced_drafter", "server_only").run()
        draft_events = [
            event for event in result.event_trace if event["event"] == "server_only_draft"
        ]

        self.assertEqual([event["scheduled_gamma"] for event in draft_events], [4, 4])


if __name__ == "__main__":
    unittest.main()
