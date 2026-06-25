from __future__ import annotations

import unittest

from src.entities import Segment
from src.latency import expected_emitted_tokens, verify_latency_ms
from src.methods import get_method_spec
from src.model_runner import DraftCandidateTree, DraftTreeNode, FakeModelRunner
from src.simulator import Simulator
from tests.common import accepting_model_runner, small_config


class _RecordingTreeModelRunner(FakeModelRunner):
    def __init__(self) -> None:
        super().__init__()
        self.draft_calls = 0
        self.verify_batch_calls = 0
        self.draft_tree_calls = 0
        self.verify_tree_batch_calls = 0

    def draft(self, drafter_profile, prefix_ids, gamma):
        self.draft_calls += 1
        return super().draft(drafter_profile, prefix_ids, gamma)

    def draft_tree(self, drafter_profile, prefix_ids, plan):
        self.draft_tree_calls += 1
        return super().draft_tree(drafter_profile, prefix_ids, plan)

    def verify_batch(self, requests):
        self.verify_batch_calls += 1
        return super().verify_batch(requests)

    def verify_tree_batch(self, requests):
        self.verify_tree_batch_calls += 1
        return super().verify_tree_batch(requests)


class _NonPrimaryTreePathRunner(FakeModelRunner):
    def __init__(self) -> None:
        super().__init__(target_token_fn=lambda prefix: 1)

    def draft_tree(self, drafter_profile, prefix_ids, plan):
        return DraftCandidateTree(
            list(prefix_ids),
            primary_ids=[2],
            primary_node_ids=[1],
            nodes=[
                DraftTreeNode(1, None, 2, 1),
                DraftTreeNode(2, None, 1, 1),
                DraftTreeNode(3, 2, 1, 2),
            ],
            processed_candidate_count=3,
            retained_tree_nodes=3,
            target_verify_tree_nodes=3,
        )


class SpecEdgeMethodTest(unittest.TestCase):
    def test_specedge_name_maps_to_specedge_runtime(self) -> None:
        config, _, _ = small_config(num_requests=1, output_len=4)
        with self.assertWarnsRegex(FutureWarning, "specedge_tree"):
            spec = get_method_spec("SpecEdge", config)
        canonical = get_method_spec("specedge_tree", config)
        self.assertEqual(spec, canonical)
        self.assertEqual(spec.runtime, "specedge")
        self.assertEqual(spec.candidate_strategy, "tree")
        self.assertTrue(spec.global_batch)

    def test_target_only_does_not_require_specedge_section(self) -> None:
        config, model_runner, workload = small_config(num_requests=1, output_len=4)
        del config["specedge"]

        result = Simulator(config, model_runner, workload, "combined_strong_heterogeneous", "target_only").run()

        self.assertEqual(len(result.requests[0].generated_ids), 4)

    def test_specedge_uses_specexec_tree_budget_by_default(self) -> None:
        config, model_runner, workload = small_config(num_requests=2, output_len=8)
        config["sync_batch"]["B_global"] = 2
        config["specedge"]["server_batch_size"] = 2
        result = Simulator(config, model_runner, workload, "combined_strong_heterogeneous", "SpecEdge").run()
        verified = [segment for segment in result.segments if segment.accepted_count is not None]
        self.assertTrue(verified)
        self.assertTrue(all(segment.tree_strategy == "specexec_approx" for segment in verified))
        self.assertTrue(any(segment.tree_budget_nodes > segment.gamma for segment in verified))
        self.assertTrue(all(segment.processed_candidate_count == segment.draft_compute_nodes for segment in verified))
        self.assertTrue(all(segment.retained_tree_nodes == segment.tree_budget_nodes for segment in verified))
        self.assertTrue(all(segment.target_verify_tree_nodes == segment.retained_tree_nodes for segment in verified))

    def test_specedge_uses_model_runner_tree_interfaces(self) -> None:
        config, _, workload = small_config(num_requests=1, output_len=5)
        config["specedge"]["server_batch_size"] = 1
        model_runner = _RecordingTreeModelRunner()

        result = Simulator(config, model_runner, workload, "combined_strong_heterogeneous", "SpecEdge").run()

        self.assertGreater(model_runner.draft_tree_calls, 0)
        self.assertGreater(model_runner.verify_tree_batch_calls, 0)
        self.assertTrue(any(segment.draft_tree is not None for segment in result.segments))

    def test_specedge_alias_ignores_linear_config_and_uses_tree_canonical(self) -> None:
        config, _, workload = small_config(num_requests=2, output_len=8)
        config["sync_batch"]["B_global"] = 2
        config["specedge"]["server_batch_size"] = 2
        config["specedge"]["tree_draft_strategy"] = "linear"
        config["specedge"]["proactive_tree_draft_strategy"] = "linear"
        model_runner = _RecordingTreeModelRunner()

        with self.assertWarnsRegex(FutureWarning, "specedge_tree"):
            result = Simulator(config, model_runner, workload, "combined_strong_heterogeneous", "SpecEdge").run()
        verified = [segment for segment in result.segments if segment.accepted_count is not None]

        self.assertTrue(verified)
        self.assertEqual(result.method, "specedge_tree")
        self.assertTrue(all(segment.tree_strategy == "specexec_approx" for segment in verified))
        self.assertTrue(all(segment.draft_tree is not None for segment in verified))
        self.assertEqual(model_runner.draft_calls, 0)
        self.assertEqual(model_runner.verify_batch_calls, 0)
        self.assertGreater(model_runner.draft_tree_calls, 0)
        self.assertGreater(model_runner.verify_tree_batch_calls, 0)
        verify_events = [
            event for event in result.event_trace if event["event"] == "global_batch_verify"
        ]
        self.assertTrue(all(event["target_verify_tree_nodes"] > 1 for event in verify_events))

    def test_specedge_verify_compute_is_segment_forward_level(self) -> None:
        config, model_runner, workload = small_config(num_requests=1, output_len=8)
        simulator = Simulator(config, model_runner, workload, "combined_strong_heterogeneous", "SpecEdge")
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
            verify_latency_ms(config["edge"], [segment.target_verify_tree_nodes]),
        )
        self.assertEqual(simulator._segment_payload_tokens(segment), 2)

    def test_specedge_alias_dynamic_batch_uses_canonical_tree(self) -> None:
        config, model_runner, workload = small_config(num_requests=2, output_len=4)
        config["speculation"]["gamma_candidates"] = [1]
        config["specedge"]["server_batch_type"] = "dynamic"
        config["specedge"]["server_batch_size"] = 2
        config["specedge"]["tree_draft_strategy"] = "linear"
        config["specedge"]["proactive_tree_draft_strategy"] = "linear"

        with self.assertWarnsRegex(FutureWarning, "specedge_tree"):
            result = Simulator(config, model_runner, workload, "combined_strong_heterogeneous", "SpecEdge").run()
        first_batch = next(
            event for event in result.event_trace if event["event"] == "global_batch_verify"
        )

        self.assertEqual(result.method, "specedge_tree")
        self.assertEqual(first_batch["batch_type"], "dynamic")
        self.assertEqual(first_batch["tree_strategy"], "specexec_approx")

    def test_specedge_uses_configured_tree_depth_not_gamma_candidates(self) -> None:
        config, model_runner, workload = small_config(num_requests=1, output_len=8)
        config["speculation"]["gamma_candidates"] = [1]
        config["specedge"]["max_beam_len"] = 4
        with self.assertWarnsRegex(FutureWarning, "specedge_tree"):
            simulator = Simulator(config, model_runner, workload, "combined_strong_heterogeneous", "SpecEdge")
        simulator._schedule_request_arrivals()
        device = simulator.devices[0]

        gamma = simulator._select_gamma(simulator.requests[0], device, 0.0, 8)

        self.assertEqual(gamma, 4)
        self.assertFalse(simulator.spec.adaptive_gamma)

    def test_specedge_proactive_hit_reuses_linear_continuation(self) -> None:
        config, _, workload = small_config(num_requests=1, output_len=12)
        config["speculation"]["gamma_candidates"] = [1]
        config["specedge"]["server_batch_size"] = 1
        with self.assertWarnsRegex(FutureWarning, "specedge_tree"):
            result = Simulator(config, accepting_model_runner(), workload, "combined_strong_heterogeneous", "SpecEdge").run()
        self.assertTrue(any(segment.proactive_used for segment in result.segments))
        self.assertTrue(any(segment.proactive_hit for segment in result.segments))
        self.assertTrue(
            any(
                event["event"] == "draft_compute"
                and event["proactive_used"]
                and event["proactive_reused_tokens"] > 0
                for event in result.event_trace
            )
        )
        self.assertTrue(any(event["event"] == "pipeline_schedule" for event in result.event_trace))

    def test_specedge_proactive_draft_occupies_edge_device(self) -> None:
        config, _, workload = small_config(num_requests=2, output_len=8)
        config["simulation"]["num_devices"] = 1
        config["device_pools"]["heterogeneous"]["templates"]["low_end"]["count"] = 1
        config["device_pools"]["medium_only"]["templates"]["medium"]["count"] = 1
        config["speculation"]["gamma_candidates"] = [1]
        config["specedge"]["server_batch_size"] = 1
        with self.assertWarnsRegex(FutureWarning, "specedge_tree"):
            result = Simulator(config, accepting_model_runner(), workload, "combined_strong_heterogeneous", "SpecEdge").run()
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

    def test_server_only_has_no_network_and_matches_target_only(self) -> None:
        config, model_runner, workload = small_config(num_requests=2, output_len=12)
        config["specedge"]["server_batch_size"] = 1
        with self.assertWarnsRegex(FutureWarning, "server_only_tree"):
            spec = get_method_spec("server_only", config)
        self.assertEqual(spec, get_method_spec("server_only_tree", config))
        self.assertFalse(spec.global_batch)
        target = Simulator(config, model_runner, workload, "combined_strong_heterogeneous", "target_only").run()
        with self.assertWarnsRegex(FutureWarning, "server_only_tree"):
            server_only = Simulator(config, model_runner, workload, "combined_strong_heterogeneous", "server_only").run()
        self.assertEqual(
            [request.generated_ids for request in server_only.requests],
            [request.generated_ids for request in target.requests],
        )
        for server_request in server_only.requests:
            self.assertEqual(server_request.target_only_downlink_ms, 0.0)
            self.assertEqual(server_request.target_only_downlink_payload_bytes, 0)
        self.assertTrue(all(segment.uplink_delay_ms == 0.0 for segment in server_only.segments))
        self.assertTrue(all(segment.downlink_delay_ms == 0.0 for segment in server_only.segments))
        self.assertFalse(
            any(event["event"] == "server_only_request_uplink" for event in server_only.event_trace)
        )
        self.assertTrue(any(event["event"] == "server_only_draft" for event in server_only.event_trace))
        self.assertTrue(any(event["event"] == "server_only_verify" for event in server_only.event_trace))
        self.assertFalse(any(event["event"] == "global_batch_verify" for event in server_only.event_trace))
        self.assertEqual(server_only.batch_waiting_time_ms, 0.0)

    def test_server_only_processes_one_request_lifecycle_at_a_time(self) -> None:
        config, _, workload = small_config(num_requests=2, output_len=6)
        config["speculation"]["gamma_candidates"] = [1]
        config["server_only"]["draft_startup_ms"] = 3
        config["server_only"]["draft_token_rate_tok_s"] = 1000
        config["server_only"]["tree_draft_strategy"] = "linear"
        with self.assertWarnsRegex(FutureWarning, "server_only_tree"):
            result = Simulator(config, accepting_model_runner(), workload, "combined_strong_heterogeneous", "server_only").run()

        service_events = [
            event
            for event in result.event_trace
            if event["event"] in {"server_only_draft", "server_only_verify"}
        ]
        draft_events = [
            event for event in result.event_trace if event["event"] == "server_only_draft"
        ]
        self.assertEqual(result.method, "server_only_tree")
        self.assertTrue(all(event["draft_model"] == "server_only:medium" for event in draft_events))
        self.assertTrue(all(event["tree_strategy"] == "specexec_approx" for event in draft_events))
        self.assertTrue(all(segment.draft_tree is not None for segment in result.segments))

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
        self.assertLessEqual(
            second_first_draft["start_time_ms"],
            first_finish["finish_time_ms"],
        )

    def test_server_only_uses_fixed_tree_depth(self) -> None:
        config, _, workload = small_config(num_requests=1, output_len=10)
        config["speculation"]["gamma_candidates"] = [1]
        config["specedge"]["max_beam_len"] = 4

        with self.assertWarnsRegex(FutureWarning, "server_only_tree"):
            result = Simulator(config, accepting_model_runner(), workload, "combined_strong_heterogeneous", "server_only").run()
        draft_events = [
            event for event in result.event_trace if event["event"] == "server_only_draft"
        ]

        self.assertEqual([event["scheduled_gamma"] for event in draft_events], [4, 4])
        self.assertEqual([event["tree_budget_nodes"] for event in draft_events], [64, 64])
        self.assertTrue(all(event["tree_strategy"] == "specexec_approx" for event in draft_events))

    def test_server_only_tree_acceptance_uses_tree_depth_not_primary_gamma(self) -> None:
        config, _, workload = small_config(num_requests=1, output_len=2)

        with self.assertWarnsRegex(FutureWarning, "server_only_tree"):
            result = Simulator(
                config,
                _NonPrimaryTreePathRunner(),
                workload,
                "combined_strong_heterogeneous",
                "server_only",
            ).run()
        verified = [segment for segment in result.segments if segment.accepted_count is not None]

        self.assertEqual(len(verified), 1)
        self.assertEqual(verified[0].gamma, 1)
        self.assertEqual(verified[0].accepted_count, 2)
        self.assertEqual(verified[0].proposed_count, 2)
        self.assertEqual(verified[0].acceptance_rate, 1.0)

    def test_server_only_specexec_approx_uses_server_budget(self) -> None:
        config, _, workload = small_config(num_requests=1, output_len=6)
        config["server_only"]["tree_draft_strategy"] = "specexec_approx"
        config["server_only"]["max_budget"] = 64
        config["server_only"]["draft_startup_ms"] = 3
        config["server_only"]["draft_token_rate_tok_s"] = 1000

        with self.assertWarnsRegex(FutureWarning, "server_only_tree"):
            result = Simulator(config, accepting_model_runner(), workload, "combined_strong_heterogeneous", "server_only").run()
        draft_events = [
            event for event in result.event_trace if event["event"] == "server_only_draft"
        ]

        self.assertEqual([event["scheduled_gamma"] for event in draft_events], [4, 1])
        self.assertEqual(draft_events[0]["tree_budget_nodes"], 64)
        self.assertGreater(draft_events[1]["tree_budget_nodes"], 1)
        self.assertLessEqual(draft_events[1]["tree_budget_nodes"], 64)
        self.assertEqual([event["compute_ms"] for event in draft_events], [83.0, 4.0])
        self.assertEqual(draft_events[0]["processed_candidate_count"], 80)
        self.assertEqual(draft_events[0]["retained_tree_nodes"], 64)
        self.assertEqual(draft_events[0]["target_verify_tree_nodes"], 64)
        self.assertTrue(all(event["tree_strategy"] == "specexec_approx" for event in draft_events))


if __name__ == "__main__":
    unittest.main()
