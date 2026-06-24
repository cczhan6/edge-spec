from __future__ import annotations

import unittest

from src.config import load_config
from src.methods import get_method_spec
from src.model_runner import DraftCandidateTree, DraftTreeNode, FakeModelRunner
from src.simulator import Simulator
from src.tree_drafting import build_tree_draft_strategy
from tests.common import accepting_model_runner, small_config


class ServerOnlyTreeCoreTest(unittest.TestCase):
    def test_server_only_tree_method_is_registered(self) -> None:
        config = load_config("configs/default.yaml")

        spec = get_method_spec("server_only_tree", config)

        self.assertEqual(spec.runtime, "server_only_specedge")
        self.assertEqual(spec.candidate_strategy, "tree")

    def test_server_only_tree_strategy_uses_configured_budget(self) -> None:
        config = load_config("configs/default.yaml")
        config["server_only"]["max_beam_len"] = 4
        config["server_only"]["max_budget"] = 16

        strategy = build_tree_draft_strategy(config, "server_only")
        plan = strategy.plan(4)

        self.assertEqual(plan.strategy, "specexec_approx")
        self.assertEqual(plan.max_beam_len, 4)
        self.assertEqual(plan.max_budget, 16)
        self.assertGreater(plan.tree_budget_nodes, plan.path_token_count)
        self.assertGreater(plan.target_verify_nodes, 1)

    def test_tree_verification_can_commit_non_primary_target_path(self) -> None:
        model_runner = FakeModelRunner(target_token_fn=lambda prefix: 1)
        tree = DraftCandidateTree(
            prefix_ids=[9],
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

        result = model_runner.verify_tree([9], tree)

        self.assertEqual(result.accepted_count, 2)
        self.assertEqual(result.committed_tokens[:2], [1, 1])
        self.assertIsNone(result.correction_token)
        self.assertEqual(result.bonus_token, 1)

    def test_tree_verification_rejection_uses_correction_token(self) -> None:
        model_runner = FakeModelRunner(target_token_fn=lambda prefix: 7)
        tree = DraftCandidateTree(
            prefix_ids=[9],
            primary_ids=[2],
            primary_node_ids=[1],
            nodes=[DraftTreeNode(1, None, 2, 1)],
            processed_candidate_count=1,
            retained_tree_nodes=1,
            target_verify_tree_nodes=1,
        )

        result = model_runner.verify_tree([9], tree)

        self.assertEqual(result.accepted_count, 0)
        self.assertEqual(result.committed_tokens, [7])
        self.assertEqual(result.correction_token, 7)
        self.assertIsNone(result.bonus_token)

    def test_server_only_tree_forces_tree_strategy_and_interfaces(self) -> None:
        config, _, workload = small_config(num_requests=1, output_len=8)
        config["server_only"]["tree_draft_strategy"] = "linear"

        result = Simulator(
            config,
            accepting_model_runner(),
            workload,
            "combined_strong_heterogeneous",
            "server_only_tree",
        ).run()

        self.assertTrue(result.segments)
        self.assertTrue(all(segment.tree_strategy == "specexec_approx" for segment in result.segments))
        self.assertTrue(all(segment.draft_tree is not None for segment in result.segments))
        self.assertTrue(all(segment.target_verify_tree_nodes > 1 for segment in result.segments))

    def test_server_only_tree_has_no_network_or_proactive_events(self) -> None:
        config, model_runner, workload = small_config(num_requests=2, output_len=6)

        result = Simulator(
            config,
            model_runner,
            workload,
            "combined_strong_heterogeneous",
            "server_only_tree",
        ).run()

        self.assertTrue(all(request.target_only_downlink_ms == 0.0 for request in result.requests))
        self.assertTrue(all(segment.uplink_delay_ms == 0.0 for segment in result.segments))
        self.assertTrue(all(segment.downlink_delay_ms == 0.0 for segment in result.segments))
        self.assertFalse(any(event["event"] == "proactive_draft" for event in result.event_trace))
        for event in result.event_trace:
            self.assertNotIn("uplink_ms", event)
            self.assertNotIn("downlink_ms", event)
        draft_events = [event for event in result.event_trace if event["event"] == "server_only_draft"]
        verify_events = [event for event in result.event_trace if event["event"] == "server_only_verify"]
        self.assertTrue(all(event["resource"] == "server_draft_gpu" for event in draft_events))
        self.assertTrue(all(event["resource"] == "server_target_gpu" for event in verify_events))

    def test_server_only_tree_output_equals_target_only(self) -> None:
        config, _, workload = small_config(num_requests=2, output_len=7)
        model_runner = accepting_model_runner()

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
            "server_only_tree",
        ).run()

        self.assertEqual(
            [request.generated_ids for request in server_only.requests],
            [request.generated_ids for request in target.requests],
        )


if __name__ == "__main__":
    unittest.main()
