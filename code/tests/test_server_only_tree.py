from __future__ import annotations

import unittest

from src.config import load_config
from src.model_runner import DraftCandidateTree, DraftTreeNode, FakeModelRunner
from src.tree_drafting import build_tree_draft_strategy


class ServerOnlyTreeCoreTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
