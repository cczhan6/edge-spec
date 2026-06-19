from __future__ import annotations

import unittest

from src.model_runner import (
    DraftCandidateTree,
    DraftTreeNode,
    FakeModelRunner,
    build_tree_attention_mask,
)
from src.tree_drafting import LinearDraftTreeStrategy, SpecExecDraftTreeStrategy


class TreeDraftingTest(unittest.TestCase):
    def test_specexec_plan_matches_budgeted_tree_growth(self) -> None:
        strategy = SpecExecDraftTreeStrategy(
            max_n_beams=32,
            max_beam_len=4,
            max_branch_width=16,
            max_budget=32,
        )

        one_step = strategy.plan(1)
        full_depth = strategy.plan(4)

        self.assertEqual(one_step.tree_budget_nodes, 16)
        self.assertEqual(one_step.draft_compute_nodes, 1)
        self.assertEqual(full_depth.tree_budget_nodes, 32)
        self.assertEqual(full_depth.draft_compute_nodes, 81)
        self.assertEqual(full_depth.strategy, "specexec_approx")

    def test_linear_plan_preserves_path_depth(self) -> None:
        strategy = LinearDraftTreeStrategy(max_beam_len=4, max_budget=32)
        plan = strategy.plan(4)

        self.assertEqual(plan.tree_budget_nodes, 4)
        self.assertEqual(plan.draft_compute_nodes, 4)
        self.assertEqual(plan.strategy, "linear")

    def test_model_runner_tree_verify_can_follow_non_primary_branch(self) -> None:
        def target_token(prefix):
            return (sum(prefix[-4:]) + len(prefix) + 1) % 97

        runner = FakeModelRunner(
            target_token_fn=target_token,
            draft_token_fn=lambda profile, prefix: (target_token(prefix) + 5) % 97,
        )
        strategy = SpecExecDraftTreeStrategy(
            max_n_beams=4,
            max_beam_len=2,
            max_branch_width=2,
            max_budget=6,
        )

        tree = runner.draft_tree("small", [3, 4], strategy.plan(2))
        result = runner.verify_tree([3, 4], tree)

        self.assertNotEqual(tree.primary_ids[0], result.emitted_ids[0])
        self.assertEqual(result.accepted_count, 2)
        self.assertFalse(result.rejected)

    def test_specexec_tree_builder_tracks_pruned_node_counts(self) -> None:
        runner = FakeModelRunner()
        strategy = SpecExecDraftTreeStrategy(
            max_n_beams=4,
            max_beam_len=3,
            max_branch_width=4,
            max_budget=6,
        )

        tree = runner.draft_tree("medium", [3, 4], strategy.plan(3))

        self.assertLessEqual(tree.retained_tree_nodes, 6)
        self.assertEqual(tree.retained_tree_nodes, tree.node_count)
        self.assertGreater(tree.processed_candidate_count, tree.retained_tree_nodes)
        self.assertEqual(tree.target_verify_tree_nodes, tree.retained_tree_nodes)
        self.assertTrue(all(node.logprob <= 0.0 for node in tree.nodes))

    def test_proactive_tree_starts_from_best_leaf_bonus_candidate(self) -> None:
        runner = FakeModelRunner()
        strategy = SpecExecDraftTreeStrategy(
            max_n_beams=4,
            max_beam_len=3,
            max_branch_width=4,
            max_budget=8,
        )
        tree = runner.draft_tree("medium", [3, 4], strategy.plan(3))

        proactive = runner.draft_bonus_tree("medium", tree, strategy.plan(2))

        self.assertGreater(len(proactive.prefix_ids), len(tree.prefix_ids))
        self.assertTrue(proactive.primary_ids)
        self.assertGreaterEqual(proactive.processed_candidate_count, 1)

    def test_tree_attention_mask_allows_only_prefix_and_ancestor_path(self) -> None:
        tree = DraftCandidateTree(
            prefix_ids=[10, 11],
            primary_ids=[20, 21],
            primary_node_ids=[1, 3],
            nodes=[
                DraftTreeNode(1, None, 20, 1),
                DraftTreeNode(2, None, 30, 1),
                DraftTreeNode(3, 1, 21, 2),
                DraftTreeNode(4, 2, 31, 2),
            ],
        )

        mask = build_tree_attention_mask(tree)

        self.assertEqual(
            mask,
            [
                [True, False, False, False, False, False],
                [True, True, False, False, False, False],
                [True, True, True, False, False, False],
                [True, True, False, True, False, False],
                [True, True, True, False, True, False],
                [True, True, False, True, False, True],
            ],
        )


if __name__ == "__main__":
    unittest.main()
