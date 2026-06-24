from __future__ import annotations

import unittest

from src.config import load_config
from src.model_runner import FakeModelRunner
from src.tree_drafting import build_tree_draft_strategy


class SpecEdgeTreeCoreTest(unittest.TestCase):
    def test_specedge_tree_and_proactive_limits_are_configured(self) -> None:
        config = load_config("configs/default.yaml")

        initial = build_tree_draft_strategy(config, "specedge")
        proactive = build_tree_draft_strategy(config, "specedge", proactive=True)

        self.assertEqual(initial.name, "specexec_approx")
        self.assertEqual(initial.max_n_beams, int(config["specedge"]["max_n_beams"]))
        self.assertEqual(initial.max_beam_len, int(config["specedge"]["max_beam_len"]))
        self.assertEqual(proactive.max_beam_len, int(config["specedge"]["proactive_max_beam_len"]))
        self.assertEqual(proactive.max_budget, int(config["specedge"]["proactive_max_budget"]))

    def test_proactive_bonus_tree_starts_from_bonus_candidate(self) -> None:
        config = load_config("configs/default.yaml")
        initial_plan = build_tree_draft_strategy(config, "specedge").plan(2)
        proactive_plan = build_tree_draft_strategy(config, "specedge", proactive=True).plan(2)
        model_runner = FakeModelRunner(
            target_token_fn=lambda prefix: (prefix[-1] + 1) % 97,
            draft_token_fn=lambda profile, prefix: (prefix[-1] + 1) % 97,
        )
        draft_tree = model_runner.draft_tree("small", [1], initial_plan)

        proactive_tree = model_runner.draft_bonus_tree("small", draft_tree, proactive_plan)

        self.assertTrue(proactive_tree.primary_ids)
        self.assertGreaterEqual(proactive_tree.retained_tree_nodes, len(proactive_tree.primary_ids))
        self.assertEqual(
            proactive_tree.prefix_ids,
            draft_tree.prefix_ids + draft_tree.primary_ids,
        )

    def test_tree_verify_batch_matches_single_tree_verify(self) -> None:
        config = load_config("configs/default.yaml")
        plan = build_tree_draft_strategy(config, "specedge").plan(2)
        model_runner = FakeModelRunner()
        tree = model_runner.draft_tree("small", [1], plan)

        single = model_runner.verify_tree([1], tree)
        batch = model_runner.verify_tree_batch(
            [type("TreeInput", (), {"prefix_ids": [1], "draft_tree": tree})()]
        )[0]

        self.assertEqual(batch, single)


if __name__ == "__main__":
    unittest.main()
