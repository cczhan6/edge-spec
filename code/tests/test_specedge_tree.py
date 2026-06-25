from __future__ import annotations

import unittest

from src.config import load_config
from src.methods import get_method_spec
from src.model_runner import FakeModelRunner
from src.simulator import Simulator
from src.tree_drafting import build_tree_draft_strategy
from tests.common import accepting_model_runner, rejecting_model_runner, small_config


class SpecEdgeTreeCoreTest(unittest.TestCase):
    def test_specedge_tree_method_is_registered(self) -> None:
        config = load_config("configs/default.yaml")

        spec = get_method_spec("specedge_tree", config)

        self.assertEqual(spec.runtime, "specedge")
        self.assertEqual(spec.candidate_strategy, "tree")
        self.assertTrue(spec.global_batch)

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

    def test_specedge_tree_forces_tree_strategy(self) -> None:
        config, _, workload = small_config(num_requests=1, output_len=8)
        config["specedge"]["tree_draft_strategy"] = "linear"
        config["specedge"]["proactive_tree_draft_strategy"] = "linear"

        result = Simulator(
            config,
            accepting_model_runner(),
            workload,
            "combined_strong_heterogeneous",
            "specedge_tree",
        ).run()

        verified = [segment for segment in result.segments if segment.accepted_count is not None]
        self.assertTrue(verified)
        self.assertTrue(all(segment.tree_strategy == "specexec_approx" for segment in verified))
        self.assertTrue(all(segment.draft_tree is not None for segment in verified))

    def test_specedge_tree_proactive_runs_and_reuses_on_alignment(self) -> None:
        config, _, workload = small_config(num_requests=1, output_len=12)
        config["speculation"]["gamma_candidates"] = [1]
        config["specedge"]["server_batch_size"] = 1

        result = Simulator(
            config,
            accepting_model_runner(),
            workload,
            "combined_strong_heterogeneous",
            "specedge_tree",
        ).run()

        self.assertTrue(any(event["event"] == "proactive_draft" for event in result.event_trace))
        self.assertTrue(any(segment.proactive_hit for segment in result.segments))
        for segment in result.segments:
            if segment.proactive_hit and segment.proactive_draft_tree is not None:
                accepted_prefix = segment.prefix_ids + segment.emitted_ids[: int(segment.accepted_count or 0)]
                self.assertEqual(segment.proactive_draft_tree.prefix_ids, accepted_prefix)

    def test_specedge_tree_alignment_failure_discards_proactive_state(self) -> None:
        config, _, workload = small_config(num_requests=1, output_len=8)
        config["speculation"]["gamma_candidates"] = [1]
        config["specedge"]["server_batch_size"] = 1

        result = Simulator(
            config,
            rejecting_model_runner(),
            workload,
            "combined_strong_heterogeneous",
            "specedge_tree",
        ).run()

        self.assertTrue(any(segment.proactive_wasted_tokens for segment in result.segments))
        self.assertEqual(result.requests[0].proactive_draft_ids, [])

    def test_specedge_tree_proactive_alignment_success(self) -> None:
        config, _, workload = small_config(num_requests=1, output_len=12)
        config["speculation"]["gamma_candidates"] = [1]
        config["specedge"]["server_batch_size"] = 1

        result = Simulator(
            config,
            accepting_model_runner(),
            workload,
            "combined_strong_heterogeneous",
            "specedge_tree",
        ).run()

        source = next(segment for segment in result.segments if segment.proactive_hit)
        retained_suffix = source.proactive_draft_ids[1:]
        source_verify = next(
            event
            for event in result.event_trace
            if event["event"] == "verification_result"
            and event["segment_id"] == source.segment_id
        )
        reused_event = next(
            event
            for event in result.event_trace
            if event["event"] == "draft_compute"
            and event["proactive_reused_tokens"] == len(retained_suffix)
        )
        reused = result.segments[reused_event["segment_id"]]
        accepted_prefix = source.prefix_ids + source.emitted_ids[: int(source.accepted_count or 0)]

        self.assertEqual(source.tree_strategy, "specexec_approx")
        self.assertIsNotNone(source.proactive_draft_tree)
        self.assertEqual(source.proactive_draft_tree.prefix_ids, accepted_prefix)
        self.assertTrue(retained_suffix)
        self.assertEqual(reused.draft_ids[: len(retained_suffix)], retained_suffix)
        self.assertEqual(reused_event["proactive_reused_tokens"], len(retained_suffix))
        self.assertGreaterEqual(reused_event["start_time_ms"], source_verify["finish_time_ms"])
        self.assertFalse(any(token in source.emitted_ids for token in retained_suffix))

    def test_specedge_tree_proactive_alignment_failure(self) -> None:
        config, _, workload = small_config(num_requests=1, output_len=8)
        config["speculation"]["gamma_candidates"] = [1]
        config["specedge"]["server_batch_size"] = 1

        result = Simulator(
            config,
            rejecting_model_runner(),
            workload,
            "combined_strong_heterogeneous",
            "specedge_tree",
        ).run()

        wasted_segments = [segment for segment in result.segments if segment.proactive_wasted_tokens]
        self.assertTrue(wasted_segments)
        self.assertFalse(any(segment.proactive_hit for segment in wasted_segments))
        self.assertFalse(
            any(
                event["event"] == "draft_compute"
                and event["proactive_reused_tokens"] > 0
                for event in result.event_trace
            )
        )
        self.assertTrue(any(segment.tree_path_switched or segment.status == "rejected" for segment in wasted_segments))
        self.assertEqual(result.requests[0].proactive_draft_ids, [])
        self.assertNotIn(2, result.requests[0].generated_ids)

    def test_specedge_tree_output_equals_target_only(self) -> None:
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
                "specedge_tree",
            ).run()

            self.assertEqual(
                [request.generated_ids for request in specedge.requests],
                [request.generated_ids for request in target.requests],
            )


if __name__ == "__main__":
    unittest.main()
