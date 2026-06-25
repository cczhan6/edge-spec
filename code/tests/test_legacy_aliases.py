from __future__ import annotations

import unittest
from pathlib import Path

from src.methods import DEFAULT_METHODS, get_method_spec
from src.simulator import Simulator
from tests.common import accepting_model_runner, small_config


LEGACY_ALIASES = {
    "sync_batch_sd": "dip_sd",
    "SpecEdge": "specedge_tree",
    "server_only": "server_only_tree",
}


class LegacyAliasTest(unittest.TestCase):
    def test_legacy_aliases_enter_canonical_specs_with_warning(self) -> None:
        config, _, _ = small_config(num_requests=1, output_len=4)

        for alias, canonical in LEGACY_ALIASES.items():
            with self.subTest(alias=alias):
                with self.assertWarnsRegex(FutureWarning, canonical):
                    alias_spec = get_method_spec(alias, config)
                self.assertEqual(alias_spec, get_method_spec(canonical, config))

    def test_legacy_aliases_do_not_enter_old_simulator_paths(self) -> None:
        for alias, canonical in LEGACY_ALIASES.items():
            with self.subTest(alias=alias):
                config, _, workload = small_config(num_requests=2, output_len=6)
                config["specedge"]["server_batch_size"] = 2
                config["specedge"]["tree_draft_strategy"] = "linear"
                config["specedge"]["proactive_tree_draft_strategy"] = "linear"
                config["server_only"]["tree_draft_strategy"] = "linear"
                with self.assertWarnsRegex(FutureWarning, canonical):
                    result = Simulator(
                        config,
                        accepting_model_runner(),
                        workload,
                        "combined_strong_heterogeneous",
                        alias,
                    ).run()

                self.assertEqual(result.method, canonical)
                self.assertTrue(all(event["method"] == canonical for event in result.event_trace))
                if alias == "sync_batch_sd":
                    self.assertTrue(any(event["event"] == "dip_sd_epoch_plan" for event in result.event_trace))
                    self.assertFalse(any(event["event"] == "global_batch_verify" for event in result.event_trace))
                elif alias == "SpecEdge":
                    verified = [segment for segment in result.segments if segment.accepted_count is not None]
                    self.assertTrue(verified)
                    self.assertTrue(all(segment.tree_strategy == "specexec_approx" for segment in verified))
                    self.assertTrue(any(event["event"] == "global_batch_verify" for event in result.event_trace))
                elif alias == "server_only":
                    verified = [segment for segment in result.segments if segment.accepted_count is not None]
                    self.assertTrue(verified)
                    self.assertTrue(all(segment.tree_strategy == "specexec_approx" for segment in verified))
                    self.assertFalse(any(event["event"] == "global_batch_verify" for event in result.event_trace))

    def test_defaults_and_formal_labels_do_not_use_legacy_aliases(self) -> None:
        for alias in LEGACY_ALIASES:
            self.assertNotIn(alias, DEFAULT_METHODS)

        readme = Path("README.md").read_text(encoding="utf-8")
        run_sh = Path("scripts/run.sh").read_text(encoding="utf-8")
        self.assertIn(
            "full target_only server_only_linear specedge_linear dip_sd server_only_tree specedge_tree",
            readme,
        )
        self.assertIn(
            "full target_only server_only_linear specedge_linear dip_sd server_only_tree specedge_tree",
            run_sh,
        )


if __name__ == "__main__":
    unittest.main()
