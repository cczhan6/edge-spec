from __future__ import annotations

import unittest

from src.config import build_devices, load_config, validate_config
from src.tree_drafting import build_tree_draft_strategy


class ConfigTest(unittest.TestCase):
    def test_default_and_scenario_device_templates_declare_probability_one(self) -> None:
        for scenario in (None, "homogeneous", "combined_strong_heterogeneous"):
            with self.subTest(scenario=scenario):
                config = load_config("configs/default.yaml", scenario)
                for pool_name, pool in config["device_pools"].items():
                    for template_name, template in pool["templates"].items():
                        self.assertIn(
                            "block_probability",
                            template,
                            (scenario, pool_name, template_name),
                        )
                        self.assertEqual(template["block_probability"], 1.0)

    def test_block_probability_accepts_closed_unit_interval(self) -> None:
        for value in (0, 0.25, 1):
            with self.subTest(value=value):
                config = load_config("configs/default.yaml")
                config["device_pools"]["heterogeneous"]["templates"]["low_end"][
                    "block_probability"
                ] = value
                validate_config(config)

    def test_block_probability_rejects_invalid_values_with_template_path(self) -> None:
        invalid_values = (
            True,
            "0.5",
            -0.01,
            1.01,
            float("nan"),
            float("inf"),
            float("-inf"),
        )
        for value in invalid_values:
            with self.subTest(value=value):
                config = load_config("configs/default.yaml")
                config["device_pools"]["heterogeneous"]["templates"]["low_end"][
                    "block_probability"
                ] = value
                with self.assertRaisesRegex(
                    ValueError,
                    r"device_pools\.heterogeneous\.templates\.low_end\."
                    r"block_probability must be a finite number in \[0, 1\]",
                ):
                    validate_config(config)

    def test_invalid_network_bounds_are_rejected_even_for_zero_count_template(self) -> None:
        invalid_fields = (
            ("jitter_ms", -1.0, "must be a finite non-negative number"),
            ("jitter_ms", float("nan"), "must be a finite non-negative number"),
            ("rtt_ms", -1.0, "must be a finite non-negative number"),
            ("rtt_ms", float("inf"), "must be a finite non-negative number"),
            ("uplink_mbps", float("nan"), "must be a finite positive number"),
            ("downlink_mbps", 0.0, "must be a finite positive number"),
        )
        for field, value, message in invalid_fields:
            with self.subTest(field=field, value=value):
                config = load_config("configs/default.yaml")
                template = config["device_pools"]["heterogeneous"]["templates"][
                    "high_end"
                ]
                template["count"] = 0
                template[field] = value
                with self.assertRaisesRegex(
                    ValueError,
                    rf"device_pools\.heterogeneous\.templates\.high_end\."
                    rf"{field} {message}",
                ):
                    validate_config(config)

    def test_build_devices_defaults_missing_block_probability_to_one(self) -> None:
        config = load_config("configs/default.yaml")
        for pool in config["device_pools"].values():
            for template in pool["templates"].values():
                template.pop("block_probability", None)

        for pool_name in ("heterogeneous", "medium_only"):
            with self.subTest(pool_name=pool_name):
                devices = build_devices(config, pool_name)
                self.assertTrue(devices)
                self.assertTrue(
                    all(device.block_probability == 1.0 for device in devices)
                )

    def test_build_devices_propagates_template_block_probability(self) -> None:
        config = load_config("configs/default.yaml")
        templates = config["device_pools"]["heterogeneous"]["templates"]
        templates["low_end"]["block_probability"] = 0.1
        templates["mid_end"]["block_probability"] = 0.4
        templates["high_end"]["block_probability"] = 0.9

        devices = build_devices(config, "heterogeneous")

        expected = {"low_end": 0.1, "mid_end": 0.4, "high_end": 0.9}
        self.assertTrue(devices)
        for device in devices:
            self.assertEqual(
                device.block_probability,
                expected[device.device_type],
            )

    def test_default_device_and_verify_rates_match_strong_heterogeneous_profile(self) -> None:
        config = load_config("configs/default.yaml")
        templates = config["device_pools"]["heterogeneous"]["templates"]

        self.assertEqual(templates["low_end"]["draft_token_rate_tok_s"], 25)
        self.assertEqual(templates["mid_end"]["draft_token_rate_tok_s"], 60)
        self.assertEqual(templates["high_end"]["draft_token_rate_tok_s"], 100)
        self.assertEqual(
            config["device_pools"]["medium_only"]["templates"]["medium"][
                "draft_token_rate_tok_s"
            ],
            60,
        )
        self.assertEqual(config["edge"]["target_only_token_rate_tok_s"], 80)

    def test_homogeneous_uses_only_medium_devices(self) -> None:
        config = load_config("configs/default.yaml", "homogeneous")
        templates = config["device_pools"]["heterogeneous"]["templates"]

        self.assertEqual(templates["low_end"]["count"], 0)
        self.assertEqual(templates["mid_end"]["count"], 8)
        self.assertEqual(templates["high_end"]["count"], 0)
        self.assertEqual(
            {template["drafter_profile"] for template in templates.values()},
            {"medium"},
        )
        self.assertEqual(
            {template["draft_token_rate_tok_s"] for template in templates.values()},
            {60},
        )

    def test_removed_scenario_config_is_rejected(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "scenario config was removed"):
            load_config("configs/default.yaml", "balanced_drafter")

    def test_custom_scenario_label_can_use_default_config_without_override_file(self) -> None:
        config = load_config("configs/default.yaml", "smoke")
        self.assertEqual(config["simulation"]["num_devices"], 8)

    def test_requires_positive_analytical_target_rate(self) -> None:
        config = load_config("configs/default.yaml")
        config["edge"]["target_only_token_rate_tok_s"] = 0
        with self.assertRaisesRegex(ValueError, "target_only_token_rate"):
            validate_config(config)

    def test_requires_model_runner_model_for_each_drafter_profile(self) -> None:
        config = load_config("configs/default.yaml")
        del config["model_runner"]["drafter_models"]["small"]
        with self.assertRaisesRegex(ValueError, "small"):
            validate_config(config)

    def test_device_pool_count_must_match_num_devices(self) -> None:
        config = load_config("configs/default.yaml")
        config["device_pools"]["heterogeneous"]["templates"]["low_end"]["count"] = 4
        with self.assertRaisesRegex(ValueError, "defines 9 devices"):
            validate_config(config)

    def test_server_only_drafter_profile_must_be_known(self) -> None:
        config = load_config("configs/default.yaml")
        config["server_only"]["drafter_profile"] = "unknown"
        with self.assertRaisesRegex(ValueError, "server_only"):
            validate_config(config)

    def test_server_only_draft_rate_must_be_positive(self) -> None:
        config = load_config("configs/default.yaml")
        config["server_only"]["draft_token_rate_tok_s"] = 0
        with self.assertRaisesRegex(ValueError, "server_only.draft_token_rate"):
            validate_config(config)

    def test_specedge_proactive_budget_must_not_exceed_main_budget(self) -> None:
        config = load_config("configs/default.yaml")
        config["specedge"]["proactive_max_budget"] = config["specedge"]["max_budget"] + 1
        with self.assertRaisesRegex(ValueError, "proactive_max_budget"):
            validate_config(config)

    def test_specedge_scheduler_types_must_be_known(self) -> None:
        config = load_config("configs/default.yaml")
        config["specedge"]["server_batch_type"] = "round_robin"
        with self.assertRaisesRegex(ValueError, "server_batch_type"):
            validate_config(config)
        config = load_config("configs/default.yaml")
        config["specedge"]["proactive_type"] = "eager"
        with self.assertRaisesRegex(ValueError, "proactive_type"):
            validate_config(config)

    def test_tree_draft_strategy_types_must_be_known(self) -> None:
        config = load_config("configs/default.yaml")
        config["specedge"]["tree_draft_strategy"] = "unknown"
        with self.assertRaisesRegex(ValueError, "tree_draft_strategy"):
            validate_config(config)
        config = load_config("configs/default.yaml")
        config["server_only"]["tree_draft_strategy"] = "unknown"
        with self.assertRaisesRegex(ValueError, "server_only.tree_draft_strategy"):
            validate_config(config)

    def test_tree_draft_defaults_are_specexec_approx(self) -> None:
        config = load_config("configs/default.yaml")

        self.assertEqual(build_tree_draft_strategy(config, "specedge").name, "specexec_approx")
        self.assertEqual(build_tree_draft_strategy(config, "server_only").name, "specexec_approx")
        self.assertEqual(
            build_tree_draft_strategy(config, "specedge", proactive=True).name,
            "specexec_approx",
        )

    def test_legacy_specexec_name_maps_to_approximation(self) -> None:
        config = load_config("configs/default.yaml")
        config["specedge"]["tree_draft_strategy"] = "specexec"
        validate_config(config)

        self.assertEqual(
            build_tree_draft_strategy(config, "specedge").name,
            "specexec_approx",
        )

    def test_tree_budget_must_cover_beam_depth(self) -> None:
        config = load_config("configs/default.yaml")
        config["specedge"]["max_budget"] = config["specedge"]["max_beam_len"] - 1
        config["specedge"]["proactive_max_budget"] = config["specedge"]["max_budget"]
        with self.assertRaisesRegex(ValueError, "max_budget"):
            validate_config(config)
        config = load_config("configs/default.yaml")
        config["server_only"]["max_beam_len"] = 4
        config["server_only"]["max_budget"] = 3
        with self.assertRaisesRegex(ValueError, "server_only.max_budget"):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
