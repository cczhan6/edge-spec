from __future__ import annotations

import copy
import math
import unittest

from src.config import load_config, validate_config
from src.model_runner import FakeModelRunner
from src.simulator import Simulator
from tests.common import accepting_model_runner, small_config


def enable_dynamic_edge(config: dict) -> dict:
    config["dynamic_edge_compute"]["enabled"] = True
    return config


def force_one_device(config: dict) -> None:
    config["simulation"]["num_devices"] = 1
    for pool_name, pool in config["device_pools"].items():
        templates = pool["templates"]
        for template in templates.values():
            template["count"] = 0
        selected = "low_end" if pool_name == "heterogeneous" else "medium"
        templates[selected]["count"] = 1


class DynamicEdgeConfigurationTest(unittest.TestCase):
    def test_defaults_are_disabled_with_five_completion_interval_and_ranges(self) -> None:
        config = load_config("configs/default.yaml")

        self.assertEqual(
            config["dynamic_edge_compute"],
            {"enabled": False, "resample_every_completed_requests": 5},
        )
        templates = config["device_pools"]["heterogeneous"]["templates"]
        self.assertEqual(
            templates["low_end"]["dynamic_draft_token_rate_range_tok_s"],
            [20, 30],
        )
        self.assertEqual(
            templates["mid_end"]["dynamic_draft_token_rate_range_tok_s"],
            [48, 72],
        )
        self.assertEqual(
            templates["high_end"]["dynamic_draft_token_rate_range_tok_s"],
            [80, 120],
        )

    def test_enabled_mode_requires_valid_ranges_for_populated_templates(self) -> None:
        cases = (
            ("not-a-bool", "enabled"),
            ([0, 30], "range"),
            ([30, 20], "range"),
            ([20], "range"),
            ([20, math.inf], "range"),
        )
        for value, message in cases:
            with self.subTest(value=value):
                config = load_config("configs/default.yaml")
                if message == "enabled":
                    config["dynamic_edge_compute"]["enabled"] = value
                else:
                    config["dynamic_edge_compute"]["enabled"] = True
                    config["device_pools"]["heterogeneous"]["templates"][
                        "low_end"
                    ]["dynamic_draft_token_rate_range_tok_s"] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_config(config)

        config = enable_dynamic_edge(load_config("configs/default.yaml"))
        del config["device_pools"]["heterogeneous"]["templates"]["low_end"][
            "dynamic_draft_token_rate_range_tok_s"
        ]
        with self.assertRaisesRegex(ValueError, "low_end"):
            validate_config(config)

    def test_resample_interval_is_exactly_five(self) -> None:
        config = load_config("configs/default.yaml")
        config["dynamic_edge_compute"]["resample_every_completed_requests"] = 4
        with self.assertRaisesRegex(ValueError, "must be 5"):
            validate_config(config)

    def test_disabled_legacy_config_may_omit_dynamic_section_and_ranges(self) -> None:
        config = load_config("configs/default.yaml")
        del config["dynamic_edge_compute"]
        for pool in config["device_pools"].values():
            for template in pool["templates"].values():
                template.pop("dynamic_draft_token_rate_range_tok_s", None)

        validate_config(config)
