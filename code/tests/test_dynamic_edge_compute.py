from __future__ import annotations

import copy
import math
import unittest
from unittest.mock import patch

from src.config import build_devices, load_config, validate_config
from src.edge_compute import EdgeComputeModel, deterministic_draft_rate
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


class EdgeComputeModelTest(unittest.TestCase):
    def make_model(self, *, seed: int = 42) -> EdgeComputeModel:
        config = enable_dynamic_edge(load_config("configs/default.yaml"))
        config["simulation"]["seed"] = seed
        return EdgeComputeModel(config, build_devices(config), "heterogeneous")

    def test_same_type_initial_rates_are_distinct_and_in_range(self) -> None:
        model = self.make_model()
        rates = [
            model.state(device_id).draft_token_rate_tok_s
            for device_id in (0, 1, 2)
        ]

        self.assertEqual(len(set(rates)), 3)
        self.assertTrue(all(20.0 <= rate < 30.0 for rate in rates))

    def test_sampling_is_reproducible_and_call_order_independent(self) -> None:
        bounds = (20.0, 30.0)
        forward = {
            epoch: deterministic_draft_rate(7, 3, "low_end", epoch, bounds)
            for epoch in (0, 1, 2)
        }
        reverse = {
            epoch: deterministic_draft_rate(7, 3, "low_end", epoch, bounds)
            for epoch in (2, 1, 0)
        }

        self.assertEqual(forward, reverse)
        self.assertEqual(
            forward[1],
            deterministic_draft_rate(7, 3, "low_end", 1, bounds),
        )
        self.assertNotEqual(forward[0], forward[1])

    def test_fifth_completion_advances_only_that_device(self) -> None:
        model = self.make_model()
        old_zero = model.state(0)
        old_one = model.state(1)

        for _ in range(4):
            self.assertIsNone(model.record_request_completion(0))
        transition = model.record_request_completion(0)

        self.assertIsNotNone(transition)
        self.assertEqual(model.state(0).completed_requests, 5)
        self.assertEqual(model.state(0).epoch, 1)
        self.assertNotEqual(
            model.state(0).draft_token_rate_tok_s,
            old_zero.draft_token_rate_tok_s,
        )
        self.assertEqual(model.state(1), old_one)

    def test_started_snapshot_keeps_old_rate_after_resample(self) -> None:
        model = self.make_model()
        started = model.snapshot(0)
        started_ms = model.latency_ms(started, 4)

        for _ in range(5):
            model.record_request_completion(0)

        next_started = model.snapshot(0)
        self.assertEqual(model.latency_ms(started, 4), started_ms)
        self.assertEqual(started.epoch, 0)
        self.assertEqual(next_started.epoch, 1)
        self.assertNotEqual(
            started.draft_token_rate_tok_s,
            next_started.draft_token_rate_tok_s,
        )

    def test_disabled_model_uses_fixed_device_rate_and_never_transitions(self) -> None:
        config = load_config("configs/default.yaml")
        devices = build_devices(config)
        model = EdgeComputeModel(config, devices, "heterogeneous")

        self.assertEqual(
            model.snapshot(0).draft_token_rate_tok_s,
            devices[0].draft_token_rate_tok_s,
        )
        for _ in range(5):
            self.assertIsNone(model.record_request_completion(0))
        self.assertEqual(model.state(0).epoch, 0)

    def test_initial_collision_is_resolved_deterministically(self) -> None:
        config = enable_dynamic_edge(load_config("configs/default.yaml"))
        devices = build_devices(config)
        values = iter((21.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0))
        with patch("src.edge_compute.deterministic_draft_rate", side_effect=values):
            model = EdgeComputeModel(config, devices, "heterogeneous")

        low_rates = [
            model.state(index).draft_token_rate_tok_s for index in (0, 1, 2)
        ]
        self.assertEqual(low_rates, [21.0, 22.0, 23.0])
