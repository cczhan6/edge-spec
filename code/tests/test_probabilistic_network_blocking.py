from __future__ import annotations

import copy
import unittest
from unittest import mock

from scripts.baseline_trace import (
    _batch_trace_rows,
    _event_trace_rows,
    _request_trace_rows,
    _resource_timeline_rows,
    _token_trace_rows,
)
from src.communication import dssd_transmission_delay_ms, network_delay_ms
from src.simulator import Simulator
from tests.common import accepting_model_runner, small_config


NETWORKED_METHODS = (
    "full",
    "wo_async",
    "wo_scheduling",
    "conservative_rollback",
    "specedge_linear",
    "specedge_tree",
    "dip_sd",
)
NETWORK_FREE_METHODS = (
    "target_only",
    "server_only_linear",
    "server_only_tree",
)


def set_network_profile(config: dict, probability: float) -> None:
    for pool in config["device_pools"].values():
        for template in pool["templates"].values():
            template["rtt_ms"] = 20.0
            template["uplink_mbps"] = 8.0
            template["downlink_mbps"] = 16.0
            template["jitter_ms"] = 25.0
            template["block_probability"] = probability


def run_method(method: str, probability: float, seed: int = 42):
    config, _, workload = small_config(num_requests=2, output_len=6)
    config["simulation"]["seed"] = seed
    config["simulation"]["request_arrival"] = "burst"
    config["specedge"]["server_batch_size"] = 2
    config["dip_sd"]["batch_count"] = 2
    config["dip_sd"]["max_batch_size"] = 2
    set_network_profile(config, probability)
    result = Simulator(
        config,
        accepting_model_runner(),
        workload,
        "test",
        method,
    ).run()
    return config, result


def trace_rows(result) -> dict[str, list[dict]]:
    return {
        "requests": _request_trace_rows(result),
        "events": _event_trace_rows(result),
        "tokens": _token_trace_rows(result),
        "resources": _resource_timeline_rows(result),
        "batches": _batch_trace_rows(result),
    }


class ProbabilisticNetworkBlockingIntegrationTest(unittest.TestCase):
    def test_probability_zero_removes_jitter_for_every_networked_method(self) -> None:
        for method in NETWORKED_METHODS:
            with self.subTest(method=method):
                _, result = run_method(method, probability=0.0)
                self.assertTrue(result.segments)
                for segment in result.segments:
                    device = result.devices[segment.device_id].device
                    if segment.uplink_payload_bytes:
                        expected_uplink = dssd_transmission_delay_ms(
                            segment.uplink_payload_bytes,
                            device.rtt_ms,
                            device.uplink_mbps,
                        )
                        self.assertEqual(segment.uplink_delay_ms, expected_uplink)
                    if segment.downlink_payload_bytes:
                        expected_downlink = dssd_transmission_delay_ms(
                            segment.downlink_payload_bytes,
                            device.rtt_ms,
                            device.downlink_mbps,
                        )
                        self.assertEqual(segment.downlink_delay_ms, expected_downlink)

    def test_default_probability_one_and_missing_field_have_exact_trace_rows(self) -> None:
        for method in (*NETWORK_FREE_METHODS, *NETWORKED_METHODS):
            with self.subTest(method=method):
                config, _, workload = small_config(num_requests=2, output_len=6)
                config["simulation"]["request_arrival"] = "burst"
                set_network_profile(config, probability=1.0)
                explicit = Simulator(
                    copy.deepcopy(config),
                    accepting_model_runner(),
                    workload,
                    "test",
                    method,
                ).run()
                legacy = copy.deepcopy(config)
                for pool in legacy["device_pools"].values():
                    for template in pool["templates"].values():
                        template.pop("block_probability", None)
                missing = Simulator(
                    legacy,
                    accepting_model_runner(),
                    workload,
                    "test",
                    method,
                ).run()

                self.assertEqual(trace_rows(explicit), trace_rows(missing))
                self.assertEqual(explicit.requests, missing.requests)
                self.assertEqual(explicit.segments, missing.segments)

    def test_intermediate_probability_is_reproducible(self) -> None:
        _, first = run_method("full", probability=0.5, seed=42)
        _, second = run_method("full", probability=0.5, seed=42)

        first_delays = [
            (segment.uplink_delay_ms, segment.downlink_delay_ms)
            for segment in first.segments
        ]
        second_delays = [
            (segment.uplink_delay_ms, segment.downlink_delay_ms)
            for segment in second.segments
        ]
        self.assertEqual(first.event_trace, second.event_trace)
        self.assertEqual(first_delays, second_delays)

    def test_all_networked_methods_keep_current_keys_at_shared_function(self) -> None:
        expected_prefixes = {
            "full": ("estimate-up:", "estimate-down:"),
            "wo_async": ("estimate-up:", "estimate-down:"),
            "wo_scheduling": ("estimate-up:", "estimate-down:"),
            "conservative_rollback": ("estimate-up:", "estimate-down:"),
            "specedge_linear": ("pipeline-up:", "pipeline-down:"),
            "specedge_tree": ("pipeline-up:", "pipeline-down:"),
            "dip_sd": ("dip-sd-plan:", "dip-sd-up:", "dip-sd-down:"),
        }
        for method, prefixes in expected_prefixes.items():
            with self.subTest(method=method):
                with mock.patch(
                    "src.simulator.network_delay_ms",
                    wraps=network_delay_ms,
                ) as shared_delay:
                    run_method(method, probability=0.5)
                keys = [call.args[3] for call in shared_delay.call_args_list]
                self.assertTrue(keys)
                for prefix in prefixes:
                    self.assertTrue(
                        any(str(key).startswith(prefix) for key in keys),
                        (method, prefix, keys),
                    )
                if method != "dip_sd":
                    self.assertTrue(any(isinstance(key, int) for key in keys))

    def test_target_and_server_only_remain_network_free(self) -> None:
        for method in NETWORK_FREE_METHODS:
            with self.subTest(method=method):
                with mock.patch(
                    "src.simulator.network_delay_ms",
                    wraps=network_delay_ms,
                ) as shared_delay:
                    _, result = run_method(method, probability=0.5)
                shared_delay.assert_not_called()
                self.assertTrue(
                    all(segment.uplink_delay_ms == 0.0 for segment in result.segments)
                )
                self.assertTrue(
                    all(segment.downlink_delay_ms == 0.0 for segment in result.segments)
                )
                for event in result.event_trace:
                    self.assertNotIn("uplink_ms", event)
                    self.assertNotIn("downlink_ms", event)
