from __future__ import annotations

import unittest

from src.config import build_devices, load_config
from src.latency import (
    AcceptanceWindowEstimator,
    device_prefill_latency_ms,
    draft_latency_ms,
    expected_emitted_tokens,
    target_prefill_latency_ms,
    target_only_latency_ms,
    verify_latency_ms,
)


class AnalyticalLatencyTest(unittest.TestCase):
    def test_token_rate_formulas(self) -> None:
        config = load_config("configs/default.yaml")
        device = build_devices(config)[0]
        self.assertEqual(draft_latency_ms(device, 4), 161.0)
        self.assertEqual(device_prefill_latency_ms(device, 4), 161.0)
        self.assertEqual(verify_latency_ms(config["edge"], [1, 1]), 33.0)
        self.assertEqual(verify_latency_ms(config["edge"], [2, 4]), 83.0)
        self.assertEqual(target_prefill_latency_ms(config["edge"], 4), 50.0)
        self.assertEqual(target_only_latency_ms(config["edge"], 4), 50.0)

    def test_sliding_acceptance_uses_prior_then_recent_rounds(self) -> None:
        estimator = AcceptanceWindowEstimator(2)
        self.assertEqual(estimator.estimate(7, 0.45), 0.45)
        estimator.observe(7, 4, 4)
        estimator.observe(7, 0, 4)
        estimator.observe(7, 3, 4)
        self.assertEqual(estimator.estimate(7, 0.45), 3 / 8)
        self.assertEqual(expected_emitted_tokens(1.0, 4), 5.0)


if __name__ == "__main__":
    unittest.main()
