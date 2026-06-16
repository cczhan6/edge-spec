from __future__ import annotations

import unittest

from src.metrics import enrich_comparisons


class MetricsSpeedupTest(unittest.TestCase):
    def test_speedup_and_sync_ratio_use_distinct_baselines(self) -> None:
        rows = [
            {"method": "target_only", "avg_latency_ms": 100.0, "goodput_tok_s": 1.0},
            {"method": "sync_batch_sd", "avg_latency_ms": 75.0, "goodput_tok_s": 1.5},
            {"method": "SpecEdge", "avg_latency_ms": 60.0, "goodput_tok_s": 1.75},
            {"method": "full", "avg_latency_ms": 50.0, "goodput_tok_s": 2.0},
        ]
        full = enrich_comparisons(rows)[3]
        self.assertEqual(full["latency_speedup_vs_autoregressive"], 2.0)
        self.assertEqual(full["latency_ratio_vs_sync_batch_sd"], 1.5)
        self.assertEqual(full["latency_ratio_vs_specedge"], 1.2)


if __name__ == "__main__":
    unittest.main()
