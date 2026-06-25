from __future__ import annotations

import unittest

from src.simulator import Simulator
from tests.common import small_config


class SyncBatchBarrierTest(unittest.TestCase):
    def test_sync_batch_alias_uses_dip_sd_canonical_pipeline(self) -> None:
        config, model_runner, workload = small_config(num_requests=3, output_len=4)
        config["sync_batch"]["B_global"] = 4
        config["sync_batch"]["global_batch_timeout_ms"] = 20
        result = Simulator(config, model_runner, workload, "combined_strong_heterogeneous", "sync_batch_sd").run()

        self.assertEqual(result.method, "dip_sd")
        self.assertTrue(any(event["event"] == "dip_sd_epoch_plan" for event in result.event_trace))
        self.assertTrue(any(event["event"] == "dip_sd_batch_verify" for event in result.event_trace))
        self.assertFalse(any(event["event"] == "global_batch_verify" for event in result.event_trace))


if __name__ == "__main__":
    unittest.main()
