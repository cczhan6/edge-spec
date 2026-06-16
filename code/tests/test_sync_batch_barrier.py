from __future__ import annotations

import unittest

from src.simulator import Simulator
from tests.common import small_config


class SyncBatchBarrierTest(unittest.TestCase):
    def test_partial_batch_waits_for_timeout(self) -> None:
        config, model_runner, workload = small_config(num_requests=3, output_len=4)
        config["sync_batch"]["B_global"] = 4
        config["sync_batch"]["global_batch_timeout_ms"] = 20
        result = Simulator(config, model_runner, workload, "balanced_drafter", "sync_batch_sd").run()
        first_batch = next(event for event in result.event_trace if event["event"] == "global_batch_verify")
        arrivals = [segment.edge_arrival_time_ms for segment in result.segments[:3]]
        self.assertGreaterEqual(first_batch["start_time_ms"], min(arrivals) + 20)
        self.assertGreater(result.batch_waiting_time_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
