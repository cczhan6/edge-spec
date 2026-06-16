from __future__ import annotations

import unittest

from src.metrics import summarize
from src.simulator import Simulator
from tests.common import small_config


class DeterminismTest(unittest.TestCase):
    def test_same_seed_produces_identical_metrics(self) -> None:
        config, model_runner, workload = small_config(num_requests=4, output_len=12)
        first = summarize(Simulator(config, model_runner, workload, "balanced_drafter", "full").run(), 3)
        second = summarize(Simulator(config, model_runner, workload, "balanced_drafter", "full").run(), 3)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
