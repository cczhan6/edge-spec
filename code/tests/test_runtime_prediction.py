from __future__ import annotations

import unittest

from src.model_runner import FakeModelRunner
from src.simulator import Simulator
from tests.common import small_config


class _RecordingModelRunner(FakeModelRunner):
    def __init__(self) -> None:
        super().__init__()
        self.draft_calls = 0
        self.verify_calls = 0

    def draft(self, drafter_profile, prefix_ids, gamma):
        self.draft_calls += 1
        return super().draft(drafter_profile, prefix_ids, gamma)

    def verify(self, prefix_ids, draft_ids):
        self.verify_calls += 1
        return super().verify(prefix_ids, draft_ids)


class RuntimePredictionTest(unittest.TestCase):
    def test_gamma_enumeration_does_not_probe_model_runner(self) -> None:
        config, _, workload = small_config(num_requests=3, output_len=8)
        model_runner = _RecordingModelRunner()
        result = Simulator(config, model_runner, workload, "balanced_drafter", "full").run()
        candidates = set(config["speculation"]["gamma_candidates"])
        self.assertTrue(
            all(segment.scheduled_gamma in candidates for segment in result.segments)
        )
        lane_events = [
            event for event in result.event_trace if event["event"] == "lane_verify"
        ]
        self.assertEqual(model_runner.verify_calls, len(lane_events))
        self.assertEqual(model_runner.draft_calls, len(result.segments))


if __name__ == "__main__":
    unittest.main()
