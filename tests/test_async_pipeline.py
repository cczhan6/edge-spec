import unittest

from edge_spec.backends import FakeBackend
from edge_spec.dataset import fallback_items
from edge_spec.methods.base import RunConfig
from edge_spec.methods.proposed.async_runtime import ProposedAsyncRunner
from edge_spec.types import DeviceProfile, SamplingConfig


class ProposedAsyncTests(unittest.TestCase):
    def _runner(self, lane_count=2, max_inflight_segments=2, lane_batch_size=2):
        profiles = {
            "device-0": DeviceProfile("device-0", 20, 50, 40, 0),
            "device-1": DeviceProfile("device-1", 50, 100, 25, 0),
            "device-2": DeviceProfile("device-2", 100, 200, 15, 0),
        }
        return ProposedAsyncRunner(
            draft_backends=[
                FakeBackend("draft-0", seed=1),
                FakeBackend("draft-1", seed=3),
                FakeBackend("draft-2", seed=5),
            ],
            target_backend=FakeBackend("target", seed=9),
            profiles=profiles,
            config=RunConfig(
                method="proposed",
                sampling=SamplingConfig(temperature=1.0, top_p=1.0, top_k=0),
                gamma=2,
                max_new_tokens=6,
                seed=7,
                lane_count=lane_count,
                max_inflight_segments=max_inflight_segments,
                lookahead_policy="adaptive",
                scheduler="prefix-aware",
                lane_batch_size=lane_batch_size,
                lane_batch_timeout_s=0.0,
            ),
        )

    def test_proposed_runs_with_prefix_lanes_and_inflight_segments(self):
        result = self._runner().run_dataset([fallback_items()])
        records, traces, summary = result.records, result.traces, result.summary
        self.assertEqual(len(records), 3)
        self.assertGreaterEqual(len(traces), 1)
        self.assertEqual(summary["method"], "proposed")
        self.assertEqual(summary["lane_count"], 2)
        self.assertGreaterEqual(summary["lane_verification_count"], summary["verification_event_count"])
        self.assertEqual(summary["mean_barrier_wait_s"], 0.0)
        self.assertEqual(summary["barrier_wait_fraction"], 0.0)
        self.assertGreater(summary["throughput_tokens_per_s"], 0)
        seen_second_segment = False
        for trace in traces:
            self.assertIn("target_batch_size", trace)
            for device in trace["devices"]:
                self.assertIn("prefix_version", device)
                self.assertIn("base_position", device)
                self.assertIn("prefix_hash", device)
                self.assertIn("segment_id", device)
                seen_second_segment = seen_second_segment or device["segment_id"] >= 1
                self.assertEqual(device["barrier_wait_s"], 0.0)
        self.assertTrue(seen_second_segment)
        for record in records:
            self.assertEqual(record["method"], "proposed")
            self.assertGreater(record["async_rounds"], 0)
            self.assertEqual(len(record["lane_ids"]), record["async_rounds"])
            self.assertIsNotNone(record["first_token_latency_s"])

    def test_proposed_rejects_invalid_lane_count(self):
        with self.assertRaises(ValueError):
            self._runner(lane_count=0)


if __name__ == "__main__":
    unittest.main()
