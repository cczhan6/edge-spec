import unittest

from edge_spec.backends import FakeBackend
from edge_spec.dataset import fallback_items
from edge_spec.runner import HeteroAsyncPipelineRunner
from edge_spec.types import DeviceProfile, SamplingConfig


class AsyncPipelineTests(unittest.TestCase):
    def _runner(self, pipeline_count=2):
        profiles = {
            "device-0": DeviceProfile("device-0", 20, 50, 40, 0),
            "device-1": DeviceProfile("device-1", 50, 100, 25, 0),
            "device-2": DeviceProfile("device-2", 100, 200, 15, 0),
        }
        return HeteroAsyncPipelineRunner(
            draft_backends=[
                FakeBackend("draft-0", seed=1),
                FakeBackend("draft-1", seed=3),
                FakeBackend("draft-2", seed=5),
            ],
            target_backend=FakeBackend("target", seed=9),
            profiles=profiles,
            sampling=SamplingConfig(temperature=1.0, top_p=1.0, top_k=0),
            gamma=2,
            max_new_tokens=6,
            seed=7,
            run_target_baseline=False,
            pipeline_count=pipeline_count,
        )

    def test_async_pipeline_runs_without_barrier_batching(self):
        records, traces, summary = self._runner().run_dataset([fallback_items()])
        self.assertEqual(len(records), 3)
        self.assertGreaterEqual(len(traces), 1)
        self.assertEqual(summary["mode"], "async")
        self.assertEqual(summary["pipeline_count"], 2)
        self.assertEqual(summary["pipeline_verification_count"], len(traces))
        self.assertEqual(summary["mean_barrier_wait_s"], 0.0)
        self.assertEqual(summary["barrier_wait_fraction"], 0.0)
        self.assertGreater(summary["throughput_tokens_per_s"], 0)
        self.assertIn("fallback", summary["task_metrics"])
        for trace in traces:
            self.assertEqual(trace["target_batch_size"], 1)
            self.assertIn("pipeline_id", trace)
            self.assertIn("pipeline_queue_wait_s", trace)
            self.assertEqual(len(trace["devices"]), 1)
            device = trace["devices"][0]
            self.assertEqual(device["barrier_wait_s"], 0.0)
            self.assertIn("draft_start_s", device)
            self.assertIn("draft_end_s", device)
            self.assertAlmostEqual(
                device["draft_time_s"],
                device["draft_end_s"] - device["draft_start_s"],
            )
            self.assertAlmostEqual(
                device["arrival_s"],
                device["draft_end_s"] + device["uplink_s"],
            )
            self.assertNotIn("draft_flops", device)
            self.assertNotIn("draft_compute_s", device)
            self.assertNotIn("compute_extra_s", device)
        for record in records:
            self.assertEqual(record["execution_mode"], "async")
            self.assertGreater(record["async_rounds"], 0)
            self.assertEqual(len(record["pipeline_ids"]), record["async_rounds"])
            self.assertIsNotNone(record["first_token_latency_s"])

    def test_async_rejects_invalid_pipeline_count(self):
        with self.assertRaises(ValueError):
            self._runner(pipeline_count=0)


if __name__ == "__main__":
    unittest.main()
