import unittest

from edge_spec.backends import FakeBackend
from edge_spec.dataset import fallback_items
from edge_spec.methods.base import RunConfig
from edge_spec.methods.baselines.sync_batch import SyncBatchRunner
from edge_spec.methods.baselines.target_only import TargetOnlyRunner
from edge_spec.types import DeviceProfile, SamplingConfig


def make_config(method="sync_batch"):
    return RunConfig(
        method=method,
        sampling=SamplingConfig(temperature=1.0, top_p=1.0, top_k=0),
        gamma=2,
        max_new_tokens=6,
        seed=7,
    )


class ProtocolFakeTests(unittest.TestCase):
    def _profiles(self):
        return {
            "device-0": DeviceProfile("device-0", 20, 50, 40, 0),
            "device-1": DeviceProfile("device-1", 50, 100, 25, 0),
            "device-2": DeviceProfile("device-2", 100, 200, 15, 0),
        }

    def _runner(self):
        profiles = self._profiles()
        return SyncBatchRunner(
            draft_backends=[
                FakeBackend("draft-0", seed=1),
                FakeBackend("draft-1", seed=3),
                FakeBackend("draft-2", seed=5),
            ],
            target_backend=FakeBackend("target", seed=9),
            profiles=profiles,
            config=make_config(),
        ), profiles

    def test_three_clients_barrier_batch_verification(self):
        runner, profiles = self._runner()
        result = runner.run_dataset([fallback_items()])
        records, traces, summary = result.records, result.traces, result.summary
        self.assertEqual(len(records), 3)
        self.assertGreaterEqual(len(traces), 1)
        self.assertEqual(traces[0]["target_batch_size"], 3)
        self.assertEqual(summary["method"], "sync_batch")
        self.assertGreater(summary["throughput_tokens_per_s"], 0)
        self.assertIn("fallback", summary["task_metrics"])
        self.assertGreater(
            summary["task_metrics"]["fallback"]["effective_throughput_tokens_per_s"],
            0,
        )
        self.assertGreater(
            summary["task_metrics"]["fallback"][
                "effective_received_throughput_tokens_per_s"
            ],
            0,
        )
        self.assertGreater(
            summary["task_metrics"]["fallback"]["e2e_first_token_latency_s"],
            0,
        )
        for trace in traces:
            for device in trace["devices"]:
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
            self.assertIn(record["device_id"], profiles)
            self.assertGreater(record["sync_rounds"], 0)
            self.assertEqual(
                record["effective_received_token_count"],
                record["generated_token_count"],
            )
            self.assertGreater(record["effective_received_tokens_per_s"], 0)
            self.assertIsNotNone(record["first_token_latency_s"])

    def test_sync_batch_does_not_embed_target_only_baseline(self):
        runner, _ = self._runner()
        result = runner.run_dataset([fallback_items()])
        self.assertIsNone(result.summary["mean_target_only_latency_s"])
        for record in result.records:
            self.assertIsNone(record["target_only_latency_s"])
            self.assertIsNone(record["speedup_vs_target_only"])

    def test_target_only_method_includes_network_round_trip(self):
        profiles = self._profiles()
        target = FakeBackend("target", seed=9)
        runner = TargetOnlyRunner(
            draft_backends=[target, target, target],
            target_backend=target,
            profiles=profiles,
            config=make_config("target_only"),
        )
        result = runner.run_dataset([fallback_items()])
        records, summary = result.records, result.summary
        self.assertEqual(summary["method"], "target_only")
        self.assertIsNotNone(summary["mean_target_only_latency_s"])
        for record in records:
            self.assertEqual(record["method"], "target_only")
            self.assertIsNotNone(record["target_only_latency_s"])
            self.assertIsNotNone(record["target_only_model_latency_s"])
            self.assertIsNotNone(record["target_only_uplink_s"])
            self.assertIsNotNone(record["target_only_downlink_s"])
            self.assertGreater(record["target_only_uplink_s"], 0)
            self.assertGreater(record["target_only_downlink_s"], 0)
            self.assertEqual(record["speedup_vs_target_only"], 1.0)
            self.assertGreater(
                record["target_only_latency_s"],
                record["target_only_model_latency_s"],
            )


if __name__ == "__main__":
    unittest.main()
