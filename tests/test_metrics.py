import unittest

from edge_spec.metrics import summarize_by_task


class MetricsTests(unittest.TestCase):
    def test_task_effective_duration_uses_wall_clock_span(self):
        records = [
            {
                "task": "MT",
                "microbatch_id": 0,
                "generated_token_count": 10,
                "effective_received_token_count": 10,
                "latency_s": 10.0,
                "start_time_s": 0.0,
                "completion_time_s": 10.0,
                "first_token_latency_s": 1.0,
            },
            {
                "task": "MT",
                "microbatch_id": 1,
                "generated_token_count": 20,
                "effective_received_token_count": 20,
                "latency_s": 10.0,
                "start_time_s": 5.0,
                "completion_time_s": 15.0,
                "first_token_latency_s": 1.0,
            },
        ]
        metrics = summarize_by_task(records)["MT"]
        self.assertEqual(metrics["effective_received_token_count"], 30)
        self.assertEqual(metrics["effective_duration_s"], 15.0)
        self.assertEqual(metrics["microbatch_duration_sum_s"], 20.0)
        self.assertEqual(metrics["effective_received_throughput_tokens_per_s"], 2.0)

    def test_target_only_network_records_contribute_to_summary(self):
        from edge_spec.metrics import summarize

        records = [
            {
                "device_id": "device-0",
                "task": "MT",
                "microbatch_id": 0,
                "generated_token_count": 4,
                "effective_received_token_count": 4,
                "accepted_draft_tokens": 0,
                "proposed_draft_tokens": 0,
                "acceptance_rate": None,
                "latency_s": 1.0,
                "start_time_s": 0.0,
                "completion_time_s": 1.0,
                "first_token_latency_s": 1.0,
                "target_only_latency_s": 1.0,
                "speedup_vs_target_only": 1.0,
                "target_only_uplink_effective_mbps": 10.0,
                "target_only_downlink_effective_mbps": 20.0,
                "target_only_uplink_effective_rtt_ms": 30.0,
                "target_only_downlink_effective_rtt_ms": 40.0,
                "target_only_uplink_congested": True,
                "target_only_downlink_congested": False,
            }
        ]
        summary = summarize(records, [], 1.0)
        self.assertIsNone(summary["mean_acceptance_rate"])
        self.assertIsNone(summary["overall_acceptance_rate"])
        self.assertEqual(summary["mean_uplink_effective_mbps"], 10.0)
        self.assertEqual(summary["mean_downlink_effective_mbps"], 20.0)
        self.assertEqual(summary["network_congestion_events"], 1)
        self.assertEqual(summary["network_congestion_fraction"], 0.5)

    def test_lane_queue_wait_is_weighted_by_verified_segment(self):
        from edge_spec.metrics import summarize_lanes
        from edge_spec.protocol import VerifierLaneState

        records = [
            {
                "device_id": "device-0",
                "task": "MT",
                "microbatch_id": 0,
                "generated_token_count": 4,
                "effective_received_token_count": 4,
                "accepted_draft_tokens": 1,
                "proposed_draft_tokens": 2,
                "acceptance_rate": 0.5,
                "latency_s": 1.0,
                "start_time_s": 0.0,
                "completion_time_s": 1.0,
                "first_token_latency_s": 0.1,
            }
        ]
        traces = [
            {
                "target_batch_size": 2,
                "target_forward_s": 2.0,
                "devices": [
                    {"lane_queue_wait_s": 1.0, "uplink_payload_bytes": 1},
                    {"lane_queue_wait_s": 3.0, "uplink_payload_bytes": 1},
                ],
            }
        ]
        lanes = [VerifierLaneState(lane_id=0, busy_s=2.0, verification_count=2)]
        summary = summarize_lanes(records, traces, 4.0, lanes)
        self.assertEqual(summary["mean_lane_queue_wait_s"], 2.0)
        self.assertEqual(summary["lane_queue_wait_fraction"], 0.5)
        self.assertEqual(summary["mean_lane_batch_size"], 2.0)


if __name__ == "__main__":
    unittest.main()
