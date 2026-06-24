from __future__ import annotations

import unittest

from src.entities import Device, DeviceRuntime, Request, Segment, SimulationResult
from src.metrics import enrich_comparisons, summarize
from src.model_runner import DraftCandidateTree, DraftTreeNode


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

    def test_tree_acceptance_summary_uses_tree_proposed_count(self) -> None:
        request = Request(
            request_id=0,
            device_id=0,
            output_len=2,
            arrival_time_ms=0.0,
            decode_ready_time_ms=0.0,
            prompt_id="0",
            category="generic",
            category_group="generic",
            prompt="prompt",
            prompt_token_count=1,
            prompt_ids=[0],
            finish_time_ms=10.0,
            generated_ids=[1, 1],
        )
        segment = Segment(
            segment_id=0,
            request_id=0,
            device_id=0,
            draft_model="medium",
            prefix_version=0,
            base_pos=0,
            scheduled_gamma=1,
            prefix_ids=[0],
            draft_ids=[2],
            create_time_ms=0.0,
            draft_start_time_ms=0.0,
            accepted_count=2,
            draft_tree=DraftCandidateTree(
                prefix_ids=[0],
                primary_ids=[2],
                primary_node_ids=[1],
                nodes=[
                    DraftTreeNode(1, None, 2, 1),
                    DraftTreeNode(2, None, 1, 1),
                    DraftTreeNode(3, 2, 1, 2),
                ],
            ),
        )
        device = Device(
            device_id=0,
            device_type="medium",
            drafter_profile="medium",
            acceptance_prior=0.5,
            draft_token_rate_tok_s=100.0,
            draft_startup_ms=0.0,
            uplink_mbps=100.0,
            downlink_mbps=100.0,
            rtt_ms=0.0,
            jitter_ms=0.0,
        )
        runtime = DeviceRuntime(device, selected_gammas=[1])
        result = SimulationResult(
            method="server_only",
            scenario="test",
            requests=[request],
            segments=[segment],
            devices=[runtime],
            lanes=[],
            batch_waiting_time_ms=0.0,
            phase_waiting_time_ms=0.0,
            lane_queue_wait_times_ms=[],
            event_trace=[],
        )

        main, _ = summarize(result, 1)

        self.assertEqual(segment.proposed_count, 2)
        self.assertEqual(main["avg_acceptance_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
