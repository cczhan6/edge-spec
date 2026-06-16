from __future__ import annotations

import unittest

from src.entities import Segment
from src.model_runner import FakeModelRunner
from src.simulator import Simulator
from tests.common import accepting_model_runner, small_config


class _SecondDraftMismatchModelRunner(FakeModelRunner):
    def __init__(self) -> None:
        super().__init__(target_token_fn=lambda prefix: 1)
        self.calls = 0

    def draft(self, drafter_profile, prefix_ids, gamma):
        self.calls += 1
        return [1] if self.calls == 1 else [2]


class SemanticSimulatorTest(unittest.TestCase):
    def test_fake_model_runner_greedy_correction_and_bonus(self) -> None:
        model_runner = FakeModelRunner(
            target_token_fn=lambda prefix: 1,
            draft_token_fn=lambda profile, prefix: 1,
        )
        bonus = model_runner.verify([9], [1, 1])
        self.assertEqual(bonus.accepted_count, 2)
        self.assertEqual(bonus.emitted_ids, [1, 1, 1])
        self.assertEqual(bonus.bonus_token, 1)
        correction = model_runner.verify([9], [2, 1])
        self.assertEqual(correction.accepted_count, 0)
        self.assertEqual(correction.emitted_ids, [1])
        self.assertTrue(correction.rejected)

    def test_full_matches_target_only_greedy_output(self) -> None:
        config, model_runner, workload = small_config(num_requests=3, output_len=24)
        target = Simulator(config, model_runner, workload, "balanced", "target_only").run()
        full = Simulator(config, model_runner, workload, "balanced", "full").run()
        self.assertEqual(
            [request.generated_ids for request in full.requests],
            [request.generated_ids for request in target.requests],
        )

    def test_bonus_match_retargets_one_token_segment(self) -> None:
        config, _, workload = small_config(num_requests=1, output_len=4)
        config["speculation"]["gamma_candidates"] = [1]
        result = Simulator(config, accepting_model_runner(), workload, "balanced", "full").run()
        self.assertGreater(result.requests[0].bonus_reused_tokens, 0)
        self.assertTrue(
            any(
                segment.bonus_reused
                and segment.status == "accepted"
                and not segment.draft_ids
                for segment in result.segments
            )
        )

    def test_zero_token_active_segment_blocks_drafting_without_looping(self) -> None:
        config, model_runner, workload = small_config(num_requests=1, output_len=4)
        simulator = Simulator(config, model_runner, workload, "balanced", "full")
        simulator._schedule_request_arrivals()
        request = simulator.requests[0]
        segment = Segment(
            segment_id=0,
            request_id=request.request_id,
            device_id=request.device_id,
            draft_model="small",
            prefix_version=request.prefix_version,
            base_pos=request.committed_pos,
            scheduled_gamma=1,
            prefix_ids=list(request.prompt_ids),
            draft_ids=[],
            create_time_ms=0.0,
            draft_start_time_ms=0.0,
            status="verifying",
        )
        simulator.segments.append(segment)
        request.pending_segments[request.committed_pos] = segment.segment_id
        request.in_flight_segments.append(segment.segment_id)

        _, speculative_count, blocked = simulator._draft_prefix(request)

        self.assertEqual(speculative_count, 0)
        self.assertTrue(blocked)
        self.assertFalse(simulator._can_queue_draft(request))

    def test_bonus_mismatch_marks_optimistic_segment_stale(self) -> None:
        config, _, workload = small_config(num_requests=1, output_len=4)
        config["speculation"]["gamma_candidates"] = [1]
        result = Simulator(config, _SecondDraftMismatchModelRunner(), workload, "balanced", "full").run()
        self.assertTrue(any(segment.status == "stale" for segment in result.segments))

    def test_single_virtual_device_serializes_draft_segments_fifo(self) -> None:
        config, model_runner, workload = small_config(num_requests=3, output_len=8)
        config["simulation"]["num_devices"] = 1
        config["device_pools"]["heterogeneous"]["templates"]["low_end"]["count"] = 1
        config["device_pools"]["medium_only"]["templates"]["medium"]["count"] = 1
        result = Simulator(config, model_runner, workload, "balanced", "full").run()
        self.assertTrue(all(request.device_id == 0 for request in result.requests))
        events = [
            event for event in result.event_trace if event["event"] == "draft_compute"
        ]
        events.sort(key=lambda event: event["start_time_ms"])
        self.assertTrue(
            all(
                current["start_time_ms"] >= previous["finish_time_ms"]
                for previous, current in zip(events, events[1:])
            )
        )


if __name__ == "__main__":
    unittest.main()
