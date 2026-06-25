from __future__ import annotations

import unittest
from collections import defaultdict

from src.simulator import Simulator
from tests.common import accepting_model_runner, rejecting_model_runner, small_config


CANONICAL_METHODS = (
    "target_only",
    "server_only_linear",
    "server_only_tree",
    "specedge_linear",
    "specedge_tree",
    "dip_sd",
)


class BaselineSystemInvariantTest(unittest.TestCase):
    def test_token_accounting_conservation(self) -> None:
        # Count meanings:
        # drafted/proposed are candidate tokens or tree path-depth candidates.
        # verified segments are accepted/rejected verification results.
        # accepted is target-confirmed draft/tree-path depth.
        # committed is the final target-confirmed request output.
        # wasted is rejected draft suffix + invalidated in-flight candidates +
        # invalidated proactive candidates. These are method-specific, so this
        # checks each conservation component rather than one global equality.
        for method in CANONICAL_METHODS:
            with self.subTest(method=method):
                result = run_canonical(method, accepting_model_runner())
                assert_token_accounting_conservation(self, result)

    def test_no_illegal_target_resource_overlap(self) -> None:
        for method in CANONICAL_METHODS:
            with self.subTest(method=method):
                result = run_canonical(method, accepting_model_runner())
                intervals_by_resource: dict[tuple, list[tuple[float, float, str]]] = defaultdict(list)
                for event in result.event_trace:
                    name = event["event"]
                    if name == "target_only_service":
                        key = ("target_only", event["lane_id"])
                    elif name == "server_only_verify":
                        key = ("server_target_gpu",)
                    elif name in {"global_batch_verify", "dip_sd_batch_verify"}:
                        key = ("server_target_gpu",)
                    elif name == "lane_verify":
                        key = ("lane", event["lane_id"])
                    else:
                        continue
                    intervals_by_resource[key].append(
                        (event["start_time_ms"], event["finish_time_ms"], name)
                    )
                assert_no_overlap(self, intervals_by_resource)

    def test_no_illegal_draft_resource_overlap(self) -> None:
        for method in CANONICAL_METHODS:
            with self.subTest(method=method):
                result = run_canonical(method, accepting_model_runner())
                intervals_by_resource: dict[tuple, list[tuple[float, float, str]]] = defaultdict(list)
                for event in result.event_trace:
                    name = event["event"]
                    if name in {"draft_compute", "proactive_draft", "dip_sd_draft"}:
                        key = ("device", event["device_id"])
                    elif name == "server_only_draft":
                        key = ("server_draft_gpu",)
                    else:
                        continue
                    intervals_by_resource[key].append(
                        (event["start_time_ms"], event["finish_time_ms"], name)
                    )
                assert_no_overlap(self, intervals_by_resource)

    def test_event_time_monotonicity_all_methods(self) -> None:
        for method in CANONICAL_METHODS:
            with self.subTest(method=method):
                result = run_canonical(method, accepting_model_runner())
                for event in result.event_trace:
                    if "start_time_ms" in event and "finish_time_ms" in event:
                        self.assertGreaterEqual(
                            event["finish_time_ms"],
                            event["start_time_ms"],
                            event,
                        )
                    for key in ("time_ms", "start_time_ms", "finish_time_ms"):
                        if key in event:
                            self.assertGreaterEqual(event[key], 0.0, event)

    def test_all_lossless_methods_equal_target_greedy(self) -> None:
        for model_runner in (accepting_model_runner(), rejecting_model_runner()):
            config, _, workload = small_config(num_requests=2, output_len=6)
            target = Simulator(
                config,
                model_runner,
                workload,
                "combined_strong_heterogeneous",
                "target_only",
            ).run()
            expected = [request.generated_ids for request in target.requests]
            for method in CANONICAL_METHODS:
                with self.subTest(method=method, runner=model_runner):
                    result = Simulator(
                        config,
                        model_runner,
                        workload,
                        "combined_strong_heterogeneous",
                        method,
                    ).run()
                    self.assertEqual(
                        [request.generated_ids for request in result.requests],
                        expected,
                    )

    def test_no_request_finishes_with_pending_unverified_state(self) -> None:
        for method in CANONICAL_METHODS:
            with self.subTest(method=method):
                result = run_canonical(method, accepting_model_runner())
                for request in result.requests:
                    self.assertEqual(request.status, "finished")
                    self.assertFalse(request.in_flight_segments)
                    self.assertFalse(request.pending_segments)
                    self.assertFalse(request.completed_results)
                    self.assertFalse(request.draft_queued)
                    self.assertFalse(request.proactive_draft_ids)
                    self.assertIsNone(request.proactive_draft_tree)
                    self.assertIsNone(request.proactive_base_pos)
                    self.assertIsNone(request.proactive_prefix_version)


def run_canonical(method: str, model_runner):
    config, _, workload = small_config(num_requests=2, output_len=6)
    config["specedge"]["server_batch_size"] = 2
    config["dip_sd"]["batch_count"] = 2
    config["dip_sd"]["max_batch_size"] = 2
    return Simulator(
        config,
        model_runner,
        workload,
        "combined_strong_heterogeneous",
        method,
    ).run()


def assert_token_accounting_conservation(
    testcase: unittest.TestCase,
    result,
) -> None:
    expected_accepted = defaultdict(int)
    expected_waste = defaultdict(int)
    committed_positions = defaultdict(dict)

    for request in result.requests:
        testcase.assertGreaterEqual(request.accepted_tokens, 0)
        testcase.assertGreaterEqual(request.rejected_count, 0)
        testcase.assertGreaterEqual(request.rollback_count, 0)
        testcase.assertGreaterEqual(request.wasted_draft_tokens, 0)

    if result.method == "target_only":
        testcase.assertEqual(result.segments, [])
        for request in result.requests:
            testcase.assertEqual(request.accepted_tokens, 0)
            testcase.assertEqual(request.wasted_draft_tokens, 0)
            testcase.assertEqual(len(request.generated_ids), request.output_len)
        return

    for segment in result.segments:
        testcase.assertGreaterEqual(segment.scheduled_gamma, 0)
        testcase.assertGreaterEqual(segment.proposed_count, 0)
        testcase.assertGreaterEqual(segment.emitted_count, 0)
        if segment.accepted_count is not None:
            testcase.assertGreaterEqual(segment.accepted_count, 0)
            testcase.assertLessEqual(segment.accepted_count, segment.proposed_count)
            testcase.assertIsNotNone(segment.verify_start_time_ms)
            testcase.assertIsNotNone(segment.verify_done_time_ms)
            expected_accepted[segment.request_id] += segment.accepted_count
        if segment.status == "rejected":
            expected_waste[segment.request_id] += max(
                0,
                segment.proposed_count - int(segment.accepted_count or 0),
            )
        elif segment.status in {"stale", "discarded"}:
            expected_waste[segment.request_id] += segment.proposed_count
            testcase.assertEqual(segment.emitted_ids, [])
        expected_waste[segment.request_id] += segment.proactive_wasted_tokens
        if segment.proactive_wasted_tokens:
            testcase.assertFalse(segment.proactive_hit)

        if segment.emitted_ids:
            testcase.assertIsNotNone(segment.result_base_pos)
            for offset, token_id in enumerate(segment.emitted_ids):
                position = int(segment.result_base_pos) + offset
                testcase.assertNotIn(
                    position,
                    committed_positions[segment.request_id],
                    f"request {segment.request_id} commits position {position} twice",
                )
                committed_positions[segment.request_id][position] = token_id

    for request in result.requests:
        testcase.assertEqual(request.accepted_tokens, expected_accepted[request.request_id])
        testcase.assertEqual(request.wasted_draft_tokens, expected_waste[request.request_id])
        positions = committed_positions[request.request_id]
        testcase.assertEqual(
            [positions[index] for index in sorted(positions)],
            request.generated_ids,
        )
        testcase.assertEqual(sorted(positions), list(range(len(request.generated_ids))))
        testcase.assertEqual(len(request.generated_ids), request.output_len)


def assert_no_overlap(
    testcase: unittest.TestCase,
    intervals_by_resource: dict[tuple, list[tuple[float, float, str]]],
) -> None:
    for resource, intervals in intervals_by_resource.items():
        intervals.sort()
        for previous, current in zip(intervals, intervals[1:]):
            testcase.assertGreaterEqual(
                current[0],
                previous[1],
                f"{resource} overlaps {previous[2]} {previous} with {current[2]} {current}",
            )


if __name__ == "__main__":
    unittest.main()
