from __future__ import annotations

import unittest
from itertools import product

from src.dip_sd import (
    DipSDModelProfile,
    DipSDProblem,
    DipSDUser,
    evaluate_plan,
    optimize_dip_sd,
)
from src.config import validate_config
from src.methods import get_method_spec
from src.simulator import Simulator
from tests.common import accepting_model_runner, rejecting_model_runner, small_config


class DipSDTest(unittest.TestCase):
    def test_dip_sd_methods_are_registered(self) -> None:
        config, _, _ = small_config(num_requests=1, output_len=4)

        optimized = get_method_spec("dip_sd", config)

        self.assertEqual(optimized.runtime, "dip_sd")
        self.assertEqual(optimized.candidate_strategy, "linear")
        with self.assertRaises(ValueError):
            get_method_spec("dip_sd_greedy", config)

    def test_dip_sd_rejects_static_optimizer_names(self) -> None:
        config, _, _ = small_config(num_requests=1, output_len=4)
        config["dip_sd"]["optimizer"] = "deterministic_search"

        with self.assertRaisesRegex(ValueError, "paper_exact"):
            validate_config(config)

    def test_optimizer_is_deterministic(self) -> None:
        problem = paper_problem()

        first = optimize_dip_sd(problem)
        second = optimize_dip_sd(problem)

        self.assertEqual(first, second)
        self.assertEqual(first.optimizer, "paper_exact")

    def test_dip_sd_optimizer_returns_feasible_solution(self) -> None:
        plan = optimize_dip_sd(paper_problem())

        self.assertTrue(plan.feasible)
        self.assertGreater(plan.objective, 0.0)
        self.assertGreater(plan.expected_useful_tokens, 0.0)
        self.assertGreater(plan.pipeline_span, 0.0)
        self.assertEqual(plan.pipeline_span, sum(plan.stage_durations))

    def test_dip_sd_optimizer_assignment_is_complete_and_disjoint(self) -> None:
        problem = paper_problem()
        plan = optimize_dip_sd(problem)

        flattened = [request_id for batch in plan.batches for request_id in batch]
        self.assertEqual(
            sorted(flattened),
            sorted(user.request_id for user in problem.users),
        )
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(plan.assignment), set(flattened))

    def test_dip_sd_optimizer_batches_are_nonempty(self) -> None:
        plan = optimize_dip_sd(paper_problem())

        self.assertTrue(all(batch for batch in plan.batches))
        self.assertEqual(len(plan.batches), len(plan.batch_ready_times))
        self.assertEqual(len(plan.batches), len(plan.verify_times))

    def test_dip_sd_optimizer_draft_lengths_respect_bounds(self) -> None:
        problem = paper_problem(min_draft_length=1, max_draft_length=3)
        plan = optimize_dip_sd(problem)

        self.assertEqual(set(plan.draft_lengths), {0, 1, 2})
        self.assertTrue(
            all(1 <= length <= 3 for length in plan.draft_lengths.values())
        )

    def test_dip_sd_optimizer_objective_matches_manual_case(self) -> None:
        problem = paper_problem(
            users=(
                paper_user(0, alpha=0.5, prefix_length=8),
            ),
            min_draft_length=1,
            max_draft_length=1,
            max_batch_count=1,
            max_batch_size=1,
        )

        plan = optimize_dip_sd(problem)
        expected_useful = (1.0 - 0.5**2) / (1.0 - 0.5)
        expected_span = max(
            sum(plan.verify_times),
            plan.batch_ready_times[0] + plan.verify_times[0],
        )

        self.assertAlmostEqual(plan.expected_useful_tokens, expected_useful)
        self.assertAlmostEqual(plan.pipeline_span, expected_span)
        self.assertAlmostEqual(plan.objective, expected_useful / expected_span)

    def test_dip_sd_optimizer_matches_bruteforce_on_tiny_cases(self) -> None:
        problem = paper_problem(
            users=(
                paper_user(0, alpha=0.7, prefix_length=8, communication=0.1),
                paper_user(1, alpha=0.9, prefix_length=11, communication=0.2),
            ),
            min_draft_length=1,
            max_draft_length=3,
            max_batch_count=2,
            max_batch_size=2,
        )

        plan = optimize_dip_sd(problem)
        brute = brute_force_plan(problem)

        self.assertEqual(plan.batches, brute.batches)
        self.assertEqual(plan.draft_lengths, brute.draft_lengths)
        self.assertAlmostEqual(plan.objective, brute.objective)

    def test_dip_sd_optimizer_never_reads_future_acceptance(self) -> None:
        problem = paper_problem(
            users=(
                paper_user(0, alpha=0.2),
                paper_user(1, alpha=0.8),
            )
        )

        first = optimize_dip_sd(problem)
        second = optimize_dip_sd(problem)

        self.assertEqual(first, second)
        self.assertNotIn("future_acceptance", DipSDProblem.__dataclass_fields__)

    def test_optimizer_uses_estimated_acceptance_not_realized_future_acceptance(self) -> None:
        low = optimize_dip_sd(
            paper_problem(
                users=(paper_user(0, alpha=0.1),),
                min_draft_length=1,
                max_draft_length=4,
                max_batch_count=1,
                max_batch_size=1,
            )
        )
        high = optimize_dip_sd(
            paper_problem(
                users=(paper_user(0, alpha=0.95),),
                min_draft_length=1,
                max_draft_length=4,
                max_batch_count=1,
                max_batch_size=1,
            )
        )

        self.assertLessEqual(low.draft_lengths[0], high.draft_lengths[0])

        low_config, _, low_workload = small_config(num_requests=1, output_len=6)
        low_config["dip_sd"]["batch_count"] = 1
        low_config["dip_sd"]["min_draft_length"] = 1
        low_config["dip_sd"]["max_draft_length"] = 4
        low_config["drafter_profiles"]["small"]["acceptance_prior"] = 0.1
        low_result = Simulator(
            low_config,
            rejecting_model_runner(),
            low_workload,
            "combined_strong_heterogeneous",
            "dip_sd",
        ).run()
        low_first_plan = first_event(low_result, "dip_sd_epoch_plan")

        high_config, _, high_workload = small_config(num_requests=1, output_len=6)
        high_config["dip_sd"]["batch_count"] = 1
        high_config["dip_sd"]["min_draft_length"] = 1
        high_config["dip_sd"]["max_draft_length"] = 4
        high_config["drafter_profiles"]["small"]["acceptance_prior"] = 0.95
        high_result = Simulator(
            high_config,
            rejecting_model_runner(),
            high_workload,
            "combined_strong_heterogeneous",
            "dip_sd",
        ).run()
        high_first_plan = first_event(high_result, "dip_sd_epoch_plan")

        self.assertEqual(high_first_plan["optimizer"], "paper_exact")
        self.assertLessEqual(
            low_first_plan["draft_lengths"][0],
            high_first_plan["draft_lengths"][0],
        )

    def test_dip_sd_trace_uses_optimizer_batch_assignment(self) -> None:
        config, _, workload = dip_sd_trace_config(num_requests=3, output_len=6)

        result = Simulator(
            config,
            accepting_model_runner(),
            workload,
            "combined_strong_heterogeneous",
            "dip_sd",
        ).run()

        plan = first_event(result, "dip_sd_epoch_plan")
        verifies = epoch_events(result, "dip_sd_batch_verify", 0)
        self.assertEqual(len(verifies), len(plan["batches"]))
        for event in verifies:
            batch_index = event["batch_index"]
            self.assertEqual(event["request_ids"], plan["batches"][batch_index])
            self.assertEqual(event["optimizer_batch"], plan["batches"][batch_index])
            for request_id in event["request_ids"]:
                self.assertEqual(plan["assignment"][request_id], batch_index)

    def test_dip_sd_trace_uses_optimizer_assignment(self) -> None:
        result = run_four_request_dip_sd_trace()

        plan = first_event(result, "dip_sd_epoch_plan")
        verifies = epoch_events(result, "dip_sd_batch_verify", 0)
        drafts = epoch_events(result, "dip_sd_draft", 0)

        self.assertEqual(len(plan["batches"]), 2)
        self.assertTrue(all(len(batch) == 2 for batch in plan["batches"]))
        self.assertEqual(len(verifies), 2)
        for verify in verifies:
            batch_index = verify["batch_index"]
            self.assertEqual(verify["request_ids"], plan["batches"][batch_index])
            self.assertEqual(verify["optimizer_batch"], plan["batches"][batch_index])
        for draft in drafts:
            self.assertEqual(
                draft["batch_index"],
                plan["assignment"][draft["request_id"]],
            )

    def test_dip_sd_trace_uses_per_request_draft_lengths(self) -> None:
        config, _, workload = dip_sd_trace_config(num_requests=3, output_len=8)

        result = Simulator(
            config,
            accepting_model_runner(),
            workload,
            "combined_strong_heterogeneous",
            "dip_sd",
        ).run()

        plan = first_event(result, "dip_sd_epoch_plan")
        drafts = epoch_events(result, "dip_sd_draft", 0)
        self.assertEqual(len(drafts), len(plan["draft_lengths"]))
        for event in drafts:
            self.assertEqual(
                event["scheduled_gamma"],
                plan["draft_lengths"][event["request_id"]],
            )

    def test_dip_sd_trace_uses_per_request_draft_length(self) -> None:
        result = run_four_request_dip_sd_trace()

        plan = first_event(result, "dip_sd_epoch_plan")
        drafts = epoch_events(result, "dip_sd_draft", 0)

        self.assertGreater(len(set(plan["draft_lengths"].values())), 1)
        self.assertEqual(len(drafts), len(plan["draft_lengths"]))
        for draft in drafts:
            request_id = draft["request_id"]
            self.assertEqual(draft["scheduled_gamma"], plan["draft_lengths"][request_id])
            self.assertGreaterEqual(draft["finish_time_ms"], draft["start_time_ms"])

    def test_dip_sd_slow_member_blocks_assigned_batch(self) -> None:
        config, _, workload = dip_sd_trace_config(num_requests=3, output_len=6)
        split_heterogeneous_devices(config, draft_rates=(1000, 1, 1000))

        result = Simulator(
            config,
            accepting_model_runner(),
            workload,
            "combined_strong_heterogeneous",
            "dip_sd",
        ).run()

        drafts = {
            event["segment_id"]: event
            for event in epoch_events(result, "dip_sd_draft", 0)
        }
        multi_request_verify = next(
            event
            for event in epoch_events(result, "dip_sd_batch_verify", 0)
            if event["batch_size"] > 1
        )
        member_ready_times = [
            drafts[segment_id]["finish_time_ms"] + drafts[segment_id]["uplink_ms"]
            for segment_id in multi_request_verify["segment_ids"]
        ]

        self.assertGreater(max(member_ready_times), min(member_ready_times))
        self.assertGreaterEqual(
            multi_request_verify["start_time_ms"],
            max(member_ready_times),
        )

    def test_dip_sd_slow_member_blocks_own_batch(self) -> None:
        result = run_four_request_dip_sd_trace()

        plan = first_event(result, "dip_sd_epoch_plan")
        slow_request_id = 0
        slow_batch_index = plan["assignment"][slow_request_id]
        drafts = {
            event["segment_id"]: event
            for event in epoch_events(result, "dip_sd_draft", 0)
        }
        slow_batch_verify = next(
            event
            for event in epoch_events(result, "dip_sd_batch_verify", 0)
            if event["batch_index"] == slow_batch_index
        )
        member_ready_times = [
            drafts[segment_id]["finish_time_ms"] + drafts[segment_id]["uplink_ms"]
            for segment_id in slow_batch_verify["segment_ids"]
        ]

        self.assertIn(slow_request_id, slow_batch_verify["request_ids"])
        self.assertGreater(max(member_ready_times), min(member_ready_times))
        self.assertAlmostEqual(
            slow_batch_verify["start_time_ms"],
            max(member_ready_times),
            places=6,
        )

    def test_dip_sd_other_batches_overlap_drafting(self) -> None:
        config, _, workload = dip_sd_trace_config(num_requests=3, output_len=6)
        split_heterogeneous_devices(config, draft_rates=(1000, 1, 1000))

        result = Simulator(
            config,
            accepting_model_runner(),
            workload,
            "combined_strong_heterogeneous",
            "dip_sd",
        ).run()

        first_verify = epoch_events(result, "dip_sd_batch_verify", 0)[0]
        later_batch_drafts = [
            event
            for event in epoch_events(result, "dip_sd_draft", 0)
            if event["batch_index"] > first_verify["batch_index"]
        ]

        self.assertTrue(
            any(
                event["start_time_ms"] < first_verify["finish_time_ms"]
                and event["finish_time_ms"] > first_verify["start_time_ms"]
                for event in later_batch_drafts
            )
        )

    def test_dip_sd_other_batch_can_continue_drafting(self) -> None:
        result = run_four_request_dip_sd_trace()

        plan = first_event(result, "dip_sd_epoch_plan")
        slow_batch_index = plan["assignment"][0]
        slow_verify = next(
            event
            for event in epoch_events(result, "dip_sd_batch_verify", 0)
            if event["batch_index"] == slow_batch_index
        )
        other_batch_drafts = [
            event
            for event in epoch_events(result, "dip_sd_draft", 0)
            if event["batch_index"] != slow_batch_index
        ]

        self.assertTrue(other_batch_drafts)
        self.assertTrue(
            all(event["start_time_ms"] < slow_verify["start_time_ms"] for event in other_batch_drafts)
        )
        self.assertTrue(
            any(event["finish_time_ms"] < slow_verify["start_time_ms"] for event in other_batch_drafts)
        )

    def test_dip_sd_verification_follows_paper_batch_order(self) -> None:
        config, _, workload = dip_sd_trace_config(num_requests=3, output_len=6)

        result = Simulator(
            config,
            accepting_model_runner(),
            workload,
            "combined_strong_heterogeneous",
            "dip_sd",
        ).run()

        verifies = epoch_events(result, "dip_sd_batch_verify", 0)
        self.assertEqual(
            [event["batch_index"] for event in verifies],
            list(range(len(verifies))),
        )
        for previous, current in zip(verifies, verifies[1:]):
            self.assertGreaterEqual(
                current["start_time_ms"],
                previous["finish_time_ms"],
            )

    def test_dip_sd_verification_follows_batch_order(self) -> None:
        result = run_four_request_dip_sd_trace()

        verifies = epoch_events(result, "dip_sd_batch_verify", 0)

        self.assertEqual([event["batch_index"] for event in verifies], [0, 1])
        for previous, current in zip(verifies, verifies[1:]):
            self.assertGreaterEqual(
                current["start_time_ms"],
                previous["finish_time_ms"],
            )

    def test_dip_sd_request_waits_for_verification_before_redraft(self) -> None:
        config, _, workload = dip_sd_trace_config(num_requests=1, output_len=6)
        config["dip_sd"]["batch_count"] = 1
        config["dip_sd"]["max_batch_size"] = 1

        result = Simulator(
            config,
            accepting_model_runner(),
            workload,
            "combined_strong_heterogeneous",
            "dip_sd",
        ).run()

        drafts = [event for event in result.event_trace if event["event"] == "dip_sd_draft"]
        results = [event for event in result.event_trace if event["event"] == "dip_sd_result"]
        self.assertGreater(len(drafts), 1)
        for previous_result, next_draft in zip(results, drafts[1:]):
            self.assertGreaterEqual(
                next_draft["start_time_ms"],
                previous_result["finish_time_ms"],
            )

    def test_dip_sd_request_waits_for_verify_and_kv_update(self) -> None:
        result = run_four_request_dip_sd_trace()
        results_by_segment = {
            event["segment_id"]: event
            for event in result.event_trace
            if event["event"] == "dip_sd_result"
        }

        for request in result.requests:
            drafts = [
                event
                for event in result.event_trace
                if event["event"] == "dip_sd_draft"
                and event["request_id"] == request.request_id
            ]
            for previous_draft, next_draft in zip(drafts, drafts[1:]):
                previous_result = results_by_segment[previous_draft["segment_id"]]
                self.assertGreaterEqual(
                    next_draft["start_time_ms"],
                    previous_result["finish_time_ms"],
                )

    def test_dip_sd_batch_verification_contains_multiple_requests(self) -> None:
        config, _, workload = dip_sd_trace_config(num_requests=3, output_len=6)

        result = Simulator(
            config,
            accepting_model_runner(),
            workload,
            "combined_strong_heterogeneous",
            "dip_sd",
        ).run()

        self.assertTrue(
            any(
                event["batch_size"] > 1
                for event in epoch_events(result, "dip_sd_batch_verify", 0)
            )
        )

    def test_dip_sd_trace_span_matches_optimizer_model(self) -> None:
        config, _, workload = dip_sd_trace_config(num_requests=3, output_len=6)

        result = Simulator(
            config,
            accepting_model_runner(),
            workload,
            "combined_strong_heterogeneous",
            "dip_sd",
        ).run()

        plan = first_event(result, "dip_sd_epoch_plan")
        verifies = epoch_events(result, "dip_sd_batch_verify", 0)
        actual_verify_stage_span = (
            verifies[-1]["finish_time_ms"] - verifies[0]["start_time_ms"]
        )

        self.assertAlmostEqual(
            actual_verify_stage_span,
            plan["pipeline_span"],
            delta=1e-3,
        )

    def test_request_waits_for_sync_before_redraft(self) -> None:
        config, _, workload = small_config(num_requests=1, output_len=6)
        config["dip_sd"]["batch_count"] = 1
        config["dip_sd"]["draft_length"] = 2

        result = Simulator(
            config,
            accepting_model_runner(),
            workload,
            "combined_strong_heterogeneous",
            "dip_sd",
        ).run()

        drafts = [event for event in result.event_trace if event["event"] == "dip_sd_draft"]
        results = [event for event in result.event_trace if event["event"] == "dip_sd_result"]
        self.assertGreater(len(drafts), 1)
        for previous_result, next_draft in zip(results, drafts[1:]):
            self.assertGreaterEqual(
                next_draft["start_time_ms"],
                previous_result["finish_time_ms"],
            )

    def test_new_arrivals_wait_until_epoch_barrier(self) -> None:
        config, model_runner, workload = small_config(num_requests=2, output_len=4)
        config["simulation"]["request_arrival"] = "poisson"
        config["simulation"]["poisson_rate_per_s"] = 1000
        config["dip_sd"]["max_active_requests"] = 1
        config["dip_sd"]["batch_count"] = 1
        config["dip_sd"]["draft_length"] = 1

        result = Simulator(
            config,
            model_runner,
            workload,
            "combined_strong_heterogeneous",
            "dip_sd",
        ).run()

        admissions = [event for event in result.event_trace if event["event"] == "dip_sd_admit"]
        self.assertEqual([event["request_id"] for event in admissions], [0, 1])
        self.assertGreater(admissions[1]["epoch"], admissions[0]["epoch"])
        first_request_results = [
            event
            for event in result.event_trace
            if event["event"] == "dip_sd_result" and event["request_id"] == 0
        ]
        second_first_draft = next(
            event
            for event in result.event_trace
            if event["event"] == "dip_sd_draft" and event["request_id"] == 1
        )
        self.assertGreaterEqual(
            second_first_draft["start_time_ms"],
            first_request_results[0]["finish_time_ms"],
        )

    def test_dip_sd_online_arrival_waits_for_epoch_boundary(self) -> None:
        config, model_runner, workload = small_config(num_requests=2, output_len=4)
        config["simulation"]["request_arrival"] = "poisson"
        config["simulation"]["poisson_rate_per_s"] = 1000
        config["dip_sd"]["max_active_requests"] = 1
        config["dip_sd"]["batch_count"] = 1
        config["dip_sd"]["draft_length"] = 1

        result = Simulator(
            config,
            model_runner,
            workload,
            "combined_strong_heterogeneous",
            "dip_sd",
        ).run()

        admissions = [event for event in result.event_trace if event["event"] == "dip_sd_admit"]
        self.assertEqual([event["request_id"] for event in admissions], [0, 1])
        self.assertGreater(admissions[1]["epoch"], admissions[0]["epoch"])
        prior_epoch_results = [
            event
            for event in result.event_trace
            if event["event"] == "dip_sd_result"
            and event["epoch"] < admissions[1]["epoch"]
        ]
        self.assertTrue(prior_epoch_results)
        self.assertGreaterEqual(
            admissions[1]["time_ms"],
            max(event["finish_time_ms"] for event in prior_epoch_results),
        )
        second_request_drafts = [
            event
            for event in result.event_trace
            if event["event"] == "dip_sd_draft" and event["request_id"] == 1
        ]
        self.assertTrue(second_request_drafts)
        self.assertGreaterEqual(
            second_request_drafts[0]["start_time_ms"],
            admissions[1]["time_ms"],
        )

    def test_dip_sd_has_one_unverified_draft_per_request(self) -> None:
        config, _, workload = small_config(num_requests=2, output_len=6)
        config["dip_sd"]["batch_count"] = 2
        config["dip_sd"]["draft_length"] = 2

        result = Simulator(
            config,
            accepting_model_runner(),
            workload,
            "combined_strong_heterogeneous",
            "dip_sd",
        ).run()

        for request in result.requests:
            drafts = [
                event
                for event in result.event_trace
                if event["event"] == "dip_sd_draft" and event["request_id"] == request.request_id
            ]
            results = [
                event
                for event in result.event_trace
                if event["event"] == "dip_sd_result" and event["request_id"] == request.request_id
            ]
            for previous_result, next_draft in zip(results, drafts[1:]):
                self.assertGreaterEqual(
                    next_draft["start_time_ms"],
                    previous_result["finish_time_ms"],
                )

    def test_dip_sd_one_unverified_segment_per_request(self) -> None:
        result = run_four_request_dip_sd_trace()
        result_finish_by_segment = {
            event["segment_id"]: event["finish_time_ms"]
            for event in result.event_trace
            if event["event"] == "dip_sd_result"
        }

        for request in result.requests:
            intervals = [
                (
                    event["start_time_ms"],
                    result_finish_by_segment[event["segment_id"]],
                    event["segment_id"],
                )
                for event in result.event_trace
                if event["event"] == "dip_sd_draft"
                and event["request_id"] == request.request_id
            ]
            intervals.sort()
            for previous, current in zip(intervals, intervals[1:]):
                self.assertGreaterEqual(
                    current[0],
                    previous[1],
                    f"request {request.request_id} has overlapping unverified segments "
                    f"{previous[2]} and {current[2]}",
                )

    def test_dip_sd_output_equals_target_only(self) -> None:
        config, _, workload = small_config(num_requests=3, output_len=5)
        config["dip_sd"]["batch_count"] = 2
        config["dip_sd"]["draft_length"] = 2

        for model_runner in (accepting_model_runner(), rejecting_model_runner()):
            target = Simulator(
                config,
                model_runner,
                workload,
                "combined_strong_heterogeneous",
                "target_only",
            ).run()
            dip_sd = Simulator(
                config,
                model_runner,
                workload,
                "combined_strong_heterogeneous",
                "dip_sd",
            ).run()

            self.assertEqual(
                [request.generated_ids for request in dip_sd.requests],
                [request.generated_ids for request in target.requests],
            )


def dip_sd_trace_config(
    *,
    num_requests: int,
    output_len: int,
) -> tuple[dict, object, list]:
    config, model_runner, workload = small_config(
        num_requests=num_requests,
        output_len=output_len,
    )
    config["dip_sd"]["batch_count"] = 2
    config["dip_sd"]["draft_length"] = 1
    config["dip_sd"]["min_draft_length"] = 1
    config["dip_sd"]["max_draft_length"] = 3
    config["dip_sd"]["max_batch_size"] = 2
    config["edge"]["verify_startup_ms"] = 1
    config["edge"]["target_only_token_rate_tok_s"] = 1_000_000_000_000
    config["network"]["packet_header_bytes"] = 0
    config["network"]["packet_token_bytes"] = 0
    for pool in config["device_pools"].values():
        for template in pool["templates"].values():
            template["draft_startup_ms"] = 0
            template["draft_token_rate_tok_s"] = 1000
            template["uplink_mbps"] = 1000
            template["downlink_mbps"] = 1000
            template["rtt_ms"] = 0
            template["jitter_ms"] = 0
    return config, model_runner, workload


def split_heterogeneous_devices(
    config: dict,
    *,
    draft_rates: tuple[int, int, int],
) -> None:
    templates = config["device_pools"]["heterogeneous"]["templates"]
    for template in templates.values():
        template["count"] = 0
    for name, draft_rate in zip(("low_end", "mid_end", "high_end"), draft_rates):
        templates[name]["count"] = 1
        templates[name]["draft_token_rate_tok_s"] = draft_rate


def four_request_dip_sd_trace_config() -> tuple[dict, object, list]:
    config, model_runner, workload = dip_sd_trace_config(num_requests=4, output_len=8)
    config["simulation"]["num_devices"] = 4
    config["dip_sd"]["batch_count"] = 2
    config["dip_sd"]["max_batch_size"] = 2
    config["dip_sd"]["min_draft_length"] = 1
    config["dip_sd"]["max_draft_length"] = 3
    config["network"]["packet_header_bytes"] = 0
    config["network"]["packet_token_bytes"] = 0
    for profile in config["drafter_profiles"].values():
        profile["acceptance_prior"] = 0.9
    heterogeneous = config["device_pools"]["heterogeneous"]["templates"]
    for template in heterogeneous.values():
        template["count"] = 0
        template["draft_startup_ms"] = 0
        template["uplink_mbps"] = 1000
        template["downlink_mbps"] = 1000
        template["rtt_ms"] = 0
        template["jitter_ms"] = 0
    heterogeneous["low_end"]["count"] = 1
    heterogeneous["low_end"]["draft_token_rate_tok_s"] = 1
    heterogeneous["mid_end"]["count"] = 2
    heterogeneous["mid_end"]["draft_token_rate_tok_s"] = 1000
    heterogeneous["high_end"]["count"] = 1
    heterogeneous["high_end"]["draft_token_rate_tok_s"] = 1000
    config["device_pools"]["medium_only"]["templates"]["medium"]["count"] = 4
    return config, model_runner, workload


def run_four_request_dip_sd_trace():
    config, _, workload = four_request_dip_sd_trace_config()
    return Simulator(
        config,
        accepting_model_runner(),
        workload,
        "combined_strong_heterogeneous",
        "dip_sd",
    ).run()


def first_event(result, event_name: str) -> dict:
    return next(event for event in result.event_trace if event["event"] == event_name)


def epoch_events(result, event_name: str, epoch: int) -> list[dict]:
    return [
        event
        for event in result.event_trace
        if event["event"] == event_name and event["epoch"] == epoch
    ]


def paper_user(
    request_id: int,
    *,
    alpha: float = 0.8,
    prefix_length: int = 10,
    communication: float = 0.0,
    draft_overhead: float = 1.0,
) -> DipSDUser:
    return DipSDUser(
        request_id=request_id,
        prefix_length=prefix_length,
        acceptance_estimate=alpha,
        communication_latency_ms=communication,
        draft_latency_scale=0.0,
        draft_latency_overhead_ms=draft_overhead,
        draft_model=DipSDModelProfile(decoder_blocks=1, hidden_size=1, ffn_hidden_size=1),
    )


def paper_problem(
    *,
    users: tuple[DipSDUser, ...] | None = None,
    min_draft_length: int = 1,
    max_draft_length: int = 3,
    max_batch_count: int = 2,
    max_batch_size: int = 2,
) -> DipSDProblem:
    return DipSDProblem(
        users=users
        or (
            paper_user(0, alpha=0.6, prefix_length=8, communication=0.3),
            paper_user(1, alpha=0.8, prefix_length=10, communication=0.1),
            paper_user(2, alpha=0.4, prefix_length=12, communication=0.2),
        ),
        target_model=DipSDModelProfile(decoder_blocks=1, hidden_size=1, ffn_hidden_size=1),
        target_latency_scale=0.0,
        target_latency_overhead_ms=1.0,
        target_memory_cap=10_000.0,
        min_draft_length=min_draft_length,
        max_draft_length=max_draft_length,
        initial_draft_length=min_draft_length,
        max_batch_count=max_batch_count,
        max_batch_size=max_batch_size,
    )


def brute_force_plan(problem: DipSDProblem):
    users = tuple(sorted(problem.users, key=lambda user: user.request_id))
    request_ids = tuple(user.request_id for user in users)
    if len(users) == 1:
        batch_counts = (1,)
    else:
        upper = min(len(users), problem.max_batch_count or len(users))
        lower = max(2, (len(users) + int(problem.max_batch_size or len(users)) - 1) // int(problem.max_batch_size or len(users)))
        batch_counts = tuple(range(lower, upper + 1))
    best = None
    for batch_count in batch_counts:
        for labels in product(range(batch_count), repeat=len(request_ids)):
            if set(labels) != set(range(batch_count)):
                continue
            batches = tuple(
                tuple(
                    request_id
                    for request_id, label in zip(request_ids, labels)
                    if label == batch_index
                )
                for batch_index in range(batch_count)
            )
            for values in product(
                range(problem.min_draft_length, problem.max_draft_length + 1),
                repeat=len(request_ids),
            ):
                lengths = dict(zip(request_ids, values))
                try:
                    candidate = evaluate_plan(problem, batches, lengths)
                except ValueError:
                    continue
                if best is None or (
                    candidate.objective,
                    -candidate.pipeline_span,
                    -len(candidate.batches),
                    tuple(tuple(-item for item in batch) for batch in candidate.batches),
                ) > (
                    best.objective,
                    -best.pipeline_span,
                    -len(best.batches),
                    tuple(tuple(-item for item in batch) for batch in best.batches),
                ):
                    best = candidate
    assert best is not None
    return best


if __name__ == "__main__":
    unittest.main()
