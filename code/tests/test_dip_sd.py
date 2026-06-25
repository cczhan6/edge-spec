from __future__ import annotations

import unittest
from itertools import product

from src.dip_sd import (
    DipSDModelProfile,
    DipSDProblem,
    DipSDUser,
    build_fixed_epoch_plan,
    evaluate_plan,
    optimize_dip_sd,
    optimize_epoch_plan,
)
from src.methods import get_method_spec
from src.simulator import Simulator
from tests.common import accepting_model_runner, rejecting_model_runner, small_config


class DipSDTest(unittest.TestCase):
    def test_dip_sd_methods_are_registered(self) -> None:
        config, _, _ = small_config(num_requests=1, output_len=4)

        greedy = get_method_spec("dip_sd_greedy", config)
        optimized = get_method_spec("dip_sd", config)

        self.assertEqual(greedy.runtime, "dip_sd")
        self.assertEqual(greedy.candidate_strategy, "linear")
        self.assertEqual(optimized.runtime, "dip_sd")
        self.assertEqual(optimized.candidate_strategy, "linear")

    def test_fixed_epoch_partition_complete_disjoint_non_empty(self) -> None:
        plan = build_fixed_epoch_plan(
            [3, 1, 2, 0],
            batch_count=2,
            draft_length=2,
            min_draft_length=1,
            max_draft_length=4,
            max_batch_size=3,
        )

        flattened = [request_id for batch in plan.batches for request_id in batch]
        self.assertEqual(sorted(flattened), [0, 1, 2, 3])
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertTrue(all(batch for batch in plan.batches))
        self.assertEqual(set(plan.draft_lengths), {0, 1, 2, 3})
        self.assertTrue(all(length == 2 for length in plan.draft_lengths.values()))

    def test_dip_sd_batch_order_is_fixed_cyclic(self) -> None:
        config, model_runner, workload = small_config(num_requests=3, output_len=4)
        config["dip_sd"]["batch_count"] = 2
        config["dip_sd"]["draft_length"] = 1

        result = Simulator(
            config,
            model_runner,
            workload,
            "combined_strong_heterogeneous",
            "dip_sd_greedy",
        ).run()

        verify_events = [
            event for event in result.event_trace if event["event"] == "dip_sd_batch_verify"
        ]
        first_epoch = [event for event in verify_events if event["epoch"] == 0]
        self.assertEqual([event["batch_index"] for event in first_epoch], [0, 1])

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
        low = optimize_epoch_plan(
            [0],
            acceptance_estimates={0: 0.1},
            max_batch_count=1,
            min_draft_length=1,
            max_draft_length=4,
            max_batch_size=1,
        )
        high = optimize_epoch_plan(
            [0],
            acceptance_estimates={0: 0.95},
            max_batch_count=1,
            min_draft_length=1,
            max_draft_length=4,
            max_batch_size=1,
        )

        self.assertLessEqual(low.draft_lengths[0], high.draft_lengths[0])

        config, _, workload = small_config(num_requests=1, output_len=6)
        config["dip_sd"]["batch_count"] = 1
        config["dip_sd"]["min_draft_length"] = 1
        config["dip_sd"]["max_draft_length"] = 4
        config["drafter_profiles"]["small"]["acceptance_prior"] = 0.95
        result = Simulator(
            config,
            rejecting_model_runner(),
            workload,
            "combined_strong_heterogeneous",
            "dip_sd",
        ).run()
        first_plan = next(event for event in result.event_trace if event["event"] == "dip_sd_epoch_plan")

        self.assertEqual(first_plan["optimizer"], "paper_exact")
        self.assertEqual(first_plan["draft_lengths"][0], high.draft_lengths[0])

    def test_request_waits_for_sync_before_redraft(self) -> None:
        config, _, workload = small_config(num_requests=1, output_len=6)
        config["dip_sd"]["batch_count"] = 1
        config["dip_sd"]["draft_length"] = 2

        result = Simulator(
            config,
            accepting_model_runner(),
            workload,
            "combined_strong_heterogeneous",
            "dip_sd_greedy",
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
            "dip_sd_greedy",
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

    def test_dip_sd_has_one_unverified_draft_per_request(self) -> None:
        config, _, workload = small_config(num_requests=2, output_len=6)
        config["dip_sd"]["batch_count"] = 2
        config["dip_sd"]["draft_length"] = 2

        result = Simulator(
            config,
            accepting_model_runner(),
            workload,
            "combined_strong_heterogeneous",
            "dip_sd_greedy",
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
