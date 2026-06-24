from __future__ import annotations

import unittest

from src.dip_sd import build_fixed_epoch_plan, optimize_epoch_plan
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
        kwargs = dict(
            active_request_ids=[0, 1, 2],
            acceptance_estimates={0: 0.6, 1: 0.8, 2: 0.4},
            max_batch_count=2,
            min_draft_length=1,
            max_draft_length=3,
            max_batch_size=2,
        )

        first = optimize_epoch_plan(**kwargs)
        second = optimize_epoch_plan(**kwargs)

        self.assertEqual(first, second)
        self.assertEqual(first.optimizer, "deterministic_search")

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

        self.assertEqual(first_plan["optimizer"], "deterministic_search")
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


if __name__ == "__main__":
    unittest.main()
