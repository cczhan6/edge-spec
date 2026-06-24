from __future__ import annotations

import unittest
from dataclasses import fields

from src.model_runner import FakeModelRunner, VerificationResult
from src.simulator import Simulator
from tests.common import accepting_model_runner, rejecting_model_runner, small_config


class LinearSpeculativeDecodingCoreTest(unittest.TestCase):
    def test_verification_result_exposes_contract_fields(self) -> None:
        field_names = {field.name for field in fields(VerificationResult)}

        self.assertIn("accepted_count", field_names)
        self.assertIn("committed_tokens", field_names)
        self.assertIn("correction_token", field_names)
        self.assertIn("bonus_token", field_names)

    def test_linear_verify_all_accepts_returns_bonus(self) -> None:
        model_runner = accepting_model_runner()

        result = model_runner.verify([1], [2, 3])

        self.assertEqual(result.accepted_count, 2)
        self.assertEqual(result.committed_tokens, [2, 3, 4])
        self.assertIsNone(result.correction_token)
        self.assertEqual(result.bonus_token, 4)
        self.assertEqual(result.emitted_ids, result.committed_tokens)
        self.assertFalse(result.rejected)

    def test_linear_verify_rejects_first_mismatch_returns_correction(self) -> None:
        model_runner = FakeModelRunner(
            target_token_fn=lambda prefix: 7,
            draft_token_fn=lambda profile, prefix: 3,
        )

        result = model_runner.verify([1], [3, 3])

        self.assertEqual(result.accepted_count, 0)
        self.assertEqual(result.committed_tokens, [7])
        self.assertEqual(result.correction_token, 7)
        self.assertIsNone(result.bonus_token)
        self.assertEqual(result.emitted_ids, result.committed_tokens)
        self.assertTrue(result.rejected)

    def test_batch_linear_verify_matches_individual_verify(self) -> None:
        model_runner = accepting_model_runner()
        prefixes = [[1], [4]]
        drafts = [[2, 3], [5, 6]]

        batch_results = model_runner.verify_batch(
            [
                type("VerifyInput", (), {"prefix_ids": prefix, "draft_ids": draft})()
                for prefix, draft in zip(prefixes, drafts)
            ]
        )
        individual_results = [
            model_runner.verify(prefix, draft)
            for prefix, draft in zip(prefixes, drafts)
        ]

        self.assertEqual(batch_results, individual_results)

    def test_linear_speculative_output_equals_target_only(self) -> None:
        config, _, workload = small_config(num_requests=3, output_len=9)
        config["speculation"]["gamma_fixed"] = 3
        config["speculation"]["gamma_candidates"] = [3]
        model_runner = accepting_model_runner()

        target = Simulator(
            config,
            model_runner,
            workload,
            "combined_strong_heterogeneous",
            "target_only",
        ).run()
        speculative = Simulator(
            config,
            model_runner,
            workload,
            "combined_strong_heterogeneous",
            "sync_batch_sd",
        ).run()

        self.assertEqual(
            [request.generated_ids for request in speculative.requests],
            [request.generated_ids for request in target.requests],
        )

    def test_rejected_draft_tokens_are_not_committed(self) -> None:
        config, _, workload = small_config(num_requests=2, output_len=5)
        config["speculation"]["gamma_fixed"] = 2
        config["speculation"]["gamma_candidates"] = [2]
        model_runner = rejecting_model_runner()

        result = Simulator(
            config,
            model_runner,
            workload,
            "combined_strong_heterogeneous",
            "sync_batch_sd",
        ).run()

        self.assertTrue(result.segments)
        self.assertTrue(all(token == 1 for request in result.requests for token in request.generated_ids))
        self.assertTrue(all(2 not in request.generated_ids for request in result.requests))
        self.assertTrue(all(segment.accepted_count == 0 for segment in result.segments))
        self.assertTrue(all(segment.emitted_ids == [1] for segment in result.segments))

    def test_max_output_len_truncates_bonus_without_extra_commit(self) -> None:
        config, _, workload = small_config(num_requests=1, output_len=1)
        config["speculation"]["gamma_fixed"] = 1
        config["speculation"]["gamma_candidates"] = [1]
        model_runner = accepting_model_runner()

        result = Simulator(
            config,
            model_runner,
            workload,
            "combined_strong_heterogeneous",
            "sync_batch_sd",
        ).run()

        self.assertEqual(len(result.requests[0].generated_ids), 1)


if __name__ == "__main__":
    unittest.main()
