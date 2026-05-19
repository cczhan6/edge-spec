import random
import unittest

from edge_spec.sampling import (
    apply_top_k_top_p_py,
    sample_from_probs,
    verify_draft_exact,
)
from edge_spec.types import SamplingConfig, SparseProb


class SamplingTests(unittest.TestCase):
    def test_top_k_top_p_normalizes(self):
        probs = apply_top_k_top_p_py(
            [0.4, 0.3, 0.2, 0.1], SamplingConfig(temperature=1.0, top_p=0.7, top_k=3)
        )
        self.assertAlmostEqual(sum(probs), 1.0)
        self.assertEqual(sum(1 for p in probs if p > 0), 2)

    def test_exact_speculative_sampling_matches_target_distribution(self):
        rng = random.Random(123)
        target = SparseProb([0, 1, 2], [0.50, 0.30, 0.20])
        draft = SparseProb([0, 1, 2], [0.20, 0.50, 0.30])
        counts = {0: 0, 1: 0, 2: 0}
        runs = 20000
        for _ in range(runs):
            draft_token = sample_from_probs(draft.as_dict(), rng)
            result = verify_draft_exact(
                [draft_token],
                [draft],
                [target, target],
                rng,
            )
            counts[result.emitted_ids[0]] += 1
        observed = {token: count / runs for token, count in counts.items()}
        for token, expected in target.as_dict().items():
            self.assertLess(abs(observed[token] - expected), 0.025)


if __name__ == "__main__":
    unittest.main()
