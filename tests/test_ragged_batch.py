import random
import unittest

from edge_spec.backends import FakeBackend
from edge_spec.types import SamplingConfig


class RaggedBatchTests(unittest.TestCase):
    def test_fake_batch_matches_individual_for_ragged_prefixes(self):
        backend = FakeBackend("target", seed=11, vocab_size=12)
        sampling = SamplingConfig(temperature=1.0, top_p=1.0, top_k=0)
        prefixes = [[1, 2, 3], [1, 4, 5, 6, 7]]
        drafts = [[3, 4], [8]]
        batch, _ = backend.target_distributions(prefixes, drafts, sampling)
        first, _ = backend.target_distributions([prefixes[0]], [drafts[0]], sampling)
        second, _ = backend.target_distributions([prefixes[1]], [drafts[1]], sampling)
        self.assertEqual(batch[0][0].as_dict(), first[0][0].as_dict())
        self.assertEqual(batch[0][2].as_dict(), first[0][2].as_dict())
        self.assertEqual(batch[1][0].as_dict(), second[0][0].as_dict())
        self.assertEqual(batch[1][1].as_dict(), second[0][1].as_dict())

    def test_target_only_generates(self):
        backend = FakeBackend("target", seed=11, vocab_size=12)
        ids, elapsed = backend.generate_target_only(
            [1, 2, 3],
            4,
            SamplingConfig(temperature=1.0, top_p=1.0, top_k=0),
            random.Random(1),
        )
        self.assertLessEqual(len(ids), 4)
        self.assertGreaterEqual(elapsed, 0)


if __name__ == "__main__":
    unittest.main()
