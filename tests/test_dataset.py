import unittest

from edge_spec.dataset import (
    iter_microbatches,
    select_one_per_category_per_device,
)
from edge_spec.types import SpecBenchItem


class DatasetSelectionTests(unittest.TestCase):
    def test_one_per_category_per_device_keeps_category_batches(self):
        items = [
            SpecBenchItem(f"a-{index}", f"prompt {index}", category="a")
            for index in range(4)
        ] + [
            SpecBenchItem(f"b-{index}", f"prompt {index}", category="b")
            for index in range(3)
        ]
        selected = select_one_per_category_per_device(items, device_count=3)
        self.assertEqual([item.request_id for item in selected], [
            "a-0",
            "a-1",
            "a-2",
            "b-0",
            "b-1",
            "b-2",
        ])
        batches = list(iter_microbatches(selected, batch_size=3))
        self.assertEqual([item.category for item in batches[0]], ["a", "a", "a"])
        self.assertEqual([item.category for item in batches[1]], ["b", "b", "b"])

    def test_one_per_category_per_device_requires_enough_samples(self):
        items = [SpecBenchItem("a-0", "prompt", category="a")]
        with self.assertRaises(ValueError):
            select_one_per_category_per_device(items, device_count=3)

    def test_regular_microbatches_keep_partial_last_batch_by_default(self):
        items = [
            SpecBenchItem(f"a-{index}", f"prompt {index}", category="a")
            for index in range(5)
        ]
        batches = list(iter_microbatches(items, batch_size=3))
        self.assertEqual([len(batch) for batch in batches], [3, 2])


if __name__ == "__main__":
    unittest.main()
