import unittest

from edge_spec.dataset import (
    iter_microbatches,
    load_specbench,
    select_one_per_category_per_device,
    select_one_per_category,
)
from edge_spec.types import SpecBenchItem


class DatasetSelectionTests(unittest.TestCase):
    def test_load_specbench_maps_raw_categories_to_six_task_groups(self):
        items = load_specbench("data/spec_bench/question.jsonl")
        categories = {item.category for item in items}
        self.assertEqual(categories, {"Sum", "Math", "MT", "QA", "RAG", "Trans"})
        counts = {category: 0 for category in categories}
        for item in items:
            counts[item.category] += 1
        self.assertEqual(counts, {
            "Sum": 80,
            "Math": 80,
            "MT": 80,
            "QA": 80,
            "RAG": 80,
            "Trans": 80,
        })

    def test_one_per_category_uses_six_task_groups(self):
        items = load_specbench("data/spec_bench/question.jsonl")
        selected = select_one_per_category(items)
        self.assertEqual([item.category for item in selected], [
            "Sum",
            "Math",
            "MT",
            "QA",
            "RAG",
            "Trans",
        ])

    def test_load_specbench_filters_one_six_task_group(self):
        items = load_specbench("data/spec_bench/question.jsonl", category="translation")
        self.assertEqual(len(items), 80)
        self.assertEqual({item.category for item in items}, {"Trans"})

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
