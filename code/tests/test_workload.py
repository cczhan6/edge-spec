from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from src.workload import extract_prompt, load_workload, specbench_category_group


class WorkloadTest(unittest.TestCase):
    def test_extracts_specbench_first_turn(self) -> None:
        value = {"question_id": 1, "turns": ["first turn", "follow-up turn"]}
        self.assertEqual(extract_prompt(value, 1), "first turn")

    def test_seeded_selection_is_stable_and_without_replacement(self) -> None:
        first = load_workload("data/spec_bench/question.jsonl", 8, 42, len)
        second = load_workload("data/spec_bench/question.jsonl", 8, 42, len)
        self.assertEqual(first, second)
        self.assertEqual(len({item.prompt_id for item in first}), 8)

    def test_balanced_category_sampling_draws_each_top_level_category(self) -> None:
        workload = load_workload(
            "data/spec_bench/question.jsonl",
            200,
            42,
            len,
            samples_per_category=2,
        )
        self.assertEqual(len(workload), 12)
        self.assertEqual(
            Counter(item.category_group for item in workload),
            {
                "MT": 2,
                "QA": 2,
                "Math": 2,
                "RAG": 2,
                "Sum": 2,
                "Trans": 2,
            },
        )
        self.assertEqual(len({item.prompt_id for item in workload}), 12)

    def test_maps_specbench_subcategories_to_six_top_level_categories(self) -> None:
        self.assertEqual(specbench_category_group("writing"), "MT")
        self.assertEqual(specbench_category_group("coding"), "MT")
        self.assertEqual(specbench_category_group("qa"), "QA")
        self.assertEqual(specbench_category_group("math_reasoning"), "Math")
        self.assertEqual(specbench_category_group("rag"), "RAG")
        self.assertEqual(specbench_category_group("summarization"), "Sum")
        self.assertEqual(specbench_category_group("translation"), "Trans")
        self.assertEqual(specbench_category_group(""), "unknown")

    def test_rejects_request_count_larger_than_dataset(self) -> None:
        with self.assertRaisesRegex(ValueError, "dataset only contains"):
            load_workload("data/spec_bench/question.jsonl", 481, 42, len)

    def test_rejects_category_sample_count_larger_than_category(self) -> None:
        with self.assertRaisesRegex(ValueError, "per category"):
            load_workload(
                "data/spec_bench/question.jsonl",
                200,
                42,
                len,
                samples_per_category=81,
            )

    def test_rejects_unsupported_record_instead_of_using_empty_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.jsonl"
            path.write_text(json.dumps({"category": "writing"}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dataset line 1"):
                load_workload(path, 1, 42, len)


if __name__ == "__main__":
    unittest.main()
