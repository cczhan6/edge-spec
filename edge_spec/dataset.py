from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

from .types import SpecBenchItem

SPECBENCH_SIX_CATEGORIES = ("Sum", "Math", "MT", "QA", "RAG", "Trans")

SPECBENCH_RAW_TO_SIX = {
    "summarization": "Sum",
    "math_reasoning": "Math",
    "writing": "MT",
    "roleplay": "MT",
    "reasoning": "MT",
    "math": "MT",
    "coding": "MT",
    "extraction": "MT",
    "stem": "MT",
    "humanities": "MT",
    "qa": "QA",
    "rag": "RAG",
    "translation": "Trans",
}

SPECBENCH_CATEGORY_ALIASES = {
    "sum": "Sum",
    "summarization": "Sum",
    "math": "Math",
    "math reasoning": "Math",
    "math_reasoning": "Math",
    "mt": "MT",
    "writing": "MT",
    "roleplay": "MT",
    "reasoning": "MT",
    "coding": "MT",
    "extraction": "MT",
    "stem": "MT",
    "humanities": "MT",
    "multi-turn dialogue": "MT",
    "multi-turn conversation": "MT",
    "multi_turn_dialogue": "MT",
    "multi_turn_conversation": "MT",
    "qa": "QA",
    "question answering": "QA",
    "question_answering": "QA",
    "rag": "RAG",
    "retrieval-augmented generation": "RAG",
    "retrieval_augmented_generation": "RAG",
    "trans": "Trans",
    "translation": "Trans",
    "machine translation": "Trans",
    "machine_translation": "Trans",
}


def specbench_six_category(category: str) -> str:
    return SPECBENCH_RAW_TO_SIX.get(category.strip().casefold(), category)


def normalize_specbench_category(category: str) -> str:
    return SPECBENCH_CATEGORY_ALIASES.get(category.strip().casefold(), category)


def sort_specbench_categories(categories: Iterable[str]) -> list[str]:
    ordered_categories = list(dict.fromkeys(categories))
    six_order = {
        category: index for index, category in enumerate(SPECBENCH_SIX_CATEGORIES)
    }
    original_order = {
        category: index for index, category in enumerate(ordered_categories)
    }
    return sorted(
        ordered_categories,
        key=lambda category: (
            six_order.get(category, len(six_order)),
            original_order[category],
        ),
    )


def _first_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        return _first_text(value[0])
    if isinstance(value, dict):
        if "content" in value:
            return str(value["content"])
        if "text" in value:
            return str(value["text"])
    return str(value)


def resolve_specbench_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "question.jsonl"
    return candidate


def load_specbench(
    path: str | Path,
    limit: int | None = None,
    category: str | None = None,
    shuffle: bool = False,
    seed: int = 0,
) -> list[SpecBenchItem]:
    dataset_path = resolve_specbench_path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"SpecBench file not found: {dataset_path}. "
            "Run scripts/prepare_specbench.py first or pass --dataset-path."
        )

    items: list[SpecBenchItem] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            raw_category = str(raw.get("category") or raw.get("task") or "unknown")
            item_category = specbench_six_category(raw_category)
            if category and item_category != normalize_specbench_category(category):
                continue
            turns = raw.get("turns")
            if not isinstance(turns, list):
                turns = [raw.get("prompt") or raw.get("question") or raw.get("text") or ""]
            prompt = _first_text(turns[0] if turns else "")
            request_id = str(
                raw.get("question_id")
                or raw.get("id")
                or raw.get("request_id")
                or f"specbench-{index}"
            )
            items.append(
                SpecBenchItem(
                    request_id=request_id,
                    prompt=prompt,
                    category=item_category,
                    turns=[_first_text(turn) for turn in turns],
                    raw=raw,
                )
            )
            if limit is not None and len(items) >= limit:
                break
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(items)
    return items


def fallback_items() -> list[SpecBenchItem]:
    prompts = [
        "Write a short explanation of speculative decoding.",
        "Give three practical tips for reducing LLM serving latency.",
        "Solve this arithmetic problem and explain briefly: 17 * 23.",
    ]
    return [
        SpecBenchItem(
            request_id=f"fallback-{index}",
            prompt=prompt,
            category="fallback",
            turns=[prompt],
        )
        for index, prompt in enumerate(prompts)
    ]


def select_one_per_category(items: Iterable[SpecBenchItem]) -> list[SpecBenchItem]:
    selected: dict[str, SpecBenchItem] = {}
    for item in items:
        if item.category not in selected:
            selected[item.category] = item
    return [selected[category] for category in sort_specbench_categories(selected)]


def select_one_per_category_per_device(
    items: Iterable[SpecBenchItem],
    device_count: int = 3,
) -> list[SpecBenchItem]:
    grouped: dict[str, list[SpecBenchItem]] = {}
    for item in items:
        grouped.setdefault(item.category, []).append(item)

    selected: list[SpecBenchItem] = []
    for category in sort_specbench_categories(grouped):
        category_items = grouped[category]
        if len(category_items) < device_count:
            raise ValueError(
                f"category {category!r} has {len(category_items)} samples, "
                f"but {device_count} are required for one sample per device"
            )
        selected.extend(category_items[:device_count])
    return selected


def iter_microbatches(
    items: Iterable[SpecBenchItem],
    batch_size: int = 3,
    drop_last: bool = False,
):
    batch: list[SpecBenchItem] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch and not drop_last:
        yield batch
