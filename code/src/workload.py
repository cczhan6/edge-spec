from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


MT_BENCH_SUBCATEGORIES = {
    "coding",
    "extraction",
    "humanities",
    "math",
    "reasoning",
    "roleplay",
    "stem",
    "writing",
}

SPECBENCH_CATEGORY_LABELS = {
    "math_reasoning": "Math",
    "qa": "QA",
    "rag": "RAG",
    "summarization": "Sum",
    "translation": "Trans",
}

SPECBENCH_CATEGORY_ORDER = ("MT", "QA", "Math", "RAG", "Sum", "Trans")


@dataclass(frozen=True)
class WorkloadItem:
    prompt_id: str
    prompt: str
    prompt_token_count: int
    category: str = ""
    category_group: str = ""


def extract_prompt(value: Any, line_number: int) -> str:
    if isinstance(value, str):
        prompt = value
    elif isinstance(value, dict):
        prompt = value.get("prompt") or value.get("instruction") or value.get("input")
        if not prompt:
            turns = value.get("turns")
            if isinstance(turns, list) and turns:
                prompt = turns[0]
    else:
        prompt = None
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"dataset line {line_number} did not contain a supported prompt")
    return prompt


def specbench_category_group(category: str) -> str:
    if category in MT_BENCH_SUBCATEGORIES:
        return "MT"
    return SPECBENCH_CATEGORY_LABELS.get(category, category or "unknown")


def load_workload(
    path: str | Path,
    num_requests: int,
    seed: int,
    token_counter: Callable[[str], int],
    *,
    samples_per_category: int | None = None,
) -> list[WorkloadItem]:
    if num_requests <= 0:
        raise ValueError("num_requests must be > 0")
    if samples_per_category is not None and samples_per_category <= 0:
        raise ValueError("samples_per_category must be > 0")
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            prompt = extract_prompt(value, line_number)
            prompt_id = (
                str(value.get("question_id", line_number))
                if isinstance(value, dict)
                else str(line_number)
            )
            category = str(value.get("category", "")) if isinstance(value, dict) else ""
            records.append((prompt_id, prompt, category))
    rng = random.Random(seed)
    if samples_per_category is not None:
        selected = _sample_per_category(records, samples_per_category, rng)
    elif num_requests > len(records):
        raise ValueError(
            f"requested {num_requests} prompts but dataset only contains {len(records)}"
        )
    else:
        selected = rng.sample(records, num_requests)
    return [
        WorkloadItem(
            prompt_id=prompt_id,
            prompt=prompt,
            prompt_token_count=token_counter(prompt),
            category=category,
            category_group=specbench_category_group(category),
        )
        for prompt_id, prompt, category in selected
    ]


def _sample_per_category(
    records: list[tuple[str, str, str]],
    samples_per_category: int,
    rng: random.Random,
) -> list[tuple[str, str, str]]:
    grouped: dict[str, list[tuple[str, str, str]]] = {}
    for record in records:
        grouped.setdefault(specbench_category_group(record[2]), []).append(record)
    ordered_categories = [
        *[category for category in SPECBENCH_CATEGORY_ORDER if category in grouped],
        *sorted(set(grouped) - set(SPECBENCH_CATEGORY_ORDER)),
    ]
    selected: list[tuple[str, str, str]] = []
    for category in ordered_categories:
        candidates = grouped[category]
        if samples_per_category > len(candidates):
            raise ValueError(
                f"requested {samples_per_category} prompts per category but "
                f"category {category} only contains {len(candidates)}"
            )
        selected.extend(rng.sample(candidates, samples_per_category))
    rng.shuffle(selected)
    return selected
