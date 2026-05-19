from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

from .types import SpecBenchItem


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
            item_category = str(raw.get("category") or raw.get("task") or "unknown")
            if category and item_category != category:
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
    return list(selected.values())


def select_one_per_category_per_device(
    items: Iterable[SpecBenchItem],
    device_count: int = 3,
) -> list[SpecBenchItem]:
    grouped: dict[str, list[SpecBenchItem]] = {}
    for item in items:
        grouped.setdefault(item.category, []).append(item)

    selected: list[SpecBenchItem] = []
    for category, category_items in grouped.items():
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
