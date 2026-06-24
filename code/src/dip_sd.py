from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Sequence


@dataclass(frozen=True)
class DipSDEpochPlan:
    batches: tuple[tuple[int, ...], ...]
    draft_lengths: dict[int, int]


def build_fixed_epoch_plan(
    active_request_ids: Sequence[int],
    *,
    batch_count: int,
    draft_length: int,
    min_draft_length: int,
    max_draft_length: int,
    max_batch_size: int,
) -> DipSDEpochPlan:
    request_ids = tuple(sorted(int(request_id) for request_id in active_request_ids))
    if not request_ids:
        return DipSDEpochPlan((), {})
    if batch_count <= 0:
        raise ValueError("dip_sd.batch_count must be positive")
    if max_batch_size <= 0:
        raise ValueError("dip_sd.max_batch_size must be positive")
    bounded_length = max(min_draft_length, min(max_draft_length, int(draft_length)))
    needed_batches = ceil(len(request_ids) / max_batch_size)
    count = max(1, min(len(request_ids), max(int(batch_count), needed_batches)))
    batches = [[] for _ in range(count)]
    for index, request_id in enumerate(request_ids):
        batches[index % count].append(request_id)
    return DipSDEpochPlan(
        batches=tuple(tuple(batch) for batch in batches if batch),
        draft_lengths={request_id: bounded_length for request_id in request_ids},
    )
