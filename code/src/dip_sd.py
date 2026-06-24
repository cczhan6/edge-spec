from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from itertools import product
from typing import Mapping, Sequence


@dataclass(frozen=True)
class DipSDEpochPlan:
    batches: tuple[tuple[int, ...], ...]
    draft_lengths: dict[int, int]
    objective: float = 0.0
    expected_useful_tokens: float = 0.0
    pipeline_span: float = 0.0
    optimizer: str = "fixed_greedy"


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
        optimizer="fixed_greedy",
    )


def optimize_epoch_plan(
    active_request_ids: Sequence[int],
    *,
    acceptance_estimates: Mapping[int, float],
    max_batch_count: int,
    min_draft_length: int,
    max_draft_length: int,
    max_batch_size: int,
) -> DipSDEpochPlan:
    request_ids = tuple(sorted(int(request_id) for request_id in active_request_ids))
    if not request_ids:
        return DipSDEpochPlan((), {}, optimizer="deterministic_search")
    best: DipSDEpochPlan | None = None
    max_count = min(len(request_ids), max(1, int(max_batch_count)))
    for batch_count in range(1, max_count + 1):
        if ceil(len(request_ids) / max_batch_size) > batch_count:
            continue
        for lengths in product(
            range(int(min_draft_length), int(max_draft_length) + 1),
            repeat=len(request_ids),
        ):
            draft_lengths = dict(zip(request_ids, lengths))
            batches = _assign_for_lengths(request_ids, draft_lengths, batch_count, max_batch_size)
            if len(batches) != batch_count:
                continue
            useful = sum(
                _expected_useful_tokens(
                    float(acceptance_estimates.get(request_id, 0.5)),
                    draft_lengths[request_id],
                )
                for request_id in request_ids
            )
            span = _pipeline_span(batches, draft_lengths)
            objective = useful / span if span > 0 else 0.0
            candidate = DipSDEpochPlan(
                batches=tuple(tuple(batch) for batch in batches),
                draft_lengths=draft_lengths,
                objective=objective,
                expected_useful_tokens=useful,
                pipeline_span=span,
                optimizer="deterministic_search",
            )
            if _is_better(candidate, best):
                best = candidate
    if best is None:
        return build_fixed_epoch_plan(
            request_ids,
            batch_count=1,
            draft_length=min_draft_length,
            min_draft_length=min_draft_length,
            max_draft_length=max_draft_length,
            max_batch_size=max_batch_size,
        )
    return best


def _assign_for_lengths(
    request_ids: Sequence[int],
    draft_lengths: Mapping[int, int],
    batch_count: int,
    max_batch_size: int,
) -> tuple[tuple[int, ...], ...]:
    batches: list[list[int]] = [[] for _ in range(batch_count)]
    loads = [0.0 for _ in range(batch_count)]
    ordered = sorted(request_ids, key=lambda request_id: (-draft_lengths[request_id], request_id))
    for index, request_id in enumerate(ordered):
        if index < batch_count:
            target = index
        else:
            target = min(
                (
                    batch_index
                    for batch_index, batch in enumerate(batches)
                    if len(batch) < max_batch_size
                ),
                key=lambda batch_index: (loads[batch_index], len(batches[batch_index]), batch_index),
            )
        batches[target].append(request_id)
        loads[target] = _batch_span(batches[target], draft_lengths)
    return tuple(tuple(sorted(batch)) for batch in batches if batch)


def _expected_useful_tokens(alpha: float, draft_length: int) -> float:
    bounded_alpha = max(0.0, min(0.999999, alpha))
    if bounded_alpha == 0.0:
        return 1.0
    return (1.0 - bounded_alpha ** (int(draft_length) + 1)) / (1.0 - bounded_alpha)


def _pipeline_span(
    batches: Sequence[Sequence[int]],
    draft_lengths: Mapping[int, int],
) -> float:
    return sum(_batch_span(batch, draft_lengths) for batch in batches)


def _batch_span(batch: Sequence[int], draft_lengths: Mapping[int, int]) -> float:
    if not batch:
        return 0.0
    max_length = max(draft_lengths[request_id] for request_id in batch)
    return float(max_length + len(batch) * max_length)


def _is_better(candidate: DipSDEpochPlan, incumbent: DipSDEpochPlan | None) -> bool:
    if incumbent is None:
        return True
    epsilon = 1e-12
    if candidate.objective > incumbent.objective + epsilon:
        return True
    if abs(candidate.objective - incumbent.objective) > epsilon:
        return False
    if candidate.pipeline_span < incumbent.pipeline_span - epsilon:
        return True
    if abs(candidate.pipeline_span - incumbent.pipeline_span) > epsilon:
        return False
    if len(candidate.batches) != len(incumbent.batches):
        return len(candidate.batches) < len(incumbent.batches)
    if candidate.batches != incumbent.batches:
        return candidate.batches < incumbent.batches
    return tuple(sorted(candidate.draft_lengths.items())) < tuple(
        sorted(incumbent.draft_lengths.items())
    )
