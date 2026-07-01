from __future__ import annotations

import pytest

from src.async_verification import (
    AsyncRequestState,
    AsyncSegmentState,
    AsyncVerificationCoordinator,
    VerificationJob,
)
from src.model_runner import VerificationResult


def accepted(tokens: list[int], bonus: int) -> VerificationResult:
    return VerificationResult(
        accepted_count=len(tokens),
        committed_tokens=[*tokens, bonus],
        bonus_token=bonus,
    )


def accepting_coordinator(segment_tokens: list[list[int]]) -> AsyncVerificationCoordinator:
    request = AsyncRequestState(
        request_id=0,
        segments={
            index: AsyncSegmentState(
                segment_id=index,
                request_id=0,
                segment_index=index,
                path_generation=0,
                draft_ids=tuple(tokens),
            )
            for index, tokens in enumerate(segment_tokens)
        },
    )
    coordinator = AsyncVerificationCoordinator(num_channels=2, requests=[request])
    dependencies: list[int] = []
    for index, tokens in enumerate(segment_tokens):
        coordinator.enqueue(
            VerificationJob(
                job_id=index + 1,
                request_id=0,
                segment_index=index,
                path_generation=0,
                dependency_start=0,
                dependency_end=index,
                arrival_time_ms=float(index),
                arrival_sequence=index,
                verify_prefix_ids=(),
                local_start=len(dependencies),
                local_end=len(dependencies) + len(tokens),
                dependency_fingerprint=tuple(dependencies),
            )
        )
        dependencies.extend(tokens)
    return coordinator


def flatten_confirmed(actions) -> list[int]:
    return [token for action in actions for token in action.confirmed_ids]


def test_successor_completion_waits_for_current_then_drains_in_order() -> None:
    coordinator = accepting_coordinator([[1], [3]])

    assert coordinator.complete(job_id=2, result=accepted([3], bonus=4)) == []
    actions = coordinator.complete(job_id=1, result=accepted([1], bonus=2))

    assert flatten_confirmed(actions) == [1, 2, 3, 4]
    assert coordinator.request(0).server_confirmed_ids == [1, 2, 3, 4]
    assert coordinator.request(0).current_segment_index == 2


def test_gap_in_completed_results_stops_drain_at_first_missing_segment() -> None:
    coordinator = accepting_coordinator([[1], [2], [3]])
    coordinator.complete(job_id=3, result=accepted([3], bonus=30))

    actions = coordinator.complete(job_id=1, result=accepted([1], bonus=10))

    assert flatten_confirmed(actions) == [1, 10]
    assert coordinator.request(0).current_segment_index == 1
    assert 2 in coordinator.request(0).completed_results


def test_completion_rejects_dependency_fingerprint_mismatch() -> None:
    request = AsyncRequestState(
        request_id=0,
        segments={
            0: AsyncSegmentState(0, 0, 0, 0, (1,)),
            1: AsyncSegmentState(1, 0, 1, 0, (2,)),
        },
    )
    coordinator = AsyncVerificationCoordinator(num_channels=1, requests=[request])
    coordinator.enqueue(
        VerificationJob(
            job_id=2,
            request_id=0,
            segment_index=1,
            path_generation=0,
            dependency_start=0,
            dependency_end=1,
            arrival_time_ms=0.0,
            arrival_sequence=0,
            verify_prefix_ids=(),
            local_start=1,
            local_end=2,
            dependency_fingerprint=(99,),
        )
    )

    with pytest.raises(ValueError, match="verification dependency fingerprint mismatch"):
        coordinator.complete(job_id=2, result=accepted([2], bonus=3))
