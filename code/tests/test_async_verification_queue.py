from __future__ import annotations

import importlib


def async_verification():
    return importlib.import_module("src.async_verification")


def coordinator_with_requests(
    frontiers: dict[int, int],
):
    module = async_verification()
    return module.AsyncVerificationCoordinator(
        num_channels=2,
        requests=[
            module.AsyncRequestState(
                request_id=request_id,
                current_segment_index=frontier,
            )
            for request_id, frontier in frontiers.items()
        ],
    )


def job(
    request_id: int,
    segment_index: int,
    arrival: float,
    arrival_sequence: int,
):
    return async_verification().VerificationJob(
        job_id=arrival_sequence,
        request_id=request_id,
        segment_index=segment_index,
        path_generation=0,
        dependency_start=0,
        dependency_end=segment_index,
        arrival_time_ms=arrival,
        arrival_sequence=arrival_sequence,
        verify_prefix_ids=(),
        local_start=0,
        local_end=0,
    )


def test_current_jobs_strictly_precede_successors_then_distance_and_arrival() -> None:
    coordinator = coordinator_with_requests(frontiers={0: 2, 1: 4})
    coordinator.enqueue(job(0, 4, arrival=1.0, arrival_sequence=0))
    coordinator.enqueue(job(1, 4, arrival=9.0, arrival_sequence=1))
    coordinator.enqueue(job(0, 3, arrival=2.0, arrival_sequence=2))

    assert [item.key for item in coordinator.pop_waiting(3)] == [
        (1, 4),
        (0, 3),
        (0, 4),
    ]


def test_successor_ties_use_arrival_time_then_sequence() -> None:
    coordinator = coordinator_with_requests(frontiers={0: 0, 1: 0, 2: 0})
    coordinator.enqueue(job(0, 1, arrival=3.0, arrival_sequence=2))
    coordinator.enqueue(job(1, 1, arrival=2.0, arrival_sequence=1))
    coordinator.enqueue(job(2, 1, arrival=2.0, arrival_sequence=0))

    assert [item.key for item in coordinator.pop_waiting(3)] == [
        (2, 1),
        (1, 1),
        (0, 1),
    ]


def test_waiting_job_is_reprioritized_when_request_frontier_advances() -> None:
    coordinator = coordinator_with_requests(frontiers={0: 0, 1: 1})
    coordinator.enqueue(job(0, 1, arrival=1.0, arrival_sequence=0))
    coordinator.enqueue(job(1, 1, arrival=2.0, arrival_sequence=1))
    coordinator.requests[0].current_segment_index = 1

    assert [item.key for item in coordinator.pop_waiting(2)] == [
        (0, 1),
        (1, 1),
    ]
