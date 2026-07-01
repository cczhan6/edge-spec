from __future__ import annotations

import pytest

from src.async_verification import (
    AsyncRequestState,
    AsyncVerificationCoordinator,
    JobStatus,
    VerificationJob,
)


def coordinator(num_channels: int) -> AsyncVerificationCoordinator:
    return AsyncVerificationCoordinator(
        num_channels=num_channels,
        requests=[AsyncRequestState(request_id=0)],
    )


def waiting_job(job_id: int, segment_index: int = 0) -> VerificationJob:
    return VerificationJob(
        job_id=job_id,
        request_id=0,
        segment_index=segment_index,
        path_generation=0,
        dependency_start=0,
        dependency_end=segment_index,
        arrival_time_ms=float(job_id),
        arrival_sequence=job_id,
        verify_prefix_ids=(),
        local_start=0,
        local_end=0,
    )


def test_busy_channel_is_not_preempted_by_later_current_job() -> None:
    instance = coordinator(num_channels=1)
    instance.enqueue(waiting_job(1, segment_index=1))
    first = instance.dispatch_one(now_ms=0.0, duration_ms=10.0)
    instance.requests[0].current_segment_index = 0
    instance.enqueue(waiting_job(2, segment_index=0))

    assert instance.dispatch_all(now_ms=1.0, duration_ms=1.0) == []
    assert instance.channels[0].active_job_id == first.job_id


def test_dispatch_uses_lowest_idle_channels_and_never_exceeds_capacity() -> None:
    instance = coordinator(num_channels=2)
    for job_id in (1, 2, 3):
        instance.enqueue(waiting_job(job_id))

    dispatched = instance.dispatch_all(now_ms=5.0, duration_ms=4.0)

    assert [item.job_id for item in dispatched] == [1, 2]
    assert [channel.active_job_id for channel in instance.channels] == [1, 2]
    assert all(channel.busy_until_ms == 9.0 for channel in instance.channels)


def test_invalid_active_job_keeps_channel_until_matching_completion() -> None:
    instance = coordinator(num_channels=1)
    instance.enqueue(waiting_job(1))
    instance.dispatch_one(now_ms=2.0, duration_ms=5.0)

    instance.invalidate_active_job(1)

    assert instance.job(1).status is JobStatus.INVALID
    assert instance.channels[0].active_job_id == 1
    instance.complete_channel(channel_id=0, job_id=1, now_ms=7.0)
    assert instance.channels[0].active_job_id is None


def test_completion_must_match_active_job_and_cannot_repeat() -> None:
    instance = coordinator(num_channels=1)
    instance.enqueue(waiting_job(1))
    instance.dispatch_one(now_ms=0.0, duration_ms=3.0)

    with pytest.raises(
        ValueError,
        match="verification completion does not match active channel",
    ):
        instance.complete_channel(channel_id=0, job_id=2, now_ms=3.0)

    instance.complete_channel(channel_id=0, job_id=1, now_ms=3.0)
    with pytest.raises(
        ValueError,
        match="verification completion does not match active channel",
    ):
        instance.complete_channel(channel_id=0, job_id=1, now_ms=3.0)
