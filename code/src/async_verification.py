from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable


class JobStatus(str, Enum):
    WAITING = "waiting"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    INVALID = "invalid"


@dataclass(frozen=True)
class VerificationJob:
    job_id: int
    request_id: int
    segment_index: int
    path_generation: int
    dependency_start: int
    dependency_end: int
    arrival_time_ms: float
    arrival_sequence: int
    verify_prefix_ids: tuple[int, ...]
    local_start: int
    local_end: int
    status: JobStatus = JobStatus.WAITING

    @property
    def key(self) -> tuple[int, int]:
        return self.request_id, self.segment_index


@dataclass
class VerificationChannelState:
    channel_id: int
    active_job_id: int | None = None
    active_start_ms: float | None = None
    busy_until_ms: float = 0.0
    processed_jobs: int = 0
    total_busy_time_ms: float = 0.0


@dataclass
class AsyncSegmentState:
    segment_id: int
    request_id: int
    segment_index: int
    path_generation: int
    draft_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class ContiguousVerifyInput:
    prefix_ids: tuple[int, ...]
    draft_ids: tuple[int, ...]
    local_start: int
    local_end: int
    dependency_fingerprint: tuple[int, ...]

    @property
    def local_slice(self) -> slice:
        return slice(self.local_start, self.local_end)


@dataclass
class AsyncRequestState:
    request_id: int
    server_confirmed_ids: list[int] = field(default_factory=list)
    current_segment_index: int = 0
    segments: dict[int, AsyncSegmentState] = field(default_factory=dict)
    completed_results: dict[int, Any] = field(default_factory=dict)
    path_generation: int = 0
    terminal: bool = False

    def build_verify_input(
        self,
        segment_index: int,
        l_max_ver: int,
    ) -> ContiguousVerifyInput | None:
        if l_max_ver <= 0:
            raise ValueError("l_max_ver must be positive")
        if segment_index < self.current_segment_index:
            raise ValueError("segment precedes the request frontier")
        draft_ids: list[int] = []
        local_start = 0
        for index in range(self.current_segment_index, segment_index + 1):
            segment = self.segments.get(index)
            if segment is None:
                raise ValueError(f"missing dependency segment: {index}")
            if segment.path_generation != self.path_generation:
                raise ValueError(f"segment has stale path generation: {index}")
            if index == segment_index:
                local_start = len(draft_ids)
            draft_ids.extend(segment.draft_ids)
        if len(draft_ids) > l_max_ver:
            return None
        return ContiguousVerifyInput(
            prefix_ids=tuple(self.server_confirmed_ids),
            draft_ids=tuple(draft_ids),
            local_start=local_start,
            local_end=len(draft_ids),
            dependency_fingerprint=tuple(draft_ids[:local_start]),
        )


class AsyncVerificationCoordinator:
    def __init__(
        self,
        num_channels: int,
        requests: Iterable[AsyncRequestState] = (),
    ) -> None:
        if num_channels <= 0:
            raise ValueError("num_channels must be positive")
        request_states = list(requests)
        self.channels = [
            VerificationChannelState(channel_id=index)
            for index in range(num_channels)
        ]
        self.requests = {
            request.request_id: request
            for request in request_states
        }
        if len(self.requests) != len(request_states):
            raise ValueError("request ids must be unique")
        self._jobs: dict[int, VerificationJob] = {}
        self._waiting_job_ids: list[int] = []

    def enqueue(self, job: VerificationJob) -> None:
        if job.job_id in self._jobs:
            raise ValueError(f"duplicate verification job id: {job.job_id}")
        if job.request_id not in self.requests:
            raise ValueError(f"unknown request id: {job.request_id}")
        if job.segment_index < self.requests[job.request_id].current_segment_index:
            raise ValueError("verification job precedes the request frontier")
        self._jobs[job.job_id] = job
        self._waiting_job_ids.append(job.job_id)

    def priority(self, job: VerificationJob) -> tuple[int, int, float, int]:
        current = self.requests[job.request_id].current_segment_index
        return (
            int(job.segment_index != current),
            job.segment_index - current,
            job.arrival_time_ms,
            job.arrival_sequence,
        )

    def pop_waiting(self, count: int) -> list[VerificationJob]:
        if count < 0:
            raise ValueError("count must be non-negative")
        ordered = sorted(
            (self._jobs[job_id] for job_id in self._waiting_job_ids),
            key=self.priority,
        )
        selected = ordered[:count]
        selected_ids = {job.job_id for job in selected}
        self._waiting_job_ids = [
            job_id
            for job_id in self._waiting_job_ids
            if job_id not in selected_ids
        ]
        return selected

    def dispatch_one(
        self,
        now_ms: float,
        duration_ms: float,
    ) -> VerificationJob | None:
        if duration_ms < 0:
            raise ValueError("verification duration must be non-negative")
        channel = next(
            (
                item
                for item in self.channels
                if item.active_job_id is None
            ),
            None,
        )
        if channel is None:
            return None
        selected = self.pop_waiting(1)
        if not selected:
            return None
        job = replace(selected[0], status=JobStatus.VERIFYING)
        self._jobs[job.job_id] = job
        channel.active_job_id = job.job_id
        channel.active_start_ms = now_ms
        channel.busy_until_ms = now_ms + duration_ms
        return job

    def dispatch_all(
        self,
        now_ms: float,
        duration_ms: float,
    ) -> list[VerificationJob]:
        dispatched: list[VerificationJob] = []
        while True:
            job = self.dispatch_one(now_ms, duration_ms)
            if job is None:
                return dispatched
            dispatched.append(job)

    def invalidate_active_job(self, job_id: int) -> None:
        if not any(
            channel.active_job_id == job_id
            for channel in self.channels
        ):
            raise ValueError(f"verification job is not active: {job_id}")
        self._jobs[job_id] = replace(
            self._jobs[job_id],
            status=JobStatus.INVALID,
        )

    def complete_channel(
        self,
        channel_id: int,
        job_id: int,
        now_ms: float,
    ) -> VerificationJob:
        channel = self.channels[channel_id]
        if channel.active_job_id != job_id:
            raise ValueError(
                "verification completion does not match active channel"
            )
        job = self._jobs[job_id]
        if job.status is not JobStatus.INVALID:
            job = replace(job, status=JobStatus.COMPLETED)
            self._jobs[job_id] = job
        start_ms = (
            now_ms
            if channel.active_start_ms is None
            else channel.active_start_ms
        )
        channel.active_job_id = None
        channel.active_start_ms = None
        channel.busy_until_ms = now_ms
        channel.processed_jobs += 1
        channel.total_busy_time_ms += now_ms - start_ms
        return job

    def job(self, job_id: int) -> VerificationJob:
        return self._jobs[job_id]
