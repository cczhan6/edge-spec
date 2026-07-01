from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass
class AsyncRequestState:
    request_id: int
    server_confirmed_ids: list[int] = field(default_factory=list)
    current_segment_index: int = 0
    segments: dict[int, AsyncSegmentState] = field(default_factory=dict)
    completed_results: dict[int, Any] = field(default_factory=dict)
    path_generation: int = 0
    terminal: bool = False


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
