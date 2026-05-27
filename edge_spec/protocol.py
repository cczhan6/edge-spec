from __future__ import annotations

from dataclasses import dataclass, field

from .types import SparseProb, VerificationResult


@dataclass
class DraftSegment:
    microbatch_id: int
    round_index: int
    segment_id: int
    device_id: str
    request_id: str
    draft_model: str
    prefix_ids: list[int]
    draft_ids: list[int]
    draft_dists: list[SparseProb]
    draft_start_s: float
    draft_end_s: float
    draft_elapsed_s: float
    uplink_s: float
    uplink_effective_mbps: float
    uplink_effective_rtt_ms: float
    uplink_jitter_s: float
    uplink_congested: bool
    arrival_s: float
    uplink_payload_bytes: int
    prefix_version: int = 0
    base_position: int = 0
    prefix_hash: str = ""
    lookahead: int = 0


@dataclass
class VerificationTask:
    segment: DraftSegment
    lane_id: int
    enqueue_s: float
    scheduler_cost: float = 0.0
    scheduler_cost_terms: dict[str, float] = field(default_factory=dict)


@dataclass
class VerificationOutcome:
    segment: DraftSegment
    verification: VerificationResult
    emitted_ids: list[int]
    lane_id: int | None
    target_forward_s: float
    verify_start_s: float
    verify_finish_s: float
    queue_wait_s: float = 0.0
    status: str = "verified"
    downlink_s: float = 0.0
    downlink_effective_mbps: float = 0.0
    downlink_effective_rtt_ms: float = 0.0
    downlink_jitter_s: float = 0.0
    downlink_congested: bool = False
    downlink_payload_bytes: int = 0


@dataclass
class VerifierLaneState:
    lane_id: int
    available_at_s: float = 0.0
    busy_s: float = 0.0
    verification_count: int = 0
    cached_request_id: str | None = None
    cached_prefix_hashes: set[str] = field(default_factory=set)
    local_queue: list[VerificationTask] = field(default_factory=list)
    flush_s: float | None = None

