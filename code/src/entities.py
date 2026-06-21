from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.model_runner import DraftCandidateTree


ACTIVE_SEGMENT_STATUSES = {"drafting", "in_transit", "queued", "verifying", "verified"}
FINAL_SEGMENT_STATUSES = {
    "accepted",
    "rejected",
    "stale",
    "discarded",
    "absorbed",
}


@dataclass(frozen=True)
class Device:
    device_id: int
    device_type: str
    drafter_profile: str
    acceptance_prior: float
    draft_token_rate_tok_s: float
    draft_startup_ms: float
    uplink_mbps: float
    downlink_mbps: float
    rtt_ms: float
    jitter_ms: float


@dataclass
class DeviceRuntime:
    device: Device
    draft_queue: list[int] = field(default_factory=list)
    active_segment_id: int | None = None
    busy_until_ms: float = 0.0
    total_busy_time_ms: float = 0.0
    total_queue_wait_ms: float = 0.0
    assigned_requests: int = 0
    generated_draft_tokens: int = 0
    accepted_draft_tokens: int = 0
    selected_gammas: list[int] = field(default_factory=list)


@dataclass
class Request:
    request_id: int
    device_id: int
    output_len: int
    start_time_ms: float
    prompt_id: str
    category: str
    category_group: str
    prompt: str
    prompt_token_count: int
    prompt_ids: list[int]
    finish_time_ms: float | None = None
    generated_ids: list[int] = field(default_factory=list)
    edge_generated_ids: list[int] = field(default_factory=list)
    prefix_version: int = 0
    status: str = "running"
    in_flight_segments: list[int] = field(default_factory=list)
    pending_segments: dict[int, int] = field(default_factory=dict)
    completed_results: dict[int, int] = field(default_factory=dict)
    draft_queued: bool = False
    draft_queue_enter_ms: float | None = None
    accepted_tokens: int = 0
    rejected_count: int = 0
    rollback_count: int = 0
    wasted_draft_tokens: int = 0
    bonus_reused_tokens: int = 0
    overlap_credit_ms: float = 0.0
    first_token_time_ms: float | None = None
    max_outstanding_observed: int = 0
    max_unconfirmed_tokens_observed: int = 0
    target_only_uplink_ms: float = 0.0
    target_only_uplink_payload_bytes: int = 0
    target_only_queue_wait_ms: float = 0.0
    target_only_compute_ms: float = 0.0
    target_only_downlink_ms: float = 0.0
    target_only_downlink_payload_bytes: int = 0
    proactive_draft_ids: list[int] = field(default_factory=list)
    proactive_draft_tree: DraftCandidateTree | None = None
    proactive_base_pos: int | None = None
    proactive_prefix_version: int | None = None

    @property
    def latency_ms(self) -> float:
        if self.finish_time_ms is None:
            raise ValueError(f"request {self.request_id} has not finished")
        return self.finish_time_ms - self.start_time_ms

    @property
    def ttft_ms(self) -> float:
        if self.first_token_time_ms is None:
            return 0.0
        return self.first_token_time_ms - self.start_time_ms

    @property
    def committed_pos(self) -> int:
        return len(self.generated_ids)

    @property
    def edge_frontier_pos(self) -> int:
        return len(self.edge_generated_ids)

    @property
    def outstanding_count(self) -> int:
        return len(self.in_flight_segments) + int(self.draft_queued)


@dataclass
class Segment:
    segment_id: int
    request_id: int
    device_id: int
    draft_model: str
    prefix_version: int
    base_pos: int
    scheduled_gamma: int
    prefix_ids: list[int]
    draft_ids: list[int]
    create_time_ms: float
    draft_start_time_ms: float
    draft_queue_wait_ms: float = 0.0
    draft_compute_ms: float = 0.0
    draft_analytical_ms: float = 0.0
    uplink_delay_ms: float = 0.0
    uplink_payload_bytes: int = 0
    edge_arrival_time_ms: float | None = None
    lane_id: int | None = None
    verify_start_time_ms: float | None = None
    verify_done_time_ms: float | None = None
    verify_compute_ms: float = 0.0
    downlink_delay_ms: float = 0.0
    downlink_payload_bytes: int = 0
    status: str = "drafting"
    accepted_count: int | None = None
    emitted_ids: list[int] = field(default_factory=list)
    result_base_pos: int | None = None
    waste_recorded: bool = False
    result_arrived: bool = False
    bonus_reused: bool = False
    tree_strategy: str = "linear"
    tree_budget_nodes: int = 0
    draft_compute_nodes: int = 0
    processed_candidate_count: int = 0
    retained_tree_nodes: int = 0
    target_verify_tree_nodes: int = 1
    beam_len: int = 0
    draft_tree: DraftCandidateTree | None = None
    tree_path_switched: bool = False
    proactive_used: bool = False
    proactive_hit: bool = False
    proactive_wasted_tokens: int = 0
    proactive_draft_ids: list[int] = field(default_factory=list)
    proactive_draft_tree: DraftCandidateTree | None = None
    proactive_start_time_ms: float | None = None
    proactive_done_time_ms: float | None = None
    pipeline_target_ms: float = 0.0
    pipeline_edge_cycle_ms: float = 0.0
    pipeline_alignment_error_ms: float = 0.0

    @property
    def gamma(self) -> int:
        return len(self.draft_ids)

    @property
    def proposed_count(self) -> int:
        if self.draft_tree is None or not self.draft_tree.nodes:
            return self.gamma
        return max(self.gamma, max(node.depth for node in self.draft_tree.nodes))

    @property
    def verify_gamma(self) -> int:
        return len(self.draft_ids)

    @property
    def emitted_count(self) -> int:
        return len(self.emitted_ids)

    @property
    def acceptance_rate(self) -> float:
        if not self.proposed_count:
            return 0.0
        return float(self.accepted_count or 0) / self.proposed_count


@dataclass
class VerifierLane:
    lane_id: int
    queue: list[int] = field(default_factory=list)
    busy_until_ms: float = 0.0
    active_segment_id: int | None = None
    processed_segments: int = 0
    total_busy_time_ms: float = 0.0


@dataclass
class SimulationResult:
    method: str
    scenario: str
    requests: list[Request]
    segments: list[Segment]
    devices: list[DeviceRuntime]
    lanes: list[VerifierLane]
    batch_waiting_time_ms: float
    phase_waiting_time_ms: float
    lane_queue_wait_times_ms: list[float]
    event_trace: list[dict]
