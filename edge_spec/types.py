from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20

    def validate(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")


@dataclass
class SparseProb:
    ids: list[int]
    probs: list[float]

    def __post_init__(self) -> None:
        if len(self.ids) != len(self.probs):
            raise ValueError("ids and probs must have the same length")

    def as_dict(self) -> dict[int, float]:
        return {int(i): float(p) for i, p in zip(self.ids, self.probs)}

    def prob(self, token_id: int) -> float:
        for i, p in zip(self.ids, self.probs):
            if i == token_id:
                return float(p)
        return 0.0

    def mass(self) -> float:
        return float(sum(self.probs))

    def payload_bytes(self) -> int:
        return len(self.ids) * 8 + len(self.probs) * 4


@dataclass
class DraftOutput:
    draft_ids: list[int]
    draft_dists: list[SparseProb]
    elapsed_s: float


@dataclass
class VerificationResult:
    emitted_ids: list[int]
    accepted_count: int
    proposed_count: int
    rejected: bool


@dataclass
class SpecBenchItem:
    request_id: str
    prompt: str
    category: str = "unknown"
    turns: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeviceProfile:
    device_id: str
    uplink_mbps: float
    downlink_mbps: float
    rtt_ms: float
    jitter_ms: float = 0.0
    bandwidth_jitter_ratio: float = 0.0
    rtt_jitter_ms: float = 0.0
    congestion_probability: float = 0.0
    congestion_slowdown: float = 1.0

    def validate(self) -> None:
        if self.uplink_mbps <= 0 or self.downlink_mbps <= 0:
            raise ValueError("uplink_mbps and downlink_mbps must be > 0")
        if self.rtt_ms < 0 or self.jitter_ms < 0:
            raise ValueError("rtt_ms and jitter_ms must be >= 0")
        if self.bandwidth_jitter_ratio < 0:
            raise ValueError("bandwidth_jitter_ratio must be >= 0")
        if self.rtt_jitter_ms < 0:
            raise ValueError("rtt_jitter_ms must be >= 0")
        if not 0 <= self.congestion_probability <= 1:
            raise ValueError("congestion_probability must be in [0, 1]")
        if self.congestion_slowdown < 1.0:
            raise ValueError("congestion_slowdown must be >= 1.0")


@dataclass
class ClientState:
    device_id: str
    draft_model: str
    prompt_id: str
    category: str
    prompt: str
    prefix_ids: list[int]
    generated_ids: list[int] = field(default_factory=list)
    available_at_s: float = 0.0
    done: bool = False
    sync_rounds: int = 0
    proposed_draft_tokens: int = 0
    accepted_draft_tokens: int = 0
    latency_s: float = 0.0
    first_token_latency_s: float | None = None


@dataclass
class DraftPacket:
    microbatch_id: int
    round_index: int
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
