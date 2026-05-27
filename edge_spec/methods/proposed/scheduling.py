from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from edge_spec.protocol import DraftSegment, VerifierLaneState
from edge_spec.types import DeviceProfile

from .consistency import PrefixStateManager


@dataclass(frozen=True)
class LaneAssignment:
    lane: VerifierLaneState
    cost: float
    terms: dict[str, float]


class AdaptiveLookaheadPolicy:
    def __init__(
        self,
        policy: str,
        max_gamma: int,
        initial_lookahead: int = 4,
    ) -> None:
        if policy not in {"adaptive", "fixed"}:
            raise ValueError("lookahead policy must be adaptive or fixed")
        if max_gamma <= 0:
            raise ValueError("max_gamma must be > 0")
        if initial_lookahead <= 0:
            raise ValueError("initial_lookahead must be > 0")
        self.policy = policy
        self.max_gamma = max_gamma
        self.initial_lookahead = min(initial_lookahead, max_gamma)

    def select(
        self,
        *,
        profile: DeviceProfile,
        acceptance_rate: float,
        edge_queue_depth: int,
        remaining_tokens: int,
    ) -> int:
        if remaining_tokens <= 0:
            return 0
        if self.policy == "fixed":
            return min(self.max_gamma, remaining_tokens)

        lookahead = self.initial_lookahead

        if profile.rtt_ms >= 80 or profile.uplink_mbps <= 20:
            lookahead += 2
        elif profile.rtt_ms >= 50 or profile.uplink_mbps <= 40:
            lookahead += 1
        elif profile.rtt_ms <= 30 and profile.uplink_mbps >= 50:
            lookahead -= 1

        if acceptance_rate >= 0.95:
            lookahead += 2
        elif acceptance_rate >= 0.85:
            lookahead += 1
        elif acceptance_rate < 0.35:
            lookahead -= 2
        elif acceptance_rate < 0.55:
            lookahead -= 1

        if edge_queue_depth >= 8:
            lookahead -= 2
        elif edge_queue_depth >= 4:
            lookahead -= 1
        elif edge_queue_depth <= 1 and acceptance_rate >= 0.75:
            lookahead += 1

        lookahead = max(1, min(self.max_gamma, lookahead))
        return min(lookahead, remaining_tokens)


class PrefixAwareScheduler:
    def __init__(
        self,
        policy: str,
        prefix_state: PrefixStateManager,
        max_gamma: int,
    ) -> None:
        if policy not in {"prefix-aware", "queue-only"}:
            raise ValueError("scheduler must be prefix-aware or queue-only")
        self.policy = policy
        self.prefix_state = prefix_state
        self.max_gamma = max_gamma

    def assign(
        self,
        segment: DraftSegment,
        lanes: Sequence[VerifierLaneState],
        now_s: float,
    ) -> LaneAssignment:
        if not lanes:
            raise ValueError("at least one lane is required")
        assignments = [
            self._score_lane(segment, lane, now_s)
            for lane in lanes
        ]
        return min(
            assignments,
            key=lambda item: (item.cost, item.lane.available_at_s, item.lane.lane_id),
        )

    def _score_lane(
        self,
        segment: DraftSegment,
        lane: VerifierLaneState,
        now_s: float,
    ) -> LaneAssignment:
        earliest_start = max(lane.available_at_s, segment.arrival_s, now_s)
        queue_delay = earliest_start - segment.arrival_s
        if self.policy == "queue-only":
            terms = {
                "queue_delay": queue_delay,
                "verify_latency": 0.0,
                "kv_cache_miss": 0.0,
                "rollback_risk": 0.0,
            }
            return LaneAssignment(lane, queue_delay, terms)

        verify_latency = max(1, len(segment.draft_ids)) * 0.001
        kv_cache_miss = (
            0.0
            if lane.cached_request_id == segment.request_id
            or segment.prefix_hash in lane.cached_prefix_hashes
            else 0.002
        )
        acceptance_rate = self.prefix_state.acceptance_rate(segment.request_id)
        rollback_risk = (
            len(segment.draft_ids) / max(1, self.max_gamma)
        ) * (1.0 - acceptance_rate) * 0.001
        terms = {
            "queue_delay": queue_delay,
            "verify_latency": verify_latency,
            "kv_cache_miss": kv_cache_miss,
            "rollback_risk": rollback_risk,
        }
        cost = (
            queue_delay
            + 0.25 * verify_latency
            + kv_cache_miss
            + rollback_risk
        )
        return LaneAssignment(lane, cost, terms)

