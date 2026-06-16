from __future__ import annotations

from collections.abc import Callable

from src.entities import Segment, VerifierLane


class LaneScheduler:
    def __init__(self, assignment: str) -> None:
        self.assignment = assignment
        self._next_lane = 0

    def select(
        self,
        lanes: list[VerifierLane],
        segment: Segment,
        current_time_ms: float,
        predict_verify_latency: Callable[[int], float],
        segments: list[Segment],
    ) -> VerifierLane:
        if not lanes:
            raise ValueError("at least one verifier lane is required")
        if self.assignment == "round_robin":
            lane = lanes[self._next_lane % len(lanes)]
            self._next_lane += 1
            return lane

        def score(lane: VerifierLane) -> tuple[float, int]:
            queue_ms = sum(
                predict_verify_latency(segments[item].verify_gamma) for item in lane.queue
            )
            available_ms = max(current_time_ms, lane.busy_until_ms) + queue_ms
            return available_ms + predict_verify_latency(segment.verify_gamma), lane.lane_id

        return min(lanes, key=score)
