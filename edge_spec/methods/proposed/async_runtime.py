from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from itertools import count
from typing import Sequence

from edge_spec.metrics import summarize_lanes
from edge_spec.protocol import (
    DraftSegment,
    VerificationOutcome,
    VerificationTask,
    VerifierLaneState,
)
from edge_spec.sampling import verify_draft_exact
from edge_spec.simulation import estimate_downlink_payload_bytes
from edge_spec.tracing import (
    outcome_device_trace,
    segment_device_trace,
    verification_event_trace,
)
from edge_spec.types import ClientState, SpecBenchItem

from ..base import BaseMethodRunner, ExperimentResult, ProgressLike
from .consistency import PrefixStateManager, stable_prefix_hash
from .scheduling import AdaptiveLookaheadPolicy, PrefixAwareScheduler


@dataclass
class AsyncClientRuntime:
    client: ClientState
    device_index: int
    microbatch_id: int
    request_start_s: float
    local_prefix_version: int = 0
    speculative_ids: list[int] = field(default_factory=list)
    pending_segments: dict[int, DraftSegment] = field(default_factory=dict)
    next_segment_id: int = 0
    lane_history: list[int] = field(default_factory=list)
    stale_segment_count: int = 0
    invalidated_segment_count: int = 0

    @property
    def inflight_count(self) -> int:
        return len(self.pending_segments)


class ProposedAsyncRunner(BaseMethodRunner):
    method_name = "proposed"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.prefix_state = PrefixStateManager()
        self.lookahead = AdaptiveLookaheadPolicy(
            self.config.lookahead_policy,
            self.config.gamma,
            initial_lookahead=self.config.initial_lookahead,
        )
        self.scheduler = PrefixAwareScheduler(
            self.config.scheduler,
            self.prefix_state,
            self.config.gamma,
        )
        self.lanes = [
            VerifierLaneState(lane_id=index)
            for index in range(self.config.lane_count)
        ]

    def run_dataset(
        self,
        microbatches: Sequence[Sequence[SpecBenchItem]],
        progress: ProgressLike | None = None,
    ) -> ExperimentResult:
        device_queues: list[deque[tuple[int, SpecBenchItem]]] = [
            deque() for _ in self.draft_backends
        ]
        for microbatch_id, items in enumerate(microbatches):
            if len(items) > len(self.draft_backends):
                raise ValueError("microbatch has more requests than draft backends")
            for device_index, item in enumerate(items):
                device_queues[device_index].append((microbatch_id, item))

        events: list[tuple[float, int, str, object]] = []
        next_event_id = count()
        active: list[AsyncClientRuntime | None] = [None] * len(self.draft_backends)
        by_request: dict[str, AsyncClientRuntime] = {}
        waiting_segments: list[DraftSegment] = []
        draining_waiting = False
        records: list[dict] = []
        traces: list[dict] = []

        def push_event(time_s: float, kind: str, payload: object) -> None:
            heapq.heappush(events, (time_s, next(next_event_id), kind, payload))

        def start_next_request(device_index: int, start_time_s: float) -> None:
            if not device_queues[device_index]:
                active[device_index] = None
                return
            microbatch_id, item = device_queues[device_index].popleft()
            client = self.build_client(item, device_index, start_time_s)
            runtime = AsyncClientRuntime(
                client=client,
                device_index=device_index,
                microbatch_id=microbatch_id,
                request_start_s=start_time_s,
            )
            active[device_index] = runtime
            by_request[client.prompt_id] = runtime
            self.prefix_state.register_request(client.prompt_id, client.prefix_ids)
            fill_inflight(runtime)

        def reconstruct_speculation(runtime: AsyncClientRuntime) -> None:
            state = self.prefix_state.state(runtime.client.prompt_id)
            runtime.speculative_ids = []
            position = len(state.generated_ids)
            for segment in sorted(
                runtime.pending_segments.values(),
                key=lambda item: item.base_position,
            ):
                if (
                    segment.prefix_version == state.prefix_version
                    and segment.base_position == position
                ):
                    runtime.speculative_ids.extend(segment.draft_ids)
                    position += len(segment.draft_ids)

        def edge_queue_depth() -> int:
            return sum(len(lane.local_queue) for lane in self.lanes) + len(
                waiting_segments
            )

        def fill_inflight(runtime: AsyncClientRuntime) -> None:
            client = runtime.client
            if client.done:
                return
            while runtime.inflight_count < self.config.max_inflight_segments:
                committed = len(client.generated_ids)
                speculative = len(runtime.speculative_ids)
                remaining = self.config.max_new_tokens - committed - speculative
                if remaining <= 0:
                    break
                state = self.prefix_state.state(client.prompt_id)
                runtime.local_prefix_version = state.prefix_version
                profile = self.profiles[client.device_id]
                accepted = (
                    client.accepted_draft_tokens / client.proposed_draft_tokens
                    if client.proposed_draft_tokens
                    else 1.0
                )
                draft_len = self.lookahead.select(
                    profile=profile,
                    acceptance_rate=accepted,
                    edge_queue_depth=edge_queue_depth(),
                    remaining_tokens=remaining,
                )
                if draft_len <= 0:
                    break
                prefix = client.prefix_ids + client.generated_ids + runtime.speculative_ids
                base_position = committed + speculative
                segment = self.make_segment(
                    runtime.microbatch_id,
                    client.sync_rounds + runtime.inflight_count,
                    runtime.next_segment_id,
                    client,
                    self.draft_backends[runtime.device_index],
                    prefix_ids=prefix,
                    draft_len=draft_len,
                    prefix_version=runtime.local_prefix_version,
                    base_position=base_position,
                    prefix_hash=stable_prefix_hash(prefix),
                )
                runtime.next_segment_id += 1
                runtime.pending_segments[segment.segment_id] = segment
                runtime.speculative_ids.extend(segment.draft_ids)
                client.available_at_s = segment.draft_end_s
                push_event(segment.arrival_s, "arrival", segment)

        def drop_segment(segment: DraftSegment, reason: str) -> None:
            runtime = by_request.get(segment.request_id)
            if runtime is not None:
                runtime.pending_segments.pop(segment.segment_id, None)
                runtime.stale_segment_count += 1
                reconstruct_speculation(runtime)
            device_trace = segment_device_trace(
                segment,
                status=reason,
            )
            traces.append(
                verification_event_trace(
                    event_index=len(traces),
                    method=self.method_name,
                    microbatch_id=segment.microbatch_id,
                    round_index=segment.round_index,
                    target_batch_size=0,
                    target_forward_s=0.0,
                    devices=[device_trace],
                    status=reason,
                )
            )

        def schedule_segment(
            segment: DraftSegment,
            now_s: float,
            *,
            defer_pending: bool = True,
        ) -> bool:
            check = self.prefix_state.check_segment(segment)
            if check.ready:
                assignment = self.scheduler.assign(segment, self.lanes, now_s)
                task = VerificationTask(
                    segment=segment,
                    lane_id=assignment.lane.lane_id,
                    enqueue_s=max(now_s, segment.arrival_s),
                    scheduler_cost=assignment.cost,
                    scheduler_cost_terms=assignment.terms,
                )
                assignment.lane.local_queue.append(task)
                if len(assignment.lane.local_queue) >= self.config.lane_batch_size:
                    process_lane(assignment.lane, task.enqueue_s)
                elif self.config.lane_batch_timeout_s == 0:
                    process_lane(assignment.lane, task.enqueue_s)
                elif assignment.lane.flush_s is None:
                    assignment.lane.flush_s = (
                        task.enqueue_s + self.config.lane_batch_timeout_s
                    )
                    push_event(assignment.lane.flush_s, "flush", assignment.lane.lane_id)
                return True
            if check.status == "pending":
                if defer_pending:
                    waiting_segments.append(segment)
                return False
            drop_segment(segment, check.reason or check.status)
            return True

        def schedule_waiting(now_s: float) -> None:
            nonlocal draining_waiting, waiting_segments
            if draining_waiting:
                return

            draining_waiting = True
            try:
                while waiting_segments:
                    pending = waiting_segments
                    waiting_segments = []
                    remaining: list[DraftSegment] = []
                    for segment in pending:
                        scheduled = schedule_segment(
                            segment,
                            now_s,
                            defer_pending=False,
                        )
                        if not scheduled:
                            remaining.append(segment)
                    if len(remaining) == len(pending) and not waiting_segments:
                        waiting_segments = remaining
                        break
                    waiting_segments = remaining + waiting_segments
            finally:
                draining_waiting = False

        def invalidate_runtime_pending(
            runtime: AsyncClientRuntime,
            *,
            keep_segment_id: int | None = None,
        ) -> None:
            invalidated = [
                segment_id
                for segment_id in runtime.pending_segments
                if segment_id != keep_segment_id
            ]
            runtime.invalidated_segment_count += len(invalidated)
            for segment_id in invalidated:
                runtime.pending_segments.pop(segment_id, None)
            runtime.speculative_ids = []

        def finish_request(runtime: AsyncClientRuntime, completion_time_s: float) -> None:
            client = runtime.client
            client.done = True
            invalidate_runtime_pending(runtime)
            client.latency_s = completion_time_s - runtime.request_start_s
            records.append(
                self.record_client(
                    client,
                    runtime.microbatch_id,
                    runtime.request_start_s,
                    self.method_name,
                    {
                        "async_rounds": client.sync_rounds,
                        "lane_ids": runtime.lane_history.copy(),
                        "lane_switches": sum(
                            1
                            for previous, current in zip(
                                runtime.lane_history,
                                runtime.lane_history[1:],
                            )
                            if previous != current
                        ),
                        "stale_segment_count": runtime.stale_segment_count,
                        "invalidated_segment_count": runtime.invalidated_segment_count,
                    },
                )
            )
            if progress is not None:
                progress.update(1)
                progress.set_postfix(
                    {
                        "events": len(traces),
                        "virtual_s": f"{completion_time_s:.2f}",
                        "lanes": self.config.lane_count,
                    },
                    refresh=False,
                )
            active[runtime.device_index] = None
            by_request.pop(client.prompt_id, None)
            start_next_request(runtime.device_index, completion_time_s)

        def apply_outcome_to_runtime(
            runtime: AsyncClientRuntime,
            segment: DraftSegment,
            emitted: list[int],
            verification_accepted_count: int,
            rejected: bool,
            receive_time_s: float,
        ) -> None:
            client = runtime.client
            state = self.prefix_state.state(client.prompt_id)
            runtime.pending_segments.pop(segment.segment_id, None)
            client.generated_ids = state.generated_ids.copy()
            client.proposed_draft_tokens += len(segment.draft_ids)
            client.accepted_draft_tokens += verification_accepted_count
            client.sync_rounds += 1
            runtime.local_prefix_version = state.prefix_version
            if rejected:
                invalidate_runtime_pending(runtime)
            else:
                reconstruct_speculation(runtime)
            client.available_at_s = max(client.available_at_s, receive_time_s)
            if emitted and client.first_token_latency_s is None:
                client.first_token_latency_s = receive_time_s - runtime.request_start_s
            client.latency_s = receive_time_s - runtime.request_start_s

            eos = getattr(self.target_backend, "eos_token_id", None)
            if eos is not None and eos in emitted:
                finish_request(runtime, receive_time_s)
                return
            if len(client.generated_ids) >= self.config.max_new_tokens:
                finish_request(runtime, receive_time_s)
                return
            fill_inflight(runtime)

        def process_lane(lane: VerifierLaneState, now_s: float) -> None:
            lane.flush_s = None
            if not lane.local_queue:
                return
            raw_tasks = lane.local_queue[: self.config.lane_batch_size]
            lane.local_queue = lane.local_queue[self.config.lane_batch_size :]
            tasks: list[VerificationTask] = []
            for task in raw_tasks:
                check = self.prefix_state.check_segment(task.segment)
                if check.ready:
                    tasks.append(task)
                elif check.status == "pending":
                    waiting_segments.append(task.segment)
                else:
                    drop_segment(task.segment, check.reason or check.status)
            if not tasks:
                return

            start_s = max(
                lane.available_at_s,
                now_s,
                max(task.enqueue_s for task in tasks),
            )
            target_dists, target_elapsed = self.target_backend.target_distributions(
                [task.segment.prefix_ids for task in tasks],
                [task.segment.draft_ids for task in tasks],
                self.config.sampling,
            )
            finish_s = start_s + target_elapsed
            lane.available_at_s = finish_s
            lane.busy_s += target_elapsed
            lane.verification_count += len(tasks)

            device_traces: list[dict] = []
            for task, dists in zip(tasks, target_dists):
                segment = task.segment
                runtime = by_request.get(segment.request_id)
                if runtime is None:
                    drop_segment(segment, "request-completed")
                    continue
                check = self.prefix_state.check_segment(segment)
                if not check.ready:
                    if check.status == "pending":
                        waiting_segments.append(segment)
                    else:
                        drop_segment(segment, check.reason or check.status)
                    continue
                verification = verify_draft_exact(
                    segment.draft_ids,
                    segment.draft_dists,
                    dists,
                    self.rng,
                )
                remaining = self.config.max_new_tokens - len(runtime.client.generated_ids)
                if verification.rejected:
                    emitted = verification.emitted_ids[:remaining]
                else:
                    emitted = verification.emitted_ids[: len(segment.draft_ids)]
                    emitted = emitted[:remaining]
                state = self.prefix_state.apply_verification(
                    segment,
                    verification,
                    emitted,
                )
                lane.cached_request_id = segment.request_id
                lane.cached_prefix_hashes.add(state.prefix_hash)
                downlink_bytes = estimate_downlink_payload_bytes(emitted)
                downlink = self.network_delay_sample(
                    downlink_bytes,
                    self.profiles[segment.device_id],
                    "downlink",
                    finish_s,
                )
                receive_time_s = finish_s + downlink.delay_s
                runtime.lane_history.append(lane.lane_id)
                outcome_status = "rejected" if verification.rejected else "accepted"
                outcome = VerificationOutcome(
                    segment=segment,
                    verification=verification,
                    emitted_ids=emitted,
                    lane_id=lane.lane_id,
                    target_forward_s=target_elapsed,
                    verify_start_s=start_s,
                    verify_finish_s=finish_s,
                    queue_wait_s=start_s - segment.arrival_s,
                    status=outcome_status,
                    downlink_s=downlink.delay_s,
                    downlink_effective_mbps=downlink.effective_mbps,
                    downlink_effective_rtt_ms=downlink.effective_rtt_ms,
                    downlink_jitter_s=downlink.jitter_s,
                    downlink_congested=downlink.congested,
                    downlink_payload_bytes=downlink_bytes,
                )
                device_traces.append(
                    outcome_device_trace(
                        outcome,
                        status=outcome_status,
                    )
                )
                apply_outcome_to_runtime(
                    runtime,
                    segment,
                    emitted,
                    verification.accepted_count,
                    verification.rejected,
                    receive_time_s,
                )
                if verification.rejected:
                    schedule_waiting(finish_s)

            if device_traces:
                traces.append(
                    verification_event_trace(
                        event_index=len(traces),
                        method=self.method_name,
                        microbatch_id=tasks[0].segment.microbatch_id,
                        round_index=tasks[0].segment.round_index,
                        target_batch_size=len(device_traces),
                        target_forward_s=target_elapsed,
                        devices=device_traces,
                        lane_id=lane.lane_id,
                        lane_start_s=start_s,
                        lane_finish_s=finish_s,
                        lane_queue_wait_s=sum(
                            start_s - task.segment.arrival_s for task in tasks
                        )
                        / len(tasks),
                    )
                )
            if lane.local_queue and lane.flush_s is None:
                lane.flush_s = finish_s + self.config.lane_batch_timeout_s
                push_event(lane.flush_s, "flush", lane.lane_id)
            schedule_waiting(finish_s)

        for device_index in range(len(self.draft_backends)):
            start_next_request(device_index, 0.0)

        while events:
            now_s, _, kind, payload = heapq.heappop(events)
            if kind == "arrival":
                schedule_segment(payload, now_s)  # type: ignore[arg-type]
                schedule_waiting(now_s)
            elif kind == "flush":
                lane = self.lanes[int(payload)]
                if lane.flush_s is not None and abs(lane.flush_s - now_s) < 1e-12:
                    process_lane(lane, now_s)

        for lane in self.lanes:
            if lane.local_queue:
                process_lane(lane, lane.flush_s or lane.available_at_s)
        for segment in waiting_segments:
            drop_segment(segment, "unresolved-prefix")

        total_time_s = (
            max(float(record["completion_time_s"]) for record in records)
            if records
            else 0.0
        )
        summary = summarize_lanes(records, traces, total_time_s, self.lanes)
        summary["method"] = self.method_name
        summary["profiles"] = self.profiles_summary()
        return ExperimentResult(records, traces, summary)

