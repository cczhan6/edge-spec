from __future__ import annotations

import heapq
import itertools
import random
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from src.communication import network_delay_ms
from src.config import build_devices
from src.dip_sd import build_fixed_epoch_plan, optimize_epoch_plan
from src.entities import (
    ACTIVE_SEGMENT_STATUSES,
    FINAL_SEGMENT_STATUSES,
    Device,
    DeviceRuntime,
    Request,
    Segment,
    SimulationResult,
    VerifierLane,
)
from src.events import Event, EventType
from src.latency import (
    AcceptanceWindowEstimator,
    draft_latency_ms,
    expected_emitted_tokens,
    target_only_latency_ms,
    verify_latency_ms,
)
from src.methods import MethodSpec, get_method_spec
from src.scheduler import LaneScheduler
from src.model_runner import (
    DraftCandidateTree,
    ModelRunner,
    SemanticTreeVerifyInput,
    SemanticVerifyInput,
    VerificationResult,
    concat_linear_prefix_tree,
    rebase_draft_tree,
)
from src.tree_drafting import DraftTreePlan, LinearDraftTreeStrategy, build_tree_draft_strategy
from src.workload import WorkloadItem


@dataclass(frozen=True)
class _DraftBuild:
    draft_ids: list[int]
    draft_tree: DraftCandidateTree | None
    processed_candidate_count: int
    retained_tree_nodes: int
    target_verify_tree_nodes: int


class Simulator:
    """Event-driven semantic-in-the-loop simulator with analytical latency."""

    def __init__(
        self,
        config: dict[str, Any],
        model_runner: ModelRunner,
        workload: Sequence[WorkloadItem],
        scenario: str,
        method: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        expected_requests = int(config["simulation"]["num_requests"])
        if len(workload) != expected_requests:
            raise ValueError(
                f"workload contains {len(workload)} prompts, expected {expected_requests}"
            )
        self.config = config
        self.model_runner = model_runner
        self.workload = list(workload)
        self._progress_callback = progress_callback
        self.scenario = scenario
        self.spec: MethodSpec = get_method_spec(method, config)
        self.devices = build_devices(config, self.spec.device_pool)
        self.device_runtimes = [DeviceRuntime(device) for device in self.devices]
        self.requests: list[Request] = []
        self.segments: list[Segment] = []
        self.lanes = [VerifierLane(index) for index in range(self.spec.num_lanes)]
        self.scheduler = LaneScheduler(self.spec.lane_assignment)
        self.acceptance = AcceptanceWindowEstimator(
            int(config["speculation"]["acceptance_window_rounds"])
        )
        self.events: list[Event] = []
        self._event_ids = itertools.count()
        self._rng = random.Random(int(config["simulation"]["seed"]))
        self._verification_results: dict[int, VerificationResult] = {}
        self._batch_buffer: list[int] = []
        self._batch_busy_until_ms = 0.0
        self._server_only_verify_available_ms = 0.0
        self._server_only_request_queue: list[int] = []
        self._server_only_active_request_id: int | None = None
        self._batch_timeout_token = 0
        self._batch_timeout_deadline_ms: float | None = None
        self._batch_flush_token = 0
        self._batch_flush_deadline_ms: float | None = None
        self._target_only_available_ms = [0.0]
        self._batch_waiting_time_ms = 0.0
        self._phase_waiting_time_ms = 0.0
        self._lane_queue_wait_times_ms: list[float] = []
        self._trace: list[dict[str, Any]] = []
        self._specedge_tree_strategy = build_tree_draft_strategy(config, "specedge")
        self._server_only_tree_strategy = build_tree_draft_strategy(config, "server_only")
        self._proactive_tree_strategy = build_tree_draft_strategy(
            config,
            "specedge",
            proactive=True,
        )
        if self.spec.candidate_strategy == "linear" and self.spec.runtime == "specedge":
            specedge_config = config["specedge"]
            self._specedge_tree_strategy = LinearDraftTreeStrategy(
                max_beam_len=int(specedge_config["max_beam_len"]),
                max_budget=int(specedge_config["max_budget"]),
            )
            self._proactive_tree_strategy = LinearDraftTreeStrategy(
                max_beam_len=int(specedge_config["proactive_max_beam_len"]),
                max_budget=int(specedge_config["proactive_max_budget"]),
            )
        self._last_specedge_verify_ms = self._initial_specedge_verify_ms()
        self._pipeline_idle_bubble_ms = 0.0

    def run(self) -> SimulationResult:
        if self.spec.runtime == "dip_sd":
            return self._run_dip_sd_greedy()
        self._schedule_request_arrivals()
        while self.events or self._batch_buffer:
            if not self.events:
                if not self._maybe_start_batch(self._batch_flush_time_ms(), force=True):
                    break
                continue
            event = heapq.heappop(self.events)
            self._dispatch(event)
        unfinished = [
            request.request_id
            for request in self.requests
            if request.status != "finished"
        ]
        if unfinished:
            raise RuntimeError(f"simulation ended with unfinished requests: {unfinished}")
        return SimulationResult(
            method=self.spec.name,
            scenario=self.scenario,
            requests=self.requests,
            segments=self.segments,
            devices=self.device_runtimes,
            lanes=self.lanes,
            batch_waiting_time_ms=self._batch_waiting_time_ms,
            phase_waiting_time_ms=self._phase_waiting_time_ms,
            lane_queue_wait_times_ms=self._lane_queue_wait_times_ms,
            event_trace=self._trace,
        )

    def _run_dip_sd_greedy(self) -> SimulationResult:
        self._schedule_request_arrivals()
        self.events.clear()
        dip_config = self.config["dip_sd"]
        max_active = int(dip_config["max_active_requests"])
        waiting = [request.request_id for request in self.requests]
        active: list[int] = []
        request_ready_ms = {
            request.request_id: request.arrival_time_ms for request in self.requests
        }
        server_available_ms = 0.0
        epoch_index = 0

        while any(request.status != "finished" for request in self.requests):
            now_ms = server_available_ms
            if not active:
                next_arrival = min(
                    (
                        self.requests[request_id].arrival_time_ms
                        for request_id in waiting
                    ),
                    default=now_ms,
                )
                now_ms = max(now_ms, next_arrival)
            eligible = [
                request_id
                for request_id in list(waiting)
                if self.requests[request_id].arrival_time_ms <= now_ms
            ]
            for request_id in eligible:
                if len(active) >= max_active:
                    break
                waiting.remove(request_id)
                active.append(request_id)
                request_ready_ms[request_id] = max(request_ready_ms[request_id], now_ms)
                self._trace.append(
                    {
                        "event": "dip_sd_admit",
                        "method": self.spec.name,
                        "request_id": request_id,
                        "epoch": epoch_index,
                        "time_ms": now_ms,
                    }
                )
            active = [
                request_id
                for request_id in active
                if self.requests[request_id].status == "running"
            ]
            if not active:
                server_available_ms = now_ms
                continue

            if self.spec.name == "dip_sd":
                plan = optimize_epoch_plan(
                    active,
                    acceptance_estimates={
                        request_id: self.devices[self.requests[request_id].device_id].acceptance_prior
                        for request_id in active
                    },
                    max_batch_count=int(dip_config["batch_count"]),
                    min_draft_length=int(dip_config["min_draft_length"]),
                    max_draft_length=int(dip_config["max_draft_length"]),
                    max_batch_size=int(dip_config["max_batch_size"]),
                )
            else:
                plan = build_fixed_epoch_plan(
                    active,
                    batch_count=int(dip_config["batch_count"]),
                    draft_length=int(dip_config["draft_length"]),
                    min_draft_length=int(dip_config["min_draft_length"]),
                    max_draft_length=int(dip_config["max_draft_length"]),
                    max_batch_size=int(dip_config["max_batch_size"]),
                )
            self._trace.append(
                {
                    "event": "dip_sd_epoch_plan",
                    "method": self.spec.name,
                    "epoch": epoch_index,
                    "time_ms": now_ms,
                    "batches": [list(batch) for batch in plan.batches],
                    "draft_lengths": dict(plan.draft_lengths),
                    "optimizer": plan.optimizer,
                    "objective": plan.objective,
                    "expected_useful_tokens": plan.expected_useful_tokens,
                    "pipeline_span": plan.pipeline_span,
                }
            )
            epoch_result_arrivals: list[float] = []
            for batch_index, batch in enumerate(plan.batches):
                segments: list[Segment] = []
                for request_id in batch:
                    request = self.requests[request_id]
                    if request.status != "running":
                        continue
                    remaining = request.output_len - request.committed_pos
                    if remaining <= 0:
                        continue
                    device = self.devices[request.device_id]
                    prefix_ids = request.prompt_ids + request.generated_ids
                    draft_len = min(plan.draft_lengths[request_id], remaining)
                    draft_start_ms = max(
                        request_ready_ms[request_id],
                        request.arrival_time_ms,
                    )
                    draft_ids = self.model_runner.draft(
                        device.drafter_profile,
                        prefix_ids,
                        draft_len,
                    )
                    if not draft_ids:
                        raise RuntimeError("semantic drafter returned an empty dip_sd segment")
                    draft_compute_ms = draft_latency_ms(device, len(draft_ids))
                    draft_done_ms = draft_start_ms + draft_compute_ms
                    uplink_payload_bytes = self._payload_bytes(len(draft_ids))
                    uplink_delay_ms = self._network_delay_ms(
                        device,
                        "uplink",
                        f"dip-sd-up:{epoch_index}:{request_id}",
                        uplink_payload_bytes,
                    )
                    edge_arrival_ms = draft_done_ms + uplink_delay_ms
                    segment = Segment(
                        segment_id=len(self.segments),
                        request_id=request_id,
                        device_id=request.device_id,
                        draft_model=device.drafter_profile,
                        prefix_version=request.prefix_version,
                        base_pos=request.committed_pos,
                        scheduled_gamma=len(draft_ids),
                        prefix_ids=prefix_ids,
                        draft_ids=list(draft_ids),
                        create_time_ms=draft_start_ms,
                        draft_start_time_ms=draft_start_ms,
                        draft_compute_ms=draft_compute_ms,
                        draft_analytical_ms=draft_compute_ms,
                        uplink_delay_ms=uplink_delay_ms,
                        uplink_payload_tokens=len(draft_ids),
                        uplink_payload_bytes=uplink_payload_bytes,
                        edge_arrival_time_ms=edge_arrival_ms,
                        tree_strategy="linear",
                        tree_budget_nodes=len(draft_ids),
                        draft_compute_nodes=len(draft_ids),
                        processed_candidate_count=len(draft_ids),
                        retained_tree_nodes=len(draft_ids),
                        target_verify_tree_nodes=1,
                        beam_len=len(draft_ids),
                    )
                    self.segments.append(segment)
                    segments.append(segment)
                    runtime = self.device_runtimes[request.device_id]
                    runtime.total_busy_time_ms += draft_compute_ms
                    runtime.generated_draft_tokens += segment.proposed_count
                    runtime.selected_gammas.append(len(draft_ids))
                    self._trace.append(
                        {
                            "event": "dip_sd_draft",
                            "method": self.spec.name,
                            "epoch": epoch_index,
                            "batch_index": batch_index,
                            "request_id": request_id,
                            "segment_id": segment.segment_id,
                            "device_id": request.device_id,
                            "draft_model": segment.draft_model,
                            "scheduled_gamma": segment.scheduled_gamma,
                            "start_time_ms": draft_start_ms,
                            "finish_time_ms": draft_done_ms,
                            "compute_ms": draft_compute_ms,
                            "uplink_ms": uplink_delay_ms,
                            "uplink_payload_bytes": uplink_payload_bytes,
                        }
                    )
                if not segments:
                    continue
                batch_ready_ms = max(float(segment.edge_arrival_time_ms) for segment in segments)
                verify_start_ms = max(server_available_ms, batch_ready_ms)
                if verify_start_ms > server_available_ms:
                    self._pipeline_idle_bubble_ms += verify_start_ms - server_available_ms
                results = self._verify_segments(segments)
                verify_ms = self._verify_latency_for_segments(segments)
                verify_done_ms = verify_start_ms + verify_ms
                server_available_ms = verify_done_ms
                self._trace.append(
                    {
                        "event": "dip_sd_batch_verify",
                        "method": self.spec.name,
                        "epoch": epoch_index,
                        "batch_index": batch_index,
                        "segment_ids": [segment.segment_id for segment in segments],
                        "batch_size": len(segments),
                        "start_time_ms": verify_start_ms,
                        "finish_time_ms": verify_done_ms,
                        "compute_ms": verify_ms,
                    }
                )
                for segment, result in zip(segments, results):
                    request = self.requests[segment.request_id]
                    remaining = request.output_len - request.committed_pos
                    emitted_ids = list(result.committed_tokens[:remaining])
                    segment.accepted_count = min(result.accepted_count, len(segment.draft_ids))
                    segment.emitted_ids = emitted_ids
                    segment.result_base_pos = segment.base_pos
                    segment.verify_start_time_ms = verify_start_ms
                    segment.verify_done_time_ms = verify_done_ms
                    segment.verify_compute_ms = verify_ms
                    segment.status = "rejected" if result.rejected else "accepted"
                    request.accepted_tokens += segment.accepted_count
                    self.device_runtimes[segment.device_id].accepted_draft_tokens += segment.accepted_count
                    if result.rejected:
                        request.rejected_count += 1
                        request.rollback_count += 1
                        request.wasted_draft_tokens += max(
                            0,
                            segment.proposed_count - segment.accepted_count,
                        )
                    request.edge_generated_ids.extend(emitted_ids)
                    downlink_payload_bytes = self._payload_bytes(len(emitted_ids))
                    downlink_delay_ms = self._network_delay_ms(
                        self.devices[segment.device_id],
                        "downlink",
                        f"dip-sd-down:{epoch_index}:{segment.request_id}",
                        downlink_payload_bytes,
                    )
                    segment.downlink_payload_bytes = downlink_payload_bytes
                    segment.downlink_delay_ms = downlink_delay_ms
                    result_arrival_ms = verify_done_ms + downlink_delay_ms
                    request.generated_ids.extend(emitted_ids)
                    request_ready_ms[request.request_id] = result_arrival_ms
                    epoch_result_arrivals.append(result_arrival_ms)
                    self._trace.append(
                        {
                            "event": "dip_sd_result",
                            "method": self.spec.name,
                            "epoch": epoch_index,
                            "batch_index": batch_index,
                            "request_id": request.request_id,
                            "segment_id": segment.segment_id,
                            "accepted_count": segment.accepted_count,
                            "emitted_count": segment.emitted_count,
                            "finish_time_ms": result_arrival_ms,
                            "downlink_ms": downlink_delay_ms,
                            "downlink_payload_bytes": downlink_payload_bytes,
                        }
                    )
                    if request.generated_ids and (
                        len(request.generated_ids) >= request.output_len
                        or (
                            self.model_runner.eos_token_id is not None
                            and self.model_runner.eos_token_id in request.generated_ids
                        )
                    ):
                        request.status = "finished"
                        request.finish_time_ms = result_arrival_ms
                        self._trace.append(
                            {
                                "event": "request_finish",
                                "method": self.spec.name,
                                "request_id": request.request_id,
                                "device_id": request.device_id,
                                "finish_time_ms": result_arrival_ms,
                            }
                        )
                        if self._progress_callback is not None:
                            self._progress_callback(
                                sum(item.status == "finished" for item in self.requests),
                                len(self.requests),
                            )
            epoch_end_ms = max([server_available_ms, *epoch_result_arrivals], default=server_available_ms)
            server_available_ms = epoch_end_ms
            active = [
                request_id
                for request_id in active
                if self.requests[request_id].status == "running"
            ]
            epoch_index += 1

        return SimulationResult(
            method=self.spec.name,
            scenario=self.scenario,
            requests=self.requests,
            segments=self.segments,
            devices=self.device_runtimes,
            lanes=self.lanes,
            batch_waiting_time_ms=self._batch_waiting_time_ms,
            phase_waiting_time_ms=self._phase_waiting_time_ms,
            lane_queue_wait_times_ms=self._lane_queue_wait_times_ms,
            event_trace=self._trace,
        )

    def _batch_flush_time_ms(self) -> float:
        buffered_arrivals = [
            float(self.segments[item].edge_arrival_time_ms)
            for item in self._batch_buffer
            if self.segments[item].edge_arrival_time_ms is not None
        ]
        return max([self._batch_busy_until_ms, *buffered_arrivals], default=self._batch_busy_until_ms)

    def predict_verify_latency_ms(self, gamma: int) -> float:
        return verify_latency_ms(self.config["edge"], [1])

    def _is_specedge_runtime(self) -> bool:
        return self.spec.runtime in {"specedge", "server_only_specedge"}

    def _is_server_only_runtime(self) -> bool:
        return self.spec.runtime == "server_only_specedge"

    def _allows_out_of_order_verify(self) -> bool:
        return self.spec.runtime == "async"

    def _has_fixed_speculation_window(self) -> bool:
        return self.spec.window_size > 0

    def _uses_unconfirmed_token_budget(self) -> bool:
        return self.spec.runtime == "async"

    def _unconfirmed_token_budget(self) -> int:
        speculation = self.config["speculation"]
        return int(
            speculation.get(
                "unconfirmed_token_budget",
                speculation.get("W_max", 0),
            )
        )

    def _unconfirmed_draft_tokens(self, request: Request) -> int:
        if not self._uses_unconfirmed_token_budget():
            return 0
        token_count = 0
        seen_segment_ids: set[int] = set()
        for segment_id in request.pending_segments.values():
            if segment_id in seen_segment_ids:
                continue
            seen_segment_ids.add(segment_id)
            segment = self.segments[segment_id]
            if segment.status in ACTIVE_SEGMENT_STATUSES:
                token_count += segment.gamma
        return token_count

    def _remaining_unconfirmed_token_budget(self, request: Request) -> int:
        if not self._uses_unconfirmed_token_budget():
            return request.output_len
        return max(
            0,
            self._unconfirmed_token_budget() - self._unconfirmed_draft_tokens(request),
        )

    def _observe_unconfirmed_token_budget(self, request: Request) -> None:
        request.max_unconfirmed_tokens_observed = max(
            request.max_unconfirmed_tokens_observed,
            self._unconfirmed_draft_tokens(request),
        )

    def _global_batch_size(self) -> int:
        if self._is_specedge_runtime():
            batch_size = self.config["specedge"].get("server_batch_size")
            if batch_size is None:
                batch_size = 1
            return int(batch_size)
        return int(self.config["sync_batch"]["B_global"])

    def _global_batch_timeout_ms(self) -> float | None:
        if self._is_specedge_runtime():
            timeout_ms = self.config["specedge"].get("server_batch_timeout_ms")
            if timeout_ms is None:
                return None
            return float(timeout_ms)
        return float(self.config["sync_batch"]["global_batch_timeout_ms"])

    def _specedge_max_beam_len(self) -> int:
        return self._specedge_tree_strategy.max_beam_len

    def _specedge_proactive_type(self) -> str:
        return str(self.config["specedge"].get("proactive_type", "excluded"))

    def _specedge_proactive_enabled(self) -> bool:
        return (
            bool(self.config["specedge"].get("proactive_enabled", True))
            and self._specedge_proactive_type() != "disabled"
        )

    def _specedge_server_batch_type(self) -> str:
        return str(self.config["specedge"].get("server_batch_type", "static"))

    def _uses_specedge_dynamic_batch(self) -> bool:
        return self._is_specedge_runtime() and self._specedge_server_batch_type() == "dynamic"

    def _initial_specedge_verify_ms(self) -> float:
        if not self._is_specedge_runtime():
            return 0.0
        batch_size = max(1, self._global_batch_size())
        strategy = (
            self._server_only_tree_strategy
            if self._is_server_only_runtime()
            else self._specedge_tree_strategy
        )
        target_verify_nodes = strategy.plan(strategy.max_beam_len).target_verify_nodes
        return verify_latency_ms(
            self.config["edge"],
            [target_verify_nodes for _ in range(batch_size)],
        )

    def _specedge_tree_plan(self, beam_len: int) -> DraftTreePlan:
        return self._specedge_tree_strategy.plan(beam_len)

    def _server_only_tree_plan(self, beam_len: int) -> DraftTreePlan:
        return self._server_only_tree_strategy.plan(beam_len)

    def _proactive_tree_plan(self, beam_len: int) -> DraftTreePlan:
        return self._proactive_tree_strategy.plan(beam_len)

    def _specedge_tree_budget_nodes(self, beam_len: int) -> int:
        return self._specedge_tree_plan(beam_len).tree_budget_nodes

    def _draft_for_plan(
        self,
        drafter_profile: str,
        prefix_ids: Sequence[int],
        plan: DraftTreePlan,
    ) -> _DraftBuild:
        if plan.strategy == "linear":
            draft_ids = self.model_runner.draft(
                drafter_profile,
                prefix_ids,
                plan.path_token_count,
            )
            return _DraftBuild(
                draft_ids=list(draft_ids),
                draft_tree=None,
                processed_candidate_count=len(draft_ids),
                retained_tree_nodes=len(draft_ids),
                target_verify_tree_nodes=1 if draft_ids else 0,
            )
        draft_tree = self.model_runner.draft_tree(drafter_profile, prefix_ids, plan)
        return _DraftBuild(
            draft_ids=list(draft_tree.primary_ids),
            draft_tree=draft_tree,
            processed_candidate_count=draft_tree.processed_candidate_count,
            retained_tree_nodes=draft_tree.retained_tree_nodes,
            target_verify_tree_nodes=draft_tree.target_verify_tree_nodes,
        )

    def _verify_segment(self, segment: Segment) -> VerificationResult:
        if segment.draft_tree is not None:
            return self.model_runner.verify_tree(segment.prefix_ids, segment.draft_tree)
        if segment.tree_strategy != "linear":
            raise RuntimeError("tree draft segment is missing its draft tree")
        return self.model_runner.verify(segment.prefix_ids, segment.draft_ids)

    def _verify_segments(self, segments: Sequence[Segment]) -> list[VerificationResult]:
        missing_tree = [
            segment for segment in segments
            if segment.tree_strategy != "linear" and segment.draft_tree is None
        ]
        if missing_tree:
            raise RuntimeError("tree draft segment is missing its draft tree")
        tree_segments = [segment for segment in segments if segment.draft_tree is not None]
        if tree_segments and len(tree_segments) != len(segments):
            raise RuntimeError("linear and tree draft segments must not share one verify batch")
        if tree_segments:
            return self.model_runner.verify_tree_batch(
                [
                    SemanticTreeVerifyInput(
                        segment.prefix_ids,
                        segment.draft_tree,
                    )
                    for segment in segments
                    if segment.draft_tree is not None
                ]
            )
        return self.model_runner.verify_batch(
            [SemanticVerifyInput(segment.prefix_ids, segment.draft_ids) for segment in segments]
        )

    def _segment_payload_tokens(self, segment: Segment) -> int:
        return segment.draft_payload_tokens

    def _verify_latency_for_segments(self, segments: Sequence[Segment]) -> float:
        if self._is_specedge_runtime():
            return verify_latency_ms(
                self.config["edge"],
                [segment.target_verify_tree_nodes for segment in segments],
            )
        return verify_latency_ms(self.config["edge"], [1 for _ in segments])

    def _server_only_draft_latency_ms(self, token_count: int) -> float:
        if token_count <= 0:
            return 0.0
        server_only = self.config.get("server_only", {})
        startup_ms = float(
            server_only.get("draft_startup_ms", self.config["edge"]["verify_startup_ms"])
        )
        token_rate = float(
            server_only.get(
                "draft_token_rate_tok_s",
                self.config["edge"]["target_only_token_rate_tok_s"],
            )
        )
        return startup_ms + (
            1000.0 * token_count / token_rate
        )

    def _server_only_drafter_profile(self) -> str:
        return str(self.config.get("server_only", {}).get("drafter_profile", "medium"))

    def _server_only_acceptance_prior(self) -> float:
        profile = self._server_only_drafter_profile()
        return float(self.config["drafter_profiles"][profile]["acceptance_prior"])

    def _dispatch(self, event: Event) -> None:
        handlers = {
            EventType.REQUEST_ARRIVE: self._on_request_arrive,
            EventType.TARGET_ONLY_ARRIVE_EDGE: self._on_target_only_arrive_edge,
            EventType.SERVER_ONLY_ARRIVE_EDGE: self._on_server_only_arrive_edge,
            EventType.SERVER_ONLY_DRAFT_DONE: self._on_server_only_draft_done,
            EventType.SERVER_ONLY_VERIFY_DONE: self._on_server_only_verify_done,
            EventType.DRAFT_DONE: self._on_draft_done,
            EventType.PROACTIVE_DRAFT_DONE: self._on_proactive_draft_done,
            EventType.PACKET_ARRIVE_EDGE: self._on_packet_arrive_edge,
            EventType.VERIFY_DONE: self._on_verify_done,
            EventType.BATCH_FLUSH: self._on_batch_flush,
            EventType.BATCH_TIMEOUT: self._on_batch_timeout,
            EventType.BATCH_VERIFY_DONE: self._on_batch_verify_done,
            EventType.RESULT_ARRIVE_DEVICE: self._on_result_arrive_device,
            EventType.REQUEST_FINISH: self._on_request_finish,
        }
        handlers[event.event_type](event.time_ms, event.payload)

    def _schedule(self, time_ms: float, event_type: EventType, payload: Any = None) -> None:
        heapq.heappush(self.events, Event(time_ms, next(self._event_ids), event_type, payload))

    def _schedule_request_arrivals(self) -> None:
        simulation = self.config["simulation"]
        current_ms = 0.0
        for request_id, item in enumerate(self.workload):
            if request_id and simulation["request_arrival"] == "poisson":
                rate = float(simulation["poisson_rate_per_s"])
                current_ms += self._rng.expovariate(rate) * 1000.0
            prompt_ids = self.model_runner.encode_prompt(item.prompt)
            request = Request(
                request_id=request_id,
                device_id=request_id % len(self.devices),
                output_len=int(self._rng.choice(simulation["output_len_choices"])),
                arrival_time_ms=current_ms,
                decode_ready_time_ms=current_ms,
                prompt_id=item.prompt_id,
                category=item.category,
                category_group=item.category_group,
                prompt=item.prompt,
                prompt_token_count=len(prompt_ids),
                prompt_ids=prompt_ids,
            )
            self.requests.append(request)
            self.device_runtimes[request.device_id].assigned_requests += 1
            self._schedule(current_ms, EventType.REQUEST_ARRIVE, request_id)

    def _on_request_arrive(self, now_ms: float, request_id: int) -> None:
        request = self.requests[request_id]
        if self._is_server_only_runtime():
            self._schedule(now_ms, EventType.SERVER_ONLY_ARRIVE_EDGE, request_id)
            return
        if self.spec.runtime != "target_only":
            self._refresh_drafting(request, now_ms)
            return
        self._schedule(now_ms, EventType.TARGET_ONLY_ARRIVE_EDGE, request_id)

    def _on_target_only_arrive_edge(self, now_ms: float, request_id: int) -> None:
        request = self.requests[request_id]
        generated_ids = self.model_runner.target_only(request.prompt_ids, request.output_len)
        compute_ms = target_only_latency_ms(self.config["edge"], len(generated_ids))
        lane_id = min(
            range(len(self._target_only_available_ms)),
            key=self._target_only_available_ms.__getitem__,
        )
        start_ms = max(now_ms, self._target_only_available_ms[lane_id])
        finish_ms = start_ms + compute_ms
        self._target_only_available_ms[lane_id] = finish_ms
        request.generated_ids = generated_ids
        request.edge_generated_ids = generated_ids.copy()
        request.target_only_queue_wait_ms = start_ms - now_ms
        request.target_only_compute_ms = compute_ms
        self._trace.append(
            {
                "event": "target_only_service",
                "method": self.spec.name,
                "request_id": request_id,
                "device_id": request.device_id,
                "lane_id": lane_id,
                "batch_size": 1,
                "start_time_ms": start_ms,
                "finish_time_ms": finish_ms,
                "compute_ms": compute_ms,
            }
        )
        self._schedule(finish_ms, EventType.REQUEST_FINISH, request_id)

    def _on_server_only_arrive_edge(self, now_ms: float, request_id: int) -> None:
        self._enqueue_server_only_request(self.requests[request_id], now_ms)

    def _enqueue_server_only_request(self, request: Request, now_ms: float) -> None:
        if request.status != "running":
            return
        if (
            self._server_only_active_request_id != request.request_id
            and request.request_id not in self._server_only_request_queue
        ):
            self._server_only_request_queue.append(request.request_id)
        self._maybe_start_server_only_request(now_ms)

    def _maybe_start_server_only_request(self, now_ms: float) -> None:
        if not self._is_server_only_runtime() or self._server_only_active_request_id is not None:
            return
        while self._server_only_request_queue:
            request_id = self._server_only_request_queue.pop(0)
            request = self.requests[request_id]
            if request.status != "running":
                continue
            self._server_only_active_request_id = request_id
            self._start_server_only_draft(request, now_ms)
            return

    def _start_server_only_draft(self, request: Request, now_ms: float) -> None:
        if self._is_server_only_runtime():
            if self._server_only_active_request_id is None:
                self._server_only_active_request_id = request.request_id
            elif self._server_only_active_request_id != request.request_id:
                return
        if request.status != "running" or request.completed_results:
            return
        remaining = request.output_len - request.committed_pos
        if remaining <= 0:
            self._schedule(now_ms, EventType.REQUEST_FINISH, request.request_id)
            return
        device = self.devices[request.device_id]
        prefix_ids = request.prompt_ids + request.generated_ids
        gamma = min(
            self._select_gamma(request, device, now_ms, remaining),
            self._server_only_tree_strategy.max_beam_len,
            remaining,
        )
        drafter_profile = self._server_only_drafter_profile()
        if self.spec.candidate_strategy == "linear":
            tree_plan = DraftTreePlan(
                strategy="linear",
                path_token_count=gamma,
                tree_budget_nodes=gamma,
                draft_compute_nodes=gamma,
                target_verify_nodes=1 if gamma else 0,
                max_n_beams=1,
                max_beam_len=gamma,
                max_branch_width=1,
                max_budget=gamma,
            )
        else:
            tree_plan = self._server_only_tree_plan(gamma)
        draft_build = self._draft_for_plan(
            drafter_profile,
            prefix_ids,
            tree_plan,
        )
        draft_ids = draft_build.draft_ids
        if not draft_ids:
            raise RuntimeError("semantic drafter returned an empty server_only segment")
        if tree_plan.path_token_count != len(draft_ids):
            tree_plan = self._server_only_tree_plan(len(draft_ids))
        processed_candidates = draft_build.processed_candidate_count
        retained_nodes = draft_build.retained_tree_nodes
        target_verify_nodes = draft_build.target_verify_tree_nodes
        draft_ms = self._server_only_draft_latency_ms(processed_candidates)
        segment = Segment(
            segment_id=len(self.segments),
            request_id=request.request_id,
            device_id=request.device_id,
            draft_model=f"server_only:{drafter_profile}",
            prefix_version=request.prefix_version,
            base_pos=request.committed_pos,
            scheduled_gamma=len(draft_ids),
            prefix_ids=prefix_ids,
            draft_ids=draft_ids,
            create_time_ms=now_ms,
            draft_start_time_ms=now_ms,
            draft_compute_ms=draft_ms,
            draft_analytical_ms=draft_ms,
            tree_strategy=tree_plan.strategy,
            tree_budget_nodes=retained_nodes,
            draft_compute_nodes=processed_candidates,
            processed_candidate_count=processed_candidates,
            retained_tree_nodes=retained_nodes,
            target_verify_tree_nodes=target_verify_nodes,
            beam_len=len(draft_ids),
            draft_tree=draft_build.draft_tree,
        )
        self.segments.append(segment)
        request.pending_segments[segment.base_pos] = segment.segment_id
        request.in_flight_segments.append(segment.segment_id)
        runtime = self.device_runtimes[request.device_id]
        runtime.generated_draft_tokens += segment.proposed_count
        runtime.selected_gammas.append(len(draft_ids))
        self._trace.append(
            {
                "event": "server_only_draft",
                "method": self.spec.name,
                "resource": "server_draft_gpu",
                "request_id": request.request_id,
                "segment_id": segment.segment_id,
                "device_id": request.device_id,
                "draft_model": segment.draft_model,
                "scheduled_gamma": segment.scheduled_gamma,
                "verify_gamma": segment.verify_gamma,
                "batch_size": 1,
                "start_time_ms": now_ms,
                "finish_time_ms": now_ms + draft_ms,
                "compute_ms": draft_ms,
                "tree_budget_nodes": segment.tree_budget_nodes,
                "draft_compute_nodes": segment.draft_compute_nodes,
                "processed_candidate_count": segment.processed_candidate_count,
                "retained_tree_nodes": segment.retained_tree_nodes,
                "target_verify_tree_nodes": segment.target_verify_tree_nodes,
                "tree_strategy": segment.tree_strategy,
            }
        )
        self._schedule(now_ms + draft_ms, EventType.SERVER_ONLY_DRAFT_DONE, segment.segment_id)

    def _on_server_only_draft_done(self, now_ms: float, segment_id: int) -> None:
        segment = self.segments[segment_id]
        request = self.requests[segment.request_id]
        if (
            segment.status != "drafting"
            or request.status != "running"
            or request.pending_segments.get(segment.base_pos) != segment_id
        ):
            if segment.status not in FINAL_SEGMENT_STATUSES:
                self._stale_segment(segment)
            return
        segment.status = "queued"
        segment.edge_arrival_time_ms = now_ms
        self._start_server_only_verify(segment, now_ms)

    def _start_server_only_verify(self, segment: Segment, now_ms: float) -> None:
        request = self.requests[segment.request_id]
        if (
            segment.status != "queued"
            or request.status != "running"
            or request.pending_segments.get(segment.base_pos) != segment.segment_id
            or segment.base_pos != request.edge_frontier_pos
        ):
            if segment.status not in FINAL_SEGMENT_STATUSES:
                self._stale_segment(segment)
            return
        result = self._verify_segment(segment)
        duration_ms = self._verify_latency_for_segments([segment])
        start_ms = max(now_ms, self._server_only_verify_available_ms)
        finish_ms = start_ms + duration_ms
        self._server_only_verify_available_ms = finish_ms
        self._verification_results[segment.segment_id] = result
        segment.status = "verifying"
        segment.verify_start_time_ms = start_ms
        segment.verify_done_time_ms = finish_ms
        segment.verify_compute_ms = duration_ms
        self._schedule(finish_ms, EventType.SERVER_ONLY_VERIFY_DONE, segment.segment_id)
        self._trace.append(
            {
                "event": "server_only_verify",
                "method": self.spec.name,
                "resource": "server_target_gpu",
                "segment_id": segment.segment_id,
                "request_id": segment.request_id,
                "device_id": segment.device_id,
                "draft_model": segment.draft_model,
                "scheduled_gamma": segment.scheduled_gamma,
                "verify_gamma": segment.verify_gamma,
                "batch_size": 1,
                "start_time_ms": start_ms,
                "finish_time_ms": finish_ms,
                "compute_ms": duration_ms,
                "queue_wait_ms": start_ms - now_ms,
                "tree_budget_nodes": segment.tree_budget_nodes,
                "draft_compute_nodes": segment.draft_compute_nodes,
                "processed_candidate_count": segment.processed_candidate_count,
                "retained_tree_nodes": segment.retained_tree_nodes,
                "target_verify_tree_nodes": segment.target_verify_tree_nodes,
                "tree_strategy": segment.tree_strategy,
            }
        )

    def _on_server_only_verify_done(self, now_ms: float, segment_id: int) -> None:
        segment = self.segments[segment_id]
        if segment.status != "verifying":
            return
        self._resolve_verification(
            segment,
            self._verification_results.pop(segment_id),
            now_ms,
        )

    def _refresh_drafting(self, request: Request, now_ms: float) -> None:
        if not self._can_queue_draft(request):
            return
        runtime = self.device_runtimes[request.device_id]
        request.draft_queued = True
        request.draft_queue_enter_ms = now_ms
        runtime.draft_queue.append(request.request_id)
        request.max_outstanding_observed = max(
            request.max_outstanding_observed,
            request.outstanding_count,
        )
        self._try_start_device(runtime, now_ms)

    def _can_queue_draft(self, request: Request) -> bool:
        if (
            request.status != "running"
            or request.draft_queued
            or request.completed_results
            or (
                self._has_fixed_speculation_window()
                and request.outstanding_count >= self.spec.window_size
            )
        ):
            return False
        prefix_ids, speculative_count, blocked = self._draft_prefix(request)
        if blocked:
            return False
        if self.model_runner.eos_token_id is not None and prefix_ids[-1] == self.model_runner.eos_token_id:
            return False
        if self._remaining_unconfirmed_token_budget(request) <= 0:
            return False
        return request.committed_pos + speculative_count < request.output_len

    def _try_start_device(self, runtime: DeviceRuntime, now_ms: float) -> None:
        if runtime.active_segment_id is not None:
            return
        while runtime.draft_queue:
            request = self.requests[runtime.draft_queue.pop(0)]
            queued_at = float(request.draft_queue_enter_ms or now_ms)
            request.draft_queued = False
            request.draft_queue_enter_ms = None
            if not self._can_start_draft(request):
                continue
            self._start_draft(runtime, request, now_ms, queued_at)
            return

    def _can_start_draft(self, request: Request) -> bool:
        if (
            request.status != "running"
            or request.completed_results
            or (
                self._has_fixed_speculation_window()
                and len(request.in_flight_segments) >= self.spec.window_size
            )
        ):
            return False
        prefix_ids, speculative_count, blocked = self._draft_prefix(request)
        if blocked:
            return False
        if self.model_runner.eos_token_id is not None and prefix_ids[-1] == self.model_runner.eos_token_id:
            return False
        if self._remaining_unconfirmed_token_budget(request) <= 0:
            return False
        return request.committed_pos + speculative_count < request.output_len

    def _start_draft(
        self,
        runtime: DeviceRuntime,
        request: Request,
        now_ms: float,
        queued_at_ms: float,
    ) -> None:
        prefix_ids, speculative_count, blocked = self._draft_prefix(request)
        if blocked:
            return
        remaining = request.output_len - request.committed_pos - speculative_count
        remaining = min(remaining, self._remaining_unconfirmed_token_budget(request))
        if remaining <= 0:
            return
        gamma = self._select_gamma(request, runtime.device, now_ms, remaining)
        if self.spec.runtime == "specedge":
            gamma = min(gamma, self._specedge_max_beam_len())
        use_proactive = bool(
            self.spec.runtime == "specedge"
            and request.proactive_draft_ids
            and request.proactive_base_pos == request.committed_pos + speculative_count
            and request.proactive_prefix_version == request.prefix_version
        )
        retained_proactive_ids: list[int] = []
        draft_tree: DraftCandidateTree | None = None
        fresh_build: _DraftBuild | None = None
        if use_proactive:
            retained_proactive_tree = request.proactive_draft_tree
            retained_limit = min(
                len(request.proactive_draft_ids),
                remaining,
                self._specedge_tree_strategy.max_budget,
            )
            retained_proactive_ids = request.proactive_draft_ids[:retained_limit]
            self._clear_proactive(request)
            if self._specedge_proactive_type() == "included":
                fresh_budget = max(0, gamma - self._proactive_tree_strategy.max_beam_len)
            else:
                fresh_budget = gamma
            fresh_budget = min(
                fresh_budget,
                remaining - len(retained_proactive_ids),
                self._specedge_tree_strategy.max_budget - len(retained_proactive_ids),
            )
            if fresh_budget > 0:
                fresh_build = self._draft_for_plan(
                    runtime.device.drafter_profile,
                    prefix_ids + retained_proactive_ids,
                    self._specedge_tree_plan(fresh_budget),
                )
                fresh_ids = fresh_build.draft_ids
            else:
                fresh_ids = []
            draft_ids = retained_proactive_ids + fresh_ids
            if fresh_build is not None and fresh_build.draft_tree is not None:
                draft_tree = concat_linear_prefix_tree(
                    prefix_ids,
                    retained_proactive_ids,
                    fresh_build.draft_tree,
                )
            elif (
                retained_proactive_tree is not None
                and retained_proactive_tree.primary_ids == retained_proactive_ids
            ):
                draft_tree = retained_proactive_tree
            else:
                draft_tree = None
        else:
            self._clear_proactive(request)
            if self.spec.runtime == "specedge":
                draft_build = self._draft_for_plan(
                    runtime.device.drafter_profile,
                    prefix_ids,
                    self._specedge_tree_plan(gamma),
                )
                draft_ids = draft_build.draft_ids
                draft_tree = draft_build.draft_tree
                fresh_build = draft_build
            else:
                draft_ids = self.model_runner.draft(runtime.device.drafter_profile, prefix_ids, gamma)
                fresh_build = _DraftBuild(
                    draft_ids=list(draft_ids),
                    draft_tree=None,
                    processed_candidate_count=len(draft_ids),
                    retained_tree_nodes=len(draft_ids),
                    target_verify_tree_nodes=1 if draft_ids else 0,
                )
            fresh_ids = draft_ids
        if not draft_ids:
            raise RuntimeError("semantic drafter returned an empty segment")
        beam_len = len(draft_ids)
        tree_plan = (
            self._specedge_tree_plan(beam_len)
            if self.spec.runtime == "specedge"
            else DraftTreePlan(
                strategy="linear",
                path_token_count=beam_len,
                tree_budget_nodes=beam_len,
                draft_compute_nodes=beam_len,
                target_verify_nodes=1 if beam_len else 0,
                max_n_beams=1,
                max_beam_len=beam_len,
                max_branch_width=1,
                max_budget=beam_len,
            )
        )
        processed_candidates = (
            draft_tree.processed_candidate_count
            if draft_tree is not None
            else beam_len
        )
        retained_nodes = (
            draft_tree.retained_tree_nodes
            if draft_tree is not None
            else beam_len
        )
        target_verify_nodes = (
            draft_tree.target_verify_tree_nodes
            if draft_tree is not None
            else 1
        )
        base_pos = request.committed_pos + speculative_count
        fresh_processed_candidates = (
            fresh_build.processed_candidate_count
            if use_proactive and fresh_build is not None
            else processed_candidates
        )
        analytical_ms = draft_latency_ms(
            runtime.device,
            processed_candidates,
        )
        fresh_compute_ms = (
            draft_latency_ms(runtime.device, fresh_processed_candidates)
            if fresh_ids
            else 0.0
        )
        duration_ms = (
            fresh_compute_ms
            if use_proactive
            else max(0.0, analytical_ms - request.overlap_credit_ms)
        )
        request.overlap_credit_ms = max(0.0, request.overlap_credit_ms - analytical_ms)
        pipeline_target_ms = 0.0
        pipeline_edge_cycle_ms = 0.0
        pipeline_alignment_error_ms = 0.0
        if self.spec.runtime == "specedge":
            alpha = self.acceptance.estimate(
                request.request_id,
                runtime.device.acceptance_prior,
            )
            pipeline_target_ms = self._specedge_pipeline_target_ms(now_ms)
            network_cycle_ms = self._specedge_edge_cycle_ms(
                runtime.device,
                beam_len,
                expected_emitted_tokens(alpha, beam_len),
            ) - draft_latency_ms(runtime.device, processed_candidates)
            pipeline_edge_cycle_ms = duration_ms + network_cycle_ms
            pipeline_alignment_error_ms = abs(pipeline_target_ms - pipeline_edge_cycle_ms)
        segment = Segment(
            segment_id=len(self.segments),
            request_id=request.request_id,
            device_id=request.device_id,
            draft_model=runtime.device.drafter_profile,
            prefix_version=request.prefix_version,
            base_pos=base_pos,
            scheduled_gamma=beam_len,
            prefix_ids=prefix_ids,
            draft_ids=draft_ids,
            create_time_ms=now_ms,
            draft_start_time_ms=now_ms,
            draft_queue_wait_ms=now_ms - queued_at_ms,
            draft_compute_ms=duration_ms,
            draft_analytical_ms=analytical_ms,
            tree_strategy=tree_plan.strategy,
            tree_budget_nodes=retained_nodes,
            draft_compute_nodes=processed_candidates,
            processed_candidate_count=processed_candidates,
            retained_tree_nodes=retained_nodes,
            target_verify_tree_nodes=target_verify_nodes,
            beam_len=beam_len,
            draft_tree=draft_tree,
            proactive_used=use_proactive,
            pipeline_target_ms=pipeline_target_ms,
            pipeline_edge_cycle_ms=pipeline_edge_cycle_ms,
            pipeline_alignment_error_ms=pipeline_alignment_error_ms,
        )
        self.segments.append(segment)
        request.pending_segments[segment.base_pos] = segment.segment_id
        request.in_flight_segments.append(segment.segment_id)
        request.max_outstanding_observed = max(
            request.max_outstanding_observed,
            request.outstanding_count,
        )
        self._observe_unconfirmed_token_budget(request)
        runtime.active_segment_id = segment.segment_id
        runtime.busy_until_ms = now_ms + duration_ms
        runtime.total_queue_wait_ms += segment.draft_queue_wait_ms
        runtime.total_busy_time_ms += duration_ms
        runtime.generated_draft_tokens += segment.proposed_count
        runtime.selected_gammas.append(beam_len)
        self._trace.append(
            {
                "event": "draft_compute",
                "method": self.spec.name,
                "request_id": request.request_id,
                "segment_id": segment.segment_id,
                "device_id": request.device_id,
                "draft_model": segment.draft_model,
                "scheduled_gamma": beam_len,
                "verify_gamma": segment.verify_gamma,
                "batch_size": 1,
                "start_time_ms": now_ms,
                "finish_time_ms": now_ms + duration_ms,
                "compute_ms": duration_ms,
                "queue_wait_ms": segment.draft_queue_wait_ms,
                "tree_budget_nodes": segment.tree_budget_nodes,
                "draft_compute_nodes": segment.draft_compute_nodes,
                "processed_candidate_count": segment.processed_candidate_count,
                "retained_tree_nodes": segment.retained_tree_nodes,
                "target_verify_tree_nodes": segment.target_verify_tree_nodes,
                "tree_strategy": segment.tree_strategy,
                "pipeline_target_ms": pipeline_target_ms,
                "pipeline_edge_cycle_ms": pipeline_edge_cycle_ms,
                "pipeline_alignment_error_ms": pipeline_alignment_error_ms,
                "proactive_used": use_proactive,
                "proactive_reused_tokens": len(retained_proactive_ids),
            }
        )
        if self.spec.runtime == "specedge":
            self._trace.append(
                {
                    "event": "pipeline_schedule",
                    "method": self.spec.name,
                    "request_id": request.request_id,
                    "segment_id": segment.segment_id,
                    "device_id": request.device_id,
                    "draft_model": segment.draft_model,
                    "scheduled_gamma": beam_len,
                    "verify_gamma": segment.verify_gamma,
                    "batch_size": 1,
                    "start_time_ms": now_ms,
                    "finish_time_ms": now_ms + duration_ms,
                    "compute_ms": duration_ms,
                    "tree_budget_nodes": segment.tree_budget_nodes,
                    "draft_compute_nodes": segment.draft_compute_nodes,
                    "processed_candidate_count": segment.processed_candidate_count,
                    "retained_tree_nodes": segment.retained_tree_nodes,
                    "target_verify_tree_nodes": segment.target_verify_tree_nodes,
                    "tree_strategy": segment.tree_strategy,
                    "pipeline_target_ms": pipeline_target_ms,
                    "pipeline_edge_cycle_ms": pipeline_edge_cycle_ms,
                    "pipeline_alignment_error_ms": pipeline_alignment_error_ms,
                    "proactive_used": use_proactive,
                    "proactive_reused_tokens": len(retained_proactive_ids),
                }
            )
        self._schedule(now_ms + duration_ms, EventType.DRAFT_DONE, segment.segment_id)

    def _draft_prefix(self, request: Request) -> tuple[list[int], int, bool]:
        prefix_ids = request.prompt_ids + request.generated_ids
        position = request.committed_pos
        speculative_count = 0
        while position in request.pending_segments:
            segment = self.segments[request.pending_segments[position]]
            if segment.status not in ACTIVE_SEGMENT_STATUSES:
                break
            if segment.gamma <= 0:
                return prefix_ids, speculative_count, True
            prefix_ids.extend(segment.draft_ids)
            speculative_count += segment.gamma
            position += segment.gamma
        return prefix_ids, speculative_count, False

    def _select_gamma(
        self,
        request: Request,
        device: Device,
        now_ms: float,
        remaining: int,
    ) -> int:
        fixed = min(int(self.config["speculation"]["gamma_fixed"]), remaining)
        if self.spec.runtime == "specedge":
            return max(1, min(self._specedge_tree_strategy.max_beam_len, remaining))
        if self._is_server_only_runtime():
            if self.spec.candidate_strategy == "linear":
                return max(1, min(fixed, remaining))
            return max(1, min(self._server_only_tree_strategy.max_beam_len, remaining))
        if not self.spec.adaptive_gamma:
            return max(1, fixed)
        candidates = sorted(
            {
                int(value)
                for value in self.config["speculation"]["gamma_candidates"]
                if 0 < int(value) <= remaining
            }
        )
        if not candidates:
            candidates = [remaining]
        acceptance_prior = (
            self._server_only_acceptance_prior()
            if self._is_server_only_runtime()
            else device.acceptance_prior
        )
        alpha = self.acceptance.estimate(request.request_id, acceptance_prior)
        def score(gamma: int) -> tuple[float, int]:
            emitted = expected_emitted_tokens(alpha, gamma)
            uplink_ms = 0.0
            downlink_ms = 0.0
            if not self._is_server_only_runtime():
                uplink_ms = self._speculative_network_delay_ms(
                    device,
                    gamma,
                    "uplink",
                    f"estimate-up:{gamma}",
                )
                downlink_ms = self._speculative_network_delay_ms(
                    device,
                    max(1, round(emitted)),
                    "downlink",
                    f"estimate-down:{gamma}",
                )
            draft_ms = (
                self._server_only_draft_latency_ms(gamma)
                if self._is_server_only_runtime()
                else draft_latency_ms(device, gamma)
            )
            latency = (
                draft_ms
                + self._predicted_target_wait_ms(now_ms)
                + self.predict_verify_latency_ms(gamma)
                + uplink_ms
                + downlink_ms
            )
            return emitted / latency, -gamma

        return max(candidates, key=score)

    def _specedge_pipeline_target_ms(self, now_ms: float) -> float:
        server_busy_gap_ms = max(0.0, self._batch_busy_until_ms - now_ms)
        return max(self._last_specedge_verify_ms, server_busy_gap_ms)

    def _specedge_edge_cycle_ms(self, device: Device, gamma: int, emitted_tokens: float) -> float:
        tree_plan = self._specedge_tree_plan(gamma)
        uplink_ms = self._speculative_network_delay_ms(
            device,
            tree_plan.tree_budget_nodes,
            "uplink",
            f"pipeline-up:{device.device_id}:{gamma}",
        )
        downlink_ms = self._speculative_network_delay_ms(
            device,
            max(1, round(emitted_tokens)),
            "downlink",
            f"pipeline-down:{device.device_id}:{gamma}",
        )
        return draft_latency_ms(device, tree_plan.draft_compute_nodes) + uplink_ms + downlink_ms

    def _predicted_target_wait_ms(self, now_ms: float) -> float:
        if self._is_server_only_runtime():
            return max(0.0, self._server_only_verify_available_ms - now_ms)
        if self.spec.global_batch:
            return max(0.0, self._batch_busy_until_ms - now_ms)
        if not self.lanes:
            return 0.0
        return min(
            max(0.0, lane.busy_until_ms - now_ms)
            + sum(self.predict_verify_latency_ms(self.segments[item].verify_gamma) for item in lane.queue)
            for lane in self.lanes
        )

    def _on_draft_done(self, now_ms: float, segment_id: int) -> None:
        segment = self.segments[segment_id]
        runtime = self.device_runtimes[segment.device_id]
        if runtime.active_segment_id == segment_id:
            runtime.active_segment_id = None
            runtime.busy_until_ms = now_ms
        request = self.requests[segment.request_id]
        if (
            segment.status == "drafting"
            and request.status == "running"
            and segment.prefix_version == request.prefix_version
            and request.pending_segments.get(segment.base_pos) == segment_id
        ):
            segment.status = "in_transit"
            payload_tokens = self._segment_payload_tokens(segment)
            payload_bytes = self._payload_bytes(payload_tokens)
            delay_ms = self._network_delay_ms(
                runtime.device,
                "uplink",
                segment.segment_id,
                payload_bytes,
            )
            segment.uplink_payload_tokens = payload_tokens
            segment.uplink_payload_bytes = payload_bytes
            segment.uplink_delay_ms = delay_ms
            self._schedule(now_ms + delay_ms, EventType.PACKET_ARRIVE_EDGE, segment_id)
            if self.spec.runtime == "specedge":
                self._start_proactive_draft(segment, now_ms)
            self._refresh_drafting(request, now_ms)
        elif segment.status not in FINAL_SEGMENT_STATUSES:
            self._stale_segment(segment)
        self._try_start_device(runtime, now_ms)

    def _on_packet_arrive_edge(self, now_ms: float, segment_id: int) -> None:
        segment = self.segments[segment_id]
        request = self.requests[segment.request_id]
        segment.edge_arrival_time_ms = now_ms
        if segment.status in FINAL_SEGMENT_STATUSES:
            return
        if request.status != "running":
            self._discard_segment(segment)
            return
        if segment.prefix_version != request.prefix_version:
            self._stale_segment(segment)
            return
        if self.spec.global_batch:
            if segment.base_pos != request.edge_frontier_pos:
                return
            segment.status = "queued"
            self._batch_buffer.append(segment_id)
            if self._uses_specedge_dynamic_batch():
                self._ensure_batch_flush(now_ms)
                return
            self._ensure_batch_timeout(now_ms)
            self._maybe_start_batch(now_ms)
            return
        self._enqueue_ready_segments(request, now_ms)

    def _enqueue_ready_segments(self, request: Request, now_ms: float) -> None:
        if self._allows_out_of_order_verify():
            for segment_id in sorted(
                request.pending_segments.values(),
                key=lambda item: (self.segments[item].base_pos, item),
            ):
                self._enqueue_segment_if_ready(self.segments[segment_id], now_ms)
            return
        segment_id = request.pending_segments.get(request.edge_frontier_pos)
        if segment_id is None:
            return
        self._enqueue_segment_if_ready(self.segments[segment_id], now_ms)

    def _enqueue_segment_if_ready(self, segment: Segment, now_ms: float) -> None:
        request = self.requests[segment.request_id]
        if (
            segment.status != "in_transit"
            or segment.edge_arrival_time_ms is None
            or segment.prefix_version != request.prefix_version
            or request.status != "running"
        ):
            return
        lane = self.scheduler.select(
            self.lanes,
            segment,
            now_ms,
            self.predict_verify_latency_ms,
            self.segments,
        )
        segment.lane_id = lane.lane_id
        segment.status = "queued"
        lane.queue.append(segment.segment_id)
        self._try_start_lane(lane, now_ms)

    def _try_start_lane(self, lane: VerifierLane, now_ms: float) -> None:
        if lane.active_segment_id is not None:
            return
        while lane.queue:
            segment = self.segments[lane.queue.pop(0)]
            request = self.requests[segment.request_id]
            if (
                segment.status != "queued"
                or segment.prefix_version != request.prefix_version
                or request.status != "running"
                or (
                    not self._allows_out_of_order_verify()
                    and segment.base_pos != request.edge_frontier_pos
                )
            ):
                if segment.status not in FINAL_SEGMENT_STATUSES:
                    self._stale_segment(segment)
                continue
            result = self._verify_segment(segment)
            duration_ms = self.predict_verify_latency_ms(segment.verify_gamma)
            self._verification_results[segment.segment_id] = result
            segment.status = "verifying"
            segment.verify_start_time_ms = now_ms
            segment.verify_compute_ms = duration_ms
            queue_wait_ms = now_ms - float(segment.edge_arrival_time_ms)
            self._lane_queue_wait_times_ms.append(queue_wait_ms)
            segment.verify_done_time_ms = now_ms + duration_ms
            lane.busy_until_ms = segment.verify_done_time_ms
            lane.active_segment_id = segment.segment_id
            lane.total_busy_time_ms += duration_ms
            lane.processed_segments += 1
            self._schedule(segment.verify_done_time_ms, EventType.VERIFY_DONE, segment.segment_id)
            self._trace.append(
                {
                    "event": "lane_verify",
                    "method": self.spec.name,
                    "segment_id": segment.segment_id,
                    "request_id": segment.request_id,
                    "device_id": segment.device_id,
                    "draft_model": segment.draft_model,
                    "lane_id": lane.lane_id,
                    "scheduled_gamma": segment.scheduled_gamma,
                    "verify_gamma": segment.verify_gamma,
                    "batch_size": 1,
                    "start_time_ms": now_ms,
                    "finish_time_ms": segment.verify_done_time_ms,
                    "compute_ms": duration_ms,
                    "queue_wait_ms": queue_wait_ms,
                    "tree_budget_nodes": segment.tree_budget_nodes,
                    "draft_compute_nodes": segment.draft_compute_nodes,
                    "processed_candidate_count": segment.processed_candidate_count,
                    "retained_tree_nodes": segment.retained_tree_nodes,
                    "target_verify_tree_nodes": segment.target_verify_tree_nodes,
                    "tree_strategy": segment.tree_strategy,
                }
            )
            return

    def _on_verify_done(self, now_ms: float, segment_id: int) -> None:
        segment = self.segments[segment_id]
        lane = self.lanes[int(segment.lane_id)]
        if lane.active_segment_id != segment_id:
            return
        lane.active_segment_id = None
        lane.busy_until_ms = now_ms
        if segment.status == "verifying":
            if self._allows_out_of_order_verify():
                segment.status = "verified"
                self._drain_verified_results(self.requests[segment.request_id], now_ms)
            else:
                self._resolve_verification(
                    segment,
                    self._verification_results.pop(segment_id),
                    now_ms,
                )
        self._try_start_lane(lane, now_ms)

    def _ensure_batch_timeout(self, now_ms: float) -> None:
        if not self._batch_buffer or self._batch_timeout_deadline_ms is not None:
            return
        timeout_ms = self._global_batch_timeout_ms()
        if timeout_ms is None:
            return
        oldest_ms = min(float(self.segments[item].edge_arrival_time_ms) for item in self._batch_buffer)
        deadline_ms = max(
            oldest_ms + timeout_ms,
            self._batch_busy_until_ms,
            now_ms,
        )
        self._batch_timeout_token += 1
        self._batch_timeout_deadline_ms = deadline_ms
        self._schedule(deadline_ms, EventType.BATCH_TIMEOUT, self._batch_timeout_token)

    def _ensure_batch_flush(self, now_ms: float) -> None:
        if (
            not self._batch_buffer
            or self._batch_flush_deadline_ms is not None
            or self._batch_busy_until_ms > now_ms
        ):
            return
        self._batch_flush_token += 1
        self._batch_flush_deadline_ms = now_ms
        self._schedule(now_ms, EventType.BATCH_FLUSH, self._batch_flush_token)

    def _on_batch_flush(self, now_ms: float, token: int) -> None:
        if token != self._batch_flush_token:
            return
        self._batch_flush_deadline_ms = None
        if self._batch_busy_until_ms > now_ms:
            return
        if not self._maybe_start_batch(now_ms, force=True) and self._batch_buffer:
            self._ensure_batch_flush(now_ms)

    def _on_batch_timeout(self, now_ms: float, token: int) -> None:
        if token != self._batch_timeout_token:
            return
        self._batch_timeout_deadline_ms = None
        if self._batch_busy_until_ms > now_ms:
            self._ensure_batch_timeout(now_ms)
            return
        self._maybe_start_batch(now_ms, force=True)

    def _maybe_start_batch(self, now_ms: float, force: bool = False) -> bool:
        if self._batch_busy_until_ms > now_ms or not self._batch_buffer:
            return False
        batch_size = self._global_batch_size()
        if self._uses_specedge_dynamic_batch():
            pass
        elif len(self._batch_buffer) < batch_size and not force:
            return False
        segment_ids = self._batch_buffer[:batch_size]
        del self._batch_buffer[:batch_size]
        self._batch_timeout_token += 1
        self._batch_timeout_deadline_ms = None
        segments = [
            self.segments[item]
            for item in segment_ids
            if self.segments[item].status == "queued"
            and self.segments[item].base_pos == self.requests[self.segments[item].request_id].edge_frontier_pos
        ]
        if not segments:
            self._ensure_batch_timeout(now_ms)
            return False
        waiting_ms = sum(now_ms - float(segment.edge_arrival_time_ms) for segment in segments)
        self._batch_waiting_time_ms += waiting_ms
        results = self._verify_segments(segments)
        duration_ms = self._verify_latency_for_segments(segments)
        finish_ms = now_ms + duration_ms
        if self._is_specedge_runtime():
            idle_bubble_ms = max(0.0, now_ms - self._batch_busy_until_ms)
            self._pipeline_idle_bubble_ms += idle_bubble_ms
            self._last_specedge_verify_ms = duration_ms
        self._batch_busy_until_ms = finish_ms
        for segment, result in zip(segments, results):
            self._verification_results[segment.segment_id] = result
            segment.status = "verifying"
            segment.verify_start_time_ms = now_ms
            segment.verify_done_time_ms = finish_ms
            segment.verify_compute_ms = duration_ms
        self._schedule(finish_ms, EventType.BATCH_VERIFY_DONE, [segment.segment_id for segment in segments])
        self._trace.append(
            {
                "event": "global_batch_verify",
                "method": self.spec.name,
                "segment_ids": [segment.segment_id for segment in segments],
                "batch_size": len(segments),
                "start_time_ms": now_ms,
                "finish_time_ms": finish_ms,
                "compute_ms": duration_ms,
                "tree_budget_nodes": max(segment.tree_budget_nodes for segment in segments),
                "draft_compute_nodes": max(segment.draft_compute_nodes for segment in segments),
                "processed_candidate_count": max(segment.processed_candidate_count for segment in segments),
                "retained_tree_nodes": max(segment.retained_tree_nodes for segment in segments),
                "target_verify_tree_nodes": max(segment.target_verify_tree_nodes for segment in segments),
                "tree_strategy": segments[0].tree_strategy,
                "batch_type": self._specedge_server_batch_type() if self._is_specedge_runtime() else "timeout",
                "pipeline_idle_bubble_ms": idle_bubble_ms if self._is_specedge_runtime() else 0.0,
            }
        )
        self._ensure_batch_timeout(now_ms)
        return True

    def _on_batch_verify_done(self, now_ms: float, segment_ids: list[int]) -> None:
        self._batch_busy_until_ms = now_ms
        for segment_id in segment_ids:
            segment = self.segments[segment_id]
            if segment.status == "verifying":
                self._resolve_verification(
                    segment,
                    self._verification_results.pop(segment_id),
                    now_ms,
                )
        if self._uses_specedge_dynamic_batch():
            self._ensure_batch_flush(now_ms)
        else:
            self._maybe_start_batch(now_ms)
        self._ensure_batch_timeout(now_ms)

    def _drain_verified_results(self, request: Request, now_ms: float) -> None:
        while request.status == "running":
            segment_id = request.pending_segments.get(request.edge_frontier_pos)
            if segment_id is None:
                return
            segment = self.segments[segment_id]
            if segment.status != "verified":
                return
            result = self._verification_results.pop(segment_id)
            self._resolve_verification(segment, result, now_ms)
            if result.rejected:
                return

    def _resolve_verification(
        self,
        segment: Segment,
        result: VerificationResult,
        now_ms: float,
    ) -> None:
        request = self.requests[segment.request_id]
        if segment.base_pos != request.edge_frontier_pos:
            self._stale_segment(segment)
            return
        result_base_pos = segment.base_pos
        remaining = request.output_len - request.edge_frontier_pos
        emitted_ids = list(result.emitted_ids[:remaining])
        accepted_path = list(result.emitted_ids[: result.accepted_count])
        tree_path_switched = bool(
            segment.draft_tree is not None
            and accepted_path
            and accepted_path != segment.draft_ids[: len(accepted_path)]
        )
        request.pending_segments.pop(segment.base_pos, None)
        segment.result_base_pos = result_base_pos
        segment.accepted_count = result.accepted_count
        segment.emitted_ids = emitted_ids
        segment.tree_path_switched = tree_path_switched
        request.accepted_tokens += result.accepted_count
        self.device_runtimes[segment.device_id].accepted_draft_tokens += result.accepted_count
        self.acceptance.observe(
            request.request_id,
            result.accepted_count,
            segment.proposed_count,
        )
        request.edge_generated_ids.extend(emitted_ids)
        if result.rejected:
            segment.status = "rejected"
            request.rejected_count += 1
            request.rollback_count += 1
            request.prefix_version += 1
            self._record_proactive_miss(request, segment)
            self._clear_proactive(request)
            self._record_rejection_waste(segment)
            self._invalidate_pending(request)
        else:
            segment.status = "accepted"
            if tree_path_switched:
                request.prefix_version += 1
                self._record_proactive_miss(request, segment)
                self._clear_proactive(request)
                self._invalidate_pending(request)
            else:
                self._resolve_proactive_after_accept(request, segment, result)
            if (
                not tree_path_switched
                and result.bonus_token is not None
                and emitted_ids
                and emitted_ids[-1] == result.bonus_token
            ):
                self._retarget_after_bonus(request, segment, result.bonus_token)
        if request.edge_frontier_pos >= request.output_len:
            self._invalidate_pending(request)
        request.completed_results[result_base_pos] = segment.segment_id
        if self._is_server_only_runtime():
            payload_bytes = 0
            delay_ms = 0.0
        else:
            downlink_tokens = len(emitted_ids)
            payload_bytes = self._payload_bytes(downlink_tokens)
            delay_ms = self._network_delay_ms(
                self.devices[segment.device_id],
                "downlink",
                segment.segment_id,
                payload_bytes,
            )
        segment.downlink_payload_bytes = payload_bytes
        segment.downlink_delay_ms = delay_ms
        trace_event = {
            "event": "verification_result",
            "method": self.spec.name,
            "segment_id": segment.segment_id,
            "request_id": segment.request_id,
            "device_id": segment.device_id,
            "draft_model": segment.draft_model,
            "scheduled_gamma": segment.scheduled_gamma,
            "verify_gamma": segment.verify_gamma,
            "accepted_count": segment.accepted_count,
            "proposed_count": segment.proposed_count,
            "emitted_count": segment.emitted_count,
            "finish_time_ms": now_ms,
            "tree_budget_nodes": segment.tree_budget_nodes,
            "draft_compute_nodes": segment.draft_compute_nodes,
            "processed_candidate_count": segment.processed_candidate_count,
            "retained_tree_nodes": segment.retained_tree_nodes,
            "target_verify_tree_nodes": segment.target_verify_tree_nodes,
            "tree_strategy": segment.tree_strategy,
            "tree_path_switched": segment.tree_path_switched,
        }
        if not self._is_server_only_runtime():
            trace_event["downlink_ms"] = delay_ms
            trace_event["downlink_payload_bytes"] = payload_bytes
        self._trace.append(trace_event)
        result_arrival_ms = now_ms + delay_ms
        if segment.proactive_done_time_ms is not None:
            result_arrival_ms = max(result_arrival_ms, segment.proactive_done_time_ms)
        self._schedule(result_arrival_ms, EventType.RESULT_ARRIVE_DEVICE, segment.segment_id)
        if not self.spec.global_batch and not self._is_server_only_runtime():
            self._enqueue_ready_segments(request, now_ms)

    def _start_proactive_draft(self, segment: Segment, start_ms: float) -> None:
        if not self._specedge_proactive_enabled():
            return
        request = self.requests[segment.request_id]
        if request.status != "running":
            return
        remaining_after_bonus = request.output_len - (segment.base_pos + segment.gamma + 1)
        if remaining_after_bonus <= 0:
            return
        proactive_len = min(
            self._proactive_tree_strategy.max_beam_len,
            remaining_after_bonus + 1,
        )
        proactive_tree_plan = self._proactive_tree_plan(proactive_len)
        if proactive_tree_plan.strategy == "linear":
            proactive_ids = self.model_runner.draft(
                self.devices[segment.device_id].drafter_profile,
                segment.prefix_ids + segment.draft_ids,
                proactive_tree_plan.path_token_count,
            )
            proactive_tree = None
            processed_candidates = len(proactive_ids)
            retained_nodes = len(proactive_ids)
            target_verify_nodes = 1 if proactive_ids else 0
            proposed_count = len(proactive_ids)
        elif segment.draft_tree is None:
            return
        else:
            proactive_tree = self.model_runner.draft_bonus_tree(
                self.devices[segment.device_id].drafter_profile,
                segment.draft_tree,
                proactive_tree_plan,
            )
            proactive_ids = list(proactive_tree.primary_ids)
            processed_candidates = proactive_tree.processed_candidate_count
            retained_nodes = proactive_tree.retained_tree_nodes
            target_verify_nodes = proactive_tree.target_verify_tree_nodes
            proposed_count = _tree_proposed_count(proactive_tree, proactive_ids)
        if not proactive_ids:
            return
        proactive_compute_ms = draft_latency_ms(
            self.devices[segment.device_id],
            processed_candidates,
        )
        runtime = self.device_runtimes[segment.device_id]
        if runtime.active_segment_id is not None:
            return
        finish_ms = start_ms + proactive_compute_ms
        runtime.active_segment_id = segment.segment_id
        runtime.busy_until_ms = finish_ms
        runtime.total_busy_time_ms += proactive_compute_ms
        runtime.generated_draft_tokens += proposed_count
        segment.proactive_used = True
        segment.proactive_draft_ids = proactive_ids
        segment.proactive_draft_tree = proactive_tree
        segment.proactive_start_time_ms = start_ms
        segment.proactive_done_time_ms = finish_ms
        self._trace.append(
            {
                "event": "proactive_draft",
                "method": self.spec.name,
                "request_id": segment.request_id,
                "segment_id": segment.segment_id,
                "device_id": segment.device_id,
                "draft_model": segment.draft_model,
                "batch_size": 1,
                "scheduled_gamma": len(proactive_ids),
                "verify_gamma": len(proactive_ids),
                "start_time_ms": start_ms,
                "finish_time_ms": finish_ms,
                "compute_ms": proactive_compute_ms,
                "tree_budget_nodes": retained_nodes,
                "draft_compute_nodes": processed_candidates,
                "processed_candidate_count": processed_candidates,
                "retained_tree_nodes": retained_nodes,
                "target_verify_tree_nodes": target_verify_nodes,
                "tree_strategy": proactive_tree_plan.strategy,
            }
        )
        self._schedule(finish_ms, EventType.PROACTIVE_DRAFT_DONE, segment.segment_id)

    def _on_proactive_draft_done(self, now_ms: float, segment_id: int) -> None:
        segment = self.segments[segment_id]
        runtime = self.device_runtimes[segment.device_id]
        if runtime.active_segment_id == segment_id:
            runtime.active_segment_id = None
            runtime.busy_until_ms = now_ms
        self._try_start_device(runtime, now_ms)

    def _resolve_proactive_after_accept(
        self,
        request: Request,
        segment: Segment,
        result: VerificationResult,
    ) -> None:
        if self.spec.runtime != "specedge" or not segment.proactive_draft_ids:
            return
        if result.bonus_token is not None and segment.proactive_draft_ids[0] == result.bonus_token:
            proactive_tree = segment.proactive_draft_tree
            if proactive_tree is not None:
                accepted_prefix = segment.prefix_ids + list(result.emitted_ids[: result.accepted_count])
                if proactive_tree.prefix_ids != accepted_prefix:
                    self._record_proactive_miss(request, segment)
                    self._clear_proactive(request)
                    return
            segment.proactive_hit = True
            request.proactive_draft_ids = segment.proactive_draft_ids[1:]
            request.proactive_draft_tree = (
                rebase_draft_tree(proactive_tree, 1)
                if proactive_tree is not None
                else None
            )
            request.proactive_base_pos = request.edge_frontier_pos
            request.proactive_prefix_version = request.prefix_version
            return
        self._record_proactive_miss(request, segment)
        self._clear_proactive(request)

    def _record_proactive_miss(self, request: Request, segment: Segment) -> None:
        if not segment.proactive_draft_ids or segment.proactive_hit or segment.proactive_wasted_tokens:
            return
        segment.proactive_wasted_tokens = len(segment.proactive_draft_ids)
        request.wasted_draft_tokens += segment.proactive_wasted_tokens

    def _clear_proactive(self, request: Request) -> None:
        request.proactive_draft_ids = []
        request.proactive_draft_tree = None
        request.proactive_base_pos = None
        request.proactive_prefix_version = None

    def _invalidate_pending_from(
        self,
        request: Request,
        base_pos: int,
        exclude_segment_id: int,
    ) -> None:
        for segment_id in list(request.pending_segments.values()):
            if segment_id == exclude_segment_id:
                continue
            segment = self.segments[segment_id]
            if segment.base_pos < base_pos or segment.status not in ACTIVE_SEGMENT_STATUSES:
                continue
            if self.spec.prefix_control == "conservative":
                self._discard_segment(segment)
            else:
                self._stale_segment(segment)

    def _retarget_verified_result_after_bonus(
        self,
        result: VerificationResult,
        bonus_token: int,
    ) -> VerificationResult | None:
        if (
            not result.emitted_ids
            or result.emitted_ids[0] != bonus_token
            or result.accepted_count < 1
        ):
            return None
        return VerificationResult(
            accepted_count=result.accepted_count - 1,
            emitted_ids=list(result.emitted_ids[1:]),
            rejected=result.rejected,
            bonus_token=result.bonus_token,
        )

    def _retarget_after_bonus(self, request: Request, source: Segment, bonus_token: int) -> None:
        next_base = int(source.result_base_pos) + source.gamma
        segment_id = request.pending_segments.get(next_base)
        if segment_id is None:
            return
        segment = self.segments[segment_id]
        if (
            not self.spec.bonus_retarget
            or not segment.draft_ids
            or segment.draft_ids[0] != bonus_token
        ):
            request.prefix_version += 1
            self._invalidate_pending(request)
            return
        if segment.status in {"verifying", "verified"}:
            transformed = self._retarget_verified_result_after_bonus(
                self._verification_results[segment.segment_id],
                bonus_token,
            )
            if transformed is None:
                request.prefix_version += 1
                self._invalidate_pending(request)
                return
            self._verification_results[segment.segment_id] = transformed
            invalidate_descendants = True
        else:
            invalidate_descendants = False
        request.pending_segments.pop(next_base, None)
        segment.base_pos += 1
        segment.prefix_ids.append(bonus_token)
        segment.draft_ids.pop(0)
        if segment.draft_tree is not None:
            segment.draft_tree = rebase_draft_tree(segment.draft_tree, 1)
        segment.bonus_reused = True
        request.bonus_reused_tokens += 1
        if invalidate_descendants:
            self._invalidate_pending_from(request, segment.base_pos, segment.segment_id)
        if not segment.draft_ids and segment.status not in {"verifying", "verified"}:
            segment.status = "absorbed"
            if segment.segment_id in request.in_flight_segments:
                request.in_flight_segments.remove(segment.segment_id)
            return
        request.pending_segments[segment.base_pos] = segment.segment_id

    def _on_result_arrive_device(self, now_ms: float, segment_id: int) -> None:
        segment = self.segments[segment_id]
        request = self.requests[segment.request_id]
        if request.status != "running":
            return
        segment.result_arrived = True
        self._commit_ready_results(request, now_ms)
        if (
            len(request.generated_ids) >= request.output_len
            or (self.model_runner.eos_token_id is not None and self.model_runner.eos_token_id in request.generated_ids)
        ):
            if self._is_server_only_runtime():
                self._schedule(now_ms, EventType.REQUEST_FINISH, request.request_id)
            else:
                self._schedule(now_ms, EventType.REQUEST_FINISH, request.request_id)
            return
        if not self.spec.global_batch and not self._is_server_only_runtime():
            self._enqueue_ready_segments(request, now_ms)
        if self._is_server_only_runtime():
            self._start_server_only_draft(request, now_ms)
        else:
            self._refresh_drafting(request, now_ms)

    def _schedule_server_only_response_downlink(self, request: Request, now_ms: float) -> None:
        payload_bytes = self._payload_bytes(len(request.generated_ids))
        delay_ms = self._network_delay_ms(
            self.devices[request.device_id],
            "downlink",
            f"server-only:{request.request_id}",
            payload_bytes,
        )
        request.target_only_downlink_payload_bytes = payload_bytes
        request.target_only_downlink_ms = delay_ms
        self._trace.append(
            {
                "event": "server_only_response_downlink",
                "method": self.spec.name,
                "request_id": request.request_id,
                "device_id": request.device_id,
                "start_time_ms": now_ms,
                "finish_time_ms": now_ms + delay_ms,
                "downlink_ms": delay_ms,
                "downlink_payload_bytes": payload_bytes,
            }
        )
        if self._server_only_active_request_id == request.request_id:
            self._server_only_active_request_id = None
            self._maybe_start_server_only_request(now_ms)
        self._schedule(now_ms + delay_ms, EventType.REQUEST_FINISH, request.request_id)

    def _commit_ready_results(self, request: Request, now_ms: float) -> None:
        while True:
            segment_id = request.completed_results.get(request.committed_pos)
            if segment_id is None:
                return
            segment = self.segments[segment_id]
            if not segment.result_arrived:
                return
            request.completed_results.pop(request.committed_pos)
            if segment_id in request.in_flight_segments:
                request.in_flight_segments.remove(segment_id)
            request.generated_ids.extend(segment.emitted_ids)

    def _on_request_finish(self, now_ms: float, request_id: int) -> None:
        request = self.requests[request_id]
        if request.status == "finished":
            return
        request.status = "finished"
        request.finish_time_ms = now_ms
        request.draft_queued = False
        for segment_id in list(request.in_flight_segments):
            segment = self.segments[segment_id]
            if segment.status in ACTIVE_SEGMENT_STATUSES:
                self._discard_segment(segment)
        self._trace.append(
            {
                "event": "request_finish",
                "method": self.spec.name,
                "request_id": request_id,
                "device_id": request.device_id,
                "finish_time_ms": now_ms,
            }
        )
        if self._progress_callback is not None:
            self._progress_callback(sum(item.status == "finished" for item in self.requests), len(self.requests))
        if self._is_server_only_runtime() and self._server_only_active_request_id == request_id:
            self._server_only_active_request_id = None
            self._maybe_start_server_only_request(now_ms)

    def _invalidate_pending(self, request: Request) -> None:
        for segment_id in list(request.pending_segments.values()):
            segment = self.segments[segment_id]
            if segment.status not in ACTIVE_SEGMENT_STATUSES:
                continue
            if self.spec.prefix_control == "conservative":
                self._discard_segment(segment)
            else:
                self._stale_segment(segment)

    def _stale_segment(self, segment: Segment) -> None:
        if segment.status in FINAL_SEGMENT_STATUSES:
            return
        self._verification_results.pop(segment.segment_id, None)
        segment.status = "stale"
        self._remove_from_request(segment)
        self._record_waste(segment)

    def _discard_segment(self, segment: Segment) -> None:
        if segment.status in FINAL_SEGMENT_STATUSES:
            return
        self._verification_results.pop(segment.segment_id, None)
        segment.status = "discarded"
        self._remove_from_request(segment)
        self._record_waste(segment)

    def _remove_from_request(self, segment: Segment) -> None:
        request = self.requests[segment.request_id]
        if segment.segment_id in request.in_flight_segments:
            request.in_flight_segments.remove(segment.segment_id)
        if request.pending_segments.get(segment.base_pos) == segment.segment_id:
            request.pending_segments.pop(segment.base_pos, None)

    def _record_rejection_waste(self, segment: Segment) -> None:
        if segment.waste_recorded:
            return
        self.requests[segment.request_id].wasted_draft_tokens += max(
            0,
            segment.proposed_count - int(segment.accepted_count or 0),
        )
        segment.waste_recorded = True

    def _record_waste(self, segment: Segment) -> None:
        if segment.waste_recorded:
            return
        self.requests[segment.request_id].wasted_draft_tokens += segment.proposed_count
        segment.waste_recorded = True

    def _payload_bytes(self, token_count: int) -> int:
        network = self.config["network"]
        return int(network["packet_header_bytes"]) + (
            token_count * int(network["packet_token_bytes"])
        )

    def _speculative_network_delay_ms(
        self,
        device: Device,
        token_count: int,
        direction: str,
        key: Any,
    ) -> float:
        return self._network_delay_ms(device, direction, key, self._payload_bytes(token_count))

    def _network_delay_ms(
        self,
        device: Device,
        direction: str,
        key: Any,
        payload_bytes: int,
    ) -> float:
        return network_delay_ms(
            int(self.config["simulation"]["seed"]),
            device,
            direction,
            key,
            payload_bytes,
        )


def _tree_proposed_count(
    draft_tree: DraftCandidateTree | None,
    draft_ids: Sequence[int],
) -> int:
    if draft_tree is None or not draft_tree.nodes:
        return len(draft_ids)
    return max(len(draft_ids), max(node.depth for node in draft_tree.nodes))
