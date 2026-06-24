from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any, Iterable

from src.entities import SimulationResult


CATEGORY_ORDER = {
    "MT": 0,
    "QA": 1,
    "Math": 2,
    "RAG": 3,
    "Sum": 4,
    "Trans": 5,
}

MAIN_FIELDS = [
    "method",
    "scenario",
    "num_requests",
    "num_devices",
    "num_lanes",
    "avg_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "avg_tpot_ms",
    "avg_tbt_ms",
    "makespan_ms",
    "goodput_tok_s",
    "avg_acceptance_rate",
    "avg_selected_gamma",
    "latency_speedup_vs_autoregressive",
    "latency_ratio_vs_sync_batch_sd",
    "latency_ratio_vs_specedge",
    "relative_latency_reduction_vs_sync_batch_sd",
    "relative_latency_reduction_vs_specedge",
    "goodput_gain_vs_autoregressive",
    "goodput_gain_vs_sync_batch_sd",
]

CATEGORY_MAIN_FIELDS = [
    "method",
    "scenario",
    "category",
    *MAIN_FIELDS[2:],
]

SYSTEM_FIELDS = [
    "method",
    "scenario",
    "device_utilization_mean",
    "device_utilization_std",
    "device_queue_wait_ms_total",
    "batch_waiting_time_ms",
    "phase_waiting_time_ms",
    "target_utilization",
    "verify_idle_time_ms",
    "lane_utilization_mean",
    "lane_utilization_std",
    "lane_queue_wait_ms_mean",
    "lane_queue_wait_ms_p95",
    "stale_segment_ratio",
    "wasted_draft_tokens",
    "bonus_reused_tokens",
    "rollback_count",
    "total_segments",
    "total_stale_segments",
    "total_discarded_segments",
    "total_absorbed_segments",
    "proactive_segments",
    "proactive_hits",
    "proactive_wasted_tokens",
    "pipeline_idle_bubble_ms",
    "pipeline_alignment_error_ms_mean",
    "pipeline_alignment_error_ms_p95",
    "draft_compute_ms_total",
    "draft_compute_ms_mean",
    "verify_compute_ms_total",
    "verify_compute_ms_mean",
    "target_only_compute_ms_total",
    "target_only_compute_ms_mean",
    "uplink_ms_total",
    "uplink_ms_mean",
    "downlink_ms_total",
    "downlink_ms_mean",
    "uplink_payload_bytes_total",
    "downlink_payload_bytes_total",
]

DEVICE_FIELDS = [
    "method",
    "scenario",
    "device_id",
    "device_type",
    "draft_model",
    "device_utilization",
    "idle_time_ms",
    "draft_busy_time_ms",
    "draft_queue_wait_ms",
    "num_assigned_requests",
    "num_generated_draft_tokens",
    "num_accepted_tokens",
    "num_rejected_tokens",
    "avg_selected_gamma",
]

REQUEST_FIELDS = [
    "request_id",
    "device_id",
    "prompt_id",
    "category",
    "raw_category",
    "prompt_token_count",
    "method",
    "scenario",
    "arrival_time_ms",
    "decode_ready_time_ms",
    "finish_time_ms",
    "latency_ms",
    "output_len",
    "generated_tokens",
    "accepted_tokens",
    "rejected_count",
    "rollback_count",
    "wasted_draft_tokens",
    "bonus_reused_tokens",
    "max_outstanding_observed",
    "max_unconfirmed_tokens_observed",
    "target_only_queue_wait_ms",
    "target_only_compute_ms",
    "target_only_downlink_ms",
    "target_only_downlink_payload_bytes",
]

SEGMENT_FIELDS = [
    "segment_id",
    "request_id",
    "device_id",
    "draft_model",
    "lane_id",
    "method",
    "scenario",
    "prefix_version",
    "base_pos",
    "scheduled_gamma",
    "verify_gamma",
    "accepted_count",
    "proposed_count",
    "emitted_count",
    "acceptance_rate",
    "bonus_reused",
    "draft_start_time_ms",
    "create_time_ms",
    "draft_queue_wait_ms",
    "draft_compute_ms",
    "draft_analytical_ms",
    "uplink_delay_ms",
    "uplink_payload_tokens",
    "uplink_payload_bytes",
    "edge_arrival_time_ms",
    "verify_start_time_ms",
    "verify_done_time_ms",
    "verify_compute_ms",
    "downlink_delay_ms",
    "downlink_payload_bytes",
    "tree_strategy",
    "tree_budget_nodes",
    "draft_compute_nodes",
    "processed_candidate_count",
    "retained_tree_nodes",
    "target_verify_tree_nodes",
    "beam_len",
    "tree_path_switched",
    "proactive_used",
    "proactive_hit",
    "proactive_wasted_tokens",
    "proactive_start_time_ms",
    "proactive_done_time_ms",
    "pipeline_target_ms",
    "pipeline_edge_cycle_ms",
    "pipeline_alignment_error_ms",
    "status",
]

EVENT_FIELDS = [
    "event",
    "method",
    "scenario",
    "request_id",
    "segment_id",
    "segment_ids",
    "device_id",
    "draft_model",
    "lane_id",
    "batch_size",
    "scheduled_gamma",
    "verify_gamma",
    "accepted_count",
    "proposed_count",
    "emitted_count",
    "start_time_ms",
    "finish_time_ms",
    "compute_ms",
    "queue_wait_ms",
    "uplink_ms",
    "uplink_payload_bytes",
    "downlink_ms",
    "downlink_payload_bytes",
    "tree_strategy",
    "tree_budget_nodes",
    "draft_compute_nodes",
    "processed_candidate_count",
    "retained_tree_nodes",
    "target_verify_tree_nodes",
    "tree_path_switched",
    "pipeline_target_ms",
    "pipeline_edge_cycle_ms",
    "pipeline_alignment_error_ms",
    "pipeline_idle_bubble_ms",
    "batch_type",
    "proactive_used",
    "proactive_reused_tokens",
]


def percentile(values: Iterable[float], percentile_value: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    ratio = position - lower
    return ordered[lower] * (1.0 - ratio) + ordered[upper] * ratio


def summarize(result: SimulationResult, num_devices: int) -> tuple[dict[str, Any], dict[str, Any]]:
    latencies = [request.latency_ms for request in result.requests]
    output_tokens = [len(request.generated_ids) for request in result.requests]
    tpot_values = [
        request.latency_ms / token_count
        for request, token_count in zip(result.requests, output_tokens)
        if token_count
    ]
    starts = [request.decode_ready_time_ms for request in result.requests]
    finishes = [float(request.finish_time_ms) for request in result.requests]
    makespan_ms = max(finishes) - min(starts)
    goodput_tokens = sum(output_tokens)
    lane_utilizations = [
        lane.total_busy_time_ms / makespan_ms if makespan_ms else 0.0 for lane in result.lanes
    ]
    device_utilizations = [
        min(1.0, runtime.total_busy_time_ms / makespan_ms) if makespan_ms else 0.0
        for runtime in result.devices
    ]
    stale_segments = sum(segment.status == "stale" for segment in result.segments)
    discarded_segments = sum(segment.status == "discarded" for segment in result.segments)
    absorbed_segments = sum(segment.status == "absorbed" for segment in result.segments)
    proactive_segments = sum(segment.proactive_used for segment in result.segments)
    proactive_hits = sum(segment.proactive_hit for segment in result.segments)
    proactive_waste = sum(segment.proactive_wasted_tokens for segment in result.segments)
    pipeline_alignment_errors = [
        segment.pipeline_alignment_error_ms
        for segment in result.segments
        if segment.pipeline_alignment_error_ms
    ]
    pipeline_idle_bubbles = [
        event.get("pipeline_idle_bubble_ms", 0.0)
        for event in result.event_trace
        if event["event"] == "global_batch_verify"
    ]
    draft_compute = [segment.draft_compute_ms for segment in result.segments]
    verify_events = [
        event["compute_ms"]
        for event in result.event_trace
        if event["event"] in {"lane_verify", "global_batch_verify", "server_only_verify"}
    ]
    server_only_draft_events = [
        event["compute_ms"]
        for event in result.event_trace
        if event["event"] == "server_only_draft"
    ]
    target_compute = [
        request.target_only_compute_ms
        for request in result.requests
        if request.target_only_compute_ms
    ]
    uplink_delays = [
        segment.uplink_delay_ms
        for segment in result.segments
        if segment.uplink_payload_bytes
    ]
    downlink_delays = [
        *[segment.downlink_delay_ms for segment in result.segments if segment.downlink_payload_bytes],
        *[request.target_only_downlink_ms for request in result.requests if request.target_only_downlink_ms],
    ]
    proposed = sum(segment.proposed_count for segment in result.segments if segment.accepted_count is not None)
    accepted = sum(int(segment.accepted_count or 0) for segment in result.segments)
    selected_gammas = [
        gamma for runtime in result.devices for gamma in runtime.selected_gammas
    ]
    main = {
        "method": result.method,
        "scenario": result.scenario,
        "num_requests": len(result.requests),
        "num_devices": num_devices,
        "num_lanes": len(result.lanes),
        "avg_latency_ms": _mean(latencies),
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "p99_latency_ms": percentile(latencies, 99),
        "avg_tpot_ms": _mean(tpot_values),
        "avg_tbt_ms": _mean(tpot_values),
        "makespan_ms": makespan_ms,
        "goodput_tok_s": goodput_tokens / makespan_ms * 1000.0 if makespan_ms else 0.0,
        "avg_acceptance_rate": accepted / proposed if proposed else 0.0,
        "avg_selected_gamma": _mean(selected_gammas),
    }
    resource_busy_ms = sum(verify_events)
    if result.method == "target_only":
        target_capacity = max(1, len(result.lanes))
        target_utilization = sum(target_compute) / (target_capacity * makespan_ms) if makespan_ms else 0.0
        resource_busy_ms = sum(target_compute)
    elif result.method == "server_only":
        target_capacity = 1
        resource_busy_ms = sum(verify_events) + sum(server_only_draft_events)
        target_utilization = resource_busy_ms / makespan_ms if makespan_ms else 0.0
    elif result.lanes:
        target_capacity = len(result.lanes)
        target_utilization = sum(lane.total_busy_time_ms for lane in result.lanes) / (target_capacity * makespan_ms) if makespan_ms else 0.0
    else:
        target_capacity = 1
        target_utilization = sum(verify_events) / makespan_ms if makespan_ms else 0.0
    target_utilization = min(1.0, target_utilization)
    system = {
        "method": result.method,
        "scenario": result.scenario,
        "device_utilization_mean": _mean(device_utilizations),
        "device_utilization_std": statistics.pstdev(device_utilizations) if device_utilizations else 0.0,
        "device_queue_wait_ms_total": sum(runtime.total_queue_wait_ms for runtime in result.devices),
        "batch_waiting_time_ms": result.batch_waiting_time_ms,
        "phase_waiting_time_ms": result.phase_waiting_time_ms,
        "target_utilization": target_utilization,
        "verify_idle_time_ms": max(0.0, makespan_ms * target_capacity - resource_busy_ms),
        "lane_utilization_mean": _mean(lane_utilizations),
        "lane_utilization_std": statistics.pstdev(lane_utilizations) if lane_utilizations else 0.0,
        "lane_queue_wait_ms_mean": _mean(result.lane_queue_wait_times_ms),
        "lane_queue_wait_ms_p95": percentile(result.lane_queue_wait_times_ms, 95),
        "stale_segment_ratio": stale_segments / len(result.segments) if result.segments else 0.0,
        "wasted_draft_tokens": sum(request.wasted_draft_tokens for request in result.requests),
        "bonus_reused_tokens": sum(request.bonus_reused_tokens for request in result.requests),
        "rollback_count": sum(request.rollback_count for request in result.requests),
        "total_segments": len(result.segments),
        "total_stale_segments": stale_segments,
        "total_discarded_segments": discarded_segments,
        "total_absorbed_segments": absorbed_segments,
        "proactive_segments": proactive_segments,
        "proactive_hits": proactive_hits,
        "proactive_wasted_tokens": proactive_waste,
        "pipeline_idle_bubble_ms": sum(pipeline_idle_bubbles),
        "pipeline_alignment_error_ms_mean": _mean(pipeline_alignment_errors),
        "pipeline_alignment_error_ms_p95": percentile(pipeline_alignment_errors, 95),
        "draft_compute_ms_total": sum(draft_compute),
        "draft_compute_ms_mean": _mean(draft_compute),
        "verify_compute_ms_total": sum(verify_events),
        "verify_compute_ms_mean": _mean(verify_events),
        "target_only_compute_ms_total": sum(target_compute),
        "target_only_compute_ms_mean": _mean(target_compute),
        "uplink_ms_total": sum(uplink_delays),
        "uplink_ms_mean": _mean(uplink_delays),
        "downlink_ms_total": sum(downlink_delays),
        "downlink_ms_mean": _mean(downlink_delays),
        "uplink_payload_bytes_total": sum(
            segment.uplink_payload_bytes for segment in result.segments
        ),
        "downlink_payload_bytes_total": sum(segment.downlink_payload_bytes for segment in result.segments)
        + sum(request.target_only_downlink_payload_bytes for request in result.requests),
    }
    return main, system


def category_rows(result: SimulationResult, num_devices: int) -> list[dict[str, Any]]:
    rows = []
    requests_by_category: dict[str, list] = {}
    for request in result.requests:
        requests_by_category.setdefault(
            request.category_group or request.category or "unknown", []
        ).append(request)

    for category, requests in sorted(
        requests_by_category.items(),
        key=lambda item: (CATEGORY_ORDER.get(item[0], len(CATEGORY_ORDER)), item[0]),
    ):
        request_ids = {request.request_id for request in requests}
        segments = [
            segment for segment in result.segments if segment.request_id in request_ids
        ]
        latencies = [request.latency_ms for request in requests]
        output_tokens = [len(request.generated_ids) for request in requests]
        tpot_values = [
            request.latency_ms / token_count
            for request, token_count in zip(requests, output_tokens)
            if token_count
        ]
        starts = [request.decode_ready_time_ms for request in requests]
        finishes = [float(request.finish_time_ms) for request in requests]
        makespan_ms = max(finishes) - min(starts)
        goodput_tokens = sum(output_tokens)
        proposed = sum(segment.proposed_count for segment in segments if segment.accepted_count is not None)
        accepted = sum(int(segment.accepted_count or 0) for segment in segments)
        selected_gammas = [segment.scheduled_gamma for segment in segments]
        row = {
            "method": result.method,
            "scenario": result.scenario,
            "category": category,
            "num_requests": len(requests),
            "num_devices": num_devices,
            "num_lanes": len(result.lanes),
            "avg_latency_ms": _mean(latencies),
            "p50_latency_ms": percentile(latencies, 50),
            "p95_latency_ms": percentile(latencies, 95),
            "p99_latency_ms": percentile(latencies, 99),
            "avg_tpot_ms": _mean(tpot_values),
            "avg_tbt_ms": _mean(tpot_values),
            "makespan_ms": makespan_ms,
            "goodput_tok_s": goodput_tokens / makespan_ms * 1000.0 if makespan_ms else 0.0,
            "avg_acceptance_rate": accepted / proposed if proposed else 0.0,
            "avg_selected_gamma": _mean(selected_gammas),
        }
        rows.append(row)
    return rows


def enrich_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method = {row["method"]: row for row in rows}
    autoregressive = by_method.get("target_only")
    sync_batch = by_method.get("sync_batch_sd")
    specedge = by_method.get("SpecEdge")
    for row in rows:
        row["latency_speedup_vs_autoregressive"] = _latency_ratio(autoregressive, row)
        row["latency_ratio_vs_sync_batch_sd"] = _latency_ratio(sync_batch, row)
        row["latency_ratio_vs_specedge"] = _latency_ratio(specedge, row)
        row["relative_latency_reduction_vs_sync_batch_sd"] = _latency_reduction(sync_batch, row)
        row["relative_latency_reduction_vs_specedge"] = _latency_reduction(specedge, row)
        row["goodput_gain_vs_autoregressive"] = _goodput_gain(row, autoregressive)
        row["goodput_gain_vs_sync_batch_sd"] = _goodput_gain(row, sync_batch)
    return rows


def request_rows(result: SimulationResult) -> list[dict[str, Any]]:
    return [
        {
            "request_id": request.request_id,
            "device_id": request.device_id,
            "prompt_id": request.prompt_id,
            "category": request.category_group or request.category or "unknown",
            "raw_category": request.category,
            "prompt_token_count": request.prompt_token_count,
            "method": result.method,
            "scenario": result.scenario,
            "arrival_time_ms": request.arrival_time_ms,
            "decode_ready_time_ms": request.decode_ready_time_ms,
            "finish_time_ms": request.finish_time_ms,
            "latency_ms": request.latency_ms,
            "output_len": request.output_len,
            "generated_tokens": len(request.generated_ids),
            "accepted_tokens": request.accepted_tokens,
            "rejected_count": request.rejected_count,
            "rollback_count": request.rollback_count,
            "wasted_draft_tokens": request.wasted_draft_tokens,
            "bonus_reused_tokens": request.bonus_reused_tokens,
            "max_outstanding_observed": request.max_outstanding_observed,
            "max_unconfirmed_tokens_observed": request.max_unconfirmed_tokens_observed,
            "target_only_queue_wait_ms": request.target_only_queue_wait_ms,
            "target_only_compute_ms": request.target_only_compute_ms,
            "target_only_downlink_ms": request.target_only_downlink_ms,
            "target_only_downlink_payload_bytes": request.target_only_downlink_payload_bytes,
        }
        for request in result.requests
    ]


def segment_rows(result: SimulationResult) -> list[dict[str, Any]]:
    return [
        {
            "segment_id": segment.segment_id,
            "request_id": segment.request_id,
            "device_id": segment.device_id,
            "draft_model": segment.draft_model,
            "lane_id": segment.lane_id,
            "method": result.method,
            "scenario": result.scenario,
            "prefix_version": segment.prefix_version,
            "base_pos": segment.base_pos,
            "scheduled_gamma": segment.scheduled_gamma,
            "verify_gamma": segment.verify_gamma,
            "accepted_count": segment.accepted_count,
            "proposed_count": segment.proposed_count,
            "emitted_count": segment.emitted_count,
            "acceptance_rate": segment.acceptance_rate,
            "bonus_reused": segment.bonus_reused,
            "draft_start_time_ms": segment.draft_start_time_ms,
            "create_time_ms": segment.create_time_ms,
            "draft_queue_wait_ms": segment.draft_queue_wait_ms,
            "draft_compute_ms": segment.draft_compute_ms,
            "draft_analytical_ms": segment.draft_analytical_ms,
            "uplink_delay_ms": segment.uplink_delay_ms,
            "uplink_payload_tokens": segment.uplink_payload_tokens,
            "uplink_payload_bytes": segment.uplink_payload_bytes,
            "edge_arrival_time_ms": segment.edge_arrival_time_ms,
            "verify_start_time_ms": segment.verify_start_time_ms,
            "verify_done_time_ms": segment.verify_done_time_ms,
            "verify_compute_ms": segment.verify_compute_ms,
            "downlink_delay_ms": segment.downlink_delay_ms,
            "downlink_payload_bytes": segment.downlink_payload_bytes,
            "tree_strategy": segment.tree_strategy,
            "tree_budget_nodes": segment.tree_budget_nodes,
            "draft_compute_nodes": segment.draft_compute_nodes,
            "processed_candidate_count": segment.processed_candidate_count,
            "retained_tree_nodes": segment.retained_tree_nodes,
            "target_verify_tree_nodes": segment.target_verify_tree_nodes,
            "beam_len": segment.beam_len,
            "tree_path_switched": segment.tree_path_switched,
            "proactive_used": segment.proactive_used,
            "proactive_hit": segment.proactive_hit,
            "proactive_wasted_tokens": segment.proactive_wasted_tokens,
            "proactive_start_time_ms": segment.proactive_start_time_ms,
            "proactive_done_time_ms": segment.proactive_done_time_ms,
            "pipeline_target_ms": segment.pipeline_target_ms,
            "pipeline_edge_cycle_ms": segment.pipeline_edge_cycle_ms,
            "pipeline_alignment_error_ms": segment.pipeline_alignment_error_ms,
            "status": segment.status,
        }
        for segment in result.segments
    ]


def device_rows(result: SimulationResult) -> list[dict[str, Any]]:
    starts = [request.decode_ready_time_ms for request in result.requests]
    finishes = [float(request.finish_time_ms) for request in result.requests]
    makespan_ms = max(finishes) - min(starts)
    return [
        {
            "method": result.method,
            "scenario": result.scenario,
            "device_id": runtime.device.device_id,
            "device_type": runtime.device.device_type,
            "draft_model": runtime.device.drafter_profile,
            "device_utilization": min(1.0, runtime.total_busy_time_ms / makespan_ms) if makespan_ms else 0.0,
            "idle_time_ms": max(0.0, makespan_ms - runtime.total_busy_time_ms),
            "draft_busy_time_ms": runtime.total_busy_time_ms,
            "draft_queue_wait_ms": runtime.total_queue_wait_ms,
            "num_assigned_requests": runtime.assigned_requests,
            "num_generated_draft_tokens": runtime.generated_draft_tokens,
            "num_accepted_tokens": runtime.accepted_draft_tokens,
            "num_rejected_tokens": runtime.generated_draft_tokens - runtime.accepted_draft_tokens,
            "avg_selected_gamma": _mean(runtime.selected_gammas),
        }
        for runtime in result.devices
    ]


def event_rows(result: SimulationResult) -> list[dict[str, Any]]:
    return [{**event, "scenario": result.scenario} for event in result.event_trace]


def write_csv(path: str | Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _latency_ratio(baseline: dict[str, Any] | None, row: dict[str, Any]) -> float | None:
    if baseline is None:
        return None
    return baseline["avg_latency_ms"] / row["avg_latency_ms"]


def _latency_reduction(baseline: dict[str, Any] | None, row: dict[str, Any]) -> float | None:
    if baseline is None:
        return None
    return (baseline["avg_latency_ms"] - row["avg_latency_ms"]) / baseline["avg_latency_ms"]


def _goodput_gain(row: dict[str, Any], baseline: dict[str, Any] | None) -> float | None:
    if baseline is None:
        return None
    return row["goodput_tok_s"] / baseline["goodput_tok_s"]


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0
