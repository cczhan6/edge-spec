from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

from src.config import load_config
from src.entities import SimulationResult
from src.metrics import MAIN_FIELDS, SYSTEM_FIELDS, write_csv


METHODS = (
    "target_only",
    "server_only_linear",
    "server_only_tree",
    "specedge_linear",
    "specedge_tree",
    "dip_sd",
)

SCENARIO = "baseline_trace"

REQUIRED_FILES = (
    "resolved_config.json",
    "metrics.csv",
    "request_trace.csv",
    "event_trace.csv",
    "token_trace.csv",
    "resource_timeline.csv",
    "batch_trace.csv",
    "system_metrics.csv",
)

REQUEST_TRACE_FIELDS = [
    "method",
    "scenario",
    "request_id",
    "device_id",
    "prompt_id",
    "category",
    "arrival_time_ms",
    "decode_ready_time_ms",
    "finish_time_ms",
    "latency_ms",
    "output_len",
    "generated_tokens",
    "generated_ids",
    "status",
    "accepted_tokens",
    "rejected_count",
    "rollback_count",
    "wasted_draft_tokens",
    "bonus_reused_tokens",
    "in_flight_segment_count",
    "pending_segment_count",
    "completed_result_count",
    "draft_queued",
    "proactive_pending_count",
]

EVENT_TRACE_FIELDS = [
    "event_index",
    "event_time_ms",
    "event",
    "method",
    "scenario",
    "epoch",
    "batch_index",
    "request_id",
    "request_ids",
    "segment_id",
    "segment_ids",
    "device_id",
    "draft_model",
    "lane_id",
    "resource",
    "batch_size",
    "scheduled_gamma",
    "verify_gamma",
    "accepted_count",
    "proposed_count",
    "emitted_count",
    "start_time_ms",
    "finish_time_ms",
    "time_ms",
    "compute_ms",
    "queue_wait_ms",
    "uplink_ms",
    "downlink_ms",
    "tree_strategy",
    "tree_budget_nodes",
    "draft_compute_nodes",
    "processed_candidate_count",
    "retained_tree_nodes",
    "target_verify_tree_nodes",
    "tree_path_switched",
    "batch_type",
    "proactive_used",
    "proactive_reused_tokens",
    "pipeline_alignment_error_ms",
    "pipeline_idle_bubble_ms",
]

TOKEN_TRACE_FIELDS = [
    "method",
    "scenario",
    "token_type",
    "request_id",
    "segment_id",
    "device_id",
    "position",
    "token_index",
    "token_id",
    "commit_time_ms",
    "count",
    "status",
    "source_event",
    "wasted_reason",
]

RESOURCE_TIMELINE_FIELDS = [
    "method",
    "scenario",
    "resource_key",
    "resource_type",
    "resource_id",
    "event",
    "epoch",
    "batch_index",
    "request_id",
    "request_ids",
    "segment_id",
    "segment_ids",
    "device_id",
    "lane_id",
    "start_time_ms",
    "finish_time_ms",
    "duration_ms",
]

BATCH_TRACE_FIELDS = [
    "method",
    "scenario",
    "event_index",
    "event",
    "epoch",
    "batch_index",
    "batch_size",
    "request_id",
    "request_ids",
    "segment_id",
    "segment_ids",
    "start_time_ms",
    "finish_time_ms",
    "compute_ms",
    "tree_strategy",
    "batch_type",
    "optimizer_batch",
    "optimizer_pipeline_span",
]


def write_trace_bundle(
    output_dir: str | Path,
    config: dict[str, Any],
    result: SimulationResult,
    main: dict[str, Any],
    system: dict[str, Any],
) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(
        directory / "resolved_config.json",
        {"method": result.method, "scenario": result.scenario, "config": config},
    )
    write_csv(directory / "metrics.csv", [main], MAIN_FIELDS)
    _write_csv(directory / "request_trace.csv", _request_trace_rows(result), REQUEST_TRACE_FIELDS)
    event_rows = _event_trace_rows(result)
    _write_csv(directory / "event_trace.csv", event_rows, EVENT_TRACE_FIELDS)
    _write_csv(directory / "token_trace.csv", _token_trace_rows(result), TOKEN_TRACE_FIELDS)
    _write_csv(
        directory / "resource_timeline.csv",
        _resource_timeline_rows(result),
        RESOURCE_TIMELINE_FIELDS,
    )
    _write_csv(directory / "batch_trace.csv", _batch_trace_rows(result), BATCH_TRACE_FIELDS)
    write_csv(directory / "system_metrics.csv", [system], SYSTEM_FIELDS)


def prepare_trace_inputs(root: str | Path, config_path: str | Path) -> tuple[Path, Path]:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    config["simulation"]["seed"] = 20260625
    config["simulation"]["num_requests"] = 4
    config["simulation"]["num_devices"] = 4
    config["simulation"]["output_len_choices"] = [8, 12, 16]
    config["simulation"]["request_arrival"] = "burst"
    config["edge"]["num_lanes"] = 2
    config["edge"]["verify_startup_ms"] = 1
    config["edge"]["target_only_startup_ms"] = 0
    config["edge"]["target_only_token_rate_tok_s"] = 1000
    config["network"]["packet_header_bytes"] = 0
    config["network"]["packet_token_bytes"] = 0
    config["speculation"]["gamma_candidates"] = [1]
    config["speculation"]["gamma_fixed"] = 1
    config["speculation"]["W_default"] = 1
    config["speculation"]["W_max"] = 4
    config["speculation"]["unconfirmed_token_budget"] = 16
    config["specedge"]["server_batch_size"] = 1
    config["specedge"]["server_batch_timeout_ms"] = None
    config["specedge"]["server_batch_type"] = "static"
    config["specedge"]["proactive_enabled"] = True
    config["specedge"]["proactive_type"] = "excluded"
    config["dip_sd"]["batch_count"] = 2
    config["dip_sd"]["max_active_requests"] = 4
    config["dip_sd"]["max_batch_size"] = 2
    config["dip_sd"]["min_draft_length"] = 1
    config["dip_sd"]["max_draft_length"] = 3
    config["dip_sd"]["draft_length"] = 1
    for profile in config["drafter_profiles"].values():
        profile["acceptance_prior"] = 0.9
    _configure_heterogeneous_devices(config)
    config_path_out = root_path / "baseline_trace_config.yaml"
    dataset_path_out = root_path / "baseline_trace_dataset.jsonl"
    _write_yaml(config_path_out, config)
    _write_dataset(dataset_path_out)
    return config_path_out, dataset_path_out


def verify_trace_outputs(root: str | Path, summary_path: str | Path) -> list[dict[str, Any]]:
    root_path = Path(root)
    method_data = {method: _load_method_outputs(root_path, method) for method in METHODS}
    failures: list[str] = []

    for method, data in method_data.items():
        requests = data["requests"]
        failures.extend(_check_all_requests_finished(method, requests))
        failures.extend(_check_no_pending_state(method, requests))
        failures.extend(_check_event_monotonicity(method, data["events"]))
        failures.extend(_check_resource_overlap(method, data["resources"]))
        failures.extend(_check_required_files(root_path / method))

    failures.extend(_check_outputs_equal_target(method_data))
    failures.extend(_check_server_only_batch_size(method_data))
    failures.extend(_check_specedge_proactive_and_waste(method_data))
    failures.extend(_check_dip_sd_batches_and_overlap(method_data["dip_sd"]))

    summaries = [_summarize_method(method, method_data[method]) for method in METHODS]
    _write_summary(summary_path, summaries, failures)
    if failures:
        raise SystemExit("baseline trace verification failed:\n" + "\n".join(f"- {item}" for item in failures))
    return summaries


def _configure_heterogeneous_devices(config: dict[str, Any]) -> None:
    heterogeneous = config["device_pools"]["heterogeneous"]["templates"]
    for template in heterogeneous.values():
        template["count"] = 0
        template["draft_startup_ms"] = 0
        template["uplink_mbps"] = 1000
        template["downlink_mbps"] = 1000
        template["rtt_ms"] = 0
        template["jitter_ms"] = 0
    heterogeneous["low_end"]["count"] = 1
    heterogeneous["low_end"]["drafter_profile"] = "small"
    heterogeneous["low_end"]["draft_token_rate_tok_s"] = 1
    heterogeneous["mid_end"]["count"] = 2
    heterogeneous["mid_end"]["drafter_profile"] = "medium"
    heterogeneous["mid_end"]["draft_token_rate_tok_s"] = 1000
    heterogeneous["high_end"]["count"] = 1
    heterogeneous["high_end"]["drafter_profile"] = "large"
    heterogeneous["high_end"]["draft_token_rate_tok_s"] = 1000
    config["device_pools"]["heterogeneous"]["templates"] = {
        "mid_end": heterogeneous["mid_end"],
        "high_end": heterogeneous["high_end"],
        "low_end": heterogeneous["low_end"],
    }

    medium_only = config["device_pools"]["medium_only"]["templates"]["medium"]
    medium_only["count"] = 4
    medium_only["draft_startup_ms"] = 0
    medium_only["draft_token_rate_tok_s"] = 1000
    medium_only["uplink_mbps"] = 1000
    medium_only["downlink_mbps"] = 1000
    medium_only["rtt_ms"] = 0
    medium_only["jitter_ms"] = 0


def _request_trace_rows(result: SimulationResult) -> list[dict[str, Any]]:
    rows = []
    for request in result.requests:
        rows.append(
            {
                "method": result.method,
                "scenario": result.scenario,
                "request_id": request.request_id,
                "device_id": request.device_id,
                "prompt_id": request.prompt_id,
                "category": request.category_group or request.category or "unknown",
                "arrival_time_ms": request.arrival_time_ms,
                "decode_ready_time_ms": request.decode_ready_time_ms,
                "finish_time_ms": request.finish_time_ms,
                "latency_ms": request.latency_ms,
                "output_len": request.output_len,
                "generated_tokens": len(request.generated_ids),
                "generated_ids": request.generated_ids,
                "status": request.status,
                "accepted_tokens": request.accepted_tokens,
                "rejected_count": request.rejected_count,
                "rollback_count": request.rollback_count,
                "wasted_draft_tokens": request.wasted_draft_tokens,
                "bonus_reused_tokens": request.bonus_reused_tokens,
                "in_flight_segment_count": len(request.in_flight_segments),
                "pending_segment_count": len(request.pending_segments),
                "completed_result_count": len(request.completed_results),
                "draft_queued": request.draft_queued,
                "proactive_pending_count": len(request.proactive_draft_ids),
            }
        )
    return rows


def _event_trace_rows(result: SimulationResult) -> list[dict[str, Any]]:
    rows = []
    for index, event in enumerate(result.event_trace):
        rows.append(
            {
                **event,
                "method": event.get("method", result.method),
                "scenario": result.scenario,
                "event_index": index,
                "event_time_ms": _event_time(event),
            }
        )
    return sorted(rows, key=lambda row: (float(row["event_time_ms"]), int(row["event_index"])))


def _token_trace_rows(result: SimulationResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for request in result.requests:
        for position, (token_id, commit_time_ms) in enumerate(
            zip(request.generated_ids, request.committed_token_times_ms, strict=True)
        ):
            rows.append(
                _token_row(
                    result,
                    "committed",
                    request.request_id,
                    position=position,
                    token_index=position,
                    token_id=token_id,
                    commit_time_ms=commit_time_ms,
                    count=1,
                    status=request.status,
                    source_event="final_output",
                )
            )
    rows.sort(key=lambda row: (int(row["request_id"]), int(row["token_index"])))
    for segment in result.segments:
        rows.append(
            _token_row(
                result,
                "drafted",
                segment.request_id,
                segment_id=segment.segment_id,
                device_id=segment.device_id,
                position=segment.base_pos,
                count=_drafted_count(segment),
                status=segment.status,
                source_event="draft",
            )
        )
        if segment.proactive_draft_ids:
            rows.append(
                _token_row(
                    result,
                    "proactive_drafted",
                    segment.request_id,
                    segment_id=segment.segment_id,
                    device_id=segment.device_id,
                    position=segment.base_pos + segment.gamma,
                    count=_proactive_proposed_count(segment),
                    status=segment.status,
                    source_event="proactive_draft",
                )
            )
        if segment.accepted_count is not None:
            rows.append(
                _token_row(
                    result,
                    "verified",
                    segment.request_id,
                    segment_id=segment.segment_id,
                    device_id=segment.device_id,
                    position=segment.base_pos,
                    count=_verified_count(segment),
                    status=segment.status,
                    source_event="verification_result",
                )
            )
            if segment.accepted_count:
                rows.append(
                    _token_row(
                        result,
                        "accepted",
                        segment.request_id,
                        segment_id=segment.segment_id,
                        device_id=segment.device_id,
                        position=segment.base_pos,
                        count=segment.accepted_count,
                        status=segment.status,
                        source_event="verification_result",
                    )
                )
        wasted = _segment_waste_count(segment)
        if wasted:
            rows.append(
                _token_row(
                    result,
                    "wasted",
                    segment.request_id,
                    segment_id=segment.segment_id,
                    device_id=segment.device_id,
                    position=segment.base_pos,
                    count=wasted,
                    status=segment.status,
                    source_event="verification_result",
                    wasted_reason=_waste_reason(segment),
                )
            )
        if segment.proactive_wasted_tokens:
            rows.append(
                _token_row(
                    result,
                    "wasted",
                    segment.request_id,
                    segment_id=segment.segment_id,
                    device_id=segment.device_id,
                    position=segment.base_pos + segment.gamma,
                    count=segment.proactive_wasted_tokens,
                    status=segment.status,
                    source_event="proactive_alignment",
                    wasted_reason="proactive_alignment",
                )
            )
    return rows


def _token_row(
    result: SimulationResult,
    token_type: str,
    request_id: int,
    *,
    segment_id: int | None = None,
    device_id: int | None = None,
    position: int | None = None,
    token_index: int | None = None,
    token_id: int | None = None,
    commit_time_ms: float | None = None,
    count: int = 1,
    status: str = "",
    source_event: str = "",
    wasted_reason: str = "",
) -> dict[str, Any]:
    return {
        "method": result.method,
        "scenario": result.scenario,
        "token_type": token_type,
        "request_id": request_id,
        "segment_id": segment_id,
        "device_id": device_id,
        "position": position,
        "token_index": token_index,
        "token_id": token_id,
        "commit_time_ms": commit_time_ms,
        "count": count,
        "status": status,
        "source_event": source_event,
        "wasted_reason": wasted_reason,
    }


def _resource_timeline_rows(result: SimulationResult) -> list[dict[str, Any]]:
    rows = []
    request_ids_by_segment = {
        segment.segment_id: segment.request_id for segment in result.segments
    }
    for event in _event_trace_rows(result):
        if event.get("start_time_ms") in (None, "") or event.get("finish_time_ms") in (None, ""):
            continue
        resource = _resource_for_event(event)
        if resource is None:
            continue
        segment_ids = _as_list(event.get("segment_ids"))
        if not segment_ids and event.get("segment_id") not in (None, ""):
            segment_ids = [int(event["segment_id"])]
        request_ids = _as_list(event.get("request_ids"))
        if not request_ids:
            request_ids = [
                request_ids_by_segment[segment_id]
                for segment_id in segment_ids
                if segment_id in request_ids_by_segment
            ]
        start = float(event["start_time_ms"])
        finish = float(event["finish_time_ms"])
        rows.append(
            {
                "method": result.method,
                "scenario": result.scenario,
                "resource_key": resource[0],
                "resource_type": resource[1],
                "resource_id": resource[2],
                "event": event["event"],
                "epoch": event.get("epoch"),
                "batch_index": event.get("batch_index"),
                "request_id": event.get("request_id"),
                "request_ids": request_ids,
                "segment_id": event.get("segment_id"),
                "segment_ids": segment_ids,
                "device_id": event.get("device_id"),
                "lane_id": event.get("lane_id"),
                "start_time_ms": start,
                "finish_time_ms": finish,
                "duration_ms": finish - start,
            }
        )
    return sorted(rows, key=lambda row: (row["resource_key"], row["start_time_ms"], row["finish_time_ms"]))


def _batch_trace_rows(result: SimulationResult) -> list[dict[str, Any]]:
    batch_events = {
        "target_only_service",
        "server_only_verify",
        "global_batch_verify",
        "dip_sd_batch_verify",
    }
    rows = []
    request_ids_by_segment = {
        segment.segment_id: segment.request_id for segment in result.segments
    }
    for event in _event_trace_rows(result):
        if event["event"] not in batch_events:
            continue
        segment_ids = _as_list(event.get("segment_ids"))
        if not segment_ids and event.get("segment_id") not in (None, ""):
            segment_ids = [int(event["segment_id"])]
        request_ids = _as_list(event.get("request_ids"))
        if not request_ids:
            request_ids = [
                request_ids_by_segment[segment_id]
                for segment_id in segment_ids
                if segment_id in request_ids_by_segment
            ]
        if not request_ids and event.get("request_id") not in (None, ""):
            request_ids = [int(event["request_id"])]
        rows.append(
            {
                "method": result.method,
                "scenario": result.scenario,
                "event_index": event["event_index"],
                "event": event["event"],
                "epoch": event.get("epoch"),
                "batch_index": event.get("batch_index"),
                "batch_size": event.get("batch_size", 1),
                "request_id": event.get("request_id"),
                "request_ids": request_ids,
                "segment_id": event.get("segment_id"),
                "segment_ids": segment_ids,
                "start_time_ms": event.get("start_time_ms"),
                "finish_time_ms": event.get("finish_time_ms"),
                "compute_ms": event.get("compute_ms"),
                "tree_strategy": event.get("tree_strategy"),
                "batch_type": event.get("batch_type"),
                "optimizer_batch": event.get("optimizer_batch"),
                "optimizer_pipeline_span": event.get("optimizer_pipeline_span"),
            }
        )
    return rows


def _resource_for_event(event: dict[str, Any]) -> tuple[str, str, str] | None:
    name = str(event["event"])
    if name == "target_only_service":
        lane_id = str(event.get("lane_id", 0))
        return (f"target_only:{lane_id}", "target", lane_id)
    if name == "server_only_verify":
        return ("server_target_gpu", "target", "server_target_gpu")
    if name in {"global_batch_verify", "dip_sd_batch_verify"}:
        return ("server_target_gpu", "target", "server_target_gpu")
    if name == "lane_verify":
        lane_id = str(event.get("lane_id", 0))
        return (f"lane:{lane_id}", "target", lane_id)
    if name == "server_only_draft":
        return ("server_draft_gpu", "draft", "server_draft_gpu")
    if name in {"draft_compute", "proactive_draft", "dip_sd_draft"}:
        device_id = str(event.get("device_id", 0))
        return (f"device:{device_id}", "draft", device_id)
    return None


def _load_method_outputs(root: Path, method: str) -> dict[str, Any]:
    directory = root / method
    data = {
        "method": method,
        "directory": directory,
        "metrics": _read_csv(directory / "metrics.csv"),
        "requests": _read_csv(directory / "request_trace.csv"),
        "events": _read_csv(directory / "event_trace.csv"),
        "tokens": _read_csv(directory / "token_trace.csv"),
        "resources": _read_csv(directory / "resource_timeline.csv"),
        "batches": _read_csv(directory / "batch_trace.csv"),
    }
    return data


def _check_required_files(directory: Path) -> list[str]:
    failures = []
    for filename in REQUIRED_FILES:
        path = directory / filename
        if not path.exists():
            failures.append(f"{directory.name}: missing {filename}")
        elif path.stat().st_size <= 0:
            failures.append(f"{directory.name}: empty {filename}")
    return failures


def _check_all_requests_finished(method: str, requests: list[dict[str, str]]) -> list[str]:
    failures = []
    if len(requests) != 4:
        failures.append(f"{method}: expected 4 requests, found {len(requests)}")
    for row in requests:
        if row.get("status") != "finished":
            failures.append(f"{method}: request {row.get('request_id')} status is {row.get('status')}")
        if _to_int(row.get("generated_tokens")) != _to_int(row.get("output_len")):
            failures.append(f"{method}: request {row.get('request_id')} did not generate output_len tokens")
        output_len = _to_int(row.get("output_len"))
        if output_len < 8 or output_len > 16:
            failures.append(f"{method}: request {row.get('request_id')} output_len {output_len} outside [8, 16]")
    return failures


def _check_no_pending_state(method: str, requests: list[dict[str, str]]) -> list[str]:
    failures = []
    for row in requests:
        for key in (
            "in_flight_segment_count",
            "pending_segment_count",
            "completed_result_count",
            "proactive_pending_count",
        ):
            if _to_int(row.get(key)) != 0:
                failures.append(f"{method}: request {row.get('request_id')} has {key}={row.get(key)}")
        if _to_bool(row.get("draft_queued")):
            failures.append(f"{method}: request {row.get('request_id')} remains draft_queued")
    return failures


def _check_event_monotonicity(method: str, events: list[dict[str, str]]) -> list[str]:
    failures = []
    previous = -1.0
    for row in events:
        event_time = _to_float(row.get("event_time_ms"))
        if event_time < previous:
            failures.append(f"{method}: event time decreased at event_index {row.get('event_index')}")
            break
        previous = event_time
        start = row.get("start_time_ms")
        finish = row.get("finish_time_ms")
        if start not in (None, "") and finish not in (None, ""):
            if _to_float(finish) < _to_float(start):
                failures.append(f"{method}: event {row.get('event')} finishes before it starts")
    return failures


def _check_resource_overlap(method: str, resources: list[dict[str, str]]) -> list[str]:
    failures = []
    by_resource: dict[str, list[dict[str, str]]] = {}
    for row in resources:
        by_resource.setdefault(str(row["resource_key"]), []).append(row)
    for resource, rows in by_resource.items():
        rows.sort(key=lambda row: (_to_float(row["start_time_ms"]), _to_float(row["finish_time_ms"])))
        for previous, current in zip(rows, rows[1:]):
            if _to_float(current["start_time_ms"]) < _to_float(previous["finish_time_ms"]) - 1e-9:
                failures.append(
                    f"{method}: resource {resource} overlaps {previous['event']} and {current['event']}"
                )
                break
    return failures


def _check_outputs_equal_target(method_data: dict[str, dict[str, Any]]) -> list[str]:
    target = _committed_tokens(method_data["target_only"]["tokens"])
    failures = []
    for method, data in method_data.items():
        observed = _committed_tokens(data["tokens"])
        if observed != target:
            failures.append(f"{method}: committed token trace differs from target_only")
    return failures


def _check_server_only_batch_size(method_data: dict[str, dict[str, Any]]) -> list[str]:
    failures = []
    for method in ("server_only_linear", "server_only_tree"):
        for row in method_data[method]["batches"]:
            if row.get("event") == "server_only_verify" and _to_int(row.get("batch_size")) != 1:
                failures.append(f"{method}: observed server-only batch_size={row.get('batch_size')}")
    return failures


def _check_specedge_proactive_and_waste(method_data: dict[str, dict[str, Any]]) -> list[str]:
    failures = []
    for method in ("specedge_linear", "specedge_tree"):
        events = method_data[method]["events"]
        tokens = method_data[method]["tokens"]
        if not any(row.get("event") == "proactive_draft" for row in events):
            failures.append(f"{method}: no proactive_draft event")
        if not any(
            row.get("token_type") == "wasted" and row.get("wasted_reason") == "proactive_alignment"
            for row in tokens
        ):
            failures.append(f"{method}: no proactive alignment waste record")
    return failures


def _check_dip_sd_batches_and_overlap(data: dict[str, Any]) -> list[str]:
    failures = []
    verify_batches = [
        row for row in data["batches"] if row.get("event") == "dip_sd_batch_verify"
    ]
    batch_keys = {
        (row.get("epoch"), row.get("batch_index"))
        for row in verify_batches
    }
    if len(batch_keys) < 2:
        failures.append("dip_sd: fewer than two batch verifies")
    drafts = [row for row in data["events"] if row.get("event") == "dip_sd_draft"]
    verifies = [row for row in data["events"] if row.get("event") == "dip_sd_batch_verify"]
    has_overlap = False
    for draft in drafts:
        for verify in verifies:
            if draft.get("batch_index") == verify.get("batch_index"):
                continue
            if _intervals_overlap(
                _to_float(draft["start_time_ms"]),
                _to_float(draft["finish_time_ms"]),
                _to_float(verify["start_time_ms"]),
                _to_float(verify["finish_time_ms"]),
            ):
                has_overlap = True
                break
        if has_overlap:
            break
    if not has_overlap:
        failures.append("dip_sd: no cross-batch draft/verify overlap")

    result_by_segment = {
        row["segment_id"]: row
        for row in data["events"]
        if row.get("event") == "dip_sd_result"
    }
    drafts_by_request: dict[str, list[dict[str, str]]] = {}
    for row in drafts:
        drafts_by_request.setdefault(row["request_id"], []).append(row)
    for request_id, request_drafts in drafts_by_request.items():
        request_drafts.sort(key=lambda row: _to_float(row["start_time_ms"]))
        for previous, current in zip(request_drafts, request_drafts[1:]):
            result = result_by_segment.get(previous["segment_id"])
            if result is None:
                failures.append(f"dip_sd: missing result for segment {previous['segment_id']}")
                continue
            if _to_float(current["start_time_ms"]) < _to_float(result["finish_time_ms"]) - 1e-9:
                failures.append(
                    f"dip_sd: request {request_id} drafted segment {current['segment_id']} before previous verify/result completion"
                )
    return failures


def _summarize_method(method: str, data: dict[str, Any]) -> dict[str, Any]:
    requests = data["requests"]
    tokens = data["tokens"]
    events = data["events"]
    batches = data["batches"]
    committed = _sum_token_type(tokens, "committed")
    drafted = _sum_token_type(tokens, "drafted") + _sum_token_type(tokens, "proactive_drafted")
    verified = _sum_token_type(tokens, "verified")
    accepted = _sum_token_type(tokens, "accepted")
    wasted = _sum_token_type(tokens, "wasted")
    finish_ms = max((_to_float(row["finish_time_ms"]) for row in requests), default=0.0)
    return {
        "method": method,
        "success": all(row.get("status") == "finished" for row in requests),
        "request_count": len(requests),
        "committed_tokens": committed,
        "drafted_tokens": drafted,
        "verified_tokens": verified,
        "accepted_tokens": accepted,
        "wasted_tokens": wasted,
        "finish_time_ms": finish_ms,
        "features": _feature_summary(method, events, batches, tokens),
        "caveat": _method_caveat(method),
    }


def _feature_summary(
    method: str,
    events: list[dict[str, str]],
    batches: list[dict[str, str]],
    tokens: list[dict[str, str]],
) -> str:
    if method == "target_only":
        services = sum(1 for row in events if row.get("event") == "target_only_service")
        return f"{services} serialized target-only services; no draft segments"
    if method.startswith("server_only"):
        verifies = [row for row in batches if row.get("event") == "server_only_verify"]
        strategy = next((row.get("tree_strategy") for row in verifies if row.get("tree_strategy")), "linear")
        return f"{len(verifies)} server-only verify rounds; batch_size=1; strategy={strategy}"
    if method.startswith("specedge"):
        proactive = sum(1 for row in events if row.get("event") == "proactive_draft")
        verifies = sum(1 for row in batches if row.get("event") == "global_batch_verify")
        waste = _sum_token_type(tokens, "wasted")
        return f"{verifies} server batch verifies; {proactive} proactive drafts; wasted={waste}"
    dip_batches = [row for row in batches if row.get("event") == "dip_sd_batch_verify"]
    epochs = sorted({row.get("epoch") for row in dip_batches})
    return f"{len(dip_batches)} DiP-SD batch verifies across {len(epochs)} epochs; optimizer batch_count>=2"


def _method_caveat(method: str) -> str:
    if method == "target_only":
        return "Decode-ready greedy reference; prefill is intentionally absent."
    if method.startswith("server_only"):
        return "Server-only is intentionally fixed to batch_size=1."
    if method == "specedge_tree":
        return "Tree path uses local specexec_approx trace semantics, not upstream CUDA replay."
    if method == "specedge_linear":
        return "Linear approximation keeps SpecEdge deployment and proactive scheduling semantics."
    if method == "dip_sd":
        return "Online epoch-barrier adaptation of the paper optimizer."
    return ""


def _write_summary(path: str | Path, summaries: list[dict[str, Any]], failures: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Baseline Trace Summary",
        "",
        f"Status: {'PASS' if not failures else 'FAIL'}",
        "",
        "| method | success | requests | committed | drafted | verified | accepted | wasted | finish_ms | trace features | caveat |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in summaries:
        lines.append(
            "| {method} | {success} | {request_count} | {committed_tokens} | "
            "{drafted_tokens} | {verified_tokens} | {accepted_tokens} | "
            "{wasted_tokens} | {finish_time_ms:.3f} | {features} | {caveat} |".format(
                **row
            )
        )
    lines.extend(["", "## Automatic Checks", ""])
    if failures:
        lines.extend(f"- FAIL: {failure}" for failure in failures)
    else:
        lines.extend(
            [
                "- PASS: six methods completed all requests.",
                "- PASS: committed token traces equal target_only.",
                "- PASS: no request finishes with pending or unverified state.",
                "- PASS: event times are monotonic in exported traces.",
                "- PASS: target and draft resources do not illegally overlap.",
                "- PASS: Server-only batch size remains 1.",
                "- PASS: SpecEdge traces contain proactive drafting and alignment waste.",
                "- PASS: DiP-SD has at least two batches with cross-batch draft/verify overlap.",
                "- PASS: DiP-SD requests do not redraft before prior verification/result completion.",
                "- PASS: all required output files exist and are non-empty.",
            ]
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _committed_tokens(rows: list[dict[str, str]]) -> dict[int, list[int]]:
    tokens: dict[int, list[tuple[int, int]]] = {}
    for row in rows:
        if row.get("token_type") != "committed":
            continue
        request_id = _to_int(row["request_id"])
        tokens.setdefault(request_id, []).append((_to_int(row["position"]), _to_int(row["token_id"])))
    return {
        request_id: [token_id for _, token_id in sorted(items)]
        for request_id, items in sorted(tokens.items())
    }


def _sum_token_type(rows: list[dict[str, str]], token_type: str) -> int:
    return sum(_to_int(row.get("count")) for row in rows if row.get("token_type") == token_type)


def _segment_waste_count(segment: Any) -> int:
    if segment.tree_strategy != "linear":
        drafted = _drafted_count(segment)
        if segment.status in {"accepted", "rejected"}:
            return max(0, drafted - int(segment.accepted_count or 0))
        if segment.status in {"stale", "discarded"}:
            return drafted
        return 0
    if segment.status == "rejected":
        return max(0, segment.proposed_count - int(segment.accepted_count or 0))
    if segment.status in {"stale", "discarded"}:
        return segment.proposed_count
    return 0


def _waste_reason(segment: Any) -> str:
    if segment.status == "rejected":
        return "rejected_suffix"
    if segment.status in {"stale", "discarded"}:
        return "invalidation"
    return ""


def _proactive_proposed_count(segment: Any) -> int:
    tree = segment.proactive_draft_tree
    if tree is None or not tree.nodes:
        return len(segment.proactive_draft_ids)
    return int(tree.processed_candidate_count or len(tree.nodes))


def _drafted_count(segment: Any) -> int:
    if segment.tree_strategy != "linear":
        return int(segment.processed_candidate_count or segment.proposed_count)
    return int(segment.proposed_count)


def _verified_count(segment: Any) -> int:
    if segment.tree_strategy != "linear":
        return int(segment.target_verify_tree_nodes or segment.verify_gamma)
    return int(segment.verify_gamma)


def _event_time(event: dict[str, Any]) -> float:
    for key in ("start_time_ms", "time_ms", "finish_time_ms"):
        if key in event and event[key] is not None:
            return float(event[key])
    return 0.0


def _as_list(value: Any) -> list[int]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [int(item) for item in value]
    if isinstance(value, tuple):
        return [int(item) for item in value]
    if isinstance(value, str):
        parsed = _parse_json(value)
        if isinstance(parsed, list):
            return [int(item) for item in parsed]
        return [int(value)]
    return [int(value)]


def _intervals_overlap(a_start: float, a_finish: float, b_start: float, b_finish: float) -> bool:
    return a_start < b_finish - 1e-9 and b_start < a_finish - 1e-9


def _write_csv(path: str | Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    extra_fields = sorted({key for row in rows for key in row} - set(fields))
    fieldnames = [*fields, *extra_fields]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _serialize(value) for key, value in row.items()})


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_yaml(path: str | Path, value: Any) -> None:
    Path(path).write_text(_simple_yaml(value), encoding="utf-8")


def _simple_yaml(value: Any, indent: int = 0) -> str:
    if not isinstance(value, dict):
        raise TypeError("top-level YAML value must be a mapping")
    lines: list[str] = []
    prefix = " " * indent
    for key, item in value.items():
        if isinstance(item, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(_simple_yaml(item, indent + 2).rstrip())
        else:
            lines.append(f"{prefix}{key}: {_yaml_scalar(item)}")
    return "\n".join(lines) + "\n"


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value))
    return json.dumps(str(value))


def _write_dataset(path: str | Path) -> None:
    records = [
        {
            "question_id": f"baseline-trace-{index}",
            "category": "qa",
            "turns": [f"baseline trace deterministic prompt {index}"],
        }
        for index in range(4)
    ]
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _serialize(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def _parse_json(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _to_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def _to_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and verify deterministic baseline traces.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--root", default="outputs/baseline_trace")
    prepare.add_argument("--config", default="configs/default.yaml")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--root", default="outputs/baseline_trace")
    verify.add_argument("--summary", default="outputs/baseline_trace/summary.md")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        config_path, dataset_path = prepare_trace_inputs(args.root, args.config)
        print(f"config: {config_path}")
        print(f"dataset: {dataset_path}")
        return
    if args.command == "verify":
        summaries = verify_trace_outputs(args.root, args.summary)
        print(f"summary: {args.summary}")
        for row in summaries:
            print(
                f"{row['method']}: success={row['success']} "
                f"requests={row['request_count']} committed={row['committed_tokens']} "
                f"finish_ms={row['finish_time_ms']:.3f}"
            )


if __name__ == "__main__":
    main()
