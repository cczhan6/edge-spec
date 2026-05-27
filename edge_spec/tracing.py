from __future__ import annotations

from .protocol import DraftSegment, VerificationOutcome
from .types import ClientState


def request_record(
    client: ClientState,
    target_model: str,
    generated_text: str,
    microbatch_id: int,
    request_start_s: float,
    method: str,
    target_only: dict | None = None,
    extra: dict | None = None,
) -> dict:
    target_only = target_only or {}
    token_count = len(client.generated_ids)
    acceptance_rate = (
        client.accepted_draft_tokens / client.proposed_draft_tokens
        if client.proposed_draft_tokens
        else None
    )
    record = {
        "microbatch_id": microbatch_id,
        "device_id": client.device_id,
        "draft_model": client.draft_model,
        "target_model": target_model,
        "task": client.category,
        "prompt_id": client.prompt_id,
        "prompt": client.prompt,
        "generated_text": generated_text,
        "generated_token_count": token_count,
        "effective_received_token_count": token_count,
        "acceptance_rate": acceptance_rate,
        "accepted_draft_tokens": client.accepted_draft_tokens,
        "proposed_draft_tokens": client.proposed_draft_tokens,
        "sync_rounds": client.sync_rounds,
        "execution_mode": method,
        "method": method,
        "first_token_latency_s": client.first_token_latency_s,
        "latency_s": client.latency_s,
        "start_time_s": request_start_s,
        "completion_time_s": request_start_s + client.latency_s,
        "tokens_per_s": token_count / client.latency_s
        if client.latency_s > 0
        else 0.0,
        "effective_received_tokens_per_s": token_count / client.latency_s
        if client.latency_s > 0
        else 0.0,
        "target_only_latency_s": target_only.get("target_only_latency_s"),
        "target_only_model_latency_s": target_only.get("target_only_model_latency_s"),
        "target_only_uplink_s": target_only.get("target_only_uplink_s"),
        "target_only_downlink_s": target_only.get("target_only_downlink_s"),
        "target_only_uplink_payload_bytes": target_only.get(
            "target_only_uplink_payload_bytes"
        ),
        "target_only_downlink_payload_bytes": target_only.get(
            "target_only_downlink_payload_bytes"
        ),
        "target_only_uplink_effective_mbps": target_only.get(
            "target_only_uplink_effective_mbps"
        ),
        "target_only_downlink_effective_mbps": target_only.get(
            "target_only_downlink_effective_mbps"
        ),
        "target_only_uplink_effective_rtt_ms": target_only.get(
            "target_only_uplink_effective_rtt_ms"
        ),
        "target_only_downlink_effective_rtt_ms": target_only.get(
            "target_only_downlink_effective_rtt_ms"
        ),
        "target_only_uplink_jitter_s": target_only.get("target_only_uplink_jitter_s"),
        "target_only_downlink_jitter_s": target_only.get(
            "target_only_downlink_jitter_s"
        ),
        "target_only_uplink_congested": target_only.get(
            "target_only_uplink_congested"
        ),
        "target_only_downlink_congested": target_only.get(
            "target_only_downlink_congested"
        ),
        "target_only_text": target_only.get("target_only_text"),
        "speedup_vs_target_only": target_only.get("speedup_vs_target_only"),
    }
    if extra:
        record.update(extra)
    return record


def segment_device_trace(
    segment: DraftSegment,
    *,
    barrier_wait_s: float = 0.0,
    accepted_count: int = 0,
    proposed_count: int = 0,
    emitted_count: int = 0,
    downlink_s: float = 0.0,
    downlink_effective_mbps: float = 0.0,
    downlink_effective_rtt_ms: float = 0.0,
    downlink_jitter_s: float = 0.0,
    downlink_congested: bool = False,
    downlink_payload_bytes: int = 0,
    lane_id: int | None = None,
    lane_start_s: float | None = None,
    lane_finish_s: float | None = None,
    lane_queue_wait_s: float | None = None,
    status: str = "verified",
    extra: dict | None = None,
) -> dict:
    row = {
        "device_id": segment.device_id,
        "request_id": segment.request_id,
        "draft_model": segment.draft_model,
        "segment_id": segment.segment_id,
        "prefix_version": segment.prefix_version,
        "base_position": segment.base_position,
        "prefix_hash": segment.prefix_hash,
        "lookahead": segment.lookahead,
        "draft_start_s": segment.draft_start_s,
        "draft_end_s": segment.draft_end_s,
        "draft_time_s": segment.draft_elapsed_s,
        "uplink_s": segment.uplink_s,
        "uplink_effective_mbps": segment.uplink_effective_mbps,
        "uplink_effective_rtt_ms": segment.uplink_effective_rtt_ms,
        "uplink_jitter_s": segment.uplink_jitter_s,
        "uplink_congested": segment.uplink_congested,
        "arrival_s": segment.arrival_s,
        "barrier_wait_s": barrier_wait_s,
        "downlink_s": downlink_s,
        "downlink_effective_mbps": downlink_effective_mbps,
        "downlink_effective_rtt_ms": downlink_effective_rtt_ms,
        "downlink_jitter_s": downlink_jitter_s,
        "downlink_congested": downlink_congested,
        "uplink_payload_bytes": segment.uplink_payload_bytes,
        "downlink_payload_bytes": downlink_payload_bytes,
        "accepted_count": accepted_count,
        "proposed_count": proposed_count,
        "emitted_count": emitted_count,
        "status": status,
    }
    if lane_id is not None:
        row["lane_id"] = lane_id
    if lane_start_s is not None:
        row["lane_start_s"] = lane_start_s
    if lane_finish_s is not None:
        row["lane_finish_s"] = lane_finish_s
    if lane_queue_wait_s is not None:
        row["lane_queue_wait_s"] = lane_queue_wait_s
    if extra:
        row.update(extra)
    return row


def verification_event_trace(
    *,
    event_index: int,
    method: str,
    microbatch_id: int,
    round_index: int,
    target_batch_size: int,
    target_forward_s: float,
    devices: list[dict],
    lane_id: int | None = None,
    lane_start_s: float | None = None,
    lane_finish_s: float | None = None,
    lane_queue_wait_s: float | None = None,
    status: str = "verified",
    extra: dict | None = None,
) -> dict:
    row = {
        "method": method,
        "event_index": event_index,
        "microbatch_id": microbatch_id,
        "round_index": round_index,
        "target_batch_size": target_batch_size,
        "target_forward_s": target_forward_s,
        "devices": devices,
        "status": status,
    }
    if lane_id is not None:
        row["lane_id"] = lane_id
    if lane_start_s is not None:
        row["lane_start_s"] = lane_start_s
    if lane_finish_s is not None:
        row["lane_finish_s"] = lane_finish_s
    if lane_queue_wait_s is not None:
        row["lane_queue_wait_s"] = lane_queue_wait_s
    if extra:
        row.update(extra)
    return row


def outcome_device_trace(outcome: VerificationOutcome, status: str | None = None) -> dict:
    return segment_device_trace(
        outcome.segment,
        accepted_count=outcome.verification.accepted_count,
        proposed_count=outcome.verification.proposed_count,
        emitted_count=len(outcome.emitted_ids),
        downlink_s=outcome.downlink_s,
        downlink_effective_mbps=outcome.downlink_effective_mbps,
        downlink_effective_rtt_ms=outcome.downlink_effective_rtt_ms,
        downlink_jitter_s=outcome.downlink_jitter_s,
        downlink_congested=outcome.downlink_congested,
        downlink_payload_bytes=outcome.downlink_payload_bytes,
        lane_id=outcome.lane_id,
        lane_start_s=outcome.verify_start_s,
        lane_finish_s=outcome.verify_finish_s,
        lane_queue_wait_s=outcome.queue_wait_s,
        status=status or outcome.status,
    )

