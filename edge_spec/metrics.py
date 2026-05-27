from __future__ import annotations

from collections.abc import Sequence

from .protocol import VerifierLaneState


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _has_transfer(row: dict, direction: str) -> bool:
    payload = float(row.get(f"{direction}_payload_bytes", 0.0) or 0.0)
    delay = float(row.get(f"{direction}_s", 0.0) or 0.0)
    mbps = float(row.get(f"{direction}_effective_mbps", 0.0) or 0.0)
    return payload > 0 or delay > 0 or mbps > 0


def _collect_network_sample(
    row: dict,
    direction: str,
    mbps: list[float],
    rtt_ms: list[float],
) -> tuple[int, int]:
    if not _has_transfer(row, direction):
        return 0, 0
    mbps.append(float(row.get(f"{direction}_effective_mbps", 0.0) or 0.0))
    rtt_ms.append(float(row.get(f"{direction}_effective_rtt_ms", 0.0) or 0.0))
    return int(bool(row.get(f"{direction}_congested", False))), 1


def _collect_target_only_network_sample(
    record: dict,
    direction: str,
    mbps: list[float],
    rtt_ms: list[float],
) -> tuple[int, int]:
    prefix = f"target_only_{direction}"
    if record.get(f"{prefix}_effective_mbps") is None:
        return 0, 0
    mbps.append(float(record.get(f"{prefix}_effective_mbps", 0.0) or 0.0))
    rtt_ms.append(float(record.get(f"{prefix}_effective_rtt_ms", 0.0) or 0.0))
    return int(bool(record.get(f"{prefix}_congested", False))), 1


def summarize(records: list[dict], traces: list[dict], total_time_s: float) -> dict:
    total_tokens = sum(record["generated_token_count"] for record in records)
    waits: list[float] = []
    uplink_mbps: list[float] = []
    downlink_mbps: list[float] = []
    uplink_rtt_ms: list[float] = []
    downlink_rtt_ms: list[float] = []
    congestion_events = 0
    network_samples = 0
    device_latency: dict[str, list[float]] = {}
    slowest_counts: dict[str, int] = {}
    accepted_tokens = 0
    proposed_tokens = 0
    acceptance_rates: list[float] = []

    for record in records:
        device_latency.setdefault(record["device_id"], []).append(record["latency_s"])
        proposed = int(record.get("proposed_draft_tokens", 0) or 0)
        accepted = int(record.get("accepted_draft_tokens", 0) or 0)
        if proposed > 0:
            proposed_tokens += proposed
            accepted_tokens += accepted
            rate = record.get("acceptance_rate")
            if rate is not None:
                acceptance_rates.append(float(rate))

    for trace in traces:
        devices = trace.get("devices", [])
        if not devices:
            continue
        for device in devices:
            waits.append(float(device.get("barrier_wait_s", 0.0) or 0.0))
            events, samples = _collect_network_sample(
                device, "uplink", uplink_mbps, uplink_rtt_ms
            )
            congestion_events += events
            network_samples += samples
            events, samples = _collect_network_sample(
                device, "downlink", downlink_mbps, downlink_rtt_ms
            )
            congestion_events += events
            network_samples += samples
        if trace.get("method") == "sync_batch" or "barrier_time_s" in trace:
            slowest = min(devices, key=lambda item: item.get("barrier_wait_s", 0.0))
            slowest_counts[slowest["device_id"]] = (
                slowest_counts.get(slowest["device_id"], 0) + 1
            )

    if network_samples == 0:
        for record in records:
            events, samples = _collect_target_only_network_sample(
                record, "uplink", uplink_mbps, uplink_rtt_ms
            )
            congestion_events += events
            network_samples += samples
            events, samples = _collect_target_only_network_sample(
                record, "downlink", downlink_mbps, downlink_rtt_ms
            )
            congestion_events += events
            network_samples += samples

    baseline_latencies = [
        record["target_only_latency_s"]
        for record in records
        if record.get("target_only_latency_s") is not None
    ]
    speedups = [
        record["speedup_vs_target_only"]
        for record in records
        if record.get("speedup_vs_target_only") is not None
    ]
    draft_uplink = sum(
        float(device.get("draft_time_s", 0.0) or 0.0)
        + float(device.get("uplink_s", 0.0) or 0.0)
        for trace in traces
        for device in trace.get("devices", [])
    )
    mean_acceptance_rate = _mean(acceptance_rates) if acceptance_rates else None
    overall_acceptance_rate = (
        accepted_tokens / proposed_tokens if proposed_tokens > 0 else None
    )
    return {
        "request_count": len(records),
        "round_count": len(traces),
        "total_generated_tokens": total_tokens,
        "total_virtual_time_s": total_time_s,
        "throughput_tokens_per_s": total_tokens / total_time_s
        if total_time_s > 0
        else 0.0,
        "mean_acceptance_rate": mean_acceptance_rate,
        "overall_acceptance_rate": overall_acceptance_rate,
        "mean_barrier_wait_s": _mean(waits),
        "mean_uplink_effective_mbps": _mean(uplink_mbps),
        "mean_downlink_effective_mbps": _mean(downlink_mbps),
        "mean_uplink_effective_rtt_ms": _mean(uplink_rtt_ms),
        "mean_downlink_effective_rtt_ms": _mean(downlink_rtt_ms),
        "network_congestion_events": congestion_events,
        "network_congestion_fraction": congestion_events / network_samples
        if network_samples
        else 0.0,
        "barrier_wait_fraction": sum(waits) / (draft_uplink + sum(waits))
        if waits and (draft_uplink + sum(waits)) > 0
        else 0.0,
        "mean_target_only_latency_s": _mean(baseline_latencies)
        if baseline_latencies
        else None,
        "mean_speedup_vs_target_only": _mean(speedups) if speedups else None,
        "mean_latency_by_device": {
            device_id: sum(values) / len(values)
            for device_id, values in device_latency.items()
        },
        "slowest_device_rounds": slowest_counts,
        "task_metrics": summarize_by_task(records),
    }


def summarize_lanes(
    records: list[dict],
    traces: list[dict],
    total_time_s: float,
    lanes: Sequence[VerifierLaneState],
) -> dict:
    summary = summarize(records, traces, total_time_s)
    target_events = [
        trace for trace in traces if int(trace.get("target_batch_size", 0)) > 0
    ]
    segment_queue_waits: list[float] = []
    segment_target_forwards: list[float] = []
    for trace in target_events:
        target_forward_s = float(trace.get("target_forward_s", 0.0) or 0.0)
        for device in trace.get("devices", []):
            if device.get("lane_queue_wait_s") is None:
                continue
            segment_queue_waits.append(float(device.get("lane_queue_wait_s", 0.0)))
            segment_target_forwards.append(target_forward_s)
    total_queue_wait = sum(segment_queue_waits)
    total_target_forward = sum(segment_target_forwards)
    summary.update(
        {
            "lane_count": len(lanes),
            "verification_event_count": len(target_events),
            "lane_verification_count": sum(lane.verification_count for lane in lanes),
            "mean_lane_queue_wait_s": _mean(segment_queue_waits),
            "lane_queue_wait_fraction": total_queue_wait
            / (total_queue_wait + total_target_forward)
            if (total_queue_wait + total_target_forward) > 0
            else 0.0,
            "mean_lane_batch_size": (
                sum(int(trace.get("target_batch_size", 0)) for trace in target_events)
                / len(target_events)
                if target_events
                else 0.0
            ),
            "lane_busy_s": {str(lane.lane_id): lane.busy_s for lane in lanes},
            "lane_verifications": {
                str(lane.lane_id): lane.verification_count for lane in lanes
            },
            "lane_utilization": {
                str(lane.lane_id): lane.busy_s / total_time_s
                if total_time_s > 0
                else 0.0
                for lane in lanes
            },
            "mean_lane_utilization": (
                sum(lane.busy_s for lane in lanes) / (len(lanes) * total_time_s)
                if lanes and total_time_s > 0
                else 0.0
            ),
            "slowest_device_rounds": {},
        }
    )
    return summary


def summarize_by_task(records: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(str(record["task"]), []).append(record)

    metrics: dict[str, dict] = {}
    for task, task_records in sorted(grouped.items()):
        generated_tokens = sum(record["generated_token_count"] for record in task_records)
        effective_received_tokens = sum(
            record.get("effective_received_token_count", record["generated_token_count"])
            for record in task_records
        )
        microbatch_durations: dict[int, float | tuple[float, float]] = {}
        task_start_s = float("inf")
        task_end_s = 0.0
        for record in task_records:
            microbatch_id = int(record["microbatch_id"])
            if "start_time_s" in record and "completion_time_s" in record:
                start_s = float(record["start_time_s"])
                end_s = float(record["completion_time_s"])
                task_start_s = min(task_start_s, start_s)
                task_end_s = max(task_end_s, end_s)
                current_start, current_end = microbatch_durations.get(
                    microbatch_id,
                    (float("inf"), 0.0),
                )
                microbatch_durations[microbatch_id] = (
                    min(current_start, start_s),
                    max(current_end, end_s),
                )
            else:
                current_duration = microbatch_durations.get(microbatch_id, 0.0)
                duration_s = float(record["latency_s"])
                microbatch_durations[microbatch_id] = max(current_duration, duration_s)
                task_start_s = min(task_start_s, 0.0)
                task_end_s = max(task_end_s, duration_s)
        microbatch_duration_sum = 0.0
        for value in microbatch_durations.values():
            if isinstance(value, tuple):
                microbatch_duration_sum += max(0.0, value[1] - value[0])
            else:
                microbatch_duration_sum += float(value)
        effective_duration = max(0.0, task_end_s - task_start_s) if task_records else 0.0
        first_latencies = [
            float(record["first_token_latency_s"])
            for record in task_records
            if record.get("first_token_latency_s") is not None
        ]
        metrics[task] = {
            "request_count": len(task_records),
            "microbatch_count": len(microbatch_durations),
            "generated_token_count": generated_tokens,
            "effective_received_token_count": effective_received_tokens,
            "effective_duration_s": effective_duration,
            "microbatch_duration_sum_s": microbatch_duration_sum,
            "effective_throughput_tokens_per_s": generated_tokens / effective_duration
            if effective_duration > 0
            else 0.0,
            "effective_received_throughput_tokens_per_s": effective_received_tokens
            / effective_duration
            if effective_duration > 0
            else 0.0,
            "e2e_first_token_latency_s": _mean(first_latencies)
            if first_latencies
            else None,
            "e2e_mean_latency_s": sum(record["latency_s"] for record in task_records)
            / len(task_records)
            if task_records
            else 0.0,
        }
    return metrics
