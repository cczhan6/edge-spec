from __future__ import annotations

import heapq
import random
from collections import deque
from dataclasses import asdict, dataclass
from itertools import count
from typing import Protocol, Sequence

from .backends import ModelBackend
from .sampling import verify_draft_exact
from .simulation import (
    estimate_downlink_payload_bytes,
    estimate_prompt_payload_bytes,
    estimate_uplink_payload_bytes,
    sample_network_delay,
)
from .types import (
    ClientState,
    DeviceProfile,
    DraftPacket,
    SamplingConfig,
    SpecBenchItem,
)


class ProgressLike(Protocol):
    def update(self, n: int = 1) -> None: ...

    def set_postfix(self, ordered_dict=None, refresh: bool = True, **kwargs) -> None: ...


@dataclass
class VerificationPipeline:
    pipeline_id: int
    available_at_s: float = 0.0
    busy_s: float = 0.0
    verification_count: int = 0


class HeteroSyncRunner:
    def __init__(
        self,
        draft_backends: Sequence[ModelBackend],
        target_backend: ModelBackend,
        profiles: dict[str, DeviceProfile],
        sampling: SamplingConfig,
        gamma: int,
        max_new_tokens: int,
        seed: int = 0,
        run_target_baseline: bool = True,
    ) -> None:
        if len(draft_backends) != 3:
            raise ValueError("this baseline expects exactly three draft backends")
        self.draft_backends = list(draft_backends)
        self.target_backend = target_backend
        self.profiles = profiles
        self.sampling = sampling
        self.gamma = gamma
        self.max_new_tokens = max_new_tokens
        self.rng = random.Random(seed)
        self.baseline_rng = random.Random(seed + 10_000)
        self.run_target_baseline = run_target_baseline

    def build_clients(
        self,
        items: Sequence[SpecBenchItem],
        microbatch_start_s: float,
    ) -> list[ClientState]:
        clients: list[ClientState] = []
        for index, item in enumerate(items):
            backend = self.draft_backends[index]
            clients.append(
                ClientState(
                    device_id=f"device-{index}",
                    draft_model=backend.model_name,
                    prompt_id=item.request_id,
                    category=item.category,
                    prompt=item.prompt,
                    prefix_ids=backend.encode_prompt(item.prompt),
                    available_at_s=microbatch_start_s,
                )
            )
        return clients

    def _make_packet(
        self,
        microbatch_id: int,
        round_index: int,
        client: ClientState,
        backend: ModelBackend,
    ) -> DraftPacket:
        remaining = self.max_new_tokens - len(client.generated_ids)
        draft_len = min(self.gamma, remaining)
        prefix = client.prefix_ids + client.generated_ids
        draft = backend.draft(prefix, draft_len, self.sampling, self.rng)
        profile = self.profiles[client.device_id]
        payload_bytes = estimate_uplink_payload_bytes(
            prefix, draft.draft_ids, draft.draft_dists
        )
        uplink = sample_network_delay(payload_bytes, profile, "uplink", self.rng)
        draft_start_s = client.available_at_s
        draft_end_s = draft_start_s + draft.elapsed_s
        arrival = draft_end_s + uplink.delay_s
        return DraftPacket(
            microbatch_id=microbatch_id,
            round_index=round_index,
            device_id=client.device_id,
            request_id=client.prompt_id,
            draft_model=client.draft_model,
            prefix_ids=prefix,
            draft_ids=draft.draft_ids,
            draft_dists=draft.draft_dists,
            draft_start_s=draft_start_s,
            draft_end_s=draft_end_s,
            draft_elapsed_s=draft.elapsed_s,
            uplink_s=uplink.delay_s,
            uplink_effective_mbps=uplink.effective_mbps,
            uplink_effective_rtt_ms=uplink.effective_rtt_ms,
            uplink_jitter_s=uplink.jitter_s,
            uplink_congested=uplink.congested,
            arrival_s=arrival,
            uplink_payload_bytes=payload_bytes,
        )

    def _record_client(
        self,
        client: ClientState,
        microbatch_id: int,
        request_start_s: float,
        extra: dict | None = None,
    ) -> dict:
        generated_text = self.target_backend.decode(client.generated_ids)
        baseline_latency = None
        baseline_model_latency = None
        baseline_uplink_s = None
        baseline_downlink_s = None
        baseline_uplink_payload_bytes = None
        baseline_downlink_payload_bytes = None
        baseline_text = None
        speedup = None
        if self.run_target_baseline:
            profile = self.profiles[client.device_id]
            baseline_uplink_payload_bytes = estimate_prompt_payload_bytes(
                client.prefix_ids
            )
            baseline_uplink = sample_network_delay(
                baseline_uplink_payload_bytes,
                profile,
                "uplink",
                self.baseline_rng,
            )
            baseline_ids, baseline_model_latency = self.target_backend.generate_target_only(
                client.prefix_ids,
                self.max_new_tokens,
                self.sampling,
                self.baseline_rng,
            )
            baseline_text = self.target_backend.decode(baseline_ids)
            baseline_downlink_payload_bytes = estimate_downlink_payload_bytes(
                baseline_ids
            )
            baseline_downlink = sample_network_delay(
                baseline_downlink_payload_bytes,
                profile,
                "downlink",
                self.baseline_rng,
            )
            baseline_uplink_s = baseline_uplink.delay_s
            baseline_downlink_s = baseline_downlink.delay_s
            baseline_latency = (
                baseline_uplink_s
                + baseline_model_latency
                + baseline_downlink_s
            )
            if client.latency_s > 0:
                speedup = baseline_latency / client.latency_s
        token_count = len(client.generated_ids)
        record = {
            "microbatch_id": microbatch_id,
            "device_id": client.device_id,
            "draft_model": client.draft_model,
            "target_model": self.target_backend.model_name,
            "task": client.category,
            "prompt_id": client.prompt_id,
            "prompt": client.prompt,
            "generated_text": generated_text,
            "generated_token_count": token_count,
            "effective_received_token_count": token_count,
            "acceptance_rate": (
                client.accepted_draft_tokens / client.proposed_draft_tokens
                if client.proposed_draft_tokens
                else 0.0
            ),
            "accepted_draft_tokens": client.accepted_draft_tokens,
            "proposed_draft_tokens": client.proposed_draft_tokens,
            "sync_rounds": client.sync_rounds,
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
            "target_only_latency_s": baseline_latency,
            "target_only_model_latency_s": baseline_model_latency,
            "target_only_uplink_s": baseline_uplink_s,
            "target_only_downlink_s": baseline_downlink_s,
            "target_only_uplink_payload_bytes": baseline_uplink_payload_bytes,
            "target_only_downlink_payload_bytes": baseline_downlink_payload_bytes,
            "target_only_text": baseline_text,
            "speedup_vs_target_only": speedup,
        }
        if extra:
            record.update(extra)
        return record

    def run_microbatch(
        self,
        items: Sequence[SpecBenchItem],
        microbatch_id: int,
        start_time_s: float,
    ) -> tuple[list[dict], list[dict], float]:
        clients = self.build_clients(items, start_time_s)
        round_traces: list[dict] = []
        round_index = 0

        while any(
            (not client.done) and len(client.generated_ids) < self.max_new_tokens
            for client in clients
        ):
            active_pairs = [
                (client, self.draft_backends[index])
                for index, client in enumerate(clients)
                if (not client.done) and len(client.generated_ids) < self.max_new_tokens
            ]
            packets = [
                self._make_packet(microbatch_id, round_index, client, backend)
                for client, backend in active_pairs
            ]
            barrier_time = max(packet.arrival_s for packet in packets)
            target_dists, target_elapsed = self.target_backend.target_distributions(
                [packet.prefix_ids for packet in packets],
                [packet.draft_ids for packet in packets],
                self.sampling,
            )
            trace_devices = []
            for packet, dists, (client, _) in zip(packets, target_dists, active_pairs):
                verification = verify_draft_exact(
                    packet.draft_ids,
                    packet.draft_dists,
                    dists,
                    self.rng,
                )
                remaining = self.max_new_tokens - len(client.generated_ids)
                emitted = verification.emitted_ids[:remaining]
                client.generated_ids.extend(emitted)
                client.proposed_draft_tokens += verification.proposed_count
                client.accepted_draft_tokens += verification.accepted_count
                client.sync_rounds += 1
                profile = self.profiles[client.device_id]
                downlink_bytes = estimate_downlink_payload_bytes(emitted)
                downlink = sample_network_delay(
                    downlink_bytes, profile, "downlink", self.rng
                )
                client.available_at_s = barrier_time + target_elapsed + downlink.delay_s
                if emitted and client.first_token_latency_s is None:
                    client.first_token_latency_s = client.available_at_s - start_time_s
                client.latency_s = client.available_at_s - start_time_s
                eos = getattr(self.target_backend, "eos_token_id", None)
                if eos is not None and eos in emitted:
                    client.done = True
                if len(client.generated_ids) >= self.max_new_tokens:
                    client.done = True
                trace_devices.append(
                    {
                        "device_id": packet.device_id,
                        "request_id": packet.request_id,
                        "draft_model": packet.draft_model,
                        "draft_start_s": packet.draft_start_s,
                        "draft_end_s": packet.draft_end_s,
                        "draft_time_s": packet.draft_elapsed_s,
                        "uplink_s": packet.uplink_s,
                        "uplink_effective_mbps": packet.uplink_effective_mbps,
                        "uplink_effective_rtt_ms": packet.uplink_effective_rtt_ms,
                        "uplink_jitter_s": packet.uplink_jitter_s,
                        "uplink_congested": packet.uplink_congested,
                        "arrival_s": packet.arrival_s,
                        "barrier_wait_s": barrier_time - packet.arrival_s,
                        "downlink_s": downlink.delay_s,
                        "downlink_effective_mbps": downlink.effective_mbps,
                        "downlink_effective_rtt_ms": downlink.effective_rtt_ms,
                        "downlink_jitter_s": downlink.jitter_s,
                        "downlink_congested": downlink.congested,
                        "uplink_payload_bytes": packet.uplink_payload_bytes,
                        "downlink_payload_bytes": downlink_bytes,
                        "accepted_count": verification.accepted_count,
                        "proposed_count": verification.proposed_count,
                        "emitted_count": len(emitted),
                    }
                )
            round_traces.append(
                {
                    "microbatch_id": microbatch_id,
                    "round_index": round_index,
                    "target_batch_size": len(packets),
                    "barrier_time_s": barrier_time,
                    "target_forward_s": target_elapsed,
                    "devices": trace_devices,
                }
            )
            round_index += 1

        records: list[dict] = []
        for client in clients:
            records.append(
                self._record_client(
                    client,
                    microbatch_id,
                    start_time_s,
                    {"execution_mode": "sync"},
                )
            )
        completion_time = max(client.available_at_s for client in clients)
        return records, round_traces, completion_time

    def run_dataset(
        self,
        microbatches: Sequence[Sequence[SpecBenchItem]],
        progress: ProgressLike | None = None,
    ) -> tuple[list[dict], list[dict], dict]:
        all_records: list[dict] = []
        all_traces: list[dict] = []
        now = 0.0
        for microbatch_id, items in enumerate(microbatches):
            records, traces, now = self.run_microbatch(items, microbatch_id, now)
            all_records.extend(records)
            all_traces.extend(traces)
            if progress is not None:
                progress.update(1)
                progress.set_postfix(
                    {
                        "requests": len(all_records),
                        "rounds": len(all_traces),
                        "virtual_s": f"{now:.2f}",
                    },
                    refresh=False,
                )
        summary = summarize(all_records, all_traces, now)
        summary["mode"] = "sync"
        summary["profiles"] = {
            device_id: asdict(profile) for device_id, profile in self.profiles.items()
        }
        return all_records, all_traces, summary


class HeteroAsyncPipelineRunner(HeteroSyncRunner):
    def __init__(
        self,
        draft_backends: Sequence[ModelBackend],
        target_backend: ModelBackend,
        profiles: dict[str, DeviceProfile],
        sampling: SamplingConfig,
        gamma: int,
        max_new_tokens: int,
        seed: int = 0,
        run_target_baseline: bool = True,
        pipeline_count: int = 3,
    ) -> None:
        super().__init__(
            draft_backends=draft_backends,
            target_backend=target_backend,
            profiles=profiles,
            sampling=sampling,
            gamma=gamma,
            max_new_tokens=max_new_tokens,
            seed=seed,
            run_target_baseline=run_target_baseline,
        )
        if pipeline_count <= 0:
            raise ValueError("pipeline_count must be > 0")
        self.pipeline_count = pipeline_count

    def _build_async_client(
        self,
        item: SpecBenchItem,
        device_index: int,
        request_start_s: float,
    ) -> ClientState:
        backend = self.draft_backends[device_index]
        return ClientState(
            device_id=f"device-{device_index}",
            draft_model=backend.model_name,
            prompt_id=item.request_id,
            category=item.category,
            prompt=item.prompt,
            prefix_ids=backend.encode_prompt(item.prompt),
            available_at_s=request_start_s,
        )

    @staticmethod
    def _select_pipeline(
        pipelines: Sequence[VerificationPipeline],
        arrival_s: float,
    ) -> VerificationPipeline:
        return min(
            pipelines,
            key=lambda pipeline: (
                max(pipeline.available_at_s, arrival_s),
                pipeline.available_at_s,
                pipeline.pipeline_id,
            ),
        )

    def run_dataset(
        self,
        microbatches: Sequence[Sequence[SpecBenchItem]],
        progress: ProgressLike | None = None,
    ) -> tuple[list[dict], list[dict], dict]:
        device_queues: list[deque[tuple[int, SpecBenchItem]]] = [
            deque() for _ in self.draft_backends
        ]
        for microbatch_id, items in enumerate(microbatches):
            if len(items) > len(self.draft_backends):
                raise ValueError("microbatch has more requests than draft backends")
            for device_index, item in enumerate(items):
                device_queues[device_index].append((microbatch_id, item))

        pipelines = [
            VerificationPipeline(pipeline_id=index)
            for index in range(self.pipeline_count)
        ]
        events: list[tuple[float, int, int, DraftPacket]] = []
        next_event_id = count()
        active_clients: list[ClientState | None] = [None] * len(self.draft_backends)
        active_starts = [0.0] * len(self.draft_backends)
        active_microbatch_ids = [-1] * len(self.draft_backends)
        pipeline_history: list[list[int]] = [[] for _ in self.draft_backends]
        records: list[dict] = []
        traces: list[dict] = []

        def enqueue_packet(device_index: int) -> None:
            client = active_clients[device_index]
            if client is None or client.done:
                return
            if len(client.generated_ids) >= self.max_new_tokens:
                return
            packet = self._make_packet(
                active_microbatch_ids[device_index],
                client.sync_rounds,
                client,
                self.draft_backends[device_index],
            )
            heapq.heappush(
                events,
                (packet.arrival_s, next(next_event_id), device_index, packet),
            )

        def start_next_request(device_index: int, start_time_s: float) -> None:
            if not device_queues[device_index]:
                active_clients[device_index] = None
                return
            microbatch_id, item = device_queues[device_index].popleft()
            active_clients[device_index] = self._build_async_client(
                item, device_index, start_time_s
            )
            active_starts[device_index] = start_time_s
            active_microbatch_ids[device_index] = microbatch_id
            pipeline_history[device_index] = []
            enqueue_packet(device_index)

        for device_index in range(len(self.draft_backends)):
            start_next_request(device_index, 0.0)

        while events:
            _, packet_sequence_id, device_index, packet = heapq.heappop(events)
            client = active_clients[device_index]
            if client is None:
                continue

            pipeline = self._select_pipeline(pipelines, packet.arrival_s)
            pipeline_start_s = max(packet.arrival_s, pipeline.available_at_s)
            pipeline_queue_wait_s = pipeline_start_s - packet.arrival_s
            target_dists, target_elapsed = self.target_backend.target_distributions(
                [packet.prefix_ids],
                [packet.draft_ids],
                self.sampling,
            )
            pipeline_finish_s = pipeline_start_s + target_elapsed
            pipeline.available_at_s = pipeline_finish_s
            pipeline.busy_s += target_elapsed
            pipeline.verification_count += 1

            verification = verify_draft_exact(
                packet.draft_ids,
                packet.draft_dists,
                target_dists[0],
                self.rng,
            )
            remaining = self.max_new_tokens - len(client.generated_ids)
            emitted = verification.emitted_ids[:remaining]
            client.generated_ids.extend(emitted)
            client.proposed_draft_tokens += verification.proposed_count
            client.accepted_draft_tokens += verification.accepted_count
            client.sync_rounds += 1
            pipeline_history[device_index].append(pipeline.pipeline_id)

            profile = self.profiles[client.device_id]
            downlink_bytes = estimate_downlink_payload_bytes(emitted)
            downlink = sample_network_delay(
                downlink_bytes, profile, "downlink", self.rng
            )
            client.available_at_s = pipeline_finish_s + downlink.delay_s
            request_start_s = active_starts[device_index]
            if emitted and client.first_token_latency_s is None:
                client.first_token_latency_s = client.available_at_s - request_start_s
            client.latency_s = client.available_at_s - request_start_s
            eos = getattr(self.target_backend, "eos_token_id", None)
            if eos is not None and eos in emitted:
                client.done = True
            if len(client.generated_ids) >= self.max_new_tokens:
                client.done = True

            traces.append(
                {
                    "mode": "async",
                    "event_index": len(traces),
                    "packet_sequence_id": packet_sequence_id,
                    "microbatch_id": packet.microbatch_id,
                    "round_index": packet.round_index,
                    "target_batch_size": 1,
                    "pipeline_id": pipeline.pipeline_id,
                    "pipeline_start_s": pipeline_start_s,
                    "pipeline_finish_s": pipeline_finish_s,
                    "pipeline_queue_wait_s": pipeline_queue_wait_s,
                    "target_forward_s": target_elapsed,
                    "devices": [
                        {
                            "device_id": packet.device_id,
                            "request_id": packet.request_id,
                            "draft_model": packet.draft_model,
                            "draft_start_s": packet.draft_start_s,
                            "draft_end_s": packet.draft_end_s,
                            "draft_time_s": packet.draft_elapsed_s,
                            "uplink_s": packet.uplink_s,
                            "uplink_effective_mbps": packet.uplink_effective_mbps,
                            "uplink_effective_rtt_ms": packet.uplink_effective_rtt_ms,
                            "uplink_jitter_s": packet.uplink_jitter_s,
                            "uplink_congested": packet.uplink_congested,
                            "arrival_s": packet.arrival_s,
                            "barrier_wait_s": 0.0,
                            "pipeline_id": pipeline.pipeline_id,
                            "pipeline_start_s": pipeline_start_s,
                            "pipeline_finish_s": pipeline_finish_s,
                            "pipeline_queue_wait_s": pipeline_queue_wait_s,
                            "downlink_s": downlink.delay_s,
                            "downlink_effective_mbps": downlink.effective_mbps,
                            "downlink_effective_rtt_ms": downlink.effective_rtt_ms,
                            "downlink_jitter_s": downlink.jitter_s,
                            "downlink_congested": downlink.congested,
                            "uplink_payload_bytes": packet.uplink_payload_bytes,
                            "downlink_payload_bytes": downlink_bytes,
                            "accepted_count": verification.accepted_count,
                            "proposed_count": verification.proposed_count,
                            "emitted_count": len(emitted),
                        }
                    ],
                }
            )

            if client.done:
                history = pipeline_history[device_index]
                pipeline_switches = sum(
                    1
                    for previous, current in zip(history, history[1:])
                    if previous != current
                )
                records.append(
                    self._record_client(
                        client,
                        active_microbatch_ids[device_index],
                        request_start_s,
                        {
                            "execution_mode": "async",
                            "async_rounds": client.sync_rounds,
                            "pipeline_ids": history.copy(),
                            "pipeline_switches": pipeline_switches,
                        },
                    )
                )
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix(
                        {
                            "events": len(traces),
                            "virtual_s": f"{client.available_at_s:.2f}",
                            "pipelines": self.pipeline_count,
                        },
                        refresh=False,
                    )
                active_clients[device_index] = None
                start_next_request(device_index, client.available_at_s)
            else:
                enqueue_packet(device_index)

        total_time_s = (
            max(float(record["completion_time_s"]) for record in records)
            if records
            else 0.0
        )
        summary = summarize_async(records, traces, total_time_s, pipelines)
        summary["profiles"] = {
            device_id: asdict(profile) for device_id, profile in self.profiles.items()
        }
        return records, traces, summary


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
    for record in records:
        device_latency.setdefault(record["device_id"], []).append(record["latency_s"])
    for trace in traces:
        devices = trace["devices"]
        if not devices:
            continue
        for device in devices:
            waits.append(device["barrier_wait_s"])
            uplink_mbps.append(device.get("uplink_effective_mbps", 0.0))
            downlink_mbps.append(device.get("downlink_effective_mbps", 0.0))
            uplink_rtt_ms.append(device.get("uplink_effective_rtt_ms", 0.0))
            downlink_rtt_ms.append(device.get("downlink_effective_rtt_ms", 0.0))
            congestion_events += int(bool(device.get("uplink_congested", False)))
            congestion_events += int(bool(device.get("downlink_congested", False)))
            network_samples += 2
        slowest = min(devices, key=lambda item: item["barrier_wait_s"])
        slowest_counts[slowest["device_id"]] = slowest_counts.get(slowest["device_id"], 0) + 1
    baseline_latencies = [
        record["target_only_latency_s"]
        for record in records
        if record["target_only_latency_s"] is not None
    ]
    task_metrics = summarize_by_task(records)
    return {
        "request_count": len(records),
        "round_count": len(traces),
        "total_generated_tokens": total_tokens,
        "total_virtual_time_s": total_time_s,
        "throughput_tokens_per_s": total_tokens / total_time_s
        if total_time_s > 0
        else 0.0,
        "mean_acceptance_rate": sum(record["acceptance_rate"] for record in records)
        / len(records)
        if records
        else 0.0,
        "mean_barrier_wait_s": sum(waits) / len(waits) if waits else 0.0,
        "mean_uplink_effective_mbps": sum(uplink_mbps) / len(uplink_mbps)
        if uplink_mbps
        else 0.0,
        "mean_downlink_effective_mbps": sum(downlink_mbps) / len(downlink_mbps)
        if downlink_mbps
        else 0.0,
        "mean_uplink_effective_rtt_ms": sum(uplink_rtt_ms) / len(uplink_rtt_ms)
        if uplink_rtt_ms
        else 0.0,
        "mean_downlink_effective_rtt_ms": sum(downlink_rtt_ms) / len(downlink_rtt_ms)
        if downlink_rtt_ms
        else 0.0,
        "network_congestion_events": congestion_events,
        "network_congestion_fraction": congestion_events / network_samples
        if network_samples
        else 0.0,
        "barrier_wait_fraction": sum(waits)
        / (
            sum(
                device["draft_time_s"] + device["uplink_s"]
                for trace in traces
                for device in trace["devices"]
            )
            + sum(waits)
        )
        if waits
        else 0.0,
        "mean_target_only_latency_s": sum(baseline_latencies) / len(baseline_latencies)
        if baseline_latencies
        else None,
        "mean_speedup_vs_target_only": sum(
            record["speedup_vs_target_only"]
            for record in records
            if record["speedup_vs_target_only"] is not None
        )
        / len(baseline_latencies)
        if baseline_latencies
        else None,
        "mean_latency_by_device": {
            device_id: sum(values) / len(values)
            for device_id, values in device_latency.items()
        },
        "slowest_device_rounds": slowest_counts,
        "task_metrics": task_metrics,
    }


def summarize_async(
    records: list[dict],
    traces: list[dict],
    total_time_s: float,
    pipelines: Sequence[VerificationPipeline],
) -> dict:
    summary = summarize(records, traces, total_time_s)
    queue_waits = [
        float(trace.get("pipeline_queue_wait_s", 0.0))
        for trace in traces
    ]
    target_forwards = [
        float(trace.get("target_forward_s", 0.0))
        for trace in traces
    ]
    total_queue_wait = sum(queue_waits)
    total_target_forward = sum(target_forwards)
    summary.update(
        {
            "mode": "async",
            "pipeline_count": len(pipelines),
            "verification_event_count": len(traces),
            "pipeline_verification_count": sum(
                pipeline.verification_count for pipeline in pipelines
            ),
            "mean_pipeline_queue_wait_s": total_queue_wait / len(queue_waits)
            if queue_waits
            else 0.0,
            "pipeline_queue_wait_fraction": total_queue_wait
            / (total_queue_wait + total_target_forward)
            if (total_queue_wait + total_target_forward) > 0
            else 0.0,
            "pipeline_busy_s": {
                str(pipeline.pipeline_id): pipeline.busy_s for pipeline in pipelines
            },
            "pipeline_verifications": {
                str(pipeline.pipeline_id): pipeline.verification_count
                for pipeline in pipelines
            },
            "pipeline_utilization": {
                str(pipeline.pipeline_id): pipeline.busy_s / total_time_s
                if total_time_s > 0
                else 0.0
                for pipeline in pipelines
            },
            "mean_pipeline_utilization": (
                sum(pipeline.busy_s for pipeline in pipelines)
                / (len(pipelines) * total_time_s)
                if pipelines and total_time_s > 0
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
        effective_received_tokens = sum(
            record.get("effective_received_token_count", record["generated_token_count"])
            for record in task_records
        )
        microbatch_durations: dict[int, float | tuple[float, float]] = {}
        for record in task_records:
            microbatch_id = int(record["microbatch_id"])
            if "start_time_s" in record and "completion_time_s" in record:
                current_start, current_end = microbatch_durations.get(
                    microbatch_id,
                    (float("inf"), 0.0),
                )
                microbatch_durations[microbatch_id] = (
                    min(current_start, float(record["start_time_s"])),
                    max(current_end, float(record["completion_time_s"])),
                )
            else:
                current_duration = microbatch_durations.get(microbatch_id, 0.0)
                microbatch_durations[microbatch_id] = max(
                    current_duration,
                    float(record["latency_s"]),
                )
        effective_duration = 0.0
        for value in microbatch_durations.values():
            if isinstance(value, tuple):
                effective_duration += max(0.0, value[1] - value[0])
            else:
                effective_duration += float(value)
        first_latencies = [
            float(record["first_token_latency_s"])
            for record in task_records
            if record.get("first_token_latency_s") is not None
        ]
        metrics[task] = {
            "request_count": len(task_records),
            "microbatch_count": len(microbatch_durations),
            "generated_token_count": effective_received_tokens,
            "effective_received_token_count": effective_received_tokens,
            "effective_duration_s": effective_duration,
            "effective_throughput_tokens_per_s": effective_received_tokens / effective_duration
            if effective_duration > 0
            else 0.0,
            "effective_received_throughput_tokens_per_s": effective_received_tokens / effective_duration
            if effective_duration > 0
            else 0.0,
            "e2e_first_token_latency_s": sum(first_latencies) / len(first_latencies)
            if first_latencies
            else None,
            "e2e_mean_latency_s": sum(record["latency_s"] for record in task_records)
            / len(task_records)
            if task_records
            else 0.0,
        }
    return metrics
