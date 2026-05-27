from __future__ import annotations

from typing import Sequence

from edge_spec.metrics import summarize
from edge_spec.protocol import DraftSegment
from edge_spec.sampling import verify_draft_exact
from edge_spec.simulation import estimate_downlink_payload_bytes
from edge_spec.tracing import segment_device_trace, verification_event_trace
from edge_spec.types import ClientState, SpecBenchItem

from ..base import BaseMethodRunner, ExperimentResult, ProgressLike


class SyncBatchRunner(BaseMethodRunner):
    method_name = "sync_batch"

    def run_microbatch(
        self,
        items: Sequence[SpecBenchItem],
        microbatch_id: int,
        start_time_s: float,
    ) -> tuple[list[dict], list[dict], float]:
        clients = [
            self.build_client(item, index, start_time_s)
            for index, item in enumerate(items)
        ]
        traces: list[dict] = []
        round_index = 0

        while any(
            (not client.done)
            and len(client.generated_ids) < self.config.max_new_tokens
            for client in clients
        ):
            active_pairs = [
                (index, client)
                for index, client in enumerate(clients)
                if (not client.done)
                and len(client.generated_ids) < self.config.max_new_tokens
            ]
            segments: list[DraftSegment] = []
            for index, client in active_pairs:
                remaining = self.config.max_new_tokens - len(client.generated_ids)
                draft_len = min(self.config.gamma, remaining)
                prefix = client.prefix_ids + client.generated_ids
                segments.append(
                    self.make_segment(
                        microbatch_id,
                        round_index,
                        round_index,
                        client,
                        self.draft_backends[index],
                        prefix_ids=prefix,
                        draft_len=draft_len,
                        base_position=len(client.generated_ids),
                    )
                )

            barrier_time = max(segment.arrival_s for segment in segments)
            target_dists, target_elapsed = self.target_backend.target_distributions(
                [segment.prefix_ids for segment in segments],
                [segment.draft_ids for segment in segments],
                self.config.sampling,
            )
            trace_devices: list[dict] = []
            for segment, dists, (_, client) in zip(
                segments, target_dists, active_pairs
            ):
                verification = verify_draft_exact(
                    segment.draft_ids,
                    segment.draft_dists,
                    dists,
                    self.rng,
                )
                remaining = self.config.max_new_tokens - len(client.generated_ids)
                emitted = verification.emitted_ids[:remaining]
                client.generated_ids.extend(emitted)
                client.proposed_draft_tokens += verification.proposed_count
                client.accepted_draft_tokens += verification.accepted_count
                client.sync_rounds += 1

                profile = self.profiles[client.device_id]
                downlink_bytes = estimate_downlink_payload_bytes(emitted)
                downlink = self.network_delay_sample(
                    downlink_bytes,
                    profile,
                    "downlink",
                    barrier_time + target_elapsed,
                )
                client.available_at_s = (
                    barrier_time + target_elapsed + downlink.delay_s
                )
                if emitted and client.first_token_latency_s is None:
                    client.first_token_latency_s = (
                        client.available_at_s - start_time_s
                    )
                client.latency_s = client.available_at_s - start_time_s
                eos = getattr(self.target_backend, "eos_token_id", None)
                if eos is not None and eos in emitted:
                    client.done = True
                if len(client.generated_ids) >= self.config.max_new_tokens:
                    client.done = True

                trace_devices.append(
                    segment_device_trace(
                        segment,
                        barrier_wait_s=barrier_time - segment.arrival_s,
                        accepted_count=verification.accepted_count,
                        proposed_count=verification.proposed_count,
                        emitted_count=len(emitted),
                        downlink_s=downlink.delay_s,
                        downlink_effective_mbps=downlink.effective_mbps,
                        downlink_effective_rtt_ms=downlink.effective_rtt_ms,
                        downlink_jitter_s=downlink.jitter_s,
                        downlink_congested=downlink.congested,
                        downlink_payload_bytes=downlink_bytes,
                    )
                )
            traces.append(
                verification_event_trace(
                    event_index=len(traces),
                    method=self.method_name,
                    microbatch_id=microbatch_id,
                    round_index=round_index,
                    target_batch_size=len(segments),
                    target_forward_s=target_elapsed,
                    devices=trace_devices,
                    extra={"barrier_time_s": barrier_time},
                )
            )
            round_index += 1

        records = [
            self.record_client(
                client,
                microbatch_id,
                start_time_s,
                self.method_name,
            )
            for client in clients
        ]
        completion_time = max(client.available_at_s for client in clients)
        return records, traces, completion_time

    def run_dataset(
        self,
        microbatches: Sequence[Sequence[SpecBenchItem]],
        progress: ProgressLike | None = None,
    ) -> ExperimentResult:
        records: list[dict] = []
        traces: list[dict] = []
        now = 0.0
        for microbatch_id, items in enumerate(microbatches):
            batch_records, batch_traces, now = self.run_microbatch(
                items,
                microbatch_id,
                now,
            )
            records.extend(batch_records)
            traces.extend(batch_traces)
            if progress is not None:
                progress.update(1)
                progress.set_postfix(
                    {
                        "requests": len(records),
                        "rounds": len(traces),
                        "virtual_s": f"{now:.2f}",
                    },
                    refresh=False,
                )
        summary = summarize(records, traces, now)
        summary["method"] = self.method_name
        summary["profiles"] = self.profiles_summary()
        return ExperimentResult(records, traces, summary)

