from __future__ import annotations

import heapq
import itertools
from typing import Sequence

from edge_spec.metrics import summarize
from edge_spec.simulation import (
    estimate_downlink_payload_bytes,
    estimate_prompt_payload_bytes,
)
from edge_spec.tracing import request_record
from edge_spec.types import ClientState, SpecBenchItem

from ..base import BaseMethodRunner, ExperimentResult, ProgressLike


class TargetOnlyRunner(BaseMethodRunner):
    method_name = "target_only"

    def run_config_summary(self) -> dict:
        summary = super().run_config_summary()
        summary["target_only_server_mode"] = "shared_serial"
        return summary

    def run_dataset(
        self,
        microbatches: Sequence[Sequence[SpecBenchItem]],
        progress: ProgressLike | None = None,
    ) -> ExperimentResult:
        records: list[dict] = []
        traces: list[dict] = []
        device_queues: list[list[tuple[int, SpecBenchItem]]] = [
            [] for _ in self.draft_backends
        ]
        for microbatch_id, items in enumerate(microbatches):
            if len(items) > len(self.draft_backends):
                raise ValueError("microbatch has more requests than devices")
            for device_index, item in enumerate(items):
                device_queues[device_index].append((microbatch_id, item))

        next_event_id = itertools.count()
        arrivals: list[tuple[float, int, ClientState, int, float, dict]] = []
        target_available_at_s = 0.0
        total_time_s = 0.0

        def start_next_request(device_index: int, start_time_s: float) -> None:
            if not device_queues[device_index]:
                return
            microbatch_id, item = device_queues[device_index].pop(0)
            client = ClientState(
                device_id=f"device-{device_index}",
                draft_model="none",
                prompt_id=item.request_id,
                category=item.category,
                prompt=item.prompt,
                prefix_ids=self.target_backend.encode_prompt(item.prompt),
                available_at_s=start_time_s,
            )
            profile = self.profiles[client.device_id]
            uplink_payload_bytes = estimate_prompt_payload_bytes(client.prefix_ids)
            uplink = self.network_delay_sample(
                uplink_payload_bytes,
                profile,
                "uplink",
                start_time_s,
            )
            arrival_s = start_time_s + uplink.delay_s
            target_only = {
                "target_only_uplink_s": uplink.delay_s,
                "target_only_uplink_payload_bytes": uplink_payload_bytes,
                "target_only_uplink_effective_mbps": uplink.effective_mbps,
                "target_only_uplink_effective_rtt_ms": uplink.effective_rtt_ms,
                "target_only_uplink_jitter_s": uplink.jitter_s,
                "target_only_uplink_congested": uplink.congested,
                "target_only_request_start_s": start_time_s,
                "target_only_arrival_s": arrival_s,
                "target_only_server_mode": "shared_serial",
            }
            heapq.heappush(
                arrivals,
                (
                    arrival_s,
                    next(next_event_id),
                    client,
                    microbatch_id,
                    start_time_s,
                    target_only,
                ),
            )

        for device_index in range(len(device_queues)):
            start_next_request(device_index, 0.0)

        while arrivals:
            (
                arrival_s,
                _,
                client,
                microbatch_id,
                request_start_s,
                target_only,
            ) = heapq.heappop(arrivals)
            target_start_s = max(arrival_s, target_available_at_s)
            queue_wait_s = target_start_s - arrival_s
            ids, model_latency = self.target_backend.generate_target_only(
                client.prefix_ids,
                self.config.max_new_tokens,
                self.config.sampling,
                self.rng,
            )
            target_finish_s = target_start_s + model_latency
            target_available_at_s = target_finish_s
            downlink_payload_bytes = estimate_downlink_payload_bytes(ids)
            downlink = self.network_delay_sample(
                downlink_payload_bytes,
                self.profiles[client.device_id],
                "downlink",
                target_finish_s,
            )
            completion_time_s = target_finish_s + downlink.delay_s
            latency = completion_time_s - request_start_s

            client.generated_ids = ids
            client.done = True
            client.sync_rounds = 1
            client.latency_s = latency
            client.first_token_latency_s = latency if ids else None
            client.available_at_s = completion_time_s
            total_time_s = max(total_time_s, completion_time_s)

            target_only.update(
                {
                    "target_only_latency_s": latency,
                    "target_only_model_latency_s": model_latency,
                    "target_only_queue_wait_s": queue_wait_s,
                    "target_only_model_start_s": target_start_s,
                    "target_only_model_finish_s": target_finish_s,
                    "target_only_downlink_s": downlink.delay_s,
                    "target_only_downlink_payload_bytes": downlink_payload_bytes,
                    "target_only_downlink_effective_mbps": downlink.effective_mbps,
                    "target_only_downlink_effective_rtt_ms": downlink.effective_rtt_ms,
                    "target_only_downlink_jitter_s": downlink.jitter_s,
                    "target_only_downlink_congested": downlink.congested,
                    "target_only_text": self.target_backend.decode(ids),
                    "speedup_vs_target_only": 1.0,
                }
            )
            records.append(
                request_record(
                    client,
                    target_model=self.target_backend.model_name,
                    generated_text=self.target_backend.decode(ids),
                    microbatch_id=microbatch_id,
                    request_start_s=request_start_s,
                    method=self.method_name,
                    target_only=target_only,
                    extra={
                        "target_only_server_mode": "shared_serial",
                        "target_only_arrival_s": arrival_s,
                        "target_only_queue_wait_s": queue_wait_s,
                        "target_only_model_start_s": target_start_s,
                        "target_only_model_finish_s": target_finish_s,
                    },
                )
            )
            traces.append(
                {
                    "method": self.method_name,
                    "event_index": len(traces),
                    "microbatch_id": microbatch_id,
                    "round_index": 0,
                    "target_batch_size": 1,
                    "target_forward_s": model_latency,
                    "target_queue_wait_s": queue_wait_s,
                    "target_start_s": target_start_s,
                    "target_finish_s": target_finish_s,
                    "devices": [
                        {
                            "device_id": client.device_id,
                            "request_id": client.prompt_id,
                            "uplink_s": target_only["target_only_uplink_s"],
                            "uplink_effective_mbps": target_only[
                                "target_only_uplink_effective_mbps"
                            ],
                            "uplink_effective_rtt_ms": target_only[
                                "target_only_uplink_effective_rtt_ms"
                            ],
                            "uplink_jitter_s": target_only[
                                "target_only_uplink_jitter_s"
                            ],
                            "uplink_congested": target_only[
                                "target_only_uplink_congested"
                            ],
                            "uplink_payload_bytes": target_only[
                                "target_only_uplink_payload_bytes"
                            ],
                            "arrival_s": arrival_s,
                            "barrier_wait_s": 0.0,
                            "target_queue_wait_s": queue_wait_s,
                            "downlink_s": downlink.delay_s,
                            "downlink_effective_mbps": downlink.effective_mbps,
                            "downlink_effective_rtt_ms": downlink.effective_rtt_ms,
                            "downlink_jitter_s": downlink.jitter_s,
                            "downlink_congested": downlink.congested,
                            "downlink_payload_bytes": downlink_payload_bytes,
                            "emitted_count": len(ids),
                            "status": "completed",
                        }
                    ],
                    "status": "completed",
                }
            )
            if progress is not None:
                progress.update(1)
                progress.set_postfix(
                    {"requests": len(records), "virtual_s": f"{total_time_s:.2f}"},
                    refresh=False,
                )
            device_index = int(client.device_id.rsplit("-", 1)[1])
            start_next_request(device_index, completion_time_s)

        summary = summarize(records, traces, total_time_s)
        summary["method"] = self.method_name
        summary["target_only_server_mode"] = "shared_serial"
        summary["target_only_total_queue_wait_s"] = sum(
            float(record.get("target_only_queue_wait_s", 0.0) or 0.0)
            for record in records
        )
        summary["target_only_mean_queue_wait_s"] = (
            summary["target_only_total_queue_wait_s"] / len(records)
            if records
            else 0.0
        )
        summary["profiles"] = self.profiles_summary()
        return ExperimentResult(records, traces, summary)
