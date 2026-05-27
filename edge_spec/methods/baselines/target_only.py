from __future__ import annotations

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

    def run_dataset(
        self,
        microbatches: Sequence[Sequence[SpecBenchItem]],
        progress: ProgressLike | None = None,
    ) -> ExperimentResult:
        records: list[dict] = []
        now = 0.0
        for microbatch_id, items in enumerate(microbatches):
            clients = [
                ClientState(
                    device_id=f"device-{index}",
                    draft_model="none",
                    prompt_id=item.request_id,
                    category=item.category,
                    prompt=item.prompt,
                    prefix_ids=self.target_backend.encode_prompt(item.prompt),
                    available_at_s=now,
                )
                for index, item in enumerate(items)
            ]
            batch_completion = now
            for client in clients:
                profile = self.profiles[client.device_id]
                uplink_payload_bytes = estimate_prompt_payload_bytes(client.prefix_ids)
                uplink = self.network_delay_sample(
                    uplink_payload_bytes,
                    profile,
                    "uplink",
                    now,
                )
                ids, model_latency = self.target_backend.generate_target_only(
                    client.prefix_ids,
                    self.config.max_new_tokens,
                    self.config.sampling,
                    self.rng,
                )
                downlink_payload_bytes = estimate_downlink_payload_bytes(ids)
                downlink = self.network_delay_sample(
                    downlink_payload_bytes,
                    profile,
                    "downlink",
                    now + uplink.delay_s + model_latency,
                )
                latency = uplink.delay_s + model_latency + downlink.delay_s
                client.generated_ids = ids
                client.done = True
                client.sync_rounds = 1
                client.latency_s = latency
                client.first_token_latency_s = latency if ids else None
                client.available_at_s = now + latency
                batch_completion = max(batch_completion, client.available_at_s)
                target_only = {
                    "target_only_latency_s": latency,
                    "target_only_model_latency_s": model_latency,
                    "target_only_uplink_s": uplink.delay_s,
                    "target_only_downlink_s": downlink.delay_s,
                    "target_only_uplink_payload_bytes": uplink_payload_bytes,
                    "target_only_downlink_payload_bytes": downlink_payload_bytes,
                    "target_only_uplink_effective_mbps": uplink.effective_mbps,
                    "target_only_downlink_effective_mbps": downlink.effective_mbps,
                    "target_only_uplink_effective_rtt_ms": uplink.effective_rtt_ms,
                    "target_only_downlink_effective_rtt_ms": downlink.effective_rtt_ms,
                    "target_only_uplink_jitter_s": uplink.jitter_s,
                    "target_only_downlink_jitter_s": downlink.jitter_s,
                    "target_only_uplink_congested": uplink.congested,
                    "target_only_downlink_congested": downlink.congested,
                    "target_only_text": self.target_backend.decode(ids),
                    "speedup_vs_target_only": 1.0,
                }
                records.append(
                    request_record(
                        client,
                        target_model=self.target_backend.model_name,
                        generated_text=self.target_backend.decode(ids),
                        microbatch_id=microbatch_id,
                        request_start_s=now,
                        method=self.method_name,
                        target_only=target_only,
                    )
                )
            now = batch_completion
            if progress is not None:
                progress.update(1)
                progress.set_postfix(
                    {"requests": len(records), "virtual_s": f"{now:.2f}"},
                    refresh=False,
                )
        summary = summarize(records, [], now)
        summary["method"] = self.method_name
        summary["profiles"] = self.profiles_summary()
        return ExperimentResult(records, [], summary)

