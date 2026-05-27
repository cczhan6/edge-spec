from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Protocol, Sequence

from edge_spec.backends import ModelBackend
from edge_spec.protocol import DraftSegment
from edge_spec.simulation import (
    NetworkDelaySample,
    SeededNetworkTrace,
    estimate_uplink_payload_bytes,
)
from edge_spec.tracing import request_record
from edge_spec.types import ClientState, DeviceProfile, SamplingConfig, SpecBenchItem


class ProgressLike(Protocol):
    def update(self, n: int = 1) -> None: ...

    def set_postfix(self, ordered_dict=None, refresh: bool = True, **kwargs) -> None: ...


@dataclass(frozen=True)
class RunConfig:
    method: str
    sampling: SamplingConfig
    gamma: int
    max_new_tokens: int
    initial_lookahead: int = 4
    seed: int = 0
    network_seed: int | None = None
    network_trace_slot_s: float = 0.05
    lane_count: int = 3
    max_inflight_segments: int = 2
    lookahead_policy: str = "adaptive"
    scheduler: str = "prefix-aware"
    lane_batch_size: int = 2
    lane_batch_timeout_s: float = 0.001

    def validate(self) -> None:
        self.sampling.validate()
        if self.gamma <= 0:
            raise ValueError("gamma must be > 0")
        if self.initial_lookahead <= 0:
            raise ValueError("initial_lookahead must be > 0")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")
        if self.network_trace_slot_s <= 0:
            raise ValueError("network_trace_slot_s must be > 0")
        if self.lane_count <= 0:
            raise ValueError("lane_count must be > 0")
        if self.max_inflight_segments <= 0:
            raise ValueError("max_inflight_segments must be > 0")
        if self.lookahead_policy not in {"adaptive", "fixed"}:
            raise ValueError("lookahead_policy must be adaptive or fixed")
        if self.scheduler not in {"prefix-aware", "queue-only"}:
            raise ValueError("scheduler must be prefix-aware or queue-only")
        if self.lane_batch_size <= 0:
            raise ValueError("lane_batch_size must be > 0")
        if self.lane_batch_timeout_s < 0:
            raise ValueError("lane_batch_timeout_s must be >= 0")


@dataclass
class ExperimentResult:
    records: list[dict]
    traces: list[dict]
    summary: dict


class MethodRunner(Protocol):
    method_name: str

    def run_dataset(
        self,
        microbatches: Sequence[Sequence[SpecBenchItem]],
        progress: ProgressLike | None = None,
    ) -> ExperimentResult: ...


class BaseMethodRunner:
    method_name = "base"

    def __init__(
        self,
        draft_backends: Sequence[ModelBackend],
        target_backend: ModelBackend,
        profiles: dict[str, DeviceProfile],
        config: RunConfig,
    ) -> None:
        config.validate()
        if len(draft_backends) != 3:
            raise ValueError("this experiment expects exactly three draft backends")
        self.draft_backends = list(draft_backends)
        self.target_backend = target_backend
        self.profiles = profiles
        self.config = config
        self.rng = random.Random(config.seed)
        network_seed = config.seed if config.network_seed is None else config.network_seed
        self.network_trace = SeededNetworkTrace(
            seed=network_seed,
            time_slot_s=config.network_trace_slot_s,
        )

    def build_client(
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

    def network_delay_sample(
        self,
        payload_bytes: int,
        profile: DeviceProfile,
        direction: str,
        time_s: float,
    ) -> NetworkDelaySample:
        return self.network_trace.sample(payload_bytes, profile, direction, time_s)

    def make_segment(
        self,
        microbatch_id: int,
        round_index: int,
        segment_id: int,
        client: ClientState,
        backend: ModelBackend,
        *,
        prefix_ids: list[int],
        draft_len: int,
        prefix_version: int = 0,
        base_position: int = 0,
        prefix_hash: str = "",
    ) -> DraftSegment:
        draft_start_s = client.available_at_s
        draft = backend.draft(prefix_ids, draft_len, self.config.sampling, self.rng)
        draft_end_s = draft_start_s + draft.elapsed_s
        profile = self.profiles[client.device_id]
        payload_bytes = estimate_uplink_payload_bytes(
            prefix_ids, draft.draft_ids, draft.draft_dists
        )
        uplink = self.network_delay_sample(
            payload_bytes,
            profile,
            "uplink",
            draft_end_s,
        )
        arrival = draft_end_s + uplink.delay_s
        return DraftSegment(
            microbatch_id=microbatch_id,
            round_index=round_index,
            segment_id=segment_id,
            device_id=client.device_id,
            request_id=client.prompt_id,
            draft_model=client.draft_model,
            prefix_ids=list(prefix_ids),
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
            prefix_version=prefix_version,
            base_position=base_position,
            prefix_hash=prefix_hash,
            lookahead=draft_len,
        )

    def record_client(
        self,
        client: ClientState,
        microbatch_id: int,
        request_start_s: float,
        method: str,
        extra: dict | None = None,
    ) -> dict:
        return request_record(
            client,
            target_model=self.target_backend.model_name,
            generated_text=self.target_backend.decode(client.generated_ids),
            microbatch_id=microbatch_id,
            request_start_s=request_start_s,
            method=method,
            extra=extra,
        )

    def profiles_summary(self) -> dict[str, dict]:
        return {
            device_id: asdict(profile) for device_id, profile in self.profiles.items()
        }

    def run_config_summary(self) -> dict:
        return {
            "method": self.config.method,
            "gamma": self.config.gamma,
            "initial_lookahead": self.config.initial_lookahead,
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.sampling.temperature,
            "top_p": self.config.sampling.top_p,
            "top_k": self.config.sampling.top_k,
            "seed": self.config.seed,
            "network_seed": self.network_trace.seed,
            "network_trace_slot_s": self.network_trace.time_slot_s,
            "lane_count": self.config.lane_count,
            "max_inflight_segments": self.config.max_inflight_segments,
            "lookahead_policy": self.config.lookahead_policy,
            "scheduler": self.config.scheduler,
            "lane_batch_size": self.config.lane_batch_size,
            "lane_batch_timeout_s": self.config.lane_batch_timeout_s,
        }

