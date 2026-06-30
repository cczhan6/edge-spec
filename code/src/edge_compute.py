from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any, Sequence

from src.entities import Device


@dataclass(frozen=True)
class EdgeComputeState:
    completed_requests: int
    epoch: int
    draft_token_rate_tok_s: float


@dataclass(frozen=True)
class EdgeComputeSnapshot:
    device_id: int
    device_type: str
    epoch: int
    draft_token_rate_tok_s: float
    draft_startup_ms: float


@dataclass(frozen=True)
class EdgeComputeTransition:
    device_id: int
    device_type: str
    completed_requests: int
    old_epoch: int
    new_epoch: int
    old_rate: float
    new_rate: float


def deterministic_draft_rate(
    seed: int,
    device_id: int,
    device_type: str,
    epoch: int,
    bounds: tuple[float, float],
    *,
    attempt: int = 0,
) -> float:
    suffix = f":{attempt}" if attempt else ""
    key = (
        f"edge-compute-v1:{seed}:{device_id}:{device_type}:{epoch}{suffix}"
    ).encode()
    ratio = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
    lower, upper = bounds
    return lower + ratio * (upper - lower)


class EdgeComputeModel:
    def __init__(
        self,
        config: dict[str, Any],
        devices: Sequence[Device],
        pool_name: str,
    ) -> None:
        dynamic = config.get("dynamic_edge_compute", {})
        self.enabled = bool(dynamic.get("enabled", False))
        self._seed = int(config["simulation"]["seed"])
        self._devices = {device.device_id: device for device in devices}
        templates = config["device_pools"][pool_name]["templates"]
        self._bounds = {
            device_type: tuple(
                float(value)
                for value in template["dynamic_draft_token_rate_range_tok_s"]
            )
            for device_type, template in templates.items()
            if "dynamic_draft_token_rate_range_tok_s" in template
        }
        self._states: dict[int, EdgeComputeState] = {}
        used_initial_rates: dict[str, set[float]] = {}
        for device in sorted(devices, key=lambda item: item.device_id):
            rate = device.draft_token_rate_tok_s
            if self.enabled:
                attempt = 0
                used = used_initial_rates.setdefault(device.device_type, set())
                while True:
                    rate = deterministic_draft_rate(
                        self._seed,
                        device.device_id,
                        device.device_type,
                        0,
                        self._bounds[device.device_type],
                        attempt=attempt,
                    )
                    if rate not in used:
                        used.add(rate)
                        break
                    attempt += 1
            self._states[device.device_id] = EdgeComputeState(0, 0, rate)

    def state(self, device_id: int) -> EdgeComputeState:
        return self._states[device_id]

    def current_rate(self, device_id: int) -> float:
        return self._states[device_id].draft_token_rate_tok_s

    def snapshot(self, device_id: int) -> EdgeComputeSnapshot:
        device = self._devices[device_id]
        state = self._states[device_id]
        return EdgeComputeSnapshot(
            device_id=device_id,
            device_type=device.device_type,
            epoch=state.epoch,
            draft_token_rate_tok_s=state.draft_token_rate_tok_s,
            draft_startup_ms=device.draft_startup_ms,
        )

    def latency_ms(self, snapshot: EdgeComputeSnapshot, work_units: int) -> float:
        if work_units < 0:
            raise ValueError("work_units must be >= 0")
        return snapshot.draft_startup_ms + (
            1000.0 * work_units / snapshot.draft_token_rate_tok_s
        )

    def current_latency_ms(self, device_id: int, work_units: int) -> float:
        return self.latency_ms(self.snapshot(device_id), work_units)

    def record_request_completion(
        self,
        device_id: int,
    ) -> EdgeComputeTransition | None:
        if not self.enabled:
            return None
        device = self._devices[device_id]
        old = self._states[device_id]
        completed = old.completed_requests + 1
        if completed % 5:
            self._states[device_id] = replace(old, completed_requests=completed)
            return None
        epoch = old.epoch + 1
        rate = deterministic_draft_rate(
            self._seed,
            device_id,
            device.device_type,
            epoch,
            self._bounds[device.device_type],
        )
        self._states[device_id] = EdgeComputeState(completed, epoch, rate)
        return EdgeComputeTransition(
            device_id=device_id,
            device_type=device.device_type,
            completed_requests=completed,
            old_epoch=old.epoch,
            new_epoch=epoch,
            old_rate=old.draft_token_rate_tok_s,
            new_rate=rate,
        )
