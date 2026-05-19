from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .types import DeviceProfile, SparseProb


@dataclass(frozen=True)
class NetworkDelaySample:
    delay_s: float
    effective_mbps: float
    effective_rtt_ms: float
    serialization_s: float
    jitter_s: float
    congested: bool


def sample_network_delay(
    payload_bytes: int,
    profile: DeviceProfile,
    direction: str,
    rng: random.Random | None = None,
) -> NetworkDelaySample:
    profile.validate()
    if payload_bytes < 0:
        raise ValueError("payload_bytes must be >= 0")
    if direction not in {"uplink", "downlink"}:
        raise ValueError("direction must be uplink or downlink")
    local_rng = rng or random.Random()
    base_mbps = profile.uplink_mbps if direction == "uplink" else profile.downlink_mbps
    effective_mbps = base_mbps
    if profile.bandwidth_jitter_ratio > 0:
        low = 1.0 - profile.bandwidth_jitter_ratio
        high = 1.0 + profile.bandwidth_jitter_ratio
        effective_mbps *= max(0.01, local_rng.uniform(low, high))

    effective_rtt_ms = profile.rtt_ms
    if profile.rtt_jitter_ms > 0:
        effective_rtt_ms += local_rng.uniform(
            -profile.rtt_jitter_ms, profile.rtt_jitter_ms
        )
        effective_rtt_ms = max(0.0, effective_rtt_ms)

    congested = False
    if (
        profile.congestion_probability > 0
        and local_rng.random() < profile.congestion_probability
    ):
        congested = True
        effective_mbps = max(0.01, effective_mbps / profile.congestion_slowdown)
        effective_rtt_ms *= profile.congestion_slowdown

    serialization = payload_bytes * 8.0 / (effective_mbps * 1_000_000.0)
    one_way_rtt = effective_rtt_ms / 2000.0
    jitter = 0.0
    if profile.jitter_ms > 0:
        jitter = local_rng.uniform(-profile.jitter_ms, profile.jitter_ms) / 1000.0
    delay = max(0.0, one_way_rtt + serialization + jitter)
    return NetworkDelaySample(
        delay_s=delay,
        effective_mbps=effective_mbps,
        effective_rtt_ms=effective_rtt_ms,
        serialization_s=serialization,
        jitter_s=jitter,
        congested=congested,
    )


def network_delay_s(
    payload_bytes: int,
    profile: DeviceProfile,
    direction: str,
    rng: random.Random | None = None,
) -> float:
    return sample_network_delay(payload_bytes, profile, direction, rng).delay_s


def barrier_waits(arrival_times: Mapping[str, float]) -> dict[str, float]:
    if not arrival_times:
        return {}
    barrier = max(arrival_times.values())
    return {device_id: barrier - arrival for device_id, arrival in arrival_times.items()}


def estimate_uplink_payload_bytes(
    prefix_ids: list[int],
    draft_ids: list[int],
    draft_dists: list[SparseProb],
) -> int:
    prefix_bytes = len(prefix_ids) * 4
    draft_bytes = len(draft_ids) * 4
    dist_bytes = sum(dist.payload_bytes() for dist in draft_dists)
    framing_bytes = 128
    return prefix_bytes + draft_bytes + dist_bytes + framing_bytes


def estimate_prompt_payload_bytes(prefix_ids: list[int]) -> int:
    return len(prefix_ids) * 4 + 128


def estimate_downlink_payload_bytes(emitted_ids: list[int]) -> int:
    return len(emitted_ids) * 4 + 64


def load_device_profiles(path: str | Path) -> dict[str, DeviceProfile]:
    try:
        import yaml
    except ImportError:
        yaml = None

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        text = handle.read()
    data = yaml.safe_load(text) if yaml is not None else _load_simple_profile_yaml(text)
    raw_devices = data.get("devices", data)
    profiles: dict[str, DeviceProfile] = {}
    for device_id, values in raw_devices.items():
        profile = DeviceProfile(
            device_id=device_id,
            uplink_mbps=float(values["uplink_mbps"]),
            downlink_mbps=float(values["downlink_mbps"]),
            rtt_ms=float(values["rtt_ms"]),
            jitter_ms=float(values.get("jitter_ms", 0.0)),
            bandwidth_jitter_ratio=float(values.get("bandwidth_jitter_ratio", 0.0)),
            rtt_jitter_ms=float(values.get("rtt_jitter_ms", 0.0)),
            congestion_probability=float(values.get("congestion_probability", 0.0)),
            congestion_slowdown=float(values.get("congestion_slowdown", 1.0)),
        )
        profile.validate()
        profiles[device_id] = profile
    return profiles


def _load_simple_profile_yaml(text: str) -> dict:
    """Parse the small configs/edge_hetero.yaml shape without PyYAML."""
    devices: dict[str, dict[str, float]] = {}
    current: str | None = None
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.strip().startswith("#"):
            continue
        stripped = raw_line.strip()
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if indent == 0 and stripped == "devices:":
            continue
        if indent == 2 and stripped.endswith(":"):
            current = stripped[:-1]
            devices[current] = {}
            continue
        if indent >= 4 and current and ":" in stripped:
            key, value = stripped.split(":", 1)
            devices[current][key.strip()] = float(value.strip())
    return {"devices": devices}
