from __future__ import annotations

import hashlib
import math
from typing import Any

from src.entities import Device

NETWORK_BLOCK_DOMAIN = "network-block-v1"


def dssd_transmission_delay_ms(
    payload_bytes: int,
    rtt_ms: float,
    bandwidth_mbps: float,
) -> float:
    """DSSD one-way delay: RTT / 2 plus payload serialization time."""
    if payload_bytes < 0:
        raise ValueError("payload_bytes must be >= 0")
    if bandwidth_mbps <= 0:
        raise ValueError("bandwidth_mbps must be > 0")
    return rtt_ms / 2.0 + payload_bytes * 8.0 / (bandwidth_mbps * 1000.0)


def deterministic_jitter_ms(seed: int, device: Device, direction: str, key: Any) -> float:
    digest_key = f"{seed}:{device.device_id}:{direction}:{key}".encode()
    ratio = int.from_bytes(hashlib.sha256(digest_key).digest()[:8], "big") / 2**64
    return ratio * device.jitter_ms


def deterministic_network_blocked(
    seed: int,
    device: Device,
    direction: str,
    key: Any,
) -> bool:
    digest_key = (
        f"{NETWORK_BLOCK_DOMAIN}:{seed}:{device.device_id}:{direction}:{key}"
    ).encode()
    ratio = int.from_bytes(hashlib.sha256(digest_key).digest()[:8], "big") / 2**64
    return ratio < device.block_probability


def _validate_device_network(device: Device) -> None:
    probability = device.block_probability
    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(float(probability))
        or not 0.0 <= float(probability) <= 1.0
    ):
        raise ValueError("block_probability must be a finite number in [0, 1]")
    if not math.isfinite(device.rtt_ms) or device.rtt_ms < 0.0:
        raise ValueError("rtt_ms must be finite and non-negative")
    if not math.isfinite(device.jitter_ms) or device.jitter_ms < 0.0:
        raise ValueError("jitter_ms must be finite and non-negative")
    for name, value in (
        ("uplink_mbps", device.uplink_mbps),
        ("downlink_mbps", device.downlink_mbps),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")


def network_delay_ms(
    seed: int,
    device: Device,
    direction: str,
    key: Any,
    payload_bytes: int,
) -> float:
    if direction not in {"uplink", "downlink"}:
        raise ValueError(f"unsupported direction: {direction}")
    _validate_device_network(device)
    bandwidth_mbps = (
        device.uplink_mbps if direction == "uplink" else device.downlink_mbps
    )
    base_delay_ms = dssd_transmission_delay_ms(
        payload_bytes,
        device.rtt_ms,
        bandwidth_mbps,
    )
    if not deterministic_network_blocked(seed, device, direction, key):
        return base_delay_ms
    return base_delay_ms + deterministic_jitter_ms(seed, device, direction, key)
