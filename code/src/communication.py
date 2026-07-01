from __future__ import annotations

import hashlib
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


def network_delay_ms(
    seed: int,
    device: Device,
    direction: str,
    key: Any,
    payload_bytes: int,
) -> float:
    if direction not in {"uplink", "downlink"}:
        raise ValueError(f"unsupported direction: {direction}")
    bandwidth_mbps = (
        device.uplink_mbps if direction == "uplink" else device.downlink_mbps
    )
    return dssd_transmission_delay_ms(
        payload_bytes,
        device.rtt_ms,
        bandwidth_mbps,
    ) + deterministic_jitter_ms(seed, device, direction, key)
