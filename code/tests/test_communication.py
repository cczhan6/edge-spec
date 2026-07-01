from __future__ import annotations

import hashlib
import unittest

import src.communication as communication
from src.communication import (
    deterministic_jitter_ms,
    dssd_transmission_delay_ms,
    network_delay_ms,
)
from src.entities import Device


def make_device(
    *,
    device_id: int = 0,
    jitter_ms: float = 25.0,
    block_probability: float = 1.0,
) -> Device:
    return Device(
        device_id=device_id,
        device_type="small_device",
        drafter_profile="small",
        acceptance_prior=0.5,
        draft_token_rate_tok_s=500.0,
        draft_startup_ms=1.0,
        uplink_mbps=8.0,
        downlink_mbps=16.0,
        rtt_ms=20.0,
        jitter_ms=jitter_ms,
        block_probability=block_probability,
    )


def reference_block_ratio(
    seed: int,
    device_id: int,
    direction: str,
    key: object,
) -> float:
    material = (
        f"network-block-v1:{seed}:{device_id}:{direction}:{key}".encode()
    )
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") / 2**64


class CommunicationTest(unittest.TestCase):
    def test_dssd_formula_adds_half_rtt_and_serialization(self) -> None:
        self.assertEqual(dssd_transmission_delay_ms(1000, 20.0, 8.0), 11.0)

    def test_network_extension_uses_directional_bandwidth_and_jitter(self) -> None:
        device = Device(0, "small_device", "small", 0.5, 500.0, 1.0, 8.0, 16.0, 20.0, 0.0)
        self.assertEqual(network_delay_ms(1, device, "uplink", "x", 1000), 11.0)
        self.assertEqual(network_delay_ms(1, device, "downlink", "x", 1000), 10.5)

    def test_block_decision_uses_documented_independent_hash_domain(self) -> None:
        cases = (
            (make_device(device_id=0, block_probability=0.5), "uplink", "segment-1"),
            (make_device(device_id=1, block_probability=0.5), "uplink", "segment-1"),
            (make_device(device_id=0, block_probability=0.5), "downlink", "segment-1"),
            (make_device(device_id=0, block_probability=0.5), "uplink", "segment-2"),
        )

        actual = [
            communication.deterministic_network_blocked(7, device, direction, key)
            for device, direction, key in cases
        ]
        expected = [
            reference_block_ratio(7, device.device_id, direction, key)
            < device.block_probability
            for device, direction, key in cases
        ]

        self.assertEqual(actual, expected)
        self.assertEqual(actual, [True, False, True, False])

    def test_block_decision_is_reproducible_and_call_order_independent(self) -> None:
        device = make_device(block_probability=0.5)
        keys = ["segment-1", "segment-2", "segment-3"]

        forward = [
            communication.deterministic_network_blocked(7, device, "uplink", key)
            for key in keys
        ]
        reverse = {
            key: communication.deterministic_network_blocked(
                7, device, "uplink", key
            )
            for key in reversed(keys)
        }

        self.assertEqual(forward, [reverse[key] for key in keys])

    def test_block_decision_includes_seed(self) -> None:
        device = make_device(block_probability=0.5)

        self.assertTrue(
            communication.deterministic_network_blocked(
                7, device, "uplink", "segment-1"
            )
        )
        self.assertFalse(
            communication.deterministic_network_blocked(
                8, device, "uplink", "segment-1"
            )
        )

    def test_block_and_wait_materials_are_distinct(self) -> None:
        block_material = b"network-block-v1:7:0:uplink:segment-2"
        wait_material = b"7:0:uplink:segment-2"

        self.assertNotEqual(
            hashlib.sha256(block_material).digest()[:8],
            hashlib.sha256(wait_material).digest()[:8],
        )


if __name__ == "__main__":
    unittest.main()
