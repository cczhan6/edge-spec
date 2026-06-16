from __future__ import annotations

import unittest

from src.communication import dssd_transmission_delay_ms, network_delay_ms
from src.entities import Device


class CommunicationTest(unittest.TestCase):
    def test_dssd_formula_adds_half_rtt_and_serialization(self) -> None:
        self.assertEqual(dssd_transmission_delay_ms(1000, 20.0, 8.0), 11.0)

    def test_network_extension_uses_directional_bandwidth_and_jitter(self) -> None:
        device = Device(0, "small_device", "small", 0.5, 500.0, 1.0, 8.0, 16.0, 20.0, 0.0)
        self.assertEqual(network_delay_ms(1, device, "uplink", "x", 1000), 11.0)
        self.assertEqual(network_delay_ms(1, device, "downlink", "x", 1000), 10.5)


if __name__ == "__main__":
    unittest.main()
