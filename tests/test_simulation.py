import random
import tempfile
import unittest
from pathlib import Path

from edge_spec.simulation import (
    barrier_waits,
    load_device_profiles,
    network_delay_s,
    sample_network_delay,
)
from edge_spec.types import DeviceProfile


class SimulationTests(unittest.TestCase):
    def test_network_delay_without_jitter(self):
        profile = DeviceProfile("device-0", 20, 50, 40, 0)
        delay = network_delay_s(1_000_000, profile, "uplink", random.Random(1))
        self.assertAlmostEqual(delay, 0.42)

    def test_barrier_waits(self):
        waits = barrier_waits({"a": 1.0, "b": 1.5, "c": 1.2})
        self.assertAlmostEqual(waits["a"], 0.5)
        self.assertAlmostEqual(waits["b"], 0.0)
        self.assertAlmostEqual(waits["c"], 0.3)

    def test_congestion_changes_effective_network(self):
        profile = DeviceProfile(
            "device-0",
            20,
            50,
            40,
            0,
            congestion_probability=1.0,
            congestion_slowdown=2.0,
        )
        sample = sample_network_delay(1_000_000, profile, "uplink", random.Random(1))
        self.assertTrue(sample.congested)
        self.assertAlmostEqual(sample.effective_mbps, 10.0)
        self.assertAlmostEqual(sample.effective_rtt_ms, 80.0)
        self.assertAlmostEqual(sample.delay_s, 0.84)

    def test_load_profile_ignores_legacy_compute_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profiles.yaml"
            path.write_text(
                """
devices:
  device-0:
    uplink_mbps: 20
    downlink_mbps: 50
    rtt_ms: 40
  device-1:
    peak_tflops: 7.5
    uplink_mbps: 20
    downlink_mbps: 50
    rtt_ms: 40
""",
                encoding="utf-8",
            )
            profiles = load_device_profiles(path)
        self.assertEqual(profiles["device-0"].uplink_mbps, 20.0)
        self.assertEqual(profiles["device-1"].uplink_mbps, 20.0)
        self.assertFalse(hasattr(profiles["device-1"], "effective_tflops"))


if __name__ == "__main__":
    unittest.main()
