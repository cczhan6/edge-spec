import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CliMethodTests(unittest.TestCase):
    def test_method_cli_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            for method in ("proposed", "sync_batch", "target_only"):
                out = Path(tmp) / method
                cmd = [
                    sys.executable,
                    "-m",
                    "edge_spec.run",
                    "--method",
                    method,
                    "--use-fake-models",
                    "--limit",
                    "1",
                    "--max-new-tokens",
                    "2",
                    "--gamma",
                    "1",
                    "--seed",
                    "123",
                    "--network-seed",
                    "456",
                    "--network-trace-slot-s",
                    "0.2",
                    "--no-progress",
                    "--results-dir",
                    str(out),
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, text=True)
                self.assertTrue((out / "request_records.jsonl").exists())
                self.assertTrue((out / "event_trace.jsonl").exists())
                summary = json.loads((out / "summary.json").read_text())
                self.assertEqual(summary["run_config"]["method"], method)
                self.assertEqual(summary["run_config"]["seed"], 123)
                self.assertEqual(summary["run_config"]["network_seed"], 456)
                self.assertEqual(summary["run_config"]["network_trace_slot_s"], 0.2)


if __name__ == "__main__":
    unittest.main()
