from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.baseline_trace import METHODS, REQUIRED_FILES


class BaselineTraceRunnerTest(unittest.TestCase):
    def test_baseline_trace_script_writes_and_verifies_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "baseline_trace"
            env = os.environ.copy()
            env["TRACE_ROOT"] = str(root)

            completed = subprocess.run(
                ["bash", "scripts/run_baseline_trace.sh"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )

            self.assertIn("target_only: success=True", completed.stdout)
            self.assertTrue((root / "summary.md").exists())
            summary = (root / "summary.md").read_text(encoding="utf-8")
            self.assertIn("Status: PASS", summary)
            self.assertIn("SpecEdge traces contain proactive drafting", summary)
            self.assertIn("DiP-SD has at least two batches", summary)
            for method in METHODS:
                for filename in REQUIRED_FILES:
                    path = root / method / filename
                    self.assertTrue(path.exists(), path)
                    self.assertGreater(path.stat().st_size, 0, path)


if __name__ == "__main__":
    unittest.main()
