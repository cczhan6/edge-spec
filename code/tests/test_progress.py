from __future__ import annotations

import io
import unittest

from scripts.progress import ProgressReporter


class ProgressReporterTest(unittest.TestCase):
    def test_updates_rewrite_current_line_until_finished(self) -> None:
        stream = io.StringIO()
        progress = ProgressReporter(3, "full", stream=stream, unit="req")

        progress.start()
        progress.update(1)
        progress.update(3)
        progress.finish_line()

        output = stream.getvalue()
        self.assertEqual(output.count("\n"), 1)
        self.assertEqual(output.count("\r"), 2)
        self.assertTrue(output.endswith("\n"))


if __name__ == "__main__":
    unittest.main()
