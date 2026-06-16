from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CliSmokeTest(unittest.TestCase):
    def test_fake_model_runner_cli_runs_without_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yaml"
            config.write_text(
                Path("configs/default.yaml")
                .read_text()
                .replace("num_requests: 200", "num_requests: 3")
                .replace("output_len_choices: [64, 128, 256]", "output_len_choices: [4]"),
                encoding="utf-8",
            )
            output = root / "raw"
            command = [
                sys.executable,
                "-m",
                "scripts.run_all",
                "--config",
                str(config),
                "--use-fake-model-runner",
                "--scenario",
                "smoke",
                "--method",
                "full",
                "--out_dir",
                str(output),
                "--summary_out",
                str(root / "summary.csv"),
                "--samples-per-category",
                "1",
            ]
            completed = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertIn("metrics: full", completed.stderr)
            self.assertIn("comparison:", completed.stderr)
            self.assertIn("goodput=", completed.stderr)
            self.assertTrue((output / "event_details_smoke_full.csv").exists())
            self.assertTrue((output / "segment_details_smoke_full.csv").exists())
            self.assertTrue((output / "device_metrics_smoke_full.csv").exists())
            self.assertTrue((output / "round_trace_smoke_full.csv").exists())
            self.assertTrue((output / "category_results_smoke.csv").exists())
            self.assertTrue((root / "summary.csv").exists())
            self.assertTrue((root / "category_results.csv").exists())
            with (output / "request_details_smoke_full.csv").open(encoding="utf-8") as handle:
                self.assertIn("category", csv.DictReader(handle).fieldnames or [])
            with (root / "summary.csv").open(encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
                self.assertEqual(row["num_requests"], "6")

    def test_run_sh_creates_isolated_run_dir_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yaml"
            config.write_text(
                Path("configs/default.yaml")
                .read_text()
                .replace("num_requests: 200", "num_requests: 2")
                .replace("output_len_choices: [64, 128, 256]", "output_len_choices: [4]"),
                encoding="utf-8",
            )
            env = os.environ.copy()
            for key in ("OUT_DIR", "SUMMARY_OUT", "RUN_ID", "RUN_DIR"):
                env.pop(key, None)
            env.update(
                {
                    "CONFIG": str(config),
                    "DATASET": "data/spec_bench/question.jsonl",
                    "RUN_ROOT": str(root / "runs"),
                    "SUMMARY_ONLY": "1",
                }
            )

            completed = subprocess.run(
                ["bash", "scripts/run.sh", "smoke"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )

            run_dirs = sorted((root / "runs").iterdir())
            self.assertEqual(len(run_dirs), 1)
            run_dir = run_dirs[0]
            self.assertRegex(run_dir.name, r"^\d{8}-\d{6}(-\d{2})?$")
            self.assertIn(f"Run directory: {run_dir}", completed.stderr)
            self.assertTrue((run_dir / "manifest.yaml").exists())
            self.assertTrue((run_dir / "raw" / "main_results_balanced_drafter.csv").exists())
            self.assertTrue((run_dir / "summary" / "all_results.csv").exists())
            self.assertTrue((run_dir / "summary" / "category_results.csv").exists())

            manifest = (run_dir / "manifest.yaml").read_text(encoding="utf-8")
            self.assertIn('command: "bash scripts/run.sh smoke"', manifest)
            self.assertIn('use_fake_model_runner: true', manifest)
            self.assertIn('summary_only: true', manifest)
            self.assertIn('  - "balanced_drafter"', manifest)
            self.assertIn('  - "full"', manifest)

    def test_summary_only_skips_per_method_detail_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yaml"
            config.write_text(
                Path("configs/default.yaml")
                .read_text()
                .replace("num_requests: 200", "num_requests: 3")
                .replace("output_len_choices: [64, 128, 256]", "output_len_choices: [4]"),
                encoding="utf-8",
            )
            output = root / "raw"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.run_all",
                    "--config",
                    str(config),
                    "--use-fake-model-runner",
                    "--scenario",
                    "smoke",
                    "--method",
                    "full",
                    "--out_dir",
                    str(output),
                    "--summary_out",
                    str(root / "summary.csv"),
                    "--samples-per-category",
                    "1",
                    "--summary-only",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertIn("metrics: full", completed.stderr)
            self.assertIn("comparison:", completed.stderr)

            self.assertTrue((output / "main_results_smoke.csv").exists())
            self.assertTrue((output / "category_results_smoke.csv").exists())
            self.assertTrue((output / "system_metrics_smoke.csv").exists())
            self.assertTrue((root / "summary.csv").exists())
            self.assertFalse((output / "request_details_smoke_full.csv").exists())
            self.assertFalse((output / "segment_details_smoke_full.csv").exists())
            self.assertFalse((output / "event_details_smoke_full.csv").exists())
            self.assertFalse((output / "device_metrics_smoke_full.csv").exists())
            self.assertFalse((output / "round_trace_smoke_full.csv").exists())

    def test_server_only_method_is_accepted_by_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yaml"
            config.write_text(
                Path("configs/default.yaml")
                .read_text()
                .replace("num_requests: 200", "num_requests: 2")
                .replace("output_len_choices: [64, 128, 256]", "output_len_choices: [4]"),
                encoding="utf-8",
            )
            output = root / "raw"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.run_all",
                    "--config",
                    str(config),
                    "--use-fake-model-runner",
                    "--scenario",
                    "smoke",
                    "--method",
                    "server_only",
                    "--out_dir",
                    str(output),
                    "--summary_out",
                    str(root / "summary.csv"),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertIn("metrics: server_only", completed.stderr)
            self.assertTrue((output / "event_details_smoke_server_only.csv").exists())

    def test_removed_wall_time_flags_are_rejected(self) -> None:
        for removed_args in (
            ["--cache-latency-by-shape"],
            ["--acceptance_trace", "trace.csv"],
        ):
            with self.subTest(removed_args=removed_args):
                completed = subprocess.run(
                    [sys.executable, "-m", "scripts.run_all", *removed_args],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("unrecognized arguments", completed.stderr)

    def test_removed_methods_are_rejected(self) -> None:
        for method in (
            "vanilla_sd",
            "hetero_drafter_sync_batch_sd",
            "optimized_sync_batch_sd",
            "phase_overlap_sync_sd",
        ):
            with self.subTest(method=method):
                completed = subprocess.run(
                    [sys.executable, "-m", "scripts.run_all", "--method", method],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("invalid choice", completed.stderr)


if __name__ == "__main__":
    unittest.main()
