from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.baseline_trace import write_trace_bundle
from src.config import load_config
from src.metrics import summarize
from src.simulator import Simulator
from tests.common import accepting_model_runner, small_config


class RealModelSmokeTest(unittest.TestCase):
    def test_prepare_writes_explicit_real_model_config_and_fixed_subset(self) -> None:
        from scripts.real_model_smoke import prepare_real_model_inputs

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "real_model_smoke"
            config_path, dataset_path = prepare_real_model_inputs(
                root=root,
                config_path="configs/default.yaml",
                target_model="/models/target",
                draft_model="/models/draft",
                dataset_path=None,
                target_device="cpu",
                draft_device="cpu",
                num_requests=4,
                output_tokens=8,
                local_files_only=True,
            )

            config = load_config(config_path)
            self.assertEqual(config["simulation"]["seed"], 20260625)
            self.assertEqual(config["simulation"]["num_requests"], 4)
            self.assertEqual(config["simulation"]["output_len_choices"], [8])
            self.assertEqual(config["model_runner"]["target_model"], "/models/target")
            self.assertEqual(config["model_runner"]["target_device"], "cpu")
            self.assertTrue(config["model_runner"]["local_files_only"])
            for values in config["model_runner"]["drafter_models"].values():
                self.assertEqual(values["model"], "/models/draft")
                self.assertEqual(values["device"], "cpu")
            self.assertEqual(config["server_only"]["batch_size"], 1)
            self.assertTrue(config["specedge"]["proactive_enabled"])
            self.assertEqual(config["dip_sd"]["optimizer"], "paper_exact")
            self.assertLessEqual(config["dip_sd"]["max_batch_size"], 2)
            self.assertEqual(dataset_path.read_text(encoding="utf-8").count("\n"), 4)

    def test_run_script_accepts_method_selection_before_model_path_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = os.environ.copy()
            for key in ("TARGET_MODEL_PATH", "DRAFT_MODEL_PATH"):
                env.pop(key, None)
            completed = subprocess.run(
                [
                    "bash",
                    "scripts/run_real_model_smoke.sh",
                    "--root",
                    str(Path(directory) / "out"),
                    "--methods",
                    "server_only_tree,specedge_tree",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )

            self.assertNotEqual(completed.returncode, 0)
            combined = completed.stdout + completed.stderr
            self.assertIn("TARGET_MODEL_PATH", combined)
            self.assertIn("DRAFT_MODEL_PATH", combined)
            self.assertNotIn("Unknown argument", combined)
            self.assertNotIn("--use-fake-model-runner", combined)

    def test_method_selection_defaults_and_adds_target_reference(self) -> None:
        from scripts.real_model_smoke import REAL_MODEL_METHODS, parse_real_model_methods

        self.assertEqual(parse_real_model_methods(None), REAL_MODEL_METHODS)
        self.assertEqual(
            parse_real_model_methods("server_only_tree,specedge_tree"),
            ("target_only", "server_only_tree", "specedge_tree"),
        )
        self.assertEqual(
            parse_real_model_methods("target_only,server_only_tree,specedge_tree"),
            ("target_only", "server_only_tree", "specedge_tree"),
        )
        with self.assertRaisesRegex(ValueError, "unknown real-model smoke method"):
            parse_real_model_methods("target_only,proposed")

    def test_verify_accepts_valid_real_runner_marked_trace_bundle(self) -> None:
        from scripts.real_model_smoke import REAL_MODEL_METHODS, verify_real_model_outputs

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "real_model_smoke"
            config, model_runner, workload = small_config(num_requests=4, output_len=8)
            config["simulation"]["seed"] = 20260625
            config["simulation"]["num_devices"] = 4
            config["simulation"]["request_arrival"] = "burst"
            config["device_pools"]["heterogeneous"]["templates"]["low_end"]["count"] = 4
            config["device_pools"]["medium_only"]["templates"]["medium"]["count"] = 4
            config["speculation"]["gamma_candidates"] = [1]
            config["speculation"]["gamma_fixed"] = 1
            config["specedge"]["server_batch_size"] = 1
            config["specedge"]["server_batch_timeout_ms"] = None
            config["specedge"]["proactive_enabled"] = True
            config["dip_sd"]["batch_count"] = 2
            config["dip_sd"]["max_batch_size"] = 2
            config["dip_sd"]["min_draft_length"] = 1
            config["dip_sd"]["max_draft_length"] = 2
            config["dip_sd"]["draft_length"] = 1
            for profile in config["drafter_profiles"].values():
                profile["acceptance_prior"] = 0.9

            for method in REAL_MODEL_METHODS:
                result = Simulator(
                    config,
                    model_runner,
                    workload,
                    "real_model_smoke",
                    method,
                ).run()
                main, system = summarize(result, int(config["simulation"]["num_devices"]))
                method_dir = root / method
                write_trace_bundle(method_dir, config, result, main, system)
                (method_dir / "resolved_config").write_text(
                    (method_dir / "resolved_config.json").read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
                (method_dir / "stdout.log").write_text("wrote 1 method rows\n", encoding="utf-8")
                (method_dir / "run_manifest.json").write_text(
                    json.dumps(
                        {
                            "method": method,
                            "runner": "HuggingFaceModelRunner",
                            "use_fake_model_runner": False,
                            "target_model": "target-real",
                            "draft_model": "draft-real",
                            "dataset": "fixed-subset.jsonl",
                            "request_count": 4,
                            "output_tokens": 8,
                            "return_code": 0,
                            "command": [sys.executable, "-m", "scripts.run_all"],
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )

            summaries = verify_real_model_outputs(root, root / "summary.md")

            self.assertEqual([row["method"] for row in summaries], list(REAL_MODEL_METHODS))
            summary = (root / "summary.md").read_text(encoding="utf-8")
            self.assertIn("Status: PASS", summary)
            self.assertIn("greedy equivalence", summary)
            self.assertIn("real target verification", summary)

    def test_verify_accepts_selected_tree_real_runner_marked_trace_bundle(self) -> None:
        from scripts.real_model_smoke import verify_real_model_outputs

        methods = ("target_only", "server_only_tree", "specedge_tree")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "real_model_smoke"
            config, model_runner, workload = small_config(num_requests=4, output_len=8)
            config["simulation"]["seed"] = 20260625
            config["simulation"]["num_devices"] = 4
            config["simulation"]["request_arrival"] = "burst"
            config["device_pools"]["heterogeneous"]["templates"]["low_end"]["count"] = 4
            config["device_pools"]["medium_only"]["templates"]["medium"]["count"] = 4
            config["speculation"]["gamma_candidates"] = [1]
            config["speculation"]["gamma_fixed"] = 1
            config["specedge"]["server_batch_size"] = 1
            config["specedge"]["server_batch_timeout_ms"] = None
            config["specedge"]["proactive_enabled"] = True
            config["specedge"]["proactive_type"] = "excluded"

            for method in methods:
                result = Simulator(
                    config,
                    model_runner,
                    workload,
                    "real_model_smoke",
                    method,
                ).run()
                main, system = summarize(result, int(config["simulation"]["num_devices"]))
                method_dir = root / method
                write_trace_bundle(method_dir, config, result, main, system)
                (method_dir / "resolved_config").write_text(
                    (method_dir / "resolved_config.json").read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
                (method_dir / "stdout.log").write_text("wrote 1 method rows\n", encoding="utf-8")
                (method_dir / "run_manifest.json").write_text(
                    json.dumps(
                        {
                            "method": method,
                            "runner": "HuggingFaceModelRunner",
                            "use_fake_model_runner": False,
                            "target_model": "target-real",
                            "draft_model": "draft-real",
                            "dataset": "fixed-subset.jsonl",
                            "request_count": 4,
                            "output_tokens": 8,
                            "return_code": 0,
                            "command": [sys.executable, "-m", "scripts.run_all"],
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )

            summaries = verify_real_model_outputs(
                root,
                root / "summary.md",
                methods=methods,
            )

            self.assertEqual([row["method"] for row in summaries], list(methods))
            summary = (root / "summary.md").read_text(encoding="utf-8")
            self.assertIn("Status: PASS", summary)
            self.assertIn("server_only_tree", summary)
            self.assertIn("specedge_tree", summary)
            self.assertIn("specexec_approx", summary)
            self.assertIn("tree baselines use real tree candidates", summary)
            self.assertIn("proactive tree drafting", summary)


if __name__ == "__main__":
    unittest.main()
