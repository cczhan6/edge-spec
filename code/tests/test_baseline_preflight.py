from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.baseline_trace import (
    _drafted_count,
    _proactive_proposed_count,
    _segment_waste_count,
    _token_trace_rows,
    _verified_count,
    write_trace_bundle,
)
from scripts.run_all import build_parser
from scripts.verify_baseline_preflight import (
    PREFLIGHT_METRIC_FIELDS,
    PREFLIGHT_METHODS,
    REQUIRED_RUN_FILES,
    apply_target_speedups,
    compute_decode_metrics,
    materialize_cell,
    prepare_preflight_config,
    verify_preflight,
)
from src.config import load_config
from src.metrics import SYSTEM_FIELDS, summarize, write_csv
from src.simulator import Simulator
from tests.common import accepting_model_runner, small_config


class _WriteOnlyCommitTimes(list[float]):
    def __iter__(self):
        raise AssertionError("commit-time observer must not be read by scheduling")

    def __getitem__(self, index):
        raise AssertionError("commit-time observer must not be indexed by scheduling")

    def __len__(self):
        raise AssertionError("commit-time observer must not affect completion")

    def __bool__(self):
        raise AssertionError("commit-time observer must not affect control flow")


class CommittedTokenTimeTest(unittest.TestCase):
    def test_observer_is_not_read_and_cannot_change_simulation_semantics(self) -> None:
        config, _, workload = small_config(num_requests=2, output_len=8)
        reference = Simulator(
            copy.deepcopy(config),
            accepting_model_runner(),
            workload,
            "combined_strong_heterogeneous",
            "server_only_linear",
        ).run()

        from src.entities import Request as RequestEntity

        def request_with_write_only_observer(*args, **kwargs):
            request = RequestEntity(*args, **kwargs)
            request.committed_token_times_ms = _WriteOnlyCommitTimes()
            return request

        with mock.patch("src.simulator.Request", side_effect=request_with_write_only_observer):
            guarded = Simulator(
                copy.deepcopy(config),
                accepting_model_runner(),
                workload,
                "combined_strong_heterogeneous",
                "server_only_linear",
            ).run()

        self.assertEqual(
            [request.generated_ids for request in guarded.requests],
            [request.generated_ids for request in reference.requests],
        )
        self.assertEqual(
            [request.finish_time_ms for request in guarded.requests],
            [request.finish_time_ms for request in reference.requests],
        )
        self.assertEqual(guarded.event_trace, reference.event_trace)

    def test_tokens_committed_by_one_result_share_one_commit_time(self) -> None:
        config, _, workload = small_config(num_requests=1, output_len=8)
        result = Simulator(
            config,
            accepting_model_runner(),
            workload,
            "combined_strong_heterogeneous",
            "server_only_linear",
        ).run()

        request = result.requests[0]
        multi_token_segments = [segment for segment in result.segments if len(segment.emitted_ids) > 1]
        self.assertTrue(multi_token_segments)
        for segment in multi_token_segments:
            commit_times = request.committed_token_times_ms[
                segment.base_pos : segment.base_pos + len(segment.emitted_ids)
            ]
            self.assertEqual(len(set(commit_times)), 1)

    def test_target_only_records_monotonic_committed_token_times(self) -> None:
        config, model_runner, workload = small_config(num_requests=2, output_len=5)

        result = Simulator(
            config,
            model_runner,
            workload,
            "combined_strong_heterogeneous",
            "target_only",
        ).run()

        for request in result.requests:
            self.assertEqual(
                len(request.committed_token_times_ms),
                len(request.generated_ids),
            )
            self.assertEqual(
                request.committed_token_times_ms,
                sorted(request.committed_token_times_ms),
            )
            self.assertEqual(
                request.committed_token_times_ms[-1],
                request.finish_time_ms,
            )


class PreflightConfigTest(unittest.TestCase):
    def test_verifier_is_directly_executable_from_repository_root(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/verify_baseline_preflight.py", "--help"],
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("decode-only", completed.stdout)

    def test_prepare_changes_only_formal_scale_seed_and_arrival(self) -> None:
        source = load_config("configs/default.yaml", "homogeneous")
        expected = copy.deepcopy(source)
        expected["simulation"].update(
            num_requests=16,
            output_len_choices=[32],
            seed=20260628,
            request_arrival="poisson",
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "preflight.yaml"
            prepare_preflight_config(
                "configs/default.yaml",
                "homogeneous",
                20260628,
                output,
            )

            self.assertEqual(load_config(output), expected)

    def test_formal_cli_accepts_multi_method_trace_bundle_root(self) -> None:
        args = build_parser().parse_args(
            [
                "--scenario",
                "homogeneous",
                "--methods",
                *PREFLIGHT_METHODS,
                "--trace-bundle-root",
                "outputs/baseline_preflight/homogeneous/20260628",
            ]
        )

        self.assertEqual(args.trace_bundle_root, "outputs/baseline_preflight/homogeneous/20260628")
        self.assertFalse(args.use_fake_model_runner)

    def test_trace_bundle_retains_system_metrics_for_formal_derivation(self) -> None:
        config, model_runner, workload = small_config(num_requests=1, output_len=4)
        result = Simulator(config, model_runner, workload, "homogeneous", "target_only").run()
        main, system = summarize(result, int(config["simulation"]["num_devices"]))

        with tempfile.TemporaryDirectory() as directory:
            write_trace_bundle(directory, config, result, main, system)
            rows = Path(directory, "system_metrics.csv").read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(rows), 2)
        self.assertIn("target_utilization", rows[0])

    def test_preflight_shell_freezes_cells_methods_and_real_runner(self) -> None:
        script = Path("scripts/run_baseline_preflight.sh").read_text(encoding="utf-8")

        for scenario in ("homogeneous", "combined_strong_heterogeneous"):
            self.assertIn(scenario, script)
        for seed in ("20260628", "20260629"):
            self.assertIn(seed, script)
        for method in PREFLIGHT_METHODS:
            self.assertIn(method, script)
        self.assertIn("--trace-bundle-root", script)
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF", script)
        self.assertIn("expandable_segments:True", script)
        self.assertNotIn("--use-fake-model-runner", script)
        self.assertNotIn("480", script)


class PreflightMetricTest(unittest.TestCase):
    def test_tree_token_counts_use_processed_and_target_verified_nodes(self) -> None:
        segment = SimpleNamespace(
            tree_strategy="specexec_approx",
            processed_candidate_count=12,
            target_verify_tree_nodes=7,
            proposed_count=3,
            accepted_count=2,
            status="rejected",
            verify_gamma=3,
            proactive_draft_tree=SimpleNamespace(
                nodes=[object()],
                processed_candidate_count=9,
            ),
            proactive_draft_ids=[1, 2],
        )

        self.assertEqual(_drafted_count(segment), 12)
        self.assertEqual(_verified_count(segment), 7)
        self.assertEqual(_segment_waste_count(segment), 10)
        self.assertEqual(_proactive_proposed_count(segment), 9)

    def test_decode_metrics_follow_frozen_formulas(self) -> None:
        requests = [
            {
                "request_id": "0",
                "decode_ready_time_ms": "0",
                "finish_time_ms": "40",
                "latency_ms": "40",
            },
            {
                "request_id": "1",
                "decode_ready_time_ms": "10",
                "finish_time_ms": "60",
                "latency_ms": "50",
            },
        ]
        tokens = [
            {"request_id": "0", "token_type": "committed", "position": "0", "commit_time_ms": "10", "count": "1"},
            {"request_id": "0", "token_type": "committed", "position": "1", "commit_time_ms": "20", "count": "1"},
            {"request_id": "0", "token_type": "committed", "position": "2", "commit_time_ms": "40", "count": "1"},
            {"request_id": "1", "token_type": "committed", "position": "0", "commit_time_ms": "20", "count": "1"},
            {"request_id": "1", "token_type": "committed", "position": "1", "commit_time_ms": "50", "count": "1"},
            {"request_id": "0", "token_type": "drafted", "count": "6"},
            {"request_id": "0", "token_type": "proactive_drafted", "count": "2"},
            {"request_id": "0", "token_type": "verified", "count": "5"},
            {"request_id": "0", "token_type": "accepted", "count": "3"},
            {"request_id": "0", "token_type": "wasted", "count": "4"},
        ]
        system = {
            "target_utilization": "0.5",
            "device_utilization_mean": "0.25",
            "lane_queue_wait_ms_mean": "1.5",
        }
        resources = [
            {"resource_key": "target", "resource_type": "target", "duration_ms": "30"},
            {"resource_key": "draft", "resource_type": "draft", "duration_ms": "20"},
        ]

        metrics = compute_decode_metrics(
            requests,
            tokens,
            system,
            resources=resources,
            target_only_latency=90.0,
        )

        self.assertEqual(list(metrics), PREFLIGHT_METRIC_FIELDS)
        self.assertAlmostEqual(metrics["decode_makespan"], 60.0)
        self.assertAlmostEqual(metrics["request_decode_latency"], 45.0)
        self.assertAlmostEqual(metrics["mean_inter_token_latency"], 20.0)
        self.assertAlmostEqual(metrics["p50_inter_token_latency"], 20.0)
        self.assertAlmostEqual(metrics["p95_inter_token_latency"], 29.0)
        self.assertAlmostEqual(
            metrics["effective_throughput_tokens_per_s"],
            5.0 / 0.06,
        )
        self.assertAlmostEqual(metrics["speedup_vs_target_only"], 2.0)
        self.assertAlmostEqual(metrics["acceptance_ratio"], 3.0 / 5.0)
        self.assertEqual(metrics["drafted_tokens"], 8)
        self.assertEqual(metrics["verified_tokens"], 5)
        self.assertEqual(metrics["accepted_tokens"], 3)
        self.assertEqual(metrics["committed_tokens"], 5)
        self.assertEqual(metrics["wasted_tokens"], 4)
        self.assertEqual(metrics["target_utilization"], 0.5)
        self.assertAlmostEqual(metrics["draft_utilization"], 1.0 / 3.0)
        self.assertEqual(metrics["verification_queue_wait"], 1.5)
        self.assertFalse(
            any(
                "ttft" in field.lower() or "first_token" in field.lower()
                for field in metrics
            )
        )

    def test_speedup_uses_target_only_from_same_scenario_and_seed(self) -> None:
        rows = [
            {"scenario": "homogeneous", "seed": 1, "method": "target_only", "request_decode_latency": 100.0},
            {"scenario": "homogeneous", "seed": 1, "method": "dip_sd", "request_decode_latency": 50.0},
            {"scenario": "homogeneous", "seed": 2, "method": "target_only", "request_decode_latency": 300.0},
            {"scenario": "homogeneous", "seed": 2, "method": "dip_sd", "request_decode_latency": 100.0},
        ]

        apply_target_speedups(rows)

        self.assertEqual([row["speedup_vs_target_only"] for row in rows], [1.0, 2.0, 1.0, 3.0])

    def test_complete_cell_materializes_required_files_and_passes_invariants(self) -> None:
        scenario = "homogeneous"
        seed = 20260628
        config, model_runner, workload = small_config(num_requests=4, output_len=8)
        model_runner = accepting_model_runner()
        config["simulation"].update(
            seed=seed,
            request_arrival="poisson",
            poisson_rate_per_s=0.001,
        )
        config["experiment"]["internal_time_unit"] = "ms"
        config["experiment"]["csv_time_unit"] = "ms"
        config["speculation"].update(gamma_candidates=[1], gamma_fixed=1)
        for profile in config["drafter_profiles"].values():
            profile["acceptance_prior"] = 0.9

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "baseline_preflight"
            cell = root / scenario / str(seed)
            environment = root / "environment_manifest.json"
            environment.parent.mkdir(parents=True, exist_ok=True)
            environment.write_text(
                json.dumps({"git": {"commit": "a" * 40}, "collection_errors": []}) + "\n",
                encoding="utf-8",
            )

            for method in PREFLIGHT_METHODS:
                result = Simulator(config, model_runner, workload, scenario, method).run()
                main, system = summarize(result, int(config["simulation"]["num_devices"]))
                method_dir = cell / method
                write_trace_bundle(method_dir, config, result, main, system)
                write_csv(method_dir / "system_metrics.csv", [system], SYSTEM_FIELDS)
                (method_dir / "stdout.log").write_text(
                    "wrote 6 method rows with HuggingFaceModelRunner\n",
                    encoding="utf-8",
                )

            materialize_cell(
                cell,
                scenario=scenario,
                seed=seed,
                environment_path=environment,
                command="python -m scripts.run_all --methods " + " ".join(PREFLIGHT_METHODS),
                request_count=4,
                output_length=8,
            )
            summaries = verify_preflight(
                root,
                scenarios=(scenario,),
                seeds=(seed,),
                request_count=4,
                output_length=8,
            )

            self.assertEqual(len(summaries), len(PREFLIGHT_METHODS))
            self.assertTrue(all(row["success"] for row in summaries))
            for method in PREFLIGHT_METHODS:
                method_dir = cell / method
                for filename in REQUIRED_RUN_FILES:
                    self.assertTrue((method_dir / filename).is_file(), method_dir / filename)
                with (method_dir / "metrics.csv").open(encoding="utf-8") as handle:
                    header = handle.readline().strip().split(",")
                self.assertEqual(header, PREFLIGHT_METRIC_FIELDS)
            summary_header = (root / "summary.csv").read_text(encoding="utf-8").splitlines()[0]
            self.assertNotIn("ttft", summary_header.lower())
            self.assertNotIn("first_token", summary_header.lower())
            self.assertIn("Status: PASS", (root / "summary.md").read_text(encoding="utf-8"))

    def test_speculative_trace_records_commit_time_for_every_committed_token(self) -> None:
        config, model_runner, workload = small_config(num_requests=2, output_len=5)

        result = Simulator(
            config,
            model_runner,
            workload,
            "combined_strong_heterogeneous",
            "server_only_linear",
        ).run()

        committed_rows = [
            row for row in _token_trace_rows(result) if row["token_type"] == "committed"
        ]
        self.assertEqual(
            len(committed_rows),
            sum(len(request.generated_ids) for request in result.requests),
        )
        self.assertTrue(all(row["commit_time_ms"] is not None for row in committed_rows))
        for request in result.requests:
            self.assertEqual(
                len(request.committed_token_times_ms),
                len(request.generated_ids),
            )
            self.assertEqual(
                request.committed_token_times_ms[-1],
                request.finish_time_ms,
            )


if __name__ == "__main__":
    unittest.main()
