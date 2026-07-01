from __future__ import annotations

import csv
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_baseline_performance_eval import (
    SharedTraceSimulator,
    build_resource_fingerprint,
    load_shared_trace,
    materialize_shared_trace,
    prepare_matrix_inputs,
    run_matrix_cells,
)
from scripts.baseline_trace import (
    BATCH_TRACE_FIELDS,
    EVENT_TRACE_FIELDS,
    REQUEST_TRACE_FIELDS,
    RESOURCE_TIMELINE_FIELDS,
    TOKEN_TRACE_FIELDS,
)
from scripts.summarize_baseline_performance_eval import (
    METHODS,
    PERFORMANCE_FIELDS,
    SCENARIO,
    SEEDS,
    aggregate_rows,
    initialize_runs_csv,
    summarize_results,
)
from src.config import load_config
from src.metrics import MAIN_FIELDS, SYSTEM_FIELDS, write_csv
from src.workload import WorkloadItem
from tests.common import small_config


def test_dynamic_heterogeneous_configuration_contract() -> None:
    config = load_config("configs/default.yaml", "dynamic_heterogeneous")

    assert config["simulation"]["num_requests"] == 80
    assert config["simulation"]["request_arrival"] == "poisson"
    assert config["dynamic_edge_compute"] == {
        "enabled": True,
        "resample_every_completed_requests": 5,
    }
    assert config["target_latency"] == {
        "mode": "profile",
        "profile_path": "outputs/profiling/target_verification_latency_full_merged.csv",
        "metric": "p50_ms",
    }
    templates = config["device_pools"]["heterogeneous"]["templates"]
    assert {name: values["count"] for name, values in templates.items()} == {
        "low_end": 3,
        "mid_end": 3,
        "high_end": 2,
    }
    assert all(values["block_probability"] == 0.2 for values in templates.values())
    assert all(
        "dynamic_draft_token_rate_range_tok_s" in values
        for values in templates.values()
    )


def test_initialize_runs_csv_contains_every_expected_cell(tmp_path: Path) -> None:
    path = initialize_runs_csv(tmp_path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 30
    assert {(int(row["seed"]), row["method"]) for row in rows} == {
        (seed, method) for seed in SEEDS for method in METHODS
    }
    assert all(row["metric_scope"] == "decode_only" for row in rows)
    assert all(row["success"] == "False" for row in rows)
    assert all(row["failure_reason"] == "not run" for row in rows)


def test_aggregate_rows_uses_sample_stats_only_for_performance_fields() -> None:
    rows = []
    for seed in SEEDS:
        row = {
            "scenario": "dynamic_heterogeneous",
            "seed": seed,
            "method": "target_only",
            "metric_scope": "decode_only",
            "num_requests": 80,
            "committed_tokens": 1000 + seed,
            "success": True,
            "failure_reason": "",
        }
        row.update({field: float(seed + 1) for field in PERFORMANCE_FIELDS})
        rows.append(row)

    summary = aggregate_rows(rows)[0]

    assert summary["num_runs"] == 5
    assert summary["successful_runs"] == 5
    assert summary["success"] is True
    assert summary["avg_latency_ms_mean"] == 3.0
    assert math.isclose(summary["avg_latency_ms_std"], math.sqrt(2.5))
    assert "seed_mean" not in summary
    assert "success_mean" not in summary
    assert "num_requests_mean" not in summary
    assert "committed_tokens_mean" not in summary


def _workload(count: int = 4) -> list[WorkloadItem]:
    rows = []
    for index in range(count):
        prompt = f"prompt text {index}"
        rows.append(
            WorkloadItem(
                prompt_id=f"prompt-{index}",
                prompt=prompt,
                prompt_token_count=len(prompt.encode("utf-8")),
                category="qa",
                category_group="QA",
            )
        )
    return rows


def test_materialized_trace_is_reused_byte_for_byte(tmp_path: Path) -> None:
    config = load_config("configs/default.yaml")
    config["simulation"].update(
        seed=3,
        num_requests=4,
        num_devices=4,
        output_len_choices=[8, 16],
        request_arrival="poisson",
        poisson_rate_per_s=20,
    )
    path = tmp_path / "seed_3.jsonl"

    first_hash = materialize_shared_trace(config, _workload(), path)
    first_bytes = path.read_bytes()
    second_hash = materialize_shared_trace(config, _workload(), path)
    rows = load_shared_trace(path, config)

    assert second_hash == first_hash
    assert path.read_bytes() == first_bytes
    assert [row.request_id for row in rows] == [0, 1, 2, 3]
    assert [row.device_id for row in rows] == [0, 1, 2, 3]
    assert rows[0].arrival_time_ms == 0.0
    assert [row.arrival_time_ms for row in rows] == sorted(
        row.arrival_time_ms for row in rows
    )


def test_existing_shared_trace_rejects_different_content(tmp_path: Path) -> None:
    config = load_config("configs/default.yaml")
    config["simulation"].update(seed=0, num_requests=4, num_devices=4)
    path = tmp_path / "seed_0.jsonl"
    materialize_shared_trace(config, _workload(), path)

    changed = [
        replace(item, prompt="changed") if index == 0 else item
        for index, item in enumerate(_workload())
    ]
    with pytest.raises(ValueError, match="existing shared trace differs"):
        materialize_shared_trace(config, changed, path)


def test_all_methods_consume_shared_arrivals_without_resampling(
    tmp_path: Path,
) -> None:
    config, runner, _ = small_config(num_requests=4, output_len=8)
    config["simulation"].update(
        seed=4,
        request_arrival="poisson",
        poisson_rate_per_s=20,
    )
    path = tmp_path / "shared.jsonl"
    materialize_shared_trace(config, _workload(), path)
    shared = load_shared_trace(path, config)

    observed = []
    for method in (
        "target_only",
        "server_only_linear",
        "specedge_linear",
        "dip_sd",
    ):
        simulator = SharedTraceSimulator(
            config,
            runner,
            shared,
            "dynamic_heterogeneous",
            method,
        )
        simulator._rng = SimpleNamespace(
            expovariate=lambda *_: (_ for _ in ()).throw(
                AssertionError("resampled arrival")
            ),
            choice=lambda *_: (_ for _ in ()).throw(
                AssertionError("resampled output length")
            ),
        )
        simulator._schedule_request_arrivals()
        observed.append(
            [
                (
                    request.request_id,
                    request.arrival_time_ms,
                    request.output_len,
                    request.device_id,
                )
                for request in simulator.requests
            ]
        )

    assert all(rows == observed[0] for rows in observed[1:])


def test_run_matrix_cells_continues_after_process_failure(tmp_path: Path) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append(tuple(command))
        return SimpleNamespace(returncode=7 if len(calls) == 2 else 0)

    initialize_runs_csv(tmp_path)
    statuses = run_matrix_cells(
        root=tmp_path,
        seeds=SEEDS,
        methods=METHODS,
        command_for=lambda seed, method: ["worker", str(seed), method],
        run_process=fake_run,
    )

    assert len(calls) == 30
    assert len(statuses) == 30
    assert statuses[1]["success"] is False
    assert "return code 7" in statuses[1]["failure_reason"]


def test_resource_fingerprint_ignores_runtime_event_order() -> None:
    config = load_config("configs/default.yaml", "dynamic_heterogeneous")
    config["simulation"]["seed"] = 2

    first = build_resource_fingerprint(config)
    second = build_resource_fingerprint(config)

    assert first == second
    assert len(first["epoch0_rates"]) == 8
    assert first["dynamic_edge_compute"] == {
        "enabled": True,
        "resample_every_completed_requests": 5,
    }
    assert first["block_probabilities"] == {
        "low_end": 0.2,
        "mid_end": 0.2,
        "high_end": 0.2,
    }
    assert "transition_times" not in first
    assert "network_events" not in first


def test_prepare_matrix_inputs_creates_provenance_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_runner = SimpleNamespace(prompt_token_count=lambda prompt: len(prompt))
    workload = [
        WorkloadItem(
            prompt_id=str(index),
            prompt="x",
            prompt_token_count=1,
            category="qa",
            category_group="QA",
        )
        for index in range(80)
    ]
    monkeypatch.setattr(
        "scripts.run_baseline_performance_eval.build_model_runner",
        lambda *args, **kwargs: fake_runner,
    )
    monkeypatch.setattr(
        "scripts.run_baseline_performance_eval.audit_experiment_config",
        lambda *args, **kwargs: {"schema_version": 1},
    )
    monkeypatch.setattr(
        "scripts.run_baseline_performance_eval.load_workload",
        lambda *args, **kwargs: workload,
    )

    prepared = prepare_matrix_inputs(
        root=tmp_path,
        config_path="configs/default.yaml",
        dataset_path="unused.jsonl",
    )

    assert set(prepared) == set(SEEDS)
    assert all(config_path.is_file() for config_path, _ in prepared.values())
    assert all(trace_path.is_file() for _, trace_path in prepared.values())
    assert all(
        (tmp_path / SCENARIO / str(seed) / "_raw").is_dir() for seed in SEEDS
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_complete_synthetic_matrix(
    root: Path,
    *,
    blank_device_ids_for: set[str] | None = None,
) -> None:
    blank = blank_device_ids_for or set()
    initialize_runs_csv(root)
    for seed in SEEDS:
        config = load_config("configs/default.yaml", SCENARIO)
        config["simulation"]["seed"] = seed
        config["simulation"]["output_len_choices"] = [2]
        workload = [
            WorkloadItem(
                prompt_id=f"prompt-{index}",
                prompt=f"prompt text {index}",
                prompt_token_count=3,
                category="qa",
                category_group="QA",
            )
            for index in range(80)
        ]
        trace_path = root / "_workloads" / f"{SCENARIO}_seed_{seed}.jsonl"
        trace_hash = materialize_shared_trace(config, workload, trace_path)
        shared = load_shared_trace(trace_path, config)
        fingerprint = build_resource_fingerprint(config)
        for method in METHODS:
            directory = root / SCENARIO / str(seed) / method
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "run_status.json").write_text(
                json.dumps(
                    {
                        "scenario": SCENARIO,
                        "seed": seed,
                        "method": method,
                        "success": True,
                        "return_code": 0,
                        "failure_reason": "",
                        "shared_trace_path": str(trace_path),
                        "shared_trace_sha256": trace_hash,
                        "resource_fingerprint": fingerprint,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            (directory / "resolved_config.json").write_text(
                json.dumps(
                    {"method": method, "scenario": SCENARIO, "config": config}
                ),
                encoding="utf-8",
            )
            metric = {field: 1.0 for field in MAIN_FIELDS}
            metric.update(method=method, scenario=SCENARIO, num_requests=80)
            write_csv(directory / "metrics.csv", [metric], MAIN_FIELDS)
            requests = []
            tokens = []
            for row in shared:
                requests.append(
                    {
                        "method": method,
                        "scenario": SCENARIO,
                        "request_id": row.request_id,
                        "device_id": "" if method in blank else row.device_id,
                        "prompt_id": row.prompt_id,
                        "arrival_time_ms": row.arrival_time_ms,
                        "decode_ready_time_ms": row.decode_ready_time_ms,
                        "finish_time_ms": row.arrival_time_ms + 1.0,
                        "latency_ms": 1.0,
                        "output_len": row.output_len,
                        "generated_tokens": row.output_len,
                        "generated_ids": list(range(row.output_len)),
                        "status": "finished",
                        "in_flight_segment_count": 0,
                        "pending_segment_count": 0,
                        "completed_result_count": 0,
                        "draft_queued": False,
                        "proactive_pending_count": 0,
                    }
                )
                tokens.extend(
                    {
                        "method": method,
                        "scenario": SCENARIO,
                        "token_type": "committed",
                        "request_id": row.request_id,
                        "device_id": row.device_id,
                        "position": position,
                        "token_index": position,
                        "token_id": position,
                        "commit_time_ms": row.arrival_time_ms + position + 1.0,
                        "count": 1,
                        "status": "finished",
                        "source_event": "final_output",
                    }
                    for position in range(row.output_len)
                )
            write_csv(
                directory / "request_trace.csv",
                requests,
                REQUEST_TRACE_FIELDS,
            )
            write_csv(directory / "event_trace.csv", [], EVENT_TRACE_FIELDS)
            write_csv(directory / "token_trace.csv", tokens, TOKEN_TRACE_FIELDS)
            write_csv(
                directory / "resource_timeline.csv",
                [],
                RESOURCE_TIMELINE_FIELDS,
            )
            write_csv(directory / "batch_trace.csv", [], BATCH_TRACE_FIELDS)
            system = {field: 0.0 for field in SYSTEM_FIELDS}
            system.update(method=method, scenario=SCENARIO)
            write_csv(directory / "system_metrics.csv", [system], SYSTEM_FIELDS)
            (directory / "stdout.log").write_text(
                "synthetic success\n",
                encoding="utf-8",
            )


def _replace_one_committed_token(path: Path) -> None:
    rows = _read_csv(path)
    rows[0]["token_id"] = str(int(rows[0]["token_id"]) + 1000)
    write_csv(path, rows, TOKEN_TRACE_FIELDS)


def test_missing_trace_retains_complete_matrix_and_fails(tmp_path: Path) -> None:
    initialize_runs_csv(tmp_path)
    failures = summarize_results(tmp_path)
    rows = _read_csv(tmp_path / "runs.csv")

    assert len(rows) == 30
    assert all(row["success"] == "False" for row in rows)
    assert any("missing" in row["failure_reason"] for row in rows)
    assert failures


def test_missing_output_device_id_does_not_fail_input_mapping(
    tmp_path: Path,
) -> None:
    _write_complete_synthetic_matrix(
        tmp_path,
        blank_device_ids_for={
            "target_only",
            "server_only_linear",
            "server_only_tree",
        },
    )

    failures = summarize_results(tmp_path)

    assert failures == []


def test_committed_token_mismatch_marks_cell_failed(tmp_path: Path) -> None:
    _write_complete_synthetic_matrix(tmp_path)
    _replace_one_committed_token(
        tmp_path
        / SCENARIO
        / "0"
        / "specedge_linear"
        / "token_trace.csv"
    )

    failures = summarize_results(tmp_path)
    rows = _read_csv(tmp_path / "runs.csv")
    failed = next(
        row
        for row in rows
        if row["seed"] == "0" and row["method"] == "specedge_linear"
    )

    assert failed["success"] == "False"
    assert "committed token trace differs from target_only" in failed[
        "failure_reason"
    ]
    assert failures
