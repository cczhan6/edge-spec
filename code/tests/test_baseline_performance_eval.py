from __future__ import annotations

import csv
import math
from pathlib import Path

from scripts.summarize_baseline_performance_eval import (
    METHODS,
    PERFORMANCE_FIELDS,
    SEEDS,
    aggregate_rows,
    initialize_runs_csv,
)
from src.config import load_config


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
