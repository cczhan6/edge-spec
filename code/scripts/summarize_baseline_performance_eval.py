from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any, Iterable

from src.metrics import write_csv


SCENARIO = "dynamic_heterogeneous"
SEEDS = (0, 1, 2, 3, 4)
METHODS = (
    "target_only",
    "server_only_linear",
    "server_only_tree",
    "specedge_linear",
    "specedge_tree",
    "dip_sd",
)
METRIC_SCOPE = "decode_only"
PERFORMANCE_FIELDS = (
    "avg_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "avg_tpot_ms",
    "avg_tbt_ms",
    "makespan_ms",
    "goodput_tok_s",
    "avg_acceptance_rate",
    "avg_selected_gamma",
)
RUN_FIELDS = (
    "scenario",
    "seed",
    "method",
    "metric_scope",
    "num_requests",
    "committed_tokens",
    "success",
    "failure_reason",
    *PERFORMANCE_FIELDS,
)
SUMMARY_FIELDS = (
    "scenario",
    "method",
    "metric_scope",
    "num_runs",
    "successful_runs",
    "success",
    *(
        field
        for metric in PERFORMANCE_FIELDS
        for field in (f"{metric}_mean", f"{metric}_std")
    ),
)


def initialize_runs_csv(root: str | Path) -> Path:
    output = Path(root) / "runs.csv"
    rows = []
    for seed in SEEDS:
        for method in METHODS:
            row: dict[str, Any] = {field: "" for field in RUN_FIELDS}
            row.update(
                scenario=SCENARIO,
                seed=seed,
                method=method,
                metric_scope=METRIC_SCOPE,
                success=False,
                failure_reason="not run",
            )
            rows.append(row)
    write_csv(output, rows, list(RUN_FIELDS))
    return output


def aggregate_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    materialized = list(rows)
    summaries = []
    for method in METHODS:
        selected = [row for row in materialized if row["method"] == method]
        successful = [row for row in selected if _as_bool(row["success"])]
        complete = len(selected) == len(SEEDS) and len(successful) == len(SEEDS)
        summary: dict[str, Any] = {
            "scenario": SCENARIO,
            "method": method,
            "metric_scope": METRIC_SCOPE,
            "num_runs": len(selected),
            "successful_runs": len(successful),
            "success": complete,
        }
        for metric in PERFORMANCE_FIELDS:
            values = [float(row[metric]) for row in successful] if complete else []
            summary[f"{metric}_mean"] = statistics.mean(values) if values else ""
            summary[f"{metric}_std"] = statistics.stdev(values) if values else ""
        summaries.append(summary)
    return summaries


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and summarize canonical baseline performance runs."
    )
    parser.add_argument("--root", default="outputs/baseline_performance_eval")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    initialize_runs_csv(args.root)


if __name__ == "__main__":
    main()
