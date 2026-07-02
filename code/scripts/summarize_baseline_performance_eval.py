from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml

from scripts.baseline_trace import (
    _check_event_monotonicity,
    _check_no_pending_state,
    _check_resource_overlap,
    _committed_tokens,
)
from src.metrics import write_csv
from src.config import load_config


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
REQUIRED_CELL_FILES = (
    "stdout.log",
    "run_status.json",
    "resolved_config.json",
    "metrics.csv",
    "request_trace.csv",
    "event_trace.csv",
    "token_trace.csv",
    "resource_timeline.csv",
    "batch_trace.csv",
    "system_metrics.csv",
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


def summarize_results(root: str | Path) -> list[str]:
    from scripts.run_baseline_performance_eval import build_resource_fingerprint

    output_root = Path(root)
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    loaded: dict[tuple[int, str], dict[str, Any]] = {}
    fingerprints: dict[int, dict[str, Any]] = {}
    row_failures: dict[tuple[int, str], list[str]] = defaultdict(list)

    for seed in SEEDS:
        config_path = (
            output_root / "_configs" / f"{SCENARIO}_seed_{seed}.yaml"
        )
        expected_fingerprint: dict[str, Any] | None = None
        seed_config_failure = ""
        try:
            expected_fingerprint = build_resource_fingerprint(
                load_config(config_path)
            )
        except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
            seed_config_failure = f"seed config cannot be loaded: {exc}"
        trace_path = (
            output_root / "_workloads" / f"{SCENARIO}_seed_{seed}.jsonl"
        )
        shared_rows = _read_jsonl(trace_path) if trace_path.is_file() else []
        for method in METHODS:
            key = (seed, method)
            row: dict[str, Any] = {field: "" for field in RUN_FIELDS}
            row.update(
                scenario=SCENARIO,
                seed=seed,
                method=method,
                metric_scope=METRIC_SCOPE,
                success=False,
            )
            if seed_config_failure:
                row_failures[key].append(seed_config_failure)
            directory = output_root / SCENARIO / str(seed) / method
            missing = [
                name for name in REQUIRED_CELL_FILES if not (directory / name).is_file()
            ]
            if not trace_path.is_file():
                missing.append(str(trace_path.relative_to(output_root)))
            if missing:
                row_failures[key].append("missing files: " + ", ".join(missing))
                rows.append(row)
                continue
            try:
                data = _load_cell(directory)
                loaded[key] = data
                local = _validate_cell(
                    data,
                    shared_rows=shared_rows,
                    shared_trace_path=trace_path,
                    seed=seed,
                    method=method,
                    expected_fingerprint=expected_fingerprint,
                )
                row_failures[key].extend(local)
                metric = data["metrics"][0]
                row["num_requests"] = len(data["requests"])
                row["committed_tokens"] = _sum_committed(data["tokens"])
                for field in PERFORMANCE_FIELDS:
                    row[field] = metric.get(field, "")
                fingerprint = data["status"].get("resource_fingerprint")
                if isinstance(fingerprint, dict):
                    reference = fingerprints.setdefault(seed, fingerprint)
                    if fingerprint != reference:
                        row_failures[key].append(
                            "resource fingerprint differs across methods"
                        )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                row_failures[key].append(f"invalid trace bundle: {exc}")
            rows.append(row)

    by_identity = {(int(row["seed"]), str(row["method"])): row for row in rows}
    for seed in SEEDS:
        target = loaded.get((seed, "target_only"))
        if target is None:
            continue
        expected_tokens = _committed_tokens(target["tokens"])
        for method in METHODS:
            data = loaded.get((seed, method))
            if data is None:
                continue
            if _committed_tokens(data["tokens"]) != expected_tokens:
                row_failures[(seed, method)].append(
                    "committed token trace differs from target_only"
                )

    for key, row in by_identity.items():
        messages = list(dict.fromkeys(row_failures[key]))
        row["success"] = not messages
        row["failure_reason"] = "; ".join(messages)
        failures.extend(
            f"{SCENARIO}/{key[0]}/{key[1]}: {message}" for message in messages
        )

    ordered_rows = [by_identity[(seed, method)] for seed in SEEDS for method in METHODS]
    write_csv(output_root / "runs.csv", ordered_rows, list(RUN_FIELDS))
    write_csv(
        output_root / "summary.csv",
        aggregate_rows(ordered_rows),
        list(SUMMARY_FIELDS),
    )
    return failures


def _load_cell(directory: Path) -> dict[str, Any]:
    return {
        "status": _read_json(directory / "run_status.json"),
        "resolved": _read_json(directory / "resolved_config.json"),
        "metrics": _read_csv(directory / "metrics.csv"),
        "requests": _read_csv(directory / "request_trace.csv"),
        "events": _read_csv(directory / "event_trace.csv"),
        "tokens": _read_csv(directory / "token_trace.csv"),
        "resources": _read_csv(directory / "resource_timeline.csv"),
    }


def _validate_cell(
    data: dict[str, Any],
    *,
    shared_rows: list[dict[str, Any]],
    shared_trace_path: Path,
    seed: int,
    method: str,
    expected_fingerprint: dict[str, Any] | None,
) -> list[str]:
    failures = []
    status = data["status"]
    resolved = data["resolved"]
    config = resolved.get("config", resolved)
    if not _as_bool(status.get("success")):
        failures.append(
            "worker failed: " + str(status.get("failure_reason") or "unknown failure")
        )
    if int(status.get("return_code", -1)) != 0:
        failures.append(f"worker return code is {status.get('return_code')}")
    if status.get("scenario") != SCENARIO or status.get("method") != method:
        failures.append("run status scenario/method mismatch")
    if int(status.get("seed", -1)) != seed:
        failures.append("run status seed mismatch")
    if resolved.get("scenario") != SCENARIO or resolved.get("method") != method:
        failures.append("resolved config scenario/method mismatch")

    simulation = config["simulation"]
    if int(simulation["seed"]) != seed:
        failures.append("resolved seed mismatch")
    if int(simulation["num_requests"]) != 80:
        failures.append("resolved num_requests is not 80")
    if config.get("dynamic_edge_compute") != {
        "enabled": True,
        "resample_every_completed_requests": 5,
    }:
        failures.append("dynamic edge compute contract mismatch")
    templates = config["device_pools"]["heterogeneous"]["templates"]
    populated = {name: values for name, values in templates.items() if int(values["count"]) > 0}
    if set(populated) != {"low_end", "mid_end", "high_end"}:
        failures.append("heterogeneous device composition mismatch")
    if any(float(values.get("block_probability", -1)) != 0.2 for values in populated.values()):
        failures.append("device block_probability is not uniformly 0.2")
    target = config.get("target_latency", {})
    if target.get("mode") != "profile" or target.get("metric") != "p50_ms":
        failures.append("target latency profile mode/metric mismatch")
    if target.get("profile_path") != (
        "outputs/profiling/target_verification_latency_full_merged.csv"
    ):
        failures.append("target latency profile path mismatch")

    expected_hash = hashlib.sha256(shared_trace_path.read_bytes()).hexdigest()
    if status.get("shared_trace_sha256") != expected_hash:
        failures.append("shared trace SHA-256 mismatch")
    recorded_path = status.get("shared_trace_path")
    if not recorded_path or Path(recorded_path).resolve() != shared_trace_path.resolve():
        failures.append("shared trace path mismatch")
    fingerprint = status.get("resource_fingerprint")
    if not isinstance(fingerprint, dict):
        failures.append("resource fingerprint is missing")
    elif expected_fingerprint is not None and fingerprint != expected_fingerprint:
        failures.append("resource fingerprint does not match seed config")

    metrics = data["metrics"]
    if len(metrics) != 1:
        failures.append(f"metrics.csv has {len(metrics)} rows, expected 1")
    else:
        for field in PERFORMANCE_FIELDS:
            try:
                value = float(metrics[0][field])
                if not math.isfinite(value) or value < 0:
                    raise ValueError("must be finite and non-negative")
            except (KeyError, TypeError, ValueError) as exc:
                failures.append(f"metric {field} is invalid: {exc}")

    requests = data["requests"]
    if len(requests) != 80:
        failures.append(f"request count is {len(requests)}, expected 80")
    if len(shared_rows) != 80:
        failures.append(f"shared trace has {len(shared_rows)} rows, expected 80")
    for expected, observed in zip(shared_rows, requests):
        request_id = str(expected["request_id"])
        if observed.get("request_id") != request_id:
            failures.append(f"request {request_id} ID mismatch")
            continue
        for field in ("prompt_id", "output_len"):
            if str(observed.get(field)) != str(expected[field]):
                failures.append(f"request {request_id} {field} mismatch")
        for field in ("arrival_time_ms", "decode_ready_time_ms"):
            if float(observed.get(field, "nan")) != float(expected[field]):
                failures.append(f"request {request_id} {field} mismatch")
        if observed.get("status") != "finished":
            failures.append(f"request {request_id} is not finished")
        if int(observed.get("generated_tokens", -1)) != int(expected["output_len"]):
            failures.append(f"request {request_id} generated token count mismatch")
    failures.extend(_check_no_pending_state(method, requests))
    failures.extend(_check_event_monotonicity(method, data["events"]))
    failures.extend(_check_resource_overlap(method, data["resources"]))
    return failures


def _sum_committed(rows: list[dict[str, str]]) -> int:
    return sum(
        int(row.get("count") or 1)
        for row in rows
        if row.get("token_type") == "committed"
    )


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
    failures = summarize_results(args.root)
    if failures:
        print("baseline performance evaluation validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
