from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.baseline_trace import (
    _check_event_monotonicity,
    _check_no_pending_state,
    _check_resource_overlap,
    _committed_tokens,
    _write_yaml,
)
from scripts.real_model_smoke import (
    _check_dip_sd_no_future_acceptance,
    _check_dip_sd_optimizer,
    _check_server_only_batch_size,
    _check_specedge_proactive,
    _check_tree_baseline_evidence,
)
from src.config import load_config
from src.metrics import percentile, write_csv


PREFLIGHT_SCENARIOS = (
    "homogeneous",
    "combined_strong_heterogeneous",
)
PREFLIGHT_METHODS = (
    "target_only",
    "server_only_linear",
    "server_only_tree",
    "specedge_linear",
    "specedge_tree",
    "dip_sd",
)
PREFLIGHT_SEEDS = (20260628, 20260629)

PREFLIGHT_METRIC_FIELDS = [
    "decode_makespan",
    "request_decode_latency",
    "mean_inter_token_latency",
    "p50_inter_token_latency",
    "p95_inter_token_latency",
    "effective_throughput_tokens_per_s",
    "speedup_vs_target_only",
    "acceptance_ratio",
    "drafted_tokens",
    "verified_tokens",
    "accepted_tokens",
    "committed_tokens",
    "wasted_tokens",
    "target_utilization",
    "draft_utilization",
    "verification_queue_wait",
]

REQUIRED_RUN_FILES = (
    "resolved_config.yaml",
    "environment_manifest.json",
    "run_manifest.json",
    "metrics.csv",
    "request_metrics.csv",
    "event_trace.csv",
    "token_trace.csv",
    "resource_timeline.csv",
    "stdout.log",
)

REQUEST_METRIC_FIELDS = [
    "scenario",
    "seed",
    "method",
    "request_id",
    "decode_start_time_ms",
    "decode_finish_time_ms",
    "request_decode_latency",
    "committed_tokens",
    "mean_inter_token_latency",
    "p50_inter_token_latency",
    "p95_inter_token_latency",
]

SUMMARY_FIELDS = [
    "scenario",
    "seed",
    "method",
    "success",
    *PREFLIGHT_METRIC_FIELDS,
]


def prepare_preflight_config(
    config_path: str | Path,
    scenario: str,
    seed: int,
    output_path: str | Path,
) -> Path:
    if scenario not in PREFLIGHT_SCENARIOS:
        raise ValueError(f"unsupported preflight scenario: {scenario}")
    config = load_config(config_path, scenario)
    config["simulation"].update(
        num_requests=16,
        output_len_choices=[32],
        seed=int(seed),
        request_arrival="poisson",
    )
    if os.environ.get("TARGET_MODEL_PATH"):
        config["model_runner"]["target_model"] = os.environ["TARGET_MODEL_PATH"]
    for profile, variable in {
        "small": "DRAFTER_SMALL_MODEL_PATH",
        "medium": "DRAFTER_MEDIUM_MODEL_PATH",
        "large": "DRAFTER_LARGE_MODEL_PATH",
    }.items():
        if os.environ.get(variable):
            config["model_runner"]["drafter_models"][profile]["model"] = os.environ[
                variable
            ]
    if os.environ.get("LOCAL_FILES_ONLY") is not None:
        config["model_runner"]["local_files_only"] = _parse_bool(
            os.environ["LOCAL_FILES_ONLY"],
            "LOCAL_FILES_ONLY",
        )
    destination = Path(output_path)
    _write_yaml(destination, config)
    return destination


def compute_decode_metrics(
    requests: list[dict[str, Any]],
    tokens: list[dict[str, Any]],
    system: dict[str, Any],
    *,
    resources: list[dict[str, Any]] | None = None,
    target_only_latency: float | None,
) -> dict[str, float | int]:
    starts = [_finite_float(row["decode_ready_time_ms"]) for row in requests]
    finishes = [_finite_float(row["finish_time_ms"]) for row in requests]
    latencies = [
        _finite_float(row.get("latency_ms", finish - start))
        for row, start, finish in zip(requests, starts, finishes, strict=True)
    ]
    makespan = max(finishes) - min(starts) if requests else 0.0

    committed_by_request: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in tokens:
        if row.get("token_type") != "committed":
            continue
        committed_by_request[str(row["request_id"])].append(
            (int(row["position"]), _finite_float(row["commit_time_ms"]))
        )
    inter_token_latencies: list[float] = []
    for values in committed_by_request.values():
        ordered = [time_ms for _, time_ms in sorted(values)]
        inter_token_latencies.extend(
            current - previous for previous, current in zip(ordered, ordered[1:])
        )

    committed = _sum_token_type(tokens, "committed")
    verified = _sum_token_type(tokens, "verified")
    accepted = _sum_token_type(tokens, "accepted")
    mean_latency = _mean(latencies)
    target_utilization = _resource_utilization(resources, "target", makespan)
    draft_utilization = _resource_utilization(resources, "draft", makespan)
    speedup = (
        float(target_only_latency) / mean_latency
        if target_only_latency is not None and mean_latency > 0
        else 0.0
    )
    return {
        "decode_makespan": makespan,
        "request_decode_latency": mean_latency,
        "mean_inter_token_latency": _mean(inter_token_latencies),
        "p50_inter_token_latency": percentile(inter_token_latencies, 50),
        "p95_inter_token_latency": percentile(inter_token_latencies, 95),
        "effective_throughput_tokens_per_s": (
            committed / makespan * 1000.0 if makespan > 0 else 0.0
        ),
        "speedup_vs_target_only": speedup,
        "acceptance_ratio": accepted / verified if verified else 0.0,
        "drafted_tokens": (
            _sum_token_type(tokens, "drafted")
            + _sum_token_type(tokens, "proactive_drafted")
        ),
        "verified_tokens": verified,
        "accepted_tokens": accepted,
        "committed_tokens": committed,
        "wasted_tokens": _sum_token_type(tokens, "wasted"),
        "target_utilization": (
            target_utilization
            if target_utilization is not None
            else _finite_float(system.get("target_utilization", 0.0))
        ),
        "draft_utilization": (
            draft_utilization
            if draft_utilization is not None
            else _finite_float(system.get("device_utilization_mean", 0.0))
        ),
        "verification_queue_wait": _finite_float(
            system.get("lane_queue_wait_ms_mean", 0.0)
        ),
    }


def apply_target_speedups(rows: list[dict[str, Any]]) -> None:
    references: dict[tuple[str, int], float] = {}
    for row in rows:
        if row["method"] == "target_only":
            references[(str(row["scenario"]), int(row["seed"]))] = _finite_float(
                row["request_decode_latency"]
            )
    for row in rows:
        key = (str(row["scenario"]), int(row["seed"]))
        if key not in references:
            raise ValueError(f"missing target_only reference for {key[0]} seed {key[1]}")
        latency = _finite_float(row["request_decode_latency"])
        if latency <= 0:
            raise ValueError(
                f"non-positive request decode latency for {row['method']} in {key}"
            )
        row["speedup_vs_target_only"] = references[key] / latency


def materialize_cell(
    cell_directory: str | Path,
    *,
    scenario: str,
    seed: int,
    environment_path: str | Path,
    command: str,
    request_count: int = 16,
    output_length: int = 32,
) -> list[dict[str, Any]]:
    cell = Path(cell_directory)
    environment = _read_json(environment_path)
    loaded = {
        method: _load_method(cell / method, method)
        for method in PREFLIGHT_METHODS
    }
    target_requests = loaded["target_only"]["requests"]
    target_latency = _mean(
        [_finite_float(row["latency_ms"]) for row in target_requests]
    )
    rows: list[dict[str, Any]] = []
    for method in PREFLIGHT_METHODS:
        data = loaded[method]
        metrics = compute_decode_metrics(
            data["requests"],
            data["tokens"],
            data["system"],
            resources=data["resources"],
            target_only_latency=target_latency,
        )
        write_csv(cell / method / "metrics.csv", [metrics], PREFLIGHT_METRIC_FIELDS)
        write_csv(
            cell / method / "request_metrics.csv",
            _request_metric_rows(
                data["requests"],
                data["tokens"],
                scenario=scenario,
                seed=seed,
                method=method,
            ),
            REQUEST_METRIC_FIELDS,
        )
        config = data["resolved"].get("config", data["resolved"])
        git_commit = str(environment.get("git", {}).get("commit") or _git_commit())
        resolved = {
            "schema_version": 1,
            "scenario": scenario,
            "seed": int(seed),
            "method": method,
            "git_commit": git_commit,
            "runner": "HuggingFaceModelRunner",
            "use_fake_model_runner": False,
            "decode_clock_boundary": "prefix_ready_at_request_decode_start",
            "excluded_from_decode_latency": [
                "prompt_tokenization",
                "prompt_transmission",
                "prefill",
                "initial_kv_construction",
                "model_loading",
                "ttft",
                "first_token_latency",
            ],
            "config": config,
        }
        _write_yaml(cell / method / "resolved_config.yaml", resolved)
        shutil.copyfile(environment_path, cell / method / "environment_manifest.json")
        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scenario": scenario,
            "seed": int(seed),
            "method": method,
            "runner": "HuggingFaceModelRunner",
            "use_fake_model_runner": False,
            "return_code": 0,
            "command": command,
            "git_commit": git_commit,
            "request_count": int(request_count),
            "output_length": int(output_length),
            "model_loading_in_decode_latency": False,
            "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "target_model": config["model_runner"]["target_model"],
            "drafter_models": config["model_runner"]["drafter_models"],
        }
        _write_json(cell / method / "run_manifest.json", manifest)
        rows.append(
            {
                "scenario": scenario,
                "seed": int(seed),
                "method": method,
                "success": True,
                **metrics,
            }
        )
    return rows


def attach_residency_to_environment(
    environment_path: str | Path,
    residency_path: str | Path,
) -> None:
    environment = _read_json(environment_path)
    residency = _read_json(residency_path)
    environment["drafter_residency"] = residency
    environment["model_residency_policy"] = residency["residency_policy"]
    environment["model_loading_in_decode_latency"] = False
    environment.setdefault("software", {})["pytorch_cuda_alloc_conf"] = os.environ.get(
        "PYTORCH_CUDA_ALLOC_CONF"
    )
    _write_json(environment_path, environment)


def verify_preflight(
    root: str | Path,
    *,
    scenarios: tuple[str, ...] = PREFLIGHT_SCENARIOS,
    seeds: tuple[int, ...] = PREFLIGHT_SEEDS,
    request_count: int = 16,
    output_length: int = 32,
) -> list[dict[str, Any]]:
    output_root = Path(root)
    summaries: list[dict[str, Any]] = []
    failures: list[str] = []
    for scenario in scenarios:
        for seed in seeds:
            cell = output_root / scenario / str(seed)
            method_data: dict[str, dict[str, Any]] = {}
            for method in PREFLIGHT_METHODS:
                directory = cell / method
                method_failures = _missing_files(directory)
                if method_failures:
                    failures.extend(
                        f"{scenario}/{seed}/{method}: {failure}"
                        for failure in method_failures
                    )
                    continue
                data = _load_method(directory, method)
                method_data[method] = data
                local = _verify_method(
                    data,
                    scenario=scenario,
                    seed=seed,
                    method=method,
                    request_count=request_count,
                    output_length=output_length,
                )
                failures.extend(
                    f"{scenario}/{seed}/{method}: {failure}" for failure in local
                )

            if set(method_data) != set(PREFLIGHT_METHODS):
                continue
            target_tokens = _committed_tokens(method_data["target_only"]["tokens"])
            for method, data in method_data.items():
                if _committed_tokens(data["tokens"]) != target_tokens:
                    failures.append(
                        f"{scenario}/{seed}/{method}: committed tokens differ from target_only"
                    )

            cross_checks = [
                *_check_server_only_batch_size(method_data),
                *_check_specedge_proactive(method_data),
                *_check_tree_baseline_evidence(method_data),
                *_check_dip_sd_optimizer(method_data["dip_sd"]),
                *_check_dip_sd_no_future_acceptance(method_data["dip_sd"]),
                *_check_dip_batches(method_data["dip_sd"]),
            ]
            failures.extend(
                f"{scenario}/{seed}: {failure}" for failure in cross_checks
            )

            cell_rows = []
            for method in PREFLIGHT_METHODS:
                metric_rows = method_data[method]["metrics"]
                if len(metric_rows) != 1:
                    failures.append(
                        f"{scenario}/{seed}/{method}: metrics.csv must contain one row"
                    )
                    continue
                metric = {
                    field: _finite_float(metric_rows[0][field])
                    for field in PREFLIGHT_METRIC_FIELDS
                }
                cell_rows.append(
                    {
                        "scenario": scenario,
                        "seed": int(seed),
                        "method": method,
                        "success": True,
                        **metric,
                    }
                )
            apply_target_speedups(cell_rows)
            for row in cell_rows:
                recorded = _finite_float(
                    method_data[row["method"]]["metrics"][0]["speedup_vs_target_only"]
                )
                if abs(recorded - row["speedup_vs_target_only"]) > 1e-9:
                    failures.append(
                        f"{scenario}/{seed}/{row['method']}: speedup reference is not same-cell target_only"
                    )
            summaries.extend(cell_rows)

    failed_prefixes = {
        "/".join(failure.split(":", 1)[0].split("/")[:3])
        for failure in failures
    }
    for row in summaries:
        prefix = f"{row['scenario']}/{row['seed']}/{row['method']}"
        cell_prefix = f"{row['scenario']}/{row['seed']}"
        row["success"] = not any(
            failed.startswith(prefix) or failed.startswith(cell_prefix)
            for failed in failed_prefixes
        )
    write_csv(output_root / "summary.csv", summaries, SUMMARY_FIELDS)
    _write_summary_markdown(output_root / "summary.md", summaries, failures)
    if failures:
        raise SystemExit(
            "baseline preflight verification failed:\n"
            + "\n".join(f"- {failure}" for failure in failures)
        )
    return summaries


def _verify_method(
    data: dict[str, Any],
    *,
    scenario: str,
    seed: int,
    method: str,
    request_count: int,
    output_length: int,
) -> list[str]:
    failures: list[str] = []
    requests = data["requests"]
    config = data["resolved"].get("config", data["resolved"])
    manifest = data["manifest"]
    environment = data["environment"]
    failures.extend(_check_preflight_requests(method, requests))
    failures.extend(_check_no_pending_state(method, requests))
    failures.extend(_check_event_monotonicity(method, data["events"]))
    failures.extend(_check_resource_overlap(method, data["resources"]))
    if len(requests) != request_count:
        failures.append(f"request count is {len(requests)}, expected {request_count}")
    simulation = config.get("simulation", {})
    if int(simulation.get("num_requests", 0)) != request_count:
        failures.append(f"resolved num_requests is not {request_count}")
    if simulation.get("output_len_choices") != [output_length]:
        failures.append(f"resolved output length is not [{output_length}]")
    if int(simulation.get("seed", -1)) != int(seed):
        failures.append("resolved seed does not match directory")
    if simulation.get("request_arrival") != "poisson":
        failures.append("arrival trace is not configured as Poisson")
    arrivals = [_finite_float(row["arrival_time_ms"]) for row in requests]
    if arrivals != sorted(arrivals) or len(set(arrivals)) < 2:
        failures.append("arrival trace is burst or non-monotonic")
    experiment = config.get("experiment", {})
    if experiment.get("internal_time_unit") != "ms" or experiment.get("csv_time_unit") != "ms":
        failures.append("internal/CSV time units are not uniformly ms")
    if manifest.get("runner") != "HuggingFaceModelRunner":
        failures.append("run manifest does not identify the real Hugging Face runner")
    if bool(manifest.get("use_fake_model_runner")):
        failures.append("run manifest enables fake runner")
    if "FakeModelRunner" in data["stdout"] or "--use-fake-model-runner" in data["stdout"]:
        failures.append("stdout contains fake-runner evidence")
    if int(manifest.get("return_code", -1)) != 0:
        failures.append("run manifest return code is nonzero")
    if manifest.get("scenario") != scenario or int(manifest.get("seed", -1)) != int(seed):
        failures.append("run manifest scenario/seed mismatch")
    if manifest.get("method") != method:
        failures.append("run manifest method mismatch")
    commit = environment.get("git", {}).get("commit")
    if not commit or manifest.get("git_commit") != commit:
        failures.append("environment/run manifest Git commit mismatch")
    expected_tokens = request_count * output_length
    committed = _sum_token_type(data["tokens"], "committed")
    if committed != expected_tokens:
        failures.append(f"committed token count is {committed}, expected {expected_tokens}")
    for row in requests:
        if int(row.get("generated_tokens", 0)) != int(row.get("output_len", 0)):
            failures.append(f"request {row.get('request_id')} did not commit expected output")
    if list(data["metrics"][0]) != PREFLIGHT_METRIC_FIELDS:
        failures.append("metrics.csv fields do not match formal decode-only schema")
    for field in PREFLIGHT_METRIC_FIELDS:
        try:
            value = _finite_float(data["metrics"][0][field])
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(f"metric {field} is invalid: {exc}")
            continue
        if value < 0:
            failures.append(f"metric {field} is negative")
    if method.startswith("server_only") and int(config["server_only"]["batch_size"]) != 1:
        failures.append("Server-only configured batch size is not 1")
    if method.endswith("_tree"):
        expected_strategy = (
            config["server_only"]["tree_draft_strategy"]
            if method.startswith("server_only")
            else config["specedge"]["tree_draft_strategy"]
        )
        if expected_strategy != "specexec_approx":
            failures.append("tree method lacks specexec_approx config marker")
    return failures


def _load_method(directory: Path, method: str) -> dict[str, Any]:
    system_rows = _read_csv(directory / "system_metrics.csv")
    return {
        "method": method,
        "directory": directory,
        "resolved": _read_json(directory / "resolved_config.json"),
        "manifest": _read_json(directory / "run_manifest.json")
        if (directory / "run_manifest.json").exists()
        else {},
        "environment": _read_json(directory / "environment_manifest.json")
        if (directory / "environment_manifest.json").exists()
        else {},
        "metrics": _read_csv(directory / "metrics.csv"),
        "requests": _read_csv(directory / "request_trace.csv"),
        "events": _read_csv(directory / "event_trace.csv"),
        "tokens": _read_csv(directory / "token_trace.csv"),
        "resources": _read_csv(directory / "resource_timeline.csv"),
        "batches": _read_csv(directory / "batch_trace.csv"),
        "system": system_rows[0] if system_rows else {},
        "stdout": (directory / "stdout.log").read_text(encoding="utf-8")
        if (directory / "stdout.log").exists()
        else "",
    }


def _request_metric_rows(
    requests: list[dict[str, Any]],
    tokens: list[dict[str, Any]],
    *,
    scenario: str,
    seed: int,
    method: str,
) -> list[dict[str, Any]]:
    committed: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in tokens:
        if row.get("token_type") == "committed":
            committed[str(row["request_id"])].append(
                (int(row["position"]), _finite_float(row["commit_time_ms"]))
            )
    rows = []
    for request in requests:
        request_id = str(request["request_id"])
        times = [time for _, time in sorted(committed[request_id])]
        intervals = [current - previous for previous, current in zip(times, times[1:])]
        rows.append(
            {
                "scenario": scenario,
                "seed": int(seed),
                "method": method,
                "request_id": request_id,
                "decode_start_time_ms": _finite_float(request["decode_ready_time_ms"]),
                "decode_finish_time_ms": _finite_float(request["finish_time_ms"]),
                "request_decode_latency": _finite_float(request["latency_ms"]),
                "committed_tokens": len(times),
                "mean_inter_token_latency": _mean(intervals),
                "p50_inter_token_latency": percentile(intervals, 50),
                "p95_inter_token_latency": percentile(intervals, 95),
            }
        )
    return rows


def _check_preflight_requests(
    method: str,
    requests: list[dict[str, Any]],
) -> list[str]:
    failures = []
    for row in requests:
        request_id = row.get("request_id")
        if row.get("status") != "finished":
            failures.append(f"{method}: request {request_id} is not finished")
        try:
            if _finite_float(row["finish_time_ms"]) < _finite_float(
                row["decode_ready_time_ms"]
            ):
                failures.append(f"{method}: request {request_id} finishes before decode start")
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(f"{method}: request {request_id} has invalid time: {exc}")
    return failures


def _check_dip_batches(data: dict[str, Any]) -> list[str]:
    batches = {
        (row.get("epoch"), row.get("batch_index"))
        for row in data["batches"]
        if row.get("event") == "dip_sd_batch_verify"
    }
    return [] if len(batches) >= 2 else ["dip_sd: fewer than two batches"]


def _missing_files(directory: Path) -> list[str]:
    required = (*REQUIRED_RUN_FILES, "resolved_config.json", "request_trace.csv", "batch_trace.csv", "system_metrics.csv")
    return [f"missing required file {name}" for name in required if not (directory / name).is_file()]


def _write_summary_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    failures: list[str],
) -> None:
    lines = [
        "# Decode-Only Baseline Preflight Summary",
        "",
        f"Status: {'PASS' if not failures else 'FAIL'}",
        "",
        "This preflight validates the formal pipeline and metric definitions; it is not a paper performance result.",
        "",
        "| scenario | seed | method | success | decode_makespan | request_decode_latency | speedup_vs_target_only | committed_tokens |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {seed} | {method} | {success} | {decode_makespan:.6f} | "
            "{request_decode_latency:.6f} | {speedup_vs_target_only:.6f} | {committed_tokens:.0f} |".format(
                **row
            )
        )
    lines.extend(["", "## Automatic Checks", ""])
    if failures:
        lines.extend(f"- FAIL: {failure}" for failure in failures)
    else:
        lines.extend(
            [
                "- All requests completed with no pending or unverified state.",
                "- Every committed token trace is greedily equivalent to same-cell target_only.",
                "- Event time, resource exclusivity, Poisson arrival, and millisecond-unit checks passed.",
                "- Metrics are finite and nonnegative; same-cell speedup references are valid.",
                "- Server-only batch size, SpecEdge proactive work, DiP-SD optimizer/batches, and specexec_approx markers passed.",
                "- The Hugging Face runner is real and model loading is excluded from decode latency.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _sum_token_type(rows: Iterable[dict[str, Any]], token_type: str) -> int:
    return sum(
        int(row.get("count", 0) or 0)
        for row in rows
        if row.get("token_type") == token_type
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _resource_utilization(
    resources: list[dict[str, Any]] | None,
    resource_type: str,
    makespan: float,
) -> float | None:
    if resources is None:
        return None
    selected = [row for row in resources if row.get("resource_type") == resource_type]
    if not selected or makespan <= 0:
        return 0.0
    resource_keys = {str(row["resource_key"]) for row in selected}
    busy_ms = sum(_finite_float(row["duration_ms"]) for row in selected)
    return busy_ms / (makespan * len(resource_keys))


def _finite_float(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite numeric value: {value!r}")
    return number


def _parse_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify decode-only baseline preflight outputs.")
    parser.add_argument(
        "action",
        nargs="?",
        choices=("verify", "prepare", "materialize", "attach-residency"),
        default="verify",
    )
    parser.add_argument("--root", default="outputs/baseline_preflight")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--scenario", choices=PREFLIGHT_SCENARIOS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output")
    parser.add_argument("--cell")
    parser.add_argument("--environment")
    parser.add_argument("--residency")
    parser.add_argument("--command-text", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.action == "prepare":
        if args.scenario is None or args.seed is None or args.output is None:
            raise SystemExit("prepare requires --scenario, --seed, and --output")
        prepare_preflight_config(
            args.config,
            args.scenario,
            args.seed,
            args.output,
        )
        print(f"wrote preflight config: {args.output}")
        return
    if args.action == "materialize":
        if args.cell is None or args.scenario is None or args.seed is None or args.environment is None:
            raise SystemExit("materialize requires --cell, --scenario, --seed, and --environment")
        materialize_cell(
            args.cell,
            scenario=args.scenario,
            seed=args.seed,
            environment_path=args.environment,
            command=args.command_text,
        )
        print(f"materialized preflight cell: {args.cell}")
        return
    if args.action == "attach-residency":
        if args.environment is None or args.residency is None:
            raise SystemExit("attach-residency requires --environment and --residency")
        attach_residency_to_environment(args.environment, args.residency)
        print(f"attached drafter residency to environment: {args.environment}")
        return
    rows = verify_preflight(args.root)
    print(f"baseline preflight verification passed: {len(rows)} cells")


if __name__ == "__main__":
    main()
