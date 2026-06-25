from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.baseline_trace import (
    _check_event_monotonicity,
    _check_no_pending_state,
    _check_resource_overlap,
    _committed_tokens,
    _sum_token_type,
    _to_float,
    _to_int,
    _write_yaml,
)
from src.config import load_config
from src.workload import extract_prompt


REAL_MODEL_METHODS = (
    "target_only",
    "server_only_linear",
    "specedge_linear",
    "dip_sd",
)

SCENARIO = "real_model_smoke"

REQUIRED_REAL_FILES = (
    "resolved_config",
    "resolved_config.json",
    "metrics.csv",
    "request_trace.csv",
    "event_trace.csv",
    "token_trace.csv",
    "resource_timeline.csv",
    "batch_trace.csv",
    "run_manifest.json",
    "stdout.log",
)

VERIFY_EVENTS_BY_METHOD = {
    "server_only_linear": "server_only_verify",
    "specedge_linear": "global_batch_verify",
    "dip_sd": "dip_sd_batch_verify",
}

LOG_FAILURE_PATTERNS = (
    re.compile(r"cuda out of memory", re.IGNORECASE),
    re.compile(r"\bout of memory\b", re.IGNORECASE),
    re.compile(r"\boom\b", re.IGNORECASE),
    re.compile(r"\bnan\b", re.IGNORECASE),
    re.compile(r"^Traceback ", re.MULTILINE),
)


def prepare_real_model_inputs(
    *,
    root: str | Path,
    config_path: str | Path,
    target_model: str,
    draft_model: str,
    dataset_path: str | Path | None,
    target_device: str,
    draft_device: str,
    num_requests: int = 4,
    output_tokens: int = 8,
    local_files_only: bool | None = None,
    cache_dir: str | None = None,
    revision: str | None = None,
) -> tuple[Path, Path]:
    if not target_model:
        raise ValueError("target_model must be explicit")
    if not draft_model:
        raise ValueError("draft_model must be explicit")
    if num_requests < 2 or num_requests > 4:
        raise ValueError("real model smoke num_requests must be between 2 and 4")
    if output_tokens < 8 or output_tokens > 16:
        raise ValueError("real model smoke output_tokens must be between 8 and 16")

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    _configure_simulation(config, num_requests, output_tokens)
    _configure_models(
        config,
        target_model=target_model,
        draft_model=draft_model,
        target_device=target_device,
        draft_device=draft_device,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
        revision=revision,
    )
    _configure_virtual_devices(config)
    _configure_baseline_knobs(config)

    config_out = root_path / "real_model_smoke_config.yaml"
    dataset_out = root_path / "real_model_smoke_dataset.jsonl"
    _write_yaml(config_out, config)
    _write_dataset_subset(dataset_out, dataset_path, num_requests)
    return config_out, dataset_out


def write_run_manifest(
    *,
    output_dir: str | Path,
    method: str,
    command: str,
    return_code: int,
    config_path: str | Path,
    dataset_path: str | Path,
    target_model: str,
    draft_model: str,
    target_device: str,
    draft_device: str,
    stdout_log: str | Path,
    skipped_reason: str | None = None,
    gpu_peak_mb: str | int | float | None = None,
) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": method,
        "scenario": SCENARIO,
        "runner": "HuggingFaceModelRunner",
        "use_fake_model_runner": False,
        "target_model": target_model,
        "draft_model": draft_model,
        "target_device": target_device,
        "draft_device": draft_device,
        "dataset": str(dataset_path),
        "config": str(config_path),
        "request_count": None,
        "output_tokens": None,
        "command": command,
        "return_code": int(return_code),
        "stdout_log": str(stdout_log),
        "gpu_peak_mb": gpu_peak_mb if gpu_peak_mb not in (None, "") else "n/a",
        "skipped_reason": skipped_reason or "",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    config_data = _read_resolved_or_config(directory, config_path)
    if config_data:
        simulation = config_data.get("simulation", {})
        manifest["request_count"] = simulation.get("num_requests")
        choices = simulation.get("output_len_choices")
        if isinstance(choices, list) and choices:
            manifest["output_tokens"] = choices[0] if len(set(choices)) == 1 else choices
    path = directory / "run_manifest.json"
    _write_json(path, manifest)
    return path


def verify_real_model_outputs(
    root: str | Path,
    summary_path: str | Path,
    *,
    expected_requests: int = 4,
) -> list[dict[str, Any]]:
    root_path = Path(root)
    method_data = {
        method: _load_method_outputs(root_path, method)
        for method in REAL_MODEL_METHODS
    }
    failures: list[str] = []
    method_failures: dict[str, list[str]] = {method: [] for method in REAL_MODEL_METHODS}

    for method, data in method_data.items():
        checks: list[str] = []
        checks.extend(_check_required_files(data["directory"]))
        checks.extend(_check_manifest(method, data["manifest"]))
        checks.extend(_check_stdout_log(method, data["stdout_log"]))
        checks.extend(_check_numeric_outputs(method, data["metrics"]))
        checks.extend(_check_all_requests_finished(method, data["requests"], expected_requests))
        checks.extend(_check_no_pending_state(method, data["requests"]))
        checks.extend(_check_event_monotonicity(method, data["events"]))
        checks.extend(_check_resource_overlap(method, data["resources"]))
        checks.extend(_check_real_target_verification(method, data))
        method_failures[method].extend(checks)
        failures.extend(checks)

    cross_checks = [
        *_check_outputs_equal_target(method_data),
        *_check_server_only_batch_size(method_data["server_only_linear"]),
        *_check_specedge_proactive(method_data["specedge_linear"]),
        *_check_dip_sd_optimizer(method_data["dip_sd"]),
        *_check_dip_sd_no_future_acceptance(method_data["dip_sd"]),
    ]
    failures.extend(cross_checks)
    for failure in cross_checks:
        for method in REAL_MODEL_METHODS:
            if failure.startswith(f"{method}:"):
                method_failures[method].append(failure)

    summaries = [
        _summarize_method(method, method_data[method], method_failures[method])
        for method in REAL_MODEL_METHODS
    ]
    _write_summary(summary_path, summaries, failures)
    if failures:
        raise SystemExit(
            "real model smoke verification failed:\n"
            + "\n".join(f"- {failure}" for failure in failures)
        )
    return summaries


def _configure_simulation(config: dict[str, Any], num_requests: int, output_tokens: int) -> None:
    config["simulation"]["seed"] = 20260625
    config["simulation"]["num_requests"] = int(num_requests)
    config["simulation"]["num_devices"] = 4
    config["simulation"]["output_len_choices"] = [int(output_tokens)]
    config["simulation"]["request_arrival"] = "burst"


def _configure_models(
    config: dict[str, Any],
    *,
    target_model: str,
    draft_model: str,
    target_device: str,
    draft_device: str,
    local_files_only: bool | None,
    cache_dir: str | None,
    revision: str | None,
) -> None:
    runner = config.setdefault("model_runner", {})
    runner["target_model"] = target_model
    runner["target_device"] = target_device
    if local_files_only is not None:
        runner["local_files_only"] = bool(local_files_only)
    elif "local_files_only" in runner:
        runner.pop("local_files_only")
    if cache_dir:
        runner["cache_dir"] = cache_dir
    elif "cache_dir" in runner:
        runner.pop("cache_dir")
    if revision:
        runner["revision"] = revision
    elif "revision" in runner:
        runner.pop("revision")
    drafter_models = {}
    for profile in config["drafter_profiles"]:
        drafter_models[profile] = {
            "model": draft_model,
            "device": draft_device,
        }
    runner["drafter_models"] = drafter_models


def _configure_virtual_devices(config: dict[str, Any]) -> None:
    config["edge"]["num_lanes"] = 2
    config["edge"]["verify_startup_ms"] = 1
    config["edge"]["target_only_startup_ms"] = 0
    config["edge"]["target_only_token_rate_tok_s"] = 1000
    config["network"]["packet_header_bytes"] = 128
    config["network"]["packet_token_bytes"] = 4

    heterogeneous = config["device_pools"]["heterogeneous"]["templates"]
    for template in heterogeneous.values():
        template["count"] = 0
        template["draft_startup_ms"] = 1
        template["draft_token_rate_tok_s"] = 500
        template["uplink_mbps"] = 25
        template["downlink_mbps"] = 100
        template["rtt_ms"] = 40
        template["jitter_ms"] = 0
    heterogeneous["low_end"]["count"] = 1
    heterogeneous["low_end"]["drafter_profile"] = "small"
    heterogeneous["low_end"]["draft_token_rate_tok_s"] = 250
    heterogeneous["low_end"]["uplink_mbps"] = 10
    heterogeneous["low_end"]["downlink_mbps"] = 50
    heterogeneous["low_end"]["rtt_ms"] = 80
    heterogeneous["mid_end"]["count"] = 2
    heterogeneous["mid_end"]["drafter_profile"] = "medium"
    heterogeneous["high_end"]["count"] = 1
    heterogeneous["high_end"]["drafter_profile"] = "large"
    heterogeneous["high_end"]["draft_token_rate_tok_s"] = 750
    heterogeneous["high_end"]["uplink_mbps"] = 100
    heterogeneous["high_end"]["downlink_mbps"] = 300
    heterogeneous["high_end"]["rtt_ms"] = 10

    medium_only = config["device_pools"]["medium_only"]["templates"]["medium"]
    medium_only["count"] = 4
    medium_only["drafter_profile"] = "medium"
    medium_only["draft_startup_ms"] = 1
    medium_only["draft_token_rate_tok_s"] = 500
    medium_only["uplink_mbps"] = 25
    medium_only["downlink_mbps"] = 100
    medium_only["rtt_ms"] = 40
    medium_only["jitter_ms"] = 0


def _configure_baseline_knobs(config: dict[str, Any]) -> None:
    config["speculation"]["W_default"] = 1
    config["speculation"]["W_max"] = 4
    config["speculation"]["unconfirmed_token_budget"] = 16
    config["speculation"]["gamma_candidates"] = [1]
    config["speculation"]["gamma_fixed"] = 1
    config["specedge"]["server_batch_size"] = 1
    config["specedge"]["server_batch_timeout_ms"] = None
    config["specedge"]["server_batch_type"] = "static"
    config["specedge"]["proactive_enabled"] = True
    config["specedge"]["proactive_type"] = "excluded"
    config["specedge"]["proactive_max_beam_len"] = 2
    config["specedge"]["proactive_max_budget"] = 4
    config["server_only"]["batch_size"] = 1
    config["server_only"]["drafter_profile"] = "medium"
    config["server_only"]["draft_startup_ms"] = 1
    config["server_only"]["draft_token_rate_tok_s"] = 500
    config["dip_sd"]["optimizer"] = "paper_exact"
    config["dip_sd"]["batch_count"] = 2
    config["dip_sd"]["max_active_requests"] = 4
    config["dip_sd"]["max_batch_size"] = 2
    config["dip_sd"]["min_draft_length"] = 1
    config["dip_sd"]["max_draft_length"] = 2
    config["dip_sd"]["draft_length"] = 1
    config["dip_sd"]["acceptance_estimator"] = "configured_profile"
    for profile in config["drafter_profiles"].values():
        profile["acceptance_prior"] = 0.9


def _write_dataset_subset(
    output_path: Path,
    dataset_path: str | Path | None,
    num_requests: int,
) -> None:
    records: list[dict[str, Any]] = []
    if dataset_path:
        with Path(dataset_path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                prompt = extract_prompt(value, line_number)
                if isinstance(value, dict):
                    record = dict(value)
                    if "turns" not in record and "prompt" not in record:
                        record["prompt"] = prompt
                    record.setdefault("question_id", f"real-smoke-{line_number}")
                    record.setdefault("category", "qa")
                else:
                    record = {
                        "question_id": f"real-smoke-{line_number}",
                        "category": "qa",
                        "turns": [prompt],
                    }
                records.append(record)
                if len(records) >= num_requests:
                    break
        if len(records) < num_requests:
            raise ValueError(
                f"requested {num_requests} smoke prompts but dataset only contains {len(records)}"
            )
    else:
        records = [
            {
                "question_id": f"real-smoke-{index}",
                "category": "qa",
                "turns": [f"real model baseline smoke prompt {index}"],
            }
            for index in range(num_requests)
        ]
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _load_method_outputs(root: Path, method: str) -> dict[str, Any]:
    directory = root / method
    return {
        "method": method,
        "directory": directory,
        "metrics": _read_csv(directory / "metrics.csv"),
        "requests": _read_csv(directory / "request_trace.csv"),
        "events": _read_csv(directory / "event_trace.csv"),
        "tokens": _read_csv(directory / "token_trace.csv"),
        "resources": _read_csv(directory / "resource_timeline.csv"),
        "batches": _read_csv(directory / "batch_trace.csv"),
        "manifest": _read_json(directory / "run_manifest.json"),
        "resolved_config": _read_json(directory / "resolved_config.json"),
        "stdout_log": _read_text(directory / "stdout.log"),
    }


def _check_required_files(directory: Path) -> list[str]:
    failures = []
    for filename in REQUIRED_REAL_FILES:
        path = directory / filename
        if not path.exists():
            failures.append(f"{directory.name}: missing {filename}")
        elif path.stat().st_size <= 0:
            failures.append(f"{directory.name}: empty {filename}")
    return failures


def _check_manifest(method: str, manifest: dict[str, Any]) -> list[str]:
    failures = []
    if not manifest:
        return [f"{method}: missing or invalid run_manifest.json"]
    if manifest.get("runner") != "HuggingFaceModelRunner":
        failures.append(f"{method}: manifest runner is {manifest.get('runner')}")
    if manifest.get("use_fake_model_runner") is not False:
        failures.append(f"{method}: manifest does not explicitly disable fake runner")
    if "--use-fake-model-runner" in str(manifest.get("command", "")):
        failures.append(f"{method}: command includes --use-fake-model-runner")
    if _to_int(manifest.get("return_code")) != 0:
        failures.append(f"{method}: command return_code={manifest.get('return_code')}")
    for key in ("target_model", "draft_model", "dataset"):
        if not manifest.get(key):
            failures.append(f"{method}: manifest missing {key}")
    return failures


def _check_stdout_log(method: str, text: str) -> list[str]:
    failures = []
    if not text:
        return [f"{method}: stdout.log is empty or missing"]
    for pattern in LOG_FAILURE_PATTERNS:
        if pattern.search(text):
            failures.append(f"{method}: stdout.log contains failure pattern {pattern.pattern!r}")
    return failures


def _check_numeric_outputs(method: str, metrics: list[dict[str, str]]) -> list[str]:
    failures = []
    if not metrics:
        return [f"{method}: metrics.csv has no rows"]
    for row_index, row in enumerate(metrics):
        for key, value in row.items():
            if value in (None, ""):
                continue
            try:
                number = float(value)
            except ValueError:
                continue
            if math.isnan(number):
                failures.append(f"{method}: metrics row {row_index} has NaN in {key}")
    return failures


def _check_all_requests_finished(
    method: str,
    requests: list[dict[str, str]],
    expected_requests: int,
) -> list[str]:
    failures = []
    if len(requests) != expected_requests:
        failures.append(f"{method}: expected {expected_requests} requests, found {len(requests)}")
    for row in requests:
        request_id = row.get("request_id")
        if row.get("status") != "finished":
            failures.append(f"{method}: request {request_id} status is {row.get('status')}")
        if _to_int(row.get("generated_tokens")) != _to_int(row.get("output_len")):
            failures.append(f"{method}: request {request_id} did not generate output_len tokens")
        output_len = _to_int(row.get("output_len"))
        if output_len < 8 or output_len > 16:
            failures.append(f"{method}: request {request_id} output_len {output_len} outside [8, 16]")
    return failures


def _check_real_target_verification(method: str, data: dict[str, Any]) -> list[str]:
    failures = []
    config = data["resolved_config"].get("config", data["resolved_config"])
    if config and not config.get("model_runner"):
        failures.append(f"{method}: resolved config missing model_runner")
    if method == "target_only":
        return failures
    expected_event = VERIFY_EVENTS_BY_METHOD[method]
    if not any(row.get("event") == expected_event for row in data["events"]):
        failures.append(f"{method}: no {expected_event} event")
    if _sum_token_type(data["tokens"], "verified") <= 0:
        failures.append(f"{method}: token_trace has no verified tokens")
    return failures


def _check_outputs_equal_target(method_data: dict[str, dict[str, Any]]) -> list[str]:
    target = _committed_tokens(method_data["target_only"]["tokens"])
    if not target:
        return ["target_only: committed token trace is empty"]
    failures = []
    for method, data in method_data.items():
        observed = _committed_tokens(data["tokens"])
        if observed != target:
            failures.append(f"{method}: committed token trace differs from target_only greedy equivalence")
    return failures


def _check_server_only_batch_size(data: dict[str, Any]) -> list[str]:
    failures = []
    for row in data["batches"]:
        if row.get("event") == "server_only_verify" and _to_int(row.get("batch_size")) != 1:
            failures.append(f"server_only_linear: observed batch_size={row.get('batch_size')}")
    return failures


def _check_specedge_proactive(data: dict[str, Any]) -> list[str]:
    events = data["events"]
    tokens = data["tokens"]
    failures = []
    if not any(row.get("event") == "proactive_draft" for row in events):
        failures.append("specedge_linear: no proactive_draft event")
    if _sum_token_type(tokens, "proactive_drafted") <= 0:
        failures.append("specedge_linear: no proactive_drafted token trace")
    return failures


def _check_dip_sd_optimizer(data: dict[str, Any]) -> list[str]:
    failures = []
    plans = [row for row in data["events"] if row.get("event") == "dip_sd_epoch_plan"]
    if not plans:
        return ["dip_sd: no dip_sd_epoch_plan event"]
    saw_assignment = False
    saw_draft_lengths = False
    for plan in plans:
        if plan.get("optimizer") and plan.get("optimizer") != "paper_exact":
            failures.append(f"dip_sd: optimizer is {plan.get('optimizer')}")
        assignment = _parse_json(plan.get("assignment", ""))
        draft_lengths = _parse_json(plan.get("draft_lengths", ""))
        if isinstance(assignment, dict) and assignment:
            saw_assignment = True
        if isinstance(draft_lengths, dict) and draft_lengths:
            saw_draft_lengths = True
    if not saw_assignment:
        failures.append("dip_sd: no optimizer assignment in epoch plan")
    if not saw_draft_lengths:
        failures.append("dip_sd: no per-request draft_lengths in epoch plan")

    plans_by_epoch = {row.get("epoch"): row for row in plans}
    for draft in (row for row in data["events"] if row.get("event") == "dip_sd_draft"):
        plan = plans_by_epoch.get(draft.get("epoch"))
        if plan is None:
            failures.append(f"dip_sd: draft event lacks epoch plan for epoch {draft.get('epoch')}")
            continue
        draft_lengths = _parse_json(plan.get("draft_lengths", ""))
        planned = None
        if isinstance(draft_lengths, dict):
            planned = draft_lengths.get(str(draft.get("request_id")))
            if planned is None:
                planned = draft_lengths.get(draft.get("request_id"))
        if planned is None:
            failures.append(f"dip_sd: draft event request {draft.get('request_id')} lacks planned length")
        elif _to_int(draft.get("scheduled_gamma")) > int(planned):
            failures.append(
                f"dip_sd: scheduled_gamma {draft.get('scheduled_gamma')} exceeds planned length {planned}"
            )
    return failures


def _check_dip_sd_no_future_acceptance(data: dict[str, Any]) -> list[str]:
    failures = []
    plans = [row for row in data["events"] if row.get("event") == "dip_sd_epoch_plan"]
    for plan in plans:
        epoch = plan.get("epoch")
        plan_time = _to_float(plan.get("time_ms") or plan.get("event_time_ms"))
        prior_results = [
            row
            for row in data["events"]
            if row.get("event") == "dip_sd_result"
            and row.get("epoch") == epoch
            and _to_float(row.get("finish_time_ms") or row.get("event_time_ms")) <= plan_time
        ]
        if prior_results:
            failures.append(f"dip_sd: epoch {epoch} has result before optimizer plan")
        first_draft = min(
            (
                _to_float(row.get("start_time_ms") or row.get("event_time_ms"))
                for row in data["events"]
                if row.get("event") == "dip_sd_draft" and row.get("epoch") == epoch
            ),
            default=plan_time,
        )
        if plan_time > first_draft + 1e-9:
            failures.append(f"dip_sd: epoch {epoch} plan occurs after draft start")
    return failures


def _summarize_method(
    method: str,
    data: dict[str, Any],
    failures: list[str],
) -> dict[str, Any]:
    tokens = data["tokens"]
    requests = data["requests"]
    manifest = data["manifest"]
    committed = _sum_token_type(tokens, "committed")
    drafted = _sum_token_type(tokens, "drafted") + _sum_token_type(tokens, "proactive_drafted")
    verified = _sum_token_type(tokens, "verified")
    accepted = _sum_token_type(tokens, "accepted")
    wasted = _sum_token_type(tokens, "wasted")
    finish_ms = max((_to_float(row.get("finish_time_ms")) for row in requests), default=0.0)
    ratio = accepted / verified if verified else None
    return {
        "method": method,
        "success": not failures,
        "model": f"{manifest.get('target_model', 'n/a')} / {manifest.get('draft_model', 'n/a')}",
        "request_count": len(requests),
        "committed_tokens": committed,
        "drafted_tokens": drafted,
        "verified_tokens": verified,
        "accepted_tokens": accepted,
        "wasted_tokens": wasted,
        "acceptance_ratio": ratio,
        "finish_time_ms": finish_ms,
        "gpu_peak_mb": manifest.get("gpu_peak_mb", "n/a") if manifest else "n/a",
        "caveat": _method_caveat(method, failures),
    }


def _method_caveat(method: str, failures: list[str]) -> str:
    if failures:
        return "; ".join(failures[:2])
    if method == "target_only":
        return "Greedy reference; no speculative verification events expected."
    if method == "server_only_linear":
        return "Server-only batch_size is fixed at 1."
    if method == "specedge_linear":
        return "Linear SpecEdge smoke requires proactive_draft evidence."
    if method == "dip_sd":
        return "Online DiP-SD smoke checks optimizer assignment and planned lengths."
    return ""


def _write_summary(path: str | Path, summaries: list[dict[str, Any]], failures: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Real Model Smoke Summary",
        "",
        f"Status: {'PASS' if not failures else 'FAIL'}",
        "",
        "| method | success | model | requests | committed | drafted | verified | accepted | wasted | acceptance_ratio | finish_ms | gpu_peak_mb | caveat |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        ratio = "n/a" if row["acceptance_ratio"] is None else f"{row['acceptance_ratio']:.4f}"
        lines.append(
            "| {method} | {success} | {model} | {request_count} | {committed_tokens} | "
            "{drafted_tokens} | {verified_tokens} | {accepted_tokens} | {wasted_tokens} | "
            "{ratio} | {finish_time_ms:.3f} | {gpu_peak_mb} | {caveat} |".format(
                **row,
                ratio=ratio,
            )
        )
    lines.extend(["", "## Automatic Checks", ""])
    if failures:
        lines.extend(f"- FAIL: {failure}" for failure in failures)
    else:
        lines.extend(
            [
                "- PASS: four methods completed all smoke requests.",
                "- PASS: speculative committed token traces satisfy greedy equivalence with target_only.",
                "- PASS: manifests require HuggingFaceModelRunner and no fake runner flag.",
                "- PASS: acceptance was produced by real target verification events.",
                "- PASS: no request finishes with pending or unverified state.",
                "- PASS: event times are monotonic and resource timelines do not overlap illegally.",
                "- PASS: server_only_linear observed batch_size=1.",
                "- PASS: specedge_linear contains proactive drafting.",
                "- PASS: dip_sd contains optimizer assignment and per-request draft lengths.",
                "- PASS: dip_sd epoch plans precede same-epoch draft/result acceptance observations.",
                "- PASS: all required output files exist and are non-empty.",
            ]
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_resolved_or_config(directory: Path, config_path: str | Path) -> dict[str, Any]:
    resolved = _read_json(directory / "resolved_config.json")
    if resolved:
        return resolved.get("config", resolved)
    try:
        return load_config(config_path)
    except Exception:
        return {}


def _parse_json(value: Any) -> Any:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare, manifest, and verify real-model baseline smoke runs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--root", default="outputs/real_model_smoke")
    prepare.add_argument("--config", default="configs/default.yaml")
    prepare.add_argument("--target-model", required=True)
    prepare.add_argument("--draft-model", required=True)
    prepare.add_argument("--dataset")
    prepare.add_argument("--target-device", default="cuda:1")
    prepare.add_argument("--draft-device", default="cuda:0")
    prepare.add_argument("--num-requests", type=int, default=4)
    prepare.add_argument("--output-tokens", type=int, default=8)
    prepare.add_argument("--local-files-only")
    prepare.add_argument("--cache-dir")
    prepare.add_argument("--revision")

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--output-dir", required=True)
    manifest.add_argument("--method", required=True, choices=REAL_MODEL_METHODS)
    manifest.add_argument("--command-text", required=True)
    manifest.add_argument("--return-code", required=True, type=int)
    manifest.add_argument("--config", required=True)
    manifest.add_argument("--dataset", required=True)
    manifest.add_argument("--target-model", required=True)
    manifest.add_argument("--draft-model", required=True)
    manifest.add_argument("--target-device", required=True)
    manifest.add_argument("--draft-device", required=True)
    manifest.add_argument("--stdout-log", required=True)
    manifest.add_argument("--skipped-reason")
    manifest.add_argument("--gpu-peak-mb")

    verify = subparsers.add_parser("verify")
    verify.add_argument("--root", default="outputs/real_model_smoke")
    verify.add_argument("--summary", default="outputs/real_model_smoke/summary.md")
    verify.add_argument("--expected-requests", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        config_path, dataset_path = prepare_real_model_inputs(
            root=args.root,
            config_path=args.config,
            target_model=args.target_model,
            draft_model=args.draft_model,
            dataset_path=args.dataset,
            target_device=args.target_device,
            draft_device=args.draft_device,
            num_requests=args.num_requests,
            output_tokens=args.output_tokens,
            local_files_only=_optional_bool(args.local_files_only),
            cache_dir=args.cache_dir,
            revision=args.revision,
        )
        print(f"config: {config_path}")
        print(f"dataset: {dataset_path}")
        return
    if args.command == "manifest":
        path = write_run_manifest(
            output_dir=args.output_dir,
            method=args.method,
            command=args.command_text,
            return_code=args.return_code,
            config_path=args.config,
            dataset_path=args.dataset,
            target_model=args.target_model,
            draft_model=args.draft_model,
            target_device=args.target_device,
            draft_device=args.draft_device,
            stdout_log=args.stdout_log,
            skipped_reason=args.skipped_reason,
            gpu_peak_mb=args.gpu_peak_mb,
        )
        print(f"manifest: {path}")
        return
    if args.command == "verify":
        summaries = verify_real_model_outputs(
            args.root,
            args.summary,
            expected_requests=args.expected_requests,
        )
        print(f"summary: {args.summary}")
        for row in summaries:
            print(
                f"{row['method']}: success={row['success']} "
                f"requests={row['request_count']} committed={row['committed_tokens']} "
                f"finish_ms={row['finish_time_ms']:.3f}"
            )
        return
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
