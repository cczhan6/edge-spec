from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from src.entities import Device
from src.tree_drafting import SUPPORTED_TREE_DRAFT_STRATEGIES

REMOVED_SCENARIOS = {"balanced_drafter", "network_heterogeneous"}
CODE_ROOT = Path(__file__).resolve().parents[1]
TARGET_LATENCY_MODES = {"analytical", "profile"}
TARGET_LATENCY_METRICS = {"p50_ms", "mean_ms", "p95_ms"}


def resolve_target_latency_profile_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return CODE_ROOT / candidate


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path, scenario: str | None = None) -> dict[str, Any]:
    config = _read_yaml(Path(path))
    if scenario:
        scenario_path = Path(path).parent / f"{scenario}.yaml"
        if scenario_path.exists():
            config = deep_merge(config, _read_yaml(scenario_path))
        elif scenario in REMOVED_SCENARIOS:
            raise FileNotFoundError(f"scenario config was removed: {scenario}")
    validate_config(config)
    return config


def apply_tree_draft_strategy(config: dict[str, Any], strategy: str | None) -> dict[str, Any]:
    if strategy is None:
        return config
    if strategy not in SUPPORTED_TREE_DRAFT_STRATEGIES:
        raise ValueError(f"tree draft strategy is unknown: {strategy}")
    config.setdefault("specedge", {})
    config["specedge"]["tree_draft_strategy"] = strategy
    config["specedge"]["proactive_tree_draft_strategy"] = strategy
    config.setdefault("server_only", {})
    config["server_only"]["tree_draft_strategy"] = strategy
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    simulation = config["simulation"]
    _validate_dynamic_edge_compute(config)
    if int(simulation["num_requests"]) <= 0 or int(simulation["num_devices"]) <= 0:
        raise ValueError("num_requests and num_devices must be positive")
    edge = config["edge"]
    if edge.get("lane_microbatch", False):
        raise ValueError("lane microbatching is not supported")
    if int(edge["num_lanes"]) <= 0:
        raise ValueError("num_lanes must be positive")
    if float(edge["target_only_token_rate_tok_s"]) <= 0:
        raise ValueError("target_only_token_rate_tok_s must be positive")
    target_latency = config.get("target_latency", {"mode": "analytical"})
    mode = str(target_latency.get("mode", ""))
    if mode not in TARGET_LATENCY_MODES:
        raise ValueError("target_latency.mode must be analytical or profile")
    metric = target_latency.get("metric")
    if metric is not None and str(metric) not in TARGET_LATENCY_METRICS:
        raise ValueError(
            "target_latency.metric must be p50_ms, mean_ms, or p95_ms"
        )
    if mode == "profile":
        profile_path = target_latency.get("profile_path")
        if not isinstance(profile_path, str) or not profile_path.strip():
            raise ValueError(
                "target_latency.profile_path must be a non-empty string in profile mode"
            )
        resolved_profile_path = resolve_target_latency_profile_path(profile_path)
        if not resolved_profile_path.is_file():
            raise ValueError(
                "target_latency.profile_path does not exist: "
                f"{resolved_profile_path}"
            )
    speculation = config["speculation"]
    if int(speculation["W_default"]) <= 0:
        raise ValueError("W_default must be positive")
    if int(speculation.get("W_max", 1)) <= 0:
        raise ValueError("W_max must be positive")
    if int(speculation.get("unconfirmed_token_budget", speculation.get("W_max", 0))) <= 0:
        raise ValueError("unconfirmed_token_budget must be positive")
    candidates = [int(value) for value in speculation["gamma_candidates"]]
    if not candidates or any(value <= 0 for value in candidates):
        raise ValueError("gamma_candidates must contain positive integers")
    if int(speculation["gamma_fixed"]) <= 0:
        raise ValueError("gamma_fixed must be positive")
    async_speculative = config["async_speculative"]
    if str(async_speculative["mode"]) != "fixed":
        raise ValueError("async_speculative.mode must be fixed")
    if int(async_speculative["num_channels"]) <= 0:
        raise ValueError("async_speculative.num_channels must be positive")
    if int(async_speculative["gamma_fixed"]) <= 0:
        raise ValueError("async_speculative.gamma_fixed must be positive")
    if int(async_speculative["lookahead_depth_fixed"]) <= 0:
        raise ValueError(
            "async_speculative.lookahead_depth_fixed must be positive"
        )
    if int(async_speculative["l_max_ver"]) <= 0:
        raise ValueError("async_speculative.l_max_ver must be positive")
    if int(async_speculative["l_max_ver"]) < int(async_speculative["gamma_fixed"]):
        raise ValueError("async_speculative.l_max_ver must be >= gamma_fixed")
    specedge = config.get("specedge", {})
    if specedge:
        for key in (
            "max_n_beams",
            "max_beam_len",
            "max_branch_width",
            "max_budget",
            "proactive_max_beam_len",
            "proactive_max_budget",
        ):
            if int(specedge[key]) <= 0:
                raise ValueError(f"specedge.{key} must be positive")
        server_batch_size = specedge.get("server_batch_size", 1)
        if server_batch_size is None:
            server_batch_size = 1
        if int(server_batch_size) <= 0:
            raise ValueError("specedge.server_batch_size must be positive")
        server_batch_timeout_ms = specedge.get(
            "server_batch_timeout_ms",
            config["sync_batch"]["global_batch_timeout_ms"],
        )
        if server_batch_timeout_ms is not None and float(server_batch_timeout_ms) < 0:
            raise ValueError("specedge.server_batch_timeout_ms must be non-negative")
        proactive_type = str(specedge.get("proactive_type", "excluded"))
        if proactive_type not in {"included", "excluded", "disabled"}:
            raise ValueError("specedge.proactive_type must be included, excluded, or disabled")
        server_batch_type = str(specedge.get("server_batch_type", "static"))
        if server_batch_type not in {"static", "dynamic"}:
            raise ValueError("specedge.server_batch_type must be static or dynamic")
        tree_strategy = str(specedge.get("tree_draft_strategy", "specexec_approx"))
        if tree_strategy not in SUPPORTED_TREE_DRAFT_STRATEGIES:
            raise ValueError("specedge.tree_draft_strategy is unknown")
        proactive_tree_strategy = str(
            specedge.get("proactive_tree_draft_strategy", tree_strategy)
        )
        if proactive_tree_strategy not in SUPPORTED_TREE_DRAFT_STRATEGIES:
            raise ValueError("specedge.proactive_tree_draft_strategy is unknown")
        if int(specedge["max_budget"]) < int(specedge["max_beam_len"]):
            raise ValueError("specedge.max_budget must be at least max_beam_len")
        if int(specedge["proactive_max_beam_len"]) > int(specedge["max_beam_len"]):
            raise ValueError("specedge.proactive_max_beam_len must not exceed max_beam_len")
        if int(specedge["proactive_max_budget"]) > int(specedge["max_budget"]):
            raise ValueError("specedge.proactive_max_budget must not exceed max_budget")
        if int(specedge["proactive_max_budget"]) < int(specedge["proactive_max_beam_len"]):
            raise ValueError(
                "specedge.proactive_max_budget must be at least proactive_max_beam_len"
            )
    if not config["drafter_profiles"]:
        raise ValueError("drafter_profiles must define at least one drafter")
    model_runner = config.get("model_runner", config.get("oracle"))
    if model_runner is None:
        raise ValueError("model_runner must define model paths")
    model_runner_models = model_runner["drafter_models"]
    missing_models = sorted(set(config["drafter_profiles"]) - set(model_runner_models))
    if missing_models:
        raise ValueError(
            "model_runner.drafter_models must define: " + ", ".join(missing_models)
        )
    server_only = config.get("server_only", {})
    if server_only:
        server_only_batch_size = int(server_only.get("batch_size", 1))
        if server_only_batch_size <= 0:
            raise ValueError("server_only.batch_size must be positive")
        if server_only_batch_size != 1:
            raise ValueError(
                "server_only.batch_size > 1 is not supported by the current "
                "single-request server-only runtime"
            )
        drafter = str(server_only["drafter_profile"])
        if drafter not in config["drafter_profiles"]:
            raise ValueError(f"server_only uses unknown drafter {drafter}")
        if float(server_only["draft_token_rate_tok_s"]) <= 0:
            raise ValueError("server_only.draft_token_rate_tok_s must be positive")
        if float(server_only.get("draft_startup_ms", 0.0)) < 0:
            raise ValueError("server_only.draft_startup_ms must be non-negative")
        server_tree_strategy = str(
            server_only.get(
                "tree_draft_strategy",
                config.get("specedge", {}).get("tree_draft_strategy", "specexec_approx"),
            )
        )
        if server_tree_strategy not in SUPPORTED_TREE_DRAFT_STRATEGIES:
            raise ValueError("server_only.tree_draft_strategy is unknown")
        for key in ("max_n_beams", "max_beam_len", "max_branch_width", "max_budget"):
            if key in server_only and int(server_only[key]) <= 0:
                raise ValueError(f"server_only.{key} must be positive")
        server_max_beam_len = int(
            server_only.get(
                "max_beam_len",
                config.get("specedge", {}).get("max_beam_len", 1),
            )
        )
        server_max_budget = int(
            server_only.get(
                "max_budget",
                config.get("specedge", {}).get("max_budget", 1),
            )
        )
        if server_max_budget < server_max_beam_len:
            raise ValueError("server_only.max_budget must be at least max_beam_len")
    dip_sd = config.get("dip_sd", {})
    if dip_sd:
        optimizer = str(dip_sd.get("optimizer", "paper_exact"))
        if optimizer != "paper_exact":
            raise ValueError("dip_sd.optimizer must be paper_exact")
        for key in (
            "batch_count",
            "draft_length",
            "max_active_requests",
            "max_batch_size",
            "min_draft_length",
            "max_draft_length",
        ):
            if int(dip_sd[key]) <= 0:
                raise ValueError(f"dip_sd.{key} must be positive")
        if int(dip_sd["min_draft_length"]) > int(dip_sd["max_draft_length"]):
            raise ValueError("dip_sd.min_draft_length must not exceed max_draft_length")
    for pool_name in ("heterogeneous", "medium_only"):
        pool = config["device_pools"].get(pool_name)
        if not pool or not pool.get("templates"):
            raise ValueError(f"device_pools.{pool_name}.templates must not be empty")
        count = 0
        for template_name, template in pool["templates"].items():
            _validate_device_network_template(pool_name, template_name, template)
            count += int(template["count"])
            drafter = str(template["drafter_profile"])
            if drafter not in config["drafter_profiles"]:
                raise ValueError(
                    f"device template {template_name} uses unknown drafter {drafter}"
                )
            if float(template["draft_token_rate_tok_s"]) <= 0:
                raise ValueError("draft_token_rate_tok_s must be positive")
        if count != int(simulation["num_devices"]):
            raise ValueError(
                f"device pool {pool_name} defines {count} devices, "
                f"expected simulation.num_devices={simulation['num_devices']}"
            )


def _is_finite_real(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_device_network_template(
    pool_name: str,
    template_name: str,
    template: dict[str, Any],
) -> None:
    prefix = f"device_pools.{pool_name}.templates.{template_name}"
    probability = template.get("block_probability", 1.0)
    if (
        not _is_finite_real(probability)
        or not 0.0 <= float(probability) <= 1.0
    ):
        raise ValueError(
            f"{prefix}.block_probability must be a finite number in [0, 1]"
        )
    for field in ("rtt_ms", "jitter_ms"):
        value = template.get(field, 0.0)
        if not _is_finite_real(value) or float(value) < 0.0:
            raise ValueError(
                f"{prefix}.{field} must be a finite non-negative number"
            )
    for field in ("uplink_mbps", "downlink_mbps"):
        value = template[field]
        if not _is_finite_real(value) or float(value) <= 0.0:
            raise ValueError(
                f"{prefix}.{field} must be a finite positive number"
            )


def _validate_dynamic_edge_compute(config: dict[str, Any]) -> None:
    dynamic = config.get(
        "dynamic_edge_compute",
        {"enabled": False, "resample_every_completed_requests": 5},
    )
    enabled = dynamic.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("dynamic_edge_compute.enabled must be a boolean")
    if dynamic.get("resample_every_completed_requests", 5) != 5:
        raise ValueError(
            "dynamic_edge_compute.resample_every_completed_requests must be 5"
        )
    for pool_name, pool in config["device_pools"].items():
        for device_type, template in pool["templates"].items():
            bounds = template.get("dynamic_draft_token_rate_range_tok_s")
            if bounds is None:
                if enabled and int(template["count"]) > 0:
                    raise ValueError(
                        f"device template {pool_name}.{device_type} requires "
                        "dynamic_draft_token_rate_range_tok_s"
                    )
                continue
            valid = (
                isinstance(bounds, list)
                and len(bounds) == 2
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and float(value) > 0.0
                    for value in bounds
                )
                and float(bounds[0]) < float(bounds[1])
            )
            if not valid:
                raise ValueError(
                    f"device template {pool_name}.{device_type} dynamic rate range "
                    "must contain two finite positive values with min < max"
                )


def build_devices(config: dict[str, Any], pool_name: str = "heterogeneous") -> list[Device]:
    profiles = config["drafter_profiles"]
    templates = config["device_pools"][pool_name]["templates"]
    devices: list[Device] = []
    for template_name, template in templates.items():
        for _ in range(int(template["count"])):
            drafter = str(template["drafter_profile"])
            devices.append(
                Device(
                    device_id=len(devices),
                    device_type=str(template_name),
                    drafter_profile=drafter,
                    acceptance_prior=float(profiles[drafter]["acceptance_prior"]),
                    draft_token_rate_tok_s=float(template["draft_token_rate_tok_s"]),
                    draft_startup_ms=float(template.get("draft_startup_ms", 0.0)),
                    uplink_mbps=float(template["uplink_mbps"]),
                    downlink_mbps=float(template["downlink_mbps"]),
                    rtt_ms=float(template["rtt_ms"]),
                    jitter_ms=float(template.get("jitter_ms", 0.0)),
                    block_probability=float(template.get("block_probability", 1.0)),
                )
            )
    return devices


def _read_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        return _parse_simple_yaml(text)
    return yaml.safe_load(text)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the mapping-only YAML subset used by the bundled configs."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        content = raw_line.split("#", 1)[0].rstrip()
        if not content:
            continue
        indent = len(content) - len(content.lstrip(" "))
        key, separator, raw_value = content.strip().partition(":")
        if not separator:
            raise ValueError(f"unsupported YAML line: {raw_line}")
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if not raw_value.strip():
            value: Any = {}
            parent[key] = value
            stack.append((indent, value))
        else:
            parent[key] = _parse_scalar(raw_value.strip())
    return root


def _parse_scalar(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
