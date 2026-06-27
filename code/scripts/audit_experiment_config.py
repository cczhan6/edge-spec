from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config


CANONICAL_METHODS = (
    "target_only",
    "server_only_linear",
    "server_only_tree",
    "specedge_linear",
    "specedge_tree",
    "dip_sd",
)
FORMAL_ONLINE_SCENARIOS = {"homogeneous", "combined_strong_heterogeneous"}


class ConfigAuditError(ValueError):
    """Raised when a formal experiment configuration violates the contract."""


def audit_experiment_config(
    config: dict[str, Any],
    *,
    scenario: str,
    methods: Sequence[str],
    use_fake_model_runner: bool,
    repo_root: str | Path,
) -> dict[str, Any]:
    errors: list[str] = []
    root = Path(repo_root).resolve()
    selected_methods = tuple(str(method) for method in methods)

    if not selected_methods:
        errors.append("at least one canonical method must be selected")
    invalid_methods = [method for method in selected_methods if method not in CANONICAL_METHODS]
    if invalid_methods:
        errors.append(
            "formal results require canonical method names; invalid: "
            + ", ".join(invalid_methods)
        )
    if len(set(selected_methods)) != len(selected_methods):
        errors.append("canonical method names must not be duplicated")

    server_only = config.get("server_only", {})
    if int(server_only.get("batch_size", 0)) != 1:
        errors.append("server_only.batch_size must be 1")
    if "server_only_tree" in selected_methods and str(
        server_only.get("tree_draft_strategy", "")
    ) != "specexec_approx":
        errors.append("server_only.tree_draft_strategy must be specexec_approx")

    specedge = config.get("specedge", {})
    if "specedge_tree" in selected_methods:
        if str(specedge.get("tree_draft_strategy", "")) != "specexec_approx":
            errors.append("specedge.tree_draft_strategy must be specexec_approx")
        if str(specedge.get("proactive_tree_draft_strategy", "")) != "specexec_approx":
            errors.append(
                "specedge.proactive_tree_draft_strategy must be specexec_approx"
            )

    simulation = config.get("simulation", {})
    arrival = str(simulation.get("request_arrival", ""))
    if scenario in FORMAL_ONLINE_SCENARIOS and arrival == "burst":
        errors.append(f"formal online scenario {scenario!r} must not use burst arrivals")
    if scenario in FORMAL_ONLINE_SCENARIOS and arrival != "poisson":
        errors.append(f"formal online scenario {scenario!r} must use poisson arrivals")
    if "seed" not in simulation or isinstance(simulation.get("seed"), bool):
        errors.append("simulation.seed must be explicit")
    else:
        try:
            int(simulation["seed"])
        except (TypeError, ValueError):
            errors.append("simulation.seed must be an integer")
    if not _is_positive_int(simulation.get("num_requests")):
        errors.append("simulation.num_requests must be positive")
    output_lengths = simulation.get("output_len_choices")
    if not isinstance(output_lengths, list) or not output_lengths or any(
        not _is_positive_int(value) for value in output_lengths
    ):
        errors.append("simulation.output_len_choices must contain positive integers")
    if arrival == "poisson" and not _is_positive_number(
        simulation.get("poisson_rate_per_s")
    ):
        errors.append("simulation.poisson_rate_per_s must be positive")

    experiment = config.get("experiment", {})
    if experiment.get("internal_time_unit") != "ms":
        errors.append("experiment.internal_time_unit must be explicit and equal to ms")
    if experiment.get("csv_time_unit") != "ms":
        errors.append("experiment.csv_time_unit must be explicit and equal to ms")
    if experiment.get("request_device_assignment") != "fixed":
        errors.append("experiment.request_device_assignment must be fixed")
    if bool(experiment.get("use_fake_model_runner", False)) or use_fake_model_runner:
        errors.append("fake runner must be disabled for formal experiments")

    model_runner = config.get("model_runner")
    if not isinstance(model_runner, dict):
        errors.append("model_runner must be explicit")
    else:
        target_device = str(model_runner.get("target_device", ""))
        _check_model_reference(
            model_runner.get("target_model"),
            "model_runner.target_model",
            root,
            errors,
        )
        drafter_models = model_runner.get("drafter_models", {})
        if not isinstance(drafter_models, dict) or not drafter_models:
            errors.append("model_runner.drafter_models must be explicit")
        else:
            for profile in ("small", "medium", "large"):
                values = drafter_models.get(profile)
                if not isinstance(values, dict):
                    errors.append(f"model_runner.drafter_models.{profile} must be explicit")
                    continue
                _check_model_reference(
                    values.get("model"),
                    f"model_runner.drafter_models.{profile}.model",
                    root,
                    errors,
                )
                if target_device == str(values.get("device", "")):
                    errors.append(
                        "target and drafter devices must differ: "
                        f"target={target_device}, {profile}={values.get('device')}"
                    )

    try:
        json.dumps(config, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        errors.append(f"resolved config is not completely JSON exportable: {exc}")

    if errors:
        raise ConfigAuditError("experiment config audit failed:\n- " + "\n- ".join(errors))

    return {
        "schema_version": 1,
        "scenario": scenario,
        "methods": list(selected_methods),
        "use_fake_model_runner": False,
        "git_commit": _git_commit(root),
        "config": config,
    }


def write_resolved_config(path: str | Path, resolved: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        resolved,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    temporary.replace(destination)


def _check_model_reference(
    value: object,
    field: str,
    repo_root: Path,
    errors: list[str],
) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{field} must be explicit")
        return
    reference = value.strip()
    expanded = Path(reference).expanduser()
    local_candidate = expanded if expanded.is_absolute() else repo_root / expanded
    if local_candidate.exists():
        return
    if _looks_like_local_path(reference):
        errors.append(f"{field} model path does not exist: {reference}")
        return
    if not _huggingface_model_is_cached(reference):
        errors.append(
            f"{field} model path does not exist and Hugging Face id is not cached: {reference}"
        )


def _looks_like_local_path(reference: str) -> bool:
    return reference.startswith(("/", "./", "../", "~"))


def _huggingface_model_is_cached(model_id: str) -> bool:
    if model_id.count("/") != 1:
        return False
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(model_id, "config.json")
        return isinstance(cached, str) and Path(cached).exists()
    except (ImportError, OSError, ValueError):
        cache_root = Path(
            os.environ.get(
                "HF_HUB_CACHE",
                Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
                / "hub",
            )
        )
        model_cache = cache_root / f"models--{model_id.replace('/', '--')}"
        ref = model_cache / "refs" / "main"
        if not ref.exists():
            return False
        revision = ref.read_text(encoding="utf-8").strip()
        return (model_cache / "snapshots" / revision / "config.json").exists()


def _git_commit(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if completed.returncode != 0 or len(commit) != 40:
        raise ConfigAuditError("unable to record a full git commit in resolved config")
    return commit


def _is_positive_int(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return int(value) > 0 and float(value) == int(value)
    except (TypeError, ValueError, OverflowError):
        return False


def _is_positive_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return float(value) > 0
    except (TypeError, ValueError, OverflowError):
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit and resolve a formal canonical-baseline experiment config."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--scenario", choices=sorted(FORMAL_ONLINE_SCENARIOS), required=True)
    parser.add_argument("--methods", nargs="+", default=list(CANONICAL_METHODS))
    parser.add_argument("--use-fake-model-runner", action="store_true")
    parser.add_argument(
        "--resolved-config-out",
        default="outputs/resolved_experiment_config.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(args.config, args.scenario)
    try:
        resolved = audit_experiment_config(
            config,
            scenario=args.scenario,
            methods=args.methods,
            use_fake_model_runner=args.use_fake_model_runner,
            repo_root=repo_root,
        )
        write_resolved_config(args.resolved_config_out, resolved)
    except (ConfigAuditError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"experiment config audit passed: {args.resolved_config_out}")


if __name__ == "__main__":
    main()
