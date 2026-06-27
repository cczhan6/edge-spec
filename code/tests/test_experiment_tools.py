from __future__ import annotations

import copy
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.config import load_config


CANONICAL_METHODS = (
    "target_only",
    "server_only_linear",
    "server_only_tree",
    "specedge_linear",
    "specedge_tree",
    "dip_sd",
)


def test_default_formal_metadata_uses_parser_compatible_string_lists() -> None:
    config = load_config("configs/default.yaml")

    assert config["experiment"]["canonical_methods"] == list(CANONICAL_METHODS)
    assert config["experiment"]["formal_online_scenarios"] == [
        "homogeneous",
        "combined_strong_heterogeneous",
    ]


def _audit_module():
    spec = importlib.util.find_spec("scripts.audit_experiment_config")
    assert spec is not None, "scripts.audit_experiment_config must exist"
    return importlib.import_module("scripts.audit_experiment_config")


def _environment_module():
    spec = importlib.util.find_spec("scripts.collect_experiment_environment")
    assert spec is not None, "scripts.collect_experiment_environment must exist"
    return importlib.import_module("scripts.collect_experiment_environment")


@pytest.fixture
def formal_config(tmp_path: Path) -> dict:
    config = load_config("configs/default.yaml", "homogeneous")
    target = tmp_path / "target"
    target.mkdir()
    config["model_runner"]["target_model"] = str(target)
    for profile, values in config["model_runner"]["drafter_models"].items():
        model = tmp_path / profile
        model.mkdir()
        values["model"] = str(model)
    config["experiment"] = {
        "internal_time_unit": "ms",
        "csv_time_unit": "ms",
        "use_fake_model_runner": False,
        "request_device_assignment": "fixed",
    }
    return config


def test_valid_formal_config_exports_complete_resolved_document(
    formal_config: dict, tmp_path: Path
) -> None:
    audit = _audit_module()

    resolved = audit.audit_experiment_config(
        formal_config,
        scenario="homogeneous",
        methods=CANONICAL_METHODS,
        use_fake_model_runner=False,
        repo_root=Path.cwd(),
    )
    output = tmp_path / "resolved_config.json"
    audit.write_resolved_config(output, resolved)

    exported = json.loads(output.read_text(encoding="utf-8"))
    assert exported == resolved
    assert exported["scenario"] == "homogeneous"
    assert tuple(exported["methods"]) == CANONICAL_METHODS
    assert exported["config"] == formal_config
    assert len(exported["git_commit"]) == 40


@pytest.mark.parametrize("method", ["server_only", "SpecEdge", "sync_batch_sd", "full"])
def test_audit_rejects_noncanonical_method(formal_config: dict, method: str) -> None:
    audit = _audit_module()

    with pytest.raises(audit.ConfigAuditError, match="canonical method"):
        audit.audit_experiment_config(
            formal_config,
            scenario="homogeneous",
            methods=(method,),
            use_fake_model_runner=False,
            repo_root=Path.cwd(),
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c["server_only"].update(batch_size=2), "server_only.batch_size must be 1"),
        (
            lambda c: c["server_only"].update(tree_draft_strategy="linear"),
            "server_only.tree_draft_strategy must be specexec_approx",
        ),
        (
            lambda c: c["specedge"].update(tree_draft_strategy="linear"),
            "specedge.tree_draft_strategy must be specexec_approx",
        ),
        (lambda c: c["simulation"].update(request_arrival="burst"), "must not use burst"),
        (
            lambda c: c["model_runner"].update(target_model="/definitely/missing/target"),
            "model path does not exist",
        ),
        (
            lambda c: c["model_runner"].update(target_device="cuda:0"),
            "target and drafter devices must differ",
        ),
        (lambda c: c["experiment"].update(use_fake_model_runner=True), "fake runner must be disabled"),
        (lambda c: c["simulation"].pop("seed"), "simulation.seed must be explicit"),
        (lambda c: c["simulation"].update(num_requests=0), "simulation.num_requests must be positive"),
        (
            lambda c: c["simulation"].update(output_len_choices=[0]),
            "simulation.output_len_choices must contain positive integers",
        ),
        (
            lambda c: c["simulation"].update(poisson_rate_per_s=0),
            "simulation.poisson_rate_per_s must be positive",
        ),
        (lambda c: c["experiment"].pop("internal_time_unit"), "internal_time_unit must be explicit"),
        (lambda c: c["experiment"].pop("csv_time_unit"), "csv_time_unit must be explicit"),
    ],
)
def test_audit_rejects_invalid_formal_config(
    formal_config: dict, mutate, message: str
) -> None:
    audit = _audit_module()
    config = copy.deepcopy(formal_config)
    mutate(config)

    with pytest.raises(audit.ConfigAuditError, match=message):
        audit.audit_experiment_config(
            config,
            scenario="homogeneous",
            methods=CANONICAL_METHODS,
            use_fake_model_runner=False,
            repo_root=Path.cwd(),
        )


def test_audit_rejects_cli_fake_runner_even_when_config_disables_it(
    formal_config: dict,
) -> None:
    audit = _audit_module()

    with pytest.raises(audit.ConfigAuditError, match="fake runner must be disabled"):
        audit.audit_experiment_config(
            formal_config,
            scenario="homogeneous",
            methods=CANONICAL_METHODS,
            use_fake_model_runner=True,
            repo_root=Path.cwd(),
        )


def test_audit_script_supports_direct_execution_from_repository_root() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/audit_experiment_config.py", "--help"],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--resolved-config-out" in completed.stdout


def test_environment_manifest_contains_required_hardware_and_software_fields(
    tmp_path: Path,
) -> None:
    collector = _environment_module()

    manifest = collector.collect_environment(repo_root=Path.cwd())
    output = tmp_path / "environment_manifest.json"
    collector.write_environment_manifest(output, manifest)

    exported = json.loads(output.read_text(encoding="utf-8"))
    assert exported == manifest
    assert exported["schema_version"] == 1
    assert exported["collected_at_utc"].endswith("Z")
    assert exported["git"]["commit"]
    assert isinstance(exported["git"]["dirty"], bool)
    for field in (
        "python_version",
        "pytorch_version",
        "transformers_version",
        "cuda_runtime_version",
    ):
        assert field in exported["software"]
    assert "cuda_driver_version" in exported["hardware"]
    assert "gpus" in exported["hardware"]
    for gpu in exported["hardware"]["gpus"]:
        assert {"index", "model", "memory_total_mib"} <= set(gpu)


def test_environment_manifest_records_missing_nvidia_smi_as_collection_error() -> None:
    collector = _environment_module()

    manifest = collector.collect_environment(
        repo_root=Path.cwd(),
        nvidia_smi_command=("definitely-not-an-installed-nvidia-smi",),
    )

    assert manifest["hardware"]["gpu_query_status"] == "unavailable"
    assert manifest["hardware"]["gpus"] == []
    assert any("nvidia-smi" in error for error in manifest["collection_errors"])
