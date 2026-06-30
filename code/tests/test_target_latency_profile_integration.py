from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import resolve_target_latency_profile_path, validate_config
from src.latency import (
    TargetLatencyModel,
    target_only_latency_ms,
    verify_latency_ms,
)
from src.simulator import Simulator
from src.verification_latency_profile import ProfileValidationError
from tests.common import accepting_model_runner, small_config


PROFILE_FIELDS = (
    "method",
    "batch_size",
    "context_length",
    "gamma",
    "tree_nodes",
    "mean_ms",
    "p50_ms",
    "p95_ms",
    "std_ms",
    "tree_mode",
    "status",
)


def _profile_row(
    method: str,
    batch_size: int,
    context_length: int,
    *,
    gamma: int | str = "",
    tree_nodes: int | str = "",
    p50_ms: float,
    status: str = "success",
) -> dict[str, object]:
    success = status == "success"
    return {
        "method": method,
        "batch_size": batch_size,
        "context_length": context_length,
        "gamma": gamma,
        "tree_nodes": tree_nodes,
        "mean_ms": p50_ms + 1.0 if success else "",
        "p50_ms": p50_ms if success else "",
        "p95_ms": p50_ms + 2.0 if success else "",
        "std_ms": 0.5 if success else "",
        "tree_mode": (
            "fixed_forward_approx" if method == "tree_verification" else ""
        ),
        "status": status,
    }


@pytest.fixture
def integration_profile_path(tmp_path: Path) -> Path:
    rows: list[dict[str, object]] = []
    for context in (128, 512, 1024, 2048):
        for batch in (1, 2, 4, 8, 16):
            status = "oom" if (batch, context) == (16, 2048) else "success"
            rows.append(
                _profile_row(
                    "target_decode",
                    batch,
                    context,
                    p50_ms=float(batch * 100 + context // 128 * 10),
                    status=status,
                )
            )
            for gamma in (1, 2, 4, 8):
                rows.append(
                    _profile_row(
                        "linear_verification",
                        batch,
                        context,
                        gamma=gamma,
                        p50_ms=float(
                            batch * 100 + context // 128 * 10 + gamma
                        ),
                        status=status,
                    )
                )
            for nodes in (8, 16):
                rows.append(
                    _profile_row(
                        "tree_verification",
                        batch,
                        context,
                        tree_nodes=nodes,
                        p50_ms=float(batch * 100 + context // 128 * 10),
                        status=status,
                    )
                )
    path = tmp_path / "target-profile.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROFILE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _profile_config(path: Path) -> dict:
    config, _, _ = small_config(num_requests=1, output_len=2)
    config["target_latency"].update(
        mode="profile",
        profile_path=str(path),
        metric="p50_ms",
    )
    return config


@pytest.mark.parametrize(
    "method",
    (
        "server_only_linear",
        "server_only_tree",
        "specedge_linear",
        "specedge_tree",
        "dip_sd",
    ),
)
def test_segment_prefix_is_verifier_kv_prefix_before_current_draft(
    method: str,
) -> None:
    config, _, workload = small_config(num_requests=2, output_len=6)
    config["speculation"]["gamma_fixed"] = 2
    config["speculation"]["gamma_candidates"] = [2]
    config["specedge"]["server_batch_size"] = 2
    simulator = Simulator(
        config,
        accepting_model_runner(),
        workload,
        "combined_strong_heterogeneous",
        method,
    )
    simulator.run()

    contexts = simulator._verification_context_lengths(simulator.segments)

    assert contexts == tuple(
        len(segment.prefix_ids) for segment in simulator.segments
    )
    for segment, context in zip(simulator.segments, contexts):
        request = simulator.requests[segment.request_id]
        assert context == len(request.prompt_ids) + segment.base_pos
        assert context + segment.verify_gamma == (
            len(segment.prefix_ids) + len(segment.draft_ids)
        )
        assert segment.verify_gamma == len(segment.draft_ids)


def test_verification_context_invariant_rejects_draft_tokens_in_prefix() -> None:
    config, model_runner, workload = small_config(num_requests=1, output_len=4)
    simulator = Simulator(
        config,
        model_runner,
        workload,
        "combined_strong_heterogeneous",
        "server_only_linear",
    )
    simulator.run()
    segment = simulator.segments[0]
    segment.prefix_ids = segment.prefix_ids + segment.draft_ids

    with pytest.raises(RuntimeError, match="verification prefix length"):
        simulator._verification_context_lengths([segment])


def test_default_target_latency_configuration_is_analytical() -> None:
    config, _, _ = small_config(num_requests=1, output_len=1)

    assert config["target_latency"] == {
        "mode": "analytical",
        "profile_path": (
            "outputs/profiling/target_verification_latency_full_merged.csv"
        ),
        "metric": "p50_ms",
    }


@pytest.mark.parametrize("mode", ("dynamic", "measured"))
def test_invalid_target_latency_mode_is_rejected(mode: str) -> None:
    config, _, _ = small_config(num_requests=1, output_len=1)
    config["target_latency"] = {"mode": mode, "metric": "p50_ms"}

    with pytest.raises(ValueError, match="target_latency.mode"):
        validate_config(config)


def test_invalid_target_latency_metric_is_rejected() -> None:
    config, _, _ = small_config(num_requests=1, output_len=1)
    config["target_latency"]["metric"] = "std_ms"

    with pytest.raises(ValueError, match="target_latency.metric"):
        validate_config(config)


@pytest.mark.parametrize("profile_path", ("", "   "))
def test_profile_mode_requires_nonempty_path(profile_path: str) -> None:
    config, _, _ = small_config(num_requests=1, output_len=1)
    config["target_latency"].update(mode="profile", profile_path=profile_path)

    with pytest.raises(ValueError, match="target_latency.profile_path"):
        validate_config(config)


def test_profile_mode_rejects_missing_path(tmp_path: Path) -> None:
    config, _, _ = small_config(num_requests=1, output_len=1)
    config["target_latency"].update(
        mode="profile",
        profile_path=str(tmp_path / "missing.csv"),
    )

    with pytest.raises(ValueError, match="profile_path.*does not exist"):
        validate_config(config)


def test_analytical_mode_does_not_require_profile_file(tmp_path: Path) -> None:
    config, _, _ = small_config(num_requests=1, output_len=1)
    config["target_latency"].update(
        mode="analytical",
        profile_path=str(tmp_path / "missing.csv"),
    )

    validate_config(config)


def test_relative_profile_path_is_code_relative_and_cwd_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = Path(__file__).resolve().parents[1] / "outputs" / "profile.csv"
    monkeypatch.chdir(tmp_path)

    assert resolve_target_latency_profile_path("outputs/profile.csv") == expected


def test_analytical_facade_preserves_existing_formulas() -> None:
    config, _, _ = small_config(num_requests=1, output_len=2)
    model = TargetLatencyModel(config)

    assert model.target_decode_latency_ms(
        context_lengths=(128,), output_tokens=4
    ) == target_only_latency_ms(config["edge"], 4)
    assert model.linear_verification_latency_ms(
        context_lengths=(128, 128),
        gamma=4,
        analytical_work_units=(1, 1),
    ) == verify_latency_ms(config["edge"], (1, 1))
    assert model.tree_verification_latency_ms(
        context_lengths=(128,),
        tree_nodes=64,
        analytical_work_units=(1,),
    ) == verify_latency_ms(config["edge"], (1,))


def test_profile_facade_routes_all_three_methods(
    integration_profile_path: Path,
) -> None:
    model = TargetLatencyModel(_profile_config(integration_profile_path))

    assert model.target_decode_latency_ms(context_lengths=(128,)) == 110.0
    assert model.linear_verification_latency_ms(
        context_lengths=(128, 500, 900),
        gamma=3,
        analytical_work_units=(1, 1, 1),
    ) == 484.0
    assert model.tree_verification_latency_ms(
        context_lengths=(512,),
        tree_nodes=64,
        analytical_work_units=(1,),
    ) == 140.0


def test_profile_facade_consumes_oom_split_total_once(
    integration_profile_path: Path,
) -> None:
    model = TargetLatencyModel(_profile_config(integration_profile_path))
    assert model._profile is not None

    with patch.object(model._profile, "query", wraps=model._profile.query) as query:
        latency = model.target_decode_latency_ms(context_lengths=(2048,) * 16)

    assert latency == 1920.0
    query.assert_called_once()


def test_profile_facade_rejects_unloadable_csv(tmp_path: Path) -> None:
    path = tmp_path / "invalid.csv"
    path.write_text("bad,column\n1,2\n", encoding="utf-8")
    config = _profile_config(path)

    with pytest.raises(ProfileValidationError, match="missing required fields"):
        TargetLatencyModel(config)
