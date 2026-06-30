from __future__ import annotations

from pathlib import Path

import pytest

from src.config import resolve_target_latency_profile_path, validate_config
from src.simulator import Simulator
from tests.common import accepting_model_runner, small_config


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
