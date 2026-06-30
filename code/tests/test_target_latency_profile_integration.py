from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.config import resolve_target_latency_profile_path, validate_config
from src.latency import (
    TargetLatencyModel,
    target_only_latency_ms,
    verify_latency_ms,
)
from src.simulator import Simulator
from src.verification_latency_profile import (
    ProfileValidationError,
    VerificationLatencyProfile,
)
from src.workload import WorkloadItem
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


def test_simulator_constructs_profile_once_and_queries_each_decode_token(
    integration_profile_path: Path,
) -> None:
    config, model_runner, workload = small_config(num_requests=1, output_len=3)
    config["target_latency"].update(
        mode="profile",
        profile_path=str(integration_profile_path),
        metric="p50_ms",
    )
    calls: list[dict[str, object]] = []

    class RecordingProfile(VerificationLatencyProfile):
        constructions = 0

        def __init__(self, *args, **kwargs):
            type(self).constructions += 1
            super().__init__(*args, **kwargs)

        def query(self, method, **kwargs):
            calls.append({"method": method, **kwargs})
            return super().query(method, **kwargs)

    with patch("src.latency.VerificationLatencyProfile", RecordingProfile):
        result = Simulator(
            config,
            model_runner,
            workload,
            "combined_strong_heterogeneous",
            "target_only",
        ).run()

    assert RecordingProfile.constructions == 1
    assert [call["method"] for call in calls] == ["target_decode"] * 3
    prompt = result.requests[0].prompt_token_count
    assert [call["context_lengths"] for call in calls] == [
        (prompt,),
        (prompt + 1,),
        (prompt + 2,),
    ]
    assert all(call["batch_size"] == 1 for call in calls)


def test_profile_target_only_uses_cumulative_commit_timestamps(
    integration_profile_path: Path,
) -> None:
    config, model_runner, _ = small_config(num_requests=1, output_len=3)
    workload = [WorkloadItem("0", "x" * 127, 2)]
    config["target_latency"].update(
        mode="profile",
        profile_path=str(integration_profile_path),
        metric="p50_ms",
    )

    result = Simulator(
        config,
        model_runner,
        workload,
        "combined_strong_heterogeneous",
        "target_only",
    ).run()

    request = result.requests[0]
    event = next(
        item for item in result.event_trace if item["event"] == "target_only_service"
    )
    assert request.committed_token_times_ms == [110.0, 220.0, 360.0]
    assert event["compute_ms"] == 360.0
    assert len(
        [
            item
            for item in result.event_trace
            if item["event"] == "target_only_service"
        ]
    ) == 1


def test_analytical_target_only_preserves_total_and_timestamps() -> None:
    config, model_runner, workload = small_config(num_requests=1, output_len=3)
    config["edge"]["target_only_startup_ms"] = 7.0
    config["target_latency"]["mode"] = "analytical"

    result = Simulator(
        config,
        model_runner,
        workload,
        "combined_strong_heterogeneous",
        "target_only",
    ).run()

    request = result.requests[0]
    expected = target_only_latency_ms(config["edge"], 3)
    interval = 1000.0 / config["edge"]["target_only_token_rate_tok_s"]
    assert request.target_only_compute_ms == expected
    assert request.committed_token_times_ms == [
        expected - interval * 2,
        expected - interval,
        expected,
    ]


def test_analytical_simulator_never_constructs_or_reads_profile(
    tmp_path: Path,
) -> None:
    config, model_runner, workload = small_config(num_requests=1, output_len=2)
    config["target_latency"].update(
        mode="analytical",
        profile_path=str(tmp_path / "missing.csv"),
    )

    with patch(
        "src.latency.VerificationLatencyProfile",
        side_effect=AssertionError("analytical mode accessed profile CSV"),
    ) as profile_type:
        Simulator(
            config,
            model_runner,
            workload,
            "combined_strong_heterogeneous",
            "target_only",
        ).run()

    profile_type.assert_not_called()


def test_profile_target_only_rejects_context_beyond_profile(
    integration_profile_path: Path,
) -> None:
    config, model_runner, _ = small_config(num_requests=1, output_len=2)
    workload = [WorkloadItem("0", "x" * 2048, 2)]
    config["target_latency"].update(
        mode="profile",
        profile_path=str(integration_profile_path),
        metric="p50_ms",
    )

    with pytest.raises(ValueError, match="context.*2048"):
        Simulator(
            config,
            model_runner,
            workload,
            "combined_strong_heterogeneous",
            "target_only",
        ).run()


def _run_with_recorded_profile(
    method: str,
    profile_path: Path,
    *,
    num_requests: int = 2,
    output_len: int = 6,
):
    config, model_runner, workload = small_config(num_requests, output_len)
    config["target_latency"].update(
        mode="profile",
        profile_path=str(profile_path),
        metric="p50_ms",
    )
    config["speculation"]["gamma_fixed"] = 2
    config["speculation"]["gamma_candidates"] = [2]
    config["specedge"]["server_batch_size"] = num_requests
    calls: list[dict[str, object]] = []

    class RecordingProfile(VerificationLatencyProfile):
        def query(self, method_name, **kwargs):
            calls.append({"method": method_name, **kwargs})
            return super().query(method_name, **kwargs)

    with patch("src.latency.VerificationLatencyProfile", RecordingProfile):
        simulator = Simulator(
            config,
            model_runner,
            workload,
            "combined_strong_heterogeneous",
            method,
        )
        result = simulator.run()
    return simulator, result, calls


@pytest.mark.parametrize(
    "method", ("server_only_linear", "specedge_linear", "dip_sd")
)
def test_canonical_linear_paths_query_once_per_logical_batch(
    method: str,
    integration_profile_path: Path,
) -> None:
    simulator, result, calls = _run_with_recorded_profile(
        method, integration_profile_path
    )
    linear_calls = [
        call for call in calls if call["method"] == "linear_verification"
    ]
    verify_events = [
        event
        for event in result.event_trace
        if event["event"]
        in {"server_only_verify", "global_batch_verify", "dip_sd_batch_verify"}
    ]

    assert len(linear_calls) == len(verify_events)
    for call, event in zip(linear_calls, verify_events):
        segment_ids = event.get("segment_ids")
        if segment_ids is None:
            segment_ids = [event["segment_id"]]
        segments = [simulator.segments[index] for index in segment_ids]
        assert call["batch_size"] == len(segments)
        assert call["context_lengths"] == tuple(
            len(segment.prefix_ids) for segment in segments
        )
        assert call["gamma"] == max(
            len(segment.draft_ids) for segment in segments
        )


def test_linear_profile_receives_mixed_context_and_actual_longest_gamma(
    integration_profile_path: Path,
) -> None:
    config, model_runner, workload = small_config(num_requests=3, output_len=4)
    config["target_latency"].update(
        mode="profile",
        profile_path=str(integration_profile_path),
        metric="p50_ms",
    )
    simulator = Simulator(
        config,
        model_runner,
        workload,
        "combined_strong_heterogeneous",
        "specedge_linear",
    )
    simulator._schedule_request_arrivals()
    simulator.requests[0].prompt_ids = [1] * 120
    simulator.requests[1].prompt_ids = [1] * 500
    simulator.requests[2].prompt_ids = [1] * 900
    segments = []
    for request_id, gamma in enumerate((2, 4, 3)):
        segments.append(
            SimpleNamespace(
                segment_id=request_id,
                request_id=request_id,
                base_pos=0,
                prefix_ids=list(simulator.requests[request_id].prompt_ids),
                draft_ids=[2] * gamma,
                verify_gamma=gamma,
                draft_tree=None,
                target_verify_tree_nodes=1,
            )
        )
    assert simulator.target_latency._profile is not None

    with patch.object(
        simulator.target_latency._profile,
        "query",
        wraps=simulator.target_latency._profile.query,
    ) as query:
        latency = simulator._verify_latency_for_segments(segments)

    assert latency == 484.0
    query.assert_called_once_with(
        "linear_verification",
        batch_size=3,
        context_lengths=(120, 500, 900),
        gamma=4,
    )


def test_scheduler_prediction_remains_analytical_in_profile_mode(
    integration_profile_path: Path,
) -> None:
    config = _profile_config(integration_profile_path)
    _, model_runner, workload = small_config(num_requests=1, output_len=2)
    simulator = Simulator(
        config,
        model_runner,
        workload,
        "combined_strong_heterogeneous",
        "specedge_linear",
    )
    assert simulator.target_latency._profile is not None

    with patch.object(simulator.target_latency._profile, "query") as query:
        predicted = simulator.predict_verify_latency_ms(8)

    assert predicted == verify_latency_ms(config["edge"], [1])
    query.assert_not_called()


@pytest.mark.parametrize("method", ("server_only_tree", "specedge_tree"))
def test_canonical_tree_paths_query_fixed_forward_metadata(
    method: str,
    integration_profile_path: Path,
) -> None:
    simulator, result, calls = _run_with_recorded_profile(
        method,
        integration_profile_path,
        num_requests=2,
        output_len=6,
    )
    tree_calls = [
        call for call in calls if call["method"] == "tree_verification"
    ]
    verify_events = [
        event
        for event in result.event_trace
        if event["event"] in {"server_only_verify", "global_batch_verify"}
    ]

    assert len(tree_calls) == len(verify_events)
    for call, event in zip(tree_calls, verify_events):
        segment_ids = event.get("segment_ids")
        if segment_ids is None:
            segment_ids = [event["segment_id"]]
        segments = [simulator.segments[index] for index in segment_ids]
        assert call["batch_size"] == len(segments)
        assert call["context_lengths"] == tuple(
            len(segment.prefix_ids) for segment in segments
        )
        assert call["tree_nodes"] == max(
            segment.target_verify_tree_nodes for segment in segments
        )
        assert call["tree_nodes"] >= 1


def test_tree_profile_mode_guard_rejects_non_approximate_result() -> None:
    config, _, _ = small_config(num_requests=1, output_len=1)
    model = TargetLatencyModel(config)
    model.mode = "profile"
    model._profile = SimpleNamespace(
        query=lambda *args, **kwargs: SimpleNamespace(
            tree_mode="real_tree_kernel",
            total_latency_ms=1.0,
        )
    )

    with pytest.raises(ValueError, match="fixed_forward_approx"):
        model.tree_verification_latency_ms(
            context_lengths=(128,),
            tree_nodes=64,
            analytical_work_units=(1,),
        )
