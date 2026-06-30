from __future__ import annotations

import csv
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.verification_latency_profile import (
    ProfileQueryError,
    ProfileValidationError,
    VerificationLatencyProfile,
)


FIELDS = (
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

BATCH_TIERS = (1, 2, 4, 8, 16)
CONTEXT_TIERS = (128, 512, 1024, 2048)
GAMMA_TIERS = (1, 2, 4, 8)
TREE_NODES = (8, 16)


def _latency(batch_size: int, context_length: int, gamma: int = 0) -> float:
    return float(batch_size * 100 + context_length // 128 * 10 + gamma)


def _row(
    method: str,
    batch_size: int,
    context_length: int,
    *,
    gamma: int | str = "",
    tree_nodes: int | str = "",
    status: str = "success",
) -> dict[str, object]:
    gamma_value = int(gamma) if gamma != "" else 0
    p50 = _latency(batch_size, context_length, gamma_value)
    success = status == "success"
    return {
        "method": method,
        "batch_size": batch_size,
        "context_length": context_length,
        "gamma": gamma,
        "tree_nodes": tree_nodes,
        "mean_ms": p50 + 1 if success else "",
        "p50_ms": p50 if success else "",
        "p95_ms": p50 + 2 if success else "",
        "std_ms": 0.5 if success else "",
        "tree_mode": "fixed_forward_approx" if method == "tree_verification" else "",
        "status": status,
    }


def _mock_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for context_length in CONTEXT_TIERS:
        for batch_size in BATCH_TIERS:
            status = "oom" if (batch_size, context_length) == (16, 2048) else "success"
            rows.append(_row("target_decode", batch_size, context_length, status=status))
            for gamma in GAMMA_TIERS:
                rows.append(
                    _row(
                        "linear_verification",
                        batch_size,
                        context_length,
                        gamma=gamma,
                        status=status,
                    )
                )
            for tree_nodes in TREE_NODES:
                rows.append(
                    _row(
                        "tree_verification",
                        batch_size,
                        context_length,
                        tree_nodes=tree_nodes,
                        status=status,
                    )
                )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def profile_path(tmp_path: Path) -> Path:
    return _write_csv(tmp_path / "profile.csv", _mock_rows())


def test_exact_query_uses_default_p50_and_immutable_provenance(profile_path: Path) -> None:
    profile = VerificationLatencyProfile(profile_path)

    result = profile.query("target_decode", batch_size=2, context_length=512)

    assert profile.metric == "p50_ms"
    assert result.actual_batch_size == 2
    assert result.profile_batch_size == 2
    assert result.profile_batch_sizes == (2,)
    assert result.subbatch_count == 1
    assert result.subbatch_sizes == (2,)
    assert result.profile_context_length == 512
    assert result.profile_gamma is None
    assert result.total_latency_ms == 240.0
    assert result.tree_mode is None
    assert isinstance(result.source_rows, tuple)
    assert result.source_rows[0].metric == "p50_ms"
    assert result.source_rows[0].metric_value == 240.0
    assert isinstance(result.source_rows[0].raw_row, tuple)
    with pytest.raises(FrozenInstanceError):
        result.total_latency_ms = 0.0


@pytest.mark.parametrize(
    ("metric", "expected"),
    [("mean_ms", 111.0), ("p95_ms", 112.0)],
)
def test_metric_can_select_mean_or_p95(
    profile_path: Path, metric: str, expected: float
) -> None:
    profile = VerificationLatencyProfile(profile_path, metric=metric)

    result = profile.query("target_decode", batch_size=1, context_length=128)

    assert result.total_latency_ms == expected
    assert result.source_rows[0].metric == metric


def test_constructor_builds_legal_tiers_from_success_and_oom_rows(
    profile_path: Path,
) -> None:
    profile = VerificationLatencyProfile(profile_path)

    assert profile.legal_batch_sizes == BATCH_TIERS
    assert profile.legal_context_lengths == CONTEXT_TIERS
    assert profile.legal_gammas == GAMMA_TIERS
    assert profile.legal_tree_nodes == TREE_NODES
    assert isinstance(profile.oom_rows, tuple)
    assert any(
        row.method == "target_decode"
        and row.batch_size == 16
        and row.context_length == 2048
        for row in profile.oom_rows
    )


def test_query_uses_loaded_index_after_csv_is_deleted(profile_path: Path) -> None:
    profile = VerificationLatencyProfile(profile_path)
    profile_path.unlink()

    result = profile.query("target_decode", batch_size=1, context_length=128)

    assert result.total_latency_ms == 110.0


def test_batch_context_and_gamma_round_up(profile_path: Path) -> None:
    profile = VerificationLatencyProfile(profile_path)

    result = profile.query(
        "linear_verification",
        batch_size=3,
        context_length=900,
        gamma=3,
    )

    assert result.actual_batch_size == 3
    assert result.profile_batch_size == 4
    assert result.profile_batch_sizes == (4,)
    assert result.subbatch_sizes == (3,)
    assert result.profile_context_length == 1024
    assert result.profile_gamma == 4
    assert result.total_latency_ms == 484.0


def test_mixed_context_uses_global_max_padding_without_virtual_requests(
    profile_path: Path,
) -> None:
    profile = VerificationLatencyProfile(profile_path)

    result = profile.query(
        "linear_verification",
        batch_size=3,
        context_lengths=(120, 500, 900),
        gamma=3,
    )

    assert result.actual_batch_size == 3
    assert result.profile_batch_sizes == (4,)
    assert result.subbatch_sizes == (3,)
    assert sum(result.subbatch_sizes) == 3
    assert result.profile_context_length == 1024


@pytest.mark.parametrize(
    ("batch_size", "context_length", "subbatches", "profile_batches", "expected"),
    [
        (16, 2048, (8, 8), (8, 8), 1920.0),
        (9, 2048, (8, 1), (8, 1), 1220.0),
        (20, 2048, (8, 8, 4), (8, 8, 4), 2480.0),
        (17, 1024, (16, 1), (16, 1), 1860.0),
    ],
)
def test_infeasible_or_oversized_batch_splits_serially(
    profile_path: Path,
    batch_size: int,
    context_length: int,
    subbatches: tuple[int, ...],
    profile_batches: tuple[int, ...],
    expected: float,
) -> None:
    profile = VerificationLatencyProfile(profile_path)

    result = profile.query(
        "target_decode",
        batch_size=batch_size,
        context_length=context_length,
    )

    assert result.subbatch_sizes == subbatches
    assert result.profile_batch_sizes == profile_batches
    assert result.profile_batch_size == max(profile_batches)
    assert result.subbatch_count == len(subbatches)
    assert result.total_latency_ms == expected
    assert result.total_latency_ms == sum(row.metric_value for row in result.source_rows)
    assert all(row.status == "success" for row in result.source_rows)


def test_tree_uses_smallest_node_canonical_row_and_keeps_requested_metadata(
    profile_path: Path,
) -> None:
    profile = VerificationLatencyProfile(profile_path)

    result = profile.query(
        "tree_verification",
        batch_size=4,
        context_length=512,
        tree_nodes=64,
    )

    assert result.total_latency_ms == 440.0
    assert result.tree_mode == "fixed_forward_approx"
    assert result.source_rows[0].tree_nodes == 8
    assert result.source_rows[0].requested_tree_nodes == 64


def test_tree_inconsistent_statistics_are_rejected(tmp_path: Path) -> None:
    rows = _mock_rows()
    row = next(
        item
        for item in rows
        if item["method"] == "tree_verification"
        and item["batch_size"] == 1
        and item["context_length"] == 128
        and item["tree_nodes"] == 16
    )
    row["p50_ms"] = float(row["p50_ms"]) + 0.001
    path = _write_csv(tmp_path / "inconsistent.csv", rows)

    with pytest.raises(ProfileValidationError, match="tree statistics"):
        VerificationLatencyProfile(path)


def test_tree_wrong_mode_is_rejected(tmp_path: Path) -> None:
    rows = _mock_rows()
    tree_row = next(row for row in rows if row["method"] == "tree_verification")
    tree_row["tree_mode"] = "real_tree_kernel"
    path = _write_csv(tmp_path / "wrong_mode.csv", rows)

    with pytest.raises(ProfileValidationError, match="fixed_forward_approx"):
        VerificationLatencyProfile(path)


def test_duplicate_original_key_is_rejected(tmp_path: Path) -> None:
    rows = _mock_rows()
    rows.append(dict(rows[0]))
    path = _write_csv(tmp_path / "duplicate.csv", rows)

    with pytest.raises(ProfileValidationError, match="duplicate profile row"):
        VerificationLatencyProfile(path)


@pytest.mark.parametrize(
    ("method", "kwargs", "message"),
    [
        ("target_decode", {"gamma": 1}, "does not accept gamma"),
        ("target_decode", {"tree_nodes": 8}, "does not accept tree_nodes"),
        ("linear_verification", {}, "requires gamma"),
        ("linear_verification", {"gamma": 0}, "positive integer gamma"),
        (
            "linear_verification",
            {"gamma": 1, "tree_nodes": 8},
            "does not accept tree_nodes",
        ),
        ("tree_verification", {}, "requires tree_nodes"),
        ("tree_verification", {"tree_nodes": 0}, "positive integer tree_nodes"),
        (
            "tree_verification",
            {"tree_nodes": 8, "gamma": 1},
            "does not accept gamma",
        ),
    ],
)
def test_method_arguments_are_strict(
    profile_path: Path, method: str, kwargs: dict[str, int], message: str
) -> None:
    profile = VerificationLatencyProfile(profile_path)

    with pytest.raises(ProfileQueryError, match=message):
        profile.query(method, batch_size=1, context_length=128, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "exactly one"),
        (
            {"context_length": 128, "context_lengths": (128,)},
            "exactly one",
        ),
        ({"context_lengths": (128,)}, "must equal batch_size"),
        ({"context_length": 0}, "positive integer context"),
        ({"context_lengths": (128, 0)}, "positive integer context"),
    ],
)
def test_context_arguments_are_strict(
    profile_path: Path, kwargs: dict[str, object], message: str
) -> None:
    profile = VerificationLatencyProfile(profile_path)

    with pytest.raises(ProfileQueryError, match=message):
        profile.query("target_decode", batch_size=2, **kwargs)


def test_bounds_and_method_errors_are_explicit(profile_path: Path) -> None:
    profile = VerificationLatencyProfile(profile_path)

    with pytest.raises(ProfileQueryError, match="context.*2048"):
        profile.query("target_decode", batch_size=1, context_length=2049)
    with pytest.raises(ProfileQueryError, match="gamma.*8"):
        profile.query(
            "linear_verification",
            batch_size=1,
            context_length=128,
            gamma=9,
        )
    with pytest.raises(ProfileQueryError, match="unsupported method"):
        profile.query("server_only", batch_size=1, context_length=128)
    with pytest.raises(ProfileQueryError, match="positive integer batch_size"):
        profile.query("target_decode", batch_size=0, context_length=128)


def test_unsupported_metric_is_rejected(profile_path: Path) -> None:
    with pytest.raises(ProfileValidationError, match="unsupported metric"):
        VerificationLatencyProfile(profile_path, metric="std_ms")


def test_no_feasible_success_batch_is_explicit(tmp_path: Path) -> None:
    rows = _mock_rows()
    for row in rows:
        if row["method"] == "target_decode" and row["context_length"] == 512:
            row.update(status="oom", mean_ms="", p50_ms="", p95_ms="", std_ms="")
    path = _write_csv(tmp_path / "no_feasible.csv", rows)
    profile = VerificationLatencyProfile(path)

    with pytest.raises(ProfileQueryError, match="no feasible success batch"):
        profile.query("target_decode", batch_size=1, context_length=500)
