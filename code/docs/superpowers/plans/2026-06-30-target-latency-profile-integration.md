# Target Latency Profile Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route canonical target decode and verification execution latency through one shared analytical/profile facade while preserving all current analytical, batching, and scheduling behavior.

**Architecture:** Add `TargetLatencyModel` to `src/latency.py`; one instance belongs to each `Simulator`, and profile mode owns one `VerificationLatencyProfile`. `Simulator` supplies real batch dimensions and consumes one returned latency total per logical operation, while `predict_verify_latency_ms` remains analytical.

**Tech Stack:** Python 3.10+, pytest/unittest, PyYAML, frozen profile query dataclasses, fake model runner, temporary CSV fixtures.

---

## File Map

- Create `code/tests/test_target_latency_profile_integration.py`: dedicated CPU-only integration coverage and temporary profile fixtures.
- Modify `code/src/latency.py`: shared `TargetLatencyModel` facade; retain all existing analytical functions unchanged.
- Modify `code/src/config.py`: target-latency validation and stable `code/`-relative path resolution.
- Modify `code/configs/default.yaml`: analytical default and profile defaults.
- Modify `code/src/simulator.py`: construct one facade, establish verifier-prefix invariants, and route canonical actual execution latency.
- Inspect but do not modify `code/src/model_runner.py`, `code/src/entities.py`, `code/src/tree_drafting.py`, and `code/src/scheduler.py`: semantic authorities for prefixes, segment lengths, tree metadata, and analytical scheduler prediction.
- Re-run `code/tests/test_verification_latency_profile.py` unchanged: query-layer padding, tiering, provenance, and OOM behavior remain authoritative.

## Phase 1: Call-Chain and Dimension Investigation

Before editing production code, run these read-only commands from the workspace root:

```bash
rtk rg -n "def verify_latency_ms|def target_only_latency_ms" code/src/latency.py
rtk rg -n "def __init__|def _on_target_only_arrive_edge|def _start_server_only_verify|def _verify_latency_for_segments|def _maybe_start_batch|def _run_dip_sd|def predict_verify_latency_ms" code/src/simulator.py
rtk rg -n "predict_verify_latency_ms" code/src/simulator.py code/src/scheduler.py
rtk rg -n "prefix_ids =|prefix_ids=|draft_ids=|target_verify_tree_nodes=" code/src/simulator.py code/src/model_runner.py code/src/tree_drafting.py
rtk rg -n "def load_config|def validate_config|def _read_yaml" code/src/config.py
rtk sed -n '1,145p' code/configs/default.yaml
```

Record these findings in the implementation commit notes before proceeding:

- `Simulator.__init__` is the correct lifetime boundary for one `TargetLatencyModel`.
- Target-only actual service is charged in `_on_target_only_arrive_edge` and currently produces one request-level event.
- Server-only actual verification calls `_verify_latency_for_segments([segment])` from `_start_server_only_verify`.
- SpecEdge actual batch verification calls `_verify_latency_for_segments(segments)` from `_maybe_start_batch`.
- DiP-SD actual batch verification calls the same helper from `_run_dip_sd`.
- Legacy lane execution and scheduler scoring call `predict_verify_latency_ms`; those call sites must remain unchanged.
- DiP-SD and server-only build `prefix_ids` from `request.prompt_ids + request.generated_ids` before drafting.
- SpecEdge builds `prefix_ids` through `_draft_prefix`; its `base_pos` advances by any prior active speculative segments included in that prefix.
- `Segment.verify_gamma` is `len(segment.draft_ids)`, so actual batch gamma is the maximum of those values.
- Linear segments receive `target_verify_tree_nodes=1`; tree segments copy the value produced by `DraftCandidateTree.target_verify_tree_nodes`, whose source is the tree-building strategy/model-runner result.

Do not infer the KV-prefix contract from the `prefix_ids` name. Task 1 makes that contract executable before profile integration.

### Task 1: Establish Verification-Prefix and Batch-Dimension Invariants

**Files:**
- Create: `code/tests/test_target_latency_profile_integration.py`
- Modify: `code/src/simulator.py` near `_verify_segments` and `_verify_latency_for_segments`
- Inspect: `code/src/model_runner.py`, `code/src/entities.py`, `code/src/tree_drafting.py`

- [ ] **Step 1: Write the failing semantic-helper tests**

Create the test module with tests that run real simulator segment construction using the fake model runner, then call a not-yet-existing helper:

```python
from __future__ import annotations

import pytest

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
def test_segment_prefix_is_verifier_kv_prefix_before_current_draft(method: str) -> None:
    config, _, workload = small_config(num_requests=2, output_len=6)
    config["speculation"]["gamma_fixed"] = 2
    config["speculation"]["gamma_candidates"] = [2]
    config["specedge"]["server_batch_size"] = 2
    result = Simulator(
        config,
        accepting_model_runner(),
        workload,
        "combined_strong_heterogeneous",
        method,
    ).run()
    simulator = Simulator(
        config,
        accepting_model_runner(),
        workload,
        "combined_strong_heterogeneous",
        method,
    )
    simulator.run()

    contexts = simulator._verification_context_lengths(simulator.segments)

    assert contexts == tuple(len(segment.prefix_ids) for segment in simulator.segments)
    for segment, context in zip(simulator.segments, contexts):
        request = simulator.requests[segment.request_id]
        assert context == len(request.prompt_ids) + segment.base_pos
        assert context + segment.verify_gamma == len(segment.prefix_ids) + len(segment.draft_ids)
        assert segment.verify_gamma == len(segment.draft_ids)
    assert result.segments


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
```

- [ ] **Step 2: Run the tests and observe RED**

Run:

```bash
cd code && rtk pytest -q tests/test_target_latency_profile_integration.py -k "segment_prefix or verification_context"
```

Expected: FAIL with `AttributeError: 'Simulator' object has no attribute '_verification_context_lengths'`. This proves the new tests require an explicit verified semantic boundary rather than merely restating a field name.

- [ ] **Step 3: Add the minimum invariant helper**

Add this helper immediately before `_verify_latency_for_segments`:

```python
    def _verification_context_lengths(
        self,
        segments: Sequence[Segment],
    ) -> tuple[int, ...]:
        contexts: list[int] = []
        for segment in segments:
            request = self.requests[segment.request_id]
            expected = len(request.prompt_ids) + segment.base_pos
            actual = len(segment.prefix_ids)
            if actual != expected:
                raise RuntimeError(
                    "verification prefix length does not match verifier KV prefix: "
                    f"segment={segment.segment_id}, expected={expected}, actual={actual}"
                )
            contexts.append(actual)
        return tuple(contexts)
```

Do not include `segment.draft_ids` in the returned lengths. Keep the existing latency helper unchanged in this task.

- [ ] **Step 4: Run the focused and existing semantic tests**

Run:

```bash
cd code && rtk pytest -q tests/test_target_latency_profile_integration.py tests/test_server_only_linear.py tests/test_server_only_tree.py tests/test_specedge_linear.py tests/test_specedge_tree.py tests/test_dip_sd.py
```

Expected: PASS. In particular, all constructed canonical segments satisfy `len(prefix_ids) == len(prompt_ids) + base_pos` before the current draft is verified.

- [ ] **Step 5: Commit this independent semantic boundary**

Commit scope:

```bash
rtk git add code/tests/test_target_latency_profile_integration.py code/src/simulator.py
rtk git commit -m "test: establish verification batch dimensions"
```

## Phase 2: Configuration and Shared Latency Facade

### Task 2: Add Target-Latency Configuration and Stable Path Resolution

**Files:**
- Modify: `code/configs/default.yaml` after `edge`
- Modify: `code/src/config.py` constants, `validate_config`, and a new path resolver
- Modify: `code/tests/test_target_latency_profile_integration.py`

- [ ] **Step 1: Add failing configuration tests**

Append:

```python
from pathlib import Path

from src.config import (
    resolve_target_latency_profile_path,
    validate_config,
)


def test_default_target_latency_configuration_is_analytical() -> None:
    config, _, _ = small_config(num_requests=1, output_len=1)
    assert config["target_latency"] == {
        "mode": "analytical",
        "profile_path": "outputs/profiling/target_verification_latency_full_merged.csv",
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
```

- [ ] **Step 2: Run the configuration tests and observe RED**

Run:

```bash
cd code && rtk pytest -q tests/test_target_latency_profile_integration.py -k "target_latency_configuration or target_latency_mode or target_latency_metric or profile_mode or analytical_mode or code_relative"
```

Expected: FAIL because `target_latency` is absent from the default YAML and `resolve_target_latency_profile_path` does not exist; invalid values are not yet validated.

- [ ] **Step 3: Add the minimum configuration implementation**

Add to `default.yaml`:

```yaml
target_latency:
  mode: analytical
  profile_path: outputs/profiling/target_verification_latency_full_merged.csv
  metric: p50_ms
```

Add to `config.py`:

```python
CODE_ROOT = Path(__file__).resolve().parents[1]
TARGET_LATENCY_MODES = {"analytical", "profile"}
TARGET_LATENCY_METRICS = {"p50_ms", "mean_ms", "p95_ms"}


def resolve_target_latency_profile_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return CODE_ROOT / candidate
```

At the start of `validate_config`, after validating `edge`, add:

```python
    target_latency = config.get("target_latency", {"mode": "analytical"})
    mode = str(target_latency.get("mode", ""))
    if mode not in TARGET_LATENCY_MODES:
        raise ValueError(
            "target_latency.mode must be analytical or profile"
        )
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
```

Do not instantiate `VerificationLatencyProfile` in `validate_config`; Task 3 performs the one full load owned by the facade. Analytical mode must not resolve or inspect `profile_path`.

- [ ] **Step 4: Run focused configuration regressions**

Run:

```bash
cd code && rtk pytest -q tests/test_target_latency_profile_integration.py tests/test_config.py
```

Expected: PASS, including initialization with a nonexistent analytical profile path.

- [ ] **Step 5: Commit configuration separately**

```bash
rtk git add code/configs/default.yaml code/src/config.py code/tests/test_target_latency_profile_integration.py
rtk git commit -m "feat: configure target latency modes"
```

### Task 3: Implement the Shared `TargetLatencyModel` Facade

**Files:**
- Modify: `code/src/latency.py`
- Modify: `code/tests/test_target_latency_profile_integration.py`
- Inspect: `code/src/verification_latency_profile.py`

- [ ] **Step 1: Add reusable temporary CSV fixtures and failing facade tests**

Add these imports and helpers to the integration test module:

```python
import csv
from types import SimpleNamespace
from unittest.mock import patch

from src.latency import (
    TargetLatencyModel,
    target_only_latency_ms,
    verify_latency_ms,
)
from src.verification_latency_profile import ProfileValidationError

PROFILE_FIELDS = (
    "method", "batch_size", "context_length", "gamma", "tree_nodes",
    "mean_ms", "p50_ms", "p95_ms", "std_ms", "tree_mode", "status",
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
        "tree_mode": "fixed_forward_approx" if method == "tree_verification" else "",
        "status": status,
    }


@pytest.fixture
def integration_profile_path(tmp_path: Path) -> Path:
    rows: list[dict[str, object]] = []
    for context in (128, 512, 1024, 2048):
        for batch in (1, 2, 4, 8, 16):
            status = "oom" if (batch, context) == (16, 2048) else "success"
            rows.append(_profile_row(
                "target_decode", batch, context,
                p50_ms=float(batch * 100 + context // 128 * 10), status=status,
            ))
            for gamma in (1, 2, 4, 8):
                rows.append(_profile_row(
                    "linear_verification", batch, context, gamma=gamma,
                    p50_ms=float(batch * 100 + context // 128 * 10 + gamma),
                    status=status,
                ))
            for nodes in (8, 16):
                rows.append(_profile_row(
                    "tree_verification", batch, context, tree_nodes=nodes,
                    p50_ms=float(batch * 100 + context // 128 * 10), status=status,
                ))
    path = tmp_path / "target-profile.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROFILE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _profile_config(path: Path) -> dict:
    config, _, _ = small_config(num_requests=1, output_len=2)
    config["target_latency"].update(
        mode="profile", profile_path=str(path), metric="p50_ms"
    )
    return config
```

Add the facade tests:

```python
def test_analytical_facade_preserves_existing_formulas() -> None:
    config, _, _ = small_config(num_requests=1, output_len=2)
    model = TargetLatencyModel(config)
    assert model.target_decode_latency_ms(
        context_lengths=(128,), output_tokens=4
    ) == target_only_latency_ms(config["edge"], 4)
    assert model.linear_verification_latency_ms(
        context_lengths=(128, 128), gamma=4, analytical_work_units=(1, 1)
    ) == verify_latency_ms(config["edge"], (1, 1))
    assert model.tree_verification_latency_ms(
        context_lengths=(128,), tree_nodes=64, analytical_work_units=(1,)
    ) == verify_latency_ms(config["edge"], (1,))


def test_profile_facade_routes_all_three_methods(integration_profile_path: Path) -> None:
    model = TargetLatencyModel(_profile_config(integration_profile_path))
    assert model.target_decode_latency_ms(context_lengths=(128,)) == 110.0
    assert model.linear_verification_latency_ms(
        context_lengths=(128, 500, 900), gamma=3, analytical_work_units=(1, 1, 1)
    ) == 484.0
    assert model.tree_verification_latency_ms(
        context_lengths=(512,), tree_nodes=64, analytical_work_units=(1,)
    ) == 140.0


def test_profile_facade_consumes_oom_split_total_once(
    integration_profile_path: Path,
) -> None:
    model = TargetLatencyModel(_profile_config(integration_profile_path))
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
```

- [ ] **Step 2: Run the facade tests and observe RED**

Run:

```bash
cd code && rtk pytest -q tests/test_target_latency_profile_integration.py -k facade
```

Expected: collection FAIL because `TargetLatencyModel` is not importable from `src.latency`.

- [ ] **Step 3: Implement the minimum facade without changing formulas**

Add imports:

```python
from pathlib import Path

from src.config import resolve_target_latency_profile_path
from src.verification_latency_profile import VerificationLatencyProfile
```

Add after `target_only_latency_ms`:

```python
class TargetLatencyModel:
    """Shared fixed-capacity target latency facade for one simulator."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.edge = config["edge"]
        target = config.get("target_latency", {"mode": "analytical"})
        self.mode = str(target.get("mode", "analytical"))
        self._profile: VerificationLatencyProfile | None = None
        if self.mode == "profile":
            path = resolve_target_latency_profile_path(str(target["profile_path"]))
            self._profile = VerificationLatencyProfile(
                path,
                metric=str(target.get("metric", "p50_ms")),
            )

    def target_decode_latency_ms(
        self,
        *,
        context_lengths: Sequence[int],
        output_tokens: int = 1,
    ) -> float:
        if self._profile is None:
            return target_only_latency_ms(self.edge, output_tokens)
        if output_tokens != 1:
            raise ValueError("profile target decode represents exactly one output token")
        return self._profile.query(
            "target_decode",
            batch_size=len(context_lengths),
            context_lengths=context_lengths,
        ).total_latency_ms

    def linear_verification_latency_ms(
        self,
        *,
        context_lengths: Sequence[int],
        gamma: int,
        analytical_work_units: Sequence[int],
    ) -> float:
        if self._profile is None:
            return verify_latency_ms(self.edge, analytical_work_units)
        return self._profile.query(
            "linear_verification",
            batch_size=len(context_lengths),
            context_lengths=context_lengths,
            gamma=gamma,
        ).total_latency_ms

    def tree_verification_latency_ms(
        self,
        *,
        context_lengths: Sequence[int],
        tree_nodes: int,
        analytical_work_units: Sequence[int],
    ) -> float:
        if self._profile is None:
            return verify_latency_ms(self.edge, analytical_work_units)
        result = self._profile.query(
            "tree_verification",
            batch_size=len(context_lengths),
            context_lengths=context_lengths,
            tree_nodes=tree_nodes,
        )
        if result.tree_mode != "fixed_forward_approx":
            raise ValueError("tree latency profile must use fixed_forward_approx")
        return result.total_latency_ms
```

Do not add compute multipliers or cache global profile instances. Remove the unused `Path` import if the implementation does not need it.

- [ ] **Step 4: Run facade and query-layer tests**

Run:

```bash
cd code && rtk pytest -q tests/test_target_latency_profile_integration.py -k facade
cd code && rtk pytest -q tests/test_verification_latency_profile.py
```

Expected: PASS. The B=16 test performs one facade query and receives the query layer's `1920.0` serial split total without caller-side summation.

- [ ] **Step 5: Commit the shared facade**

```bash
rtk git add code/src/latency.py code/tests/test_target_latency_profile_integration.py
rtk git commit -m "feat: add shared target latency facade"
```

## Phase 3: Canonical Execution Integration

### Task 4: Construct One Facade and Route Target-Only Decode

**Files:**
- Modify: `code/src/simulator.py` imports, `Simulator.__init__`, and `_on_target_only_arrive_edge`
- Modify: `code/tests/test_target_latency_profile_integration.py`
- Regress: `code/tests/test_target_only.py`, `code/tests/test_target_only_capacity.py`

- [ ] **Step 1: Add failing target-only and lifetime tests**

Append:

```python
from src.verification_latency_profile import VerificationLatencyProfile
from src.workload import WorkloadItem


def test_simulator_constructs_profile_once_and_queries_each_decode_token(
    integration_profile_path: Path,
) -> None:
    config, model_runner, workload = small_config(num_requests=1, output_len=3)
    config["target_latency"].update(
        mode="profile", profile_path=str(integration_profile_path), metric="p50_ms"
    )
    calls: list[dict[str, object]] = []
    real_type = VerificationLatencyProfile

    class RecordingProfile(real_type):
        constructions = 0

        def __init__(self, *args, **kwargs):
            type(self).constructions += 1
            super().__init__(*args, **kwargs)

        def query(self, method, **kwargs):
            calls.append({"method": method, **kwargs})
            return super().query(method, **kwargs)

    with patch("src.latency.VerificationLatencyProfile", RecordingProfile):
        result = Simulator(
            config, model_runner, workload,
            "combined_strong_heterogeneous", "target_only",
        ).run()

    assert RecordingProfile.constructions == 1
    assert [call["method"] for call in calls] == ["target_decode"] * 3
    prompt = result.requests[0].prompt_token_count
    assert [call["context_lengths"] for call in calls] == [
        (prompt,), (prompt + 1,), (prompt + 2,),
    ]
    assert all(call["batch_size"] == 1 for call in calls)


def test_profile_target_only_uses_cumulative_commit_timestamps(
    integration_profile_path: Path,
) -> None:
    config, model_runner, _ = small_config(num_requests=1, output_len=3)
    workload = [WorkloadItem("0", "x" * 127, 2)]
    config["target_latency"].update(
        mode="profile", profile_path=str(integration_profile_path), metric="p50_ms"
    )
    result = Simulator(
        config, model_runner, workload,
        "combined_strong_heterogeneous", "target_only",
    ).run()
    request = result.requests[0]
    event = next(item for item in result.event_trace if item["event"] == "target_only_service")
    assert request.committed_token_times_ms == [110.0, 220.0, 360.0]
    assert event["compute_ms"] == 360.0
    assert len([item for item in result.event_trace if item["event"] == "target_only_service"]) == 1


def test_analytical_target_only_preserves_total_and_timestamps() -> None:
    config, model_runner, workload = small_config(num_requests=1, output_len=3)
    config["edge"]["target_only_startup_ms"] = 7.0
    config["target_latency"]["mode"] = "analytical"
    result = Simulator(
        config, model_runner, workload,
        "combined_strong_heterogeneous", "target_only",
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


def test_analytical_simulator_never_constructs_or_reads_profile(tmp_path: Path) -> None:
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
            config, model_runner, workload,
            "combined_strong_heterogeneous", "target_only",
        ).run()
    profile_type.assert_not_called()


def test_profile_target_only_rejects_context_beyond_profile(
    integration_profile_path: Path,
) -> None:
    config, model_runner, _ = small_config(num_requests=1, output_len=2)
    workload = [WorkloadItem("0", "x" * 2048, 2)]
    config["target_latency"].update(
        mode="profile", profile_path=str(integration_profile_path), metric="p50_ms"
    )
    with pytest.raises(ValueError, match="context.*2048"):
        Simulator(
            config, model_runner, workload,
            "combined_strong_heterogeneous", "target_only",
        ).run()
```

- [ ] **Step 2: Run target-only tests and observe RED**

Run:

```bash
cd code && rtk pytest -q tests/test_target_latency_profile_integration.py -k "simulator_constructs or target_only"
```

Expected: FAIL because `Simulator` does not construct `TargetLatencyModel`, makes no `target_decode` queries, and still creates analytical uniform commit timestamps in profile mode.

- [ ] **Step 3: Add the minimum simulator integration**

Import `TargetLatencyModel`, construct it once in `Simulator.__init__`, and replace direct target-only latency calculation:

```python
        self.target_latency = TargetLatencyModel(config)
```

In `_on_target_only_arrive_edge`, preserve the semantic target call, FCFS resource, and one event. Use this exact mode split:

```python
        generated_ids = self.model_runner.target_only(request.prompt_ids, request.output_len)
        if self.target_latency.mode == "analytical":
            compute_ms = self.target_latency.target_decode_latency_ms(
                context_lengths=(request.prompt_token_count,),
                output_tokens=len(generated_ids),
            )
            token_interval_ms = 1000.0 / float(
                self.config["edge"]["target_only_token_rate_tok_s"]
            )
            commit_offsets_ms = [
                compute_ms - token_interval_ms * (len(generated_ids) - index - 1)
                for index in range(len(generated_ids))
            ]
        else:
            step_latencies = [
                self.target_latency.target_decode_latency_ms(
                    context_lengths=(request.prompt_token_count + index,),
                )
                for index in range(len(generated_ids))
            ]
            elapsed = 0.0
            commit_offsets_ms = []
            for latency_ms in step_latencies:
                elapsed += latency_ms
                commit_offsets_ms.append(elapsed)
            compute_ms = elapsed
```

After FCFS chooses `start_ms`, assign:

```python
        request.committed_token_times_ms = [
            start_ms + offset_ms for offset_ms in commit_offsets_ms
        ]
```

Do not add target-only batching, per-token events, draft semantics, acceptance, or bonus-token handling.

- [ ] **Step 4: Run target-only integration and existing tests**

Run:

```bash
cd code && rtk pytest -q tests/test_target_latency_profile_integration.py -k "simulator_constructs or target_only"
cd code && rtk pytest -q tests/test_target_only.py tests/test_target_only_capacity.py tests/test_latency_estimator.py
```

Expected: PASS. Analytical startup is charged once, existing analytical timestamps remain unchanged, and profile timestamps are cumulative.

- [ ] **Step 5: Commit target-only integration**

```bash
rtk git add code/src/simulator.py code/tests/test_target_latency_profile_integration.py
rtk git commit -m "feat: route target decode through latency profiles"
```

### Task 5: Route Canonical Linear Verification Batches

**Files:**
- Modify: `code/src/simulator.py` `_verify_latency_for_segments`
- Modify: `code/tests/test_target_latency_profile_integration.py`
- Regress: `code/tests/test_server_only_linear.py`, `code/tests/test_specedge_linear.py`, `code/tests/test_dip_sd.py`
- Inspect only: `code/src/scheduler.py`

- [ ] **Step 1: Add failing linear-routing tests**

Append a recording profile helper and tests:

```python
def _run_with_recorded_profile(
    method: str,
    profile_path: Path,
    *,
    num_requests: int = 2,
    output_len: int = 6,
):
    config, model_runner, workload = small_config(num_requests, output_len)
    config["target_latency"].update(
        mode="profile", profile_path=str(profile_path), metric="p50_ms"
    )
    config["speculation"]["gamma_fixed"] = 2
    config["speculation"]["gamma_candidates"] = [2]
    config["specedge"]["server_batch_size"] = num_requests
    calls: list[dict[str, object]] = []
    real_type = VerificationLatencyProfile

    class RecordingProfile(real_type):
        def query(self, method_name, **kwargs):
            calls.append({"method": method_name, **kwargs})
            return super().query(method_name, **kwargs)

    with patch("src.latency.VerificationLatencyProfile", RecordingProfile):
        simulator = Simulator(
            config, model_runner, workload,
            "combined_strong_heterogeneous", method,
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
    linear_calls = [call for call in calls if call["method"] == "linear_verification"]
    verify_events = [
        event for event in result.event_trace
        if event["event"] in {"server_only_verify", "global_batch_verify", "dip_sd_batch_verify"}
    ]
    assert len(linear_calls) == len(verify_events)
    for call, event in zip(linear_calls, verify_events):
        segments = [simulator.segments[index] for index in event.get(
            "segment_ids", [event.get("segment_id")]
        )]
        assert call["batch_size"] == len(segments)
        assert call["context_lengths"] == tuple(len(segment.prefix_ids) for segment in segments)
        assert call["gamma"] == max(len(segment.draft_ids) for segment in segments)


def test_linear_profile_receives_mixed_context_and_actual_longest_gamma(
    integration_profile_path: Path,
) -> None:
    config, model_runner, workload = small_config(num_requests=3, output_len=4)
    config["target_latency"].update(
        mode="profile", profile_path=str(integration_profile_path), metric="p50_ms"
    )
    simulator = Simulator(
        config, model_runner, workload,
        "combined_strong_heterogeneous", "specedge_linear",
    )
    simulator._schedule_request_arrivals()
    simulator.requests[0].prompt_ids = [1] * 120
    simulator.requests[1].prompt_ids = [1] * 500
    simulator.requests[2].prompt_ids = [1] * 900
    segments = []
    for request_id, gamma in enumerate((2, 4, 3)):
        segments.append(SimpleNamespace(
            segment_id=request_id,
            request_id=request_id,
            base_pos=0,
            prefix_ids=list(simulator.requests[request_id].prompt_ids),
            draft_ids=[2] * gamma,
            verify_gamma=gamma,
            draft_tree=None,
            target_verify_tree_nodes=1,
        ))
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
        config, model_runner, workload,
        "combined_strong_heterogeneous", "specedge_linear",
    )
    with patch.object(simulator.target_latency._profile, "query") as query:
        predicted = simulator.predict_verify_latency_ms(8)
    assert predicted == verify_latency_ms(config["edge"], [1])
    query.assert_not_called()
```

- [ ] **Step 2: Run linear tests and observe RED**

Run:

```bash
cd code && rtk pytest -q tests/test_target_latency_profile_integration.py -k "linear_paths or mixed_context or scheduler_prediction"
```

Expected: FAIL because `_verify_latency_for_segments` still calls `verify_latency_ms` directly, so no linear profile query is recorded.

- [ ] **Step 3: Route only canonical linear actual execution**

Refactor `_verify_latency_for_segments` while preserving its current analytical work-unit calculation:

```python
    def _verify_latency_for_segments(self, segments: Sequence[Segment]) -> float:
        contexts = self._verification_context_lengths(segments)
        analytical_work_units = (
            tuple(
                1 if segment.draft_tree is not None else segment.target_verify_tree_nodes
                for segment in segments
            )
            if self._is_specedge_runtime()
            else tuple(1 for _ in segments)
        )
        if self.spec.candidate_strategy == "linear":
            return self.target_latency.linear_verification_latency_ms(
                context_lengths=contexts,
                gamma=max(segment.verify_gamma for segment in segments),
                analytical_work_units=analytical_work_units,
            )
        return verify_latency_ms(self.config["edge"], analytical_work_units)
```

The fallback preserves tree behavior until Task 6 begins with a failing tree-routing test, and it also preserves legacy/proposed actual execution. Do not change `predict_verify_latency_ms` or any of its call sites.

- [ ] **Step 4: Run linear and scheduler regressions**

Run:

```bash
cd code && rtk pytest -q tests/test_target_latency_profile_integration.py -k "linear or scheduler_prediction"
cd code && rtk pytest -q tests/test_server_only_linear.py tests/test_specedge_linear.py tests/test_dip_sd.py tests/test_runtime_prediction.py
```

Expected: PASS. Each actual canonical logical batch makes one query; scheduler prediction remains analytical and makes none.

- [ ] **Step 5: Commit linear integration**

```bash
rtk git add code/src/simulator.py code/tests/test_target_latency_profile_integration.py
rtk git commit -m "feat: route linear verification through latency profiles"
```

### Task 6: Verify Fixed-Forward Tree Routing and Metadata

**Files:**
- Modify: `code/tests/test_target_latency_profile_integration.py`
- Modify only if tests expose a defect: `code/src/latency.py`, `code/src/simulator.py`
- Regress: `code/tests/test_server_only_tree.py`, `code/tests/test_specedge_tree.py`

- [ ] **Step 1: Add failing tree-routing and mode-guard tests before retaining the Task 5 tree branch**

Add these tests before considering tree integration complete:

```python
@pytest.mark.parametrize("method", ("server_only_tree", "specedge_tree"))
def test_canonical_tree_paths_query_fixed_forward_metadata(
    method: str,
    integration_profile_path: Path,
) -> None:
    simulator, result, calls = _run_with_recorded_profile(
        method, integration_profile_path, num_requests=2, output_len=6
    )
    tree_calls = [call for call in calls if call["method"] == "tree_verification"]
    verify_events = [
        event for event in result.event_trace
        if event["event"] in {"server_only_verify", "global_batch_verify"}
    ]
    assert len(tree_calls) == len(verify_events)
    for call, event in zip(tree_calls, verify_events):
        segments = [simulator.segments[index] for index in event.get(
            "segment_ids", [event.get("segment_id")]
        )]
        assert call["batch_size"] == len(segments)
        assert call["context_lengths"] == tuple(len(segment.prefix_ids) for segment in segments)
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
            tree_mode="real_tree_kernel", total_latency_ms=1.0
        )
    )
    with pytest.raises(ValueError, match="fixed_forward_approx"):
        model.tree_verification_latency_ms(
            context_lengths=(128,), tree_nodes=64, analytical_work_units=(1,)
        )
```

Task 5 intentionally leaves tree execution on the analytical fallback, so these routing tests establish the RED cycle before any tree production branch exists.

- [ ] **Step 2: Run tree tests and observe RED**

Run:

```bash
cd code && rtk pytest -q tests/test_target_latency_profile_integration.py -k tree
```

Expected: FAIL because canonical tree execution still falls back to analytical latency and records no `tree_verification` calls. If the mode guard was already added in Task 3, that individual guard test passes while routing tests remain RED.

- [ ] **Step 3: Add the minimal tree dispatch**

Add only this branch after the linear branch in `_verify_latency_for_segments`:

```python
        if self.spec.candidate_strategy == "tree":
            return self.target_latency.tree_verification_latency_ms(
                context_lengths=contexts,
                tree_nodes=max(segment.target_verify_tree_nodes for segment in segments),
                analytical_work_units=analytical_work_units,
            )
```

Ensure `TargetLatencyModel.tree_verification_latency_ms` checks
`result.tree_mode == "fixed_forward_approx"` and returns only
`result.total_latency_ms`. Do not inspect profile `source_rows` or
`subbatch_sizes`, and do not use `tree_nodes` in any local latency formula.

- [ ] **Step 4: Run tree and analytical regressions**

Run:

```bash
cd code && rtk pytest -q tests/test_target_latency_profile_integration.py -k tree
cd code && rtk pytest -q tests/test_server_only_tree.py tests/test_specedge_tree.py tests/test_latency_estimator.py
```

Expected: PASS. Existing analytical tree latency remains one fixed forward per logical batch member, and profile queries retain maximum node metadata without claiming a real tree kernel.

- [ ] **Step 5: Commit tree integration**

```bash
rtk git add code/src/latency.py code/src/simulator.py code/tests/test_target_latency_profile_integration.py
rtk git commit -m "feat: route fixed-forward tree verification profiles"
```

## Phase 4: Full Verification and Scope Audit

Do not add features during this phase. If a command fails, return to the task that owns the behavior, add or correct a failing test, apply the minimum fix, and rerun that task's focused commands before restarting this sequence.

- [ ] **Step 1: Run the dedicated integration test**

```bash
cd code && rtk pytest -q tests/test_target_latency_profile_integration.py
```

Expected: all target-latency integration tests pass without CUDA or model files.

- [ ] **Step 2: Re-run the unchanged query-layer contract**

```bash
cd code && rtk pytest -q tests/test_verification_latency_profile.py
```

Expected: all profile lookup, max-context padding, tier rounding, OOM splitting, provenance, and fixed-forward tree tests pass.

- [ ] **Step 3: Run the complete test suite**

```bash
cd code && rtk pytest -q
```

Expected: zero failures; existing analytical behavior remains unchanged.

- [ ] **Step 4: Run baseline reconstruction verification**

```bash
cd code && rtk bash scripts/verify_baseline_rebuild.sh
```

Expected: full and method-specific suites pass, and static checks find no forbidden prefill or obsolete DiP-SD execution paths.

- [ ] **Step 5: Run the six-method baseline trace**

```bash
cd code && rtk bash scripts/run_baseline_trace.sh
```

Expected: target-only, both server-only variants, both SpecEdge variants, and DiP-SD trace verification succeed under the default analytical mode.

- [ ] **Step 6: Audit the diff and excluded scope**

```bash
rtk git diff --check
rtk git diff -- code/src/latency.py code/src/config.py code/configs/default.yaml code/src/simulator.py code/tests/test_target_latency_profile_integration.py
rtk rg -n "compute_factor|dynamic.*compute|probabilistic.*block|real_tree_kernel" code/src code/configs code/tests/test_target_latency_profile_integration.py
```

Expected: `git diff --check` is clean. The diff contains no scheduler, batching, gamma-selection, proposed-method, legacy-lane, dynamic-compute, probabilistic-network, or real-tree-kernel implementation. Any `real_tree_kernel` match is restricted to the negative mode-guard test fixture.

- [ ] **Step 7: Commit verification-only corrections if required**

If all prior commits pass without correction, create no empty commit. If verification exposed a narrowly scoped regression and it was fixed through a new failing test, commit only that test and fix:

```bash
rtk git add code/tests/test_target_latency_profile_integration.py code/src/latency.py code/src/config.py code/src/simulator.py code/configs/default.yaml
rtk git commit -m "fix: preserve target latency integration contracts"
```

## Non-Goals Audit

The implementation must not add edge dynamic compute, server dynamic compute, server multipliers, periodic or completion-based capacity resampling, probabilistic network blocking, scheduler modifications, batching modifications, gamma-selection modifications, proposed-method changes, legacy lane behavior changes, or a real tree kernel. Fixed server capacity remains compatible with deterministic latency variation across method, actual batch, context, gamma, and metric.
