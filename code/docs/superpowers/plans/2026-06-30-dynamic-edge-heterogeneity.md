# Dynamic Edge Heterogeneity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, completion-driven edge-drafter compute model shared by every simulator method while preserving fixed server, network, scheduler, verification-profile, and disabled-mode behavior.

**Architecture:** Add one focused `EdgeComputeModel` that owns per-device rate, completion count, and epoch state and returns immutable start-time snapshots. Each `Simulator` owns exactly one model; all edge draft starts use snapshots, planning reads current capacity, and one idempotent `_finalize_request` advances completion state for both event-driven runtimes and DiP-SD. Static `Device` records, server-only draft capacity, target latency, network behavior, and scheduling algorithms remain unchanged.

**Tech Stack:** Python 3, dataclasses, SHA-256 from `hashlib`, YAML configuration, `unittest`-style pytest tests, fake model runner, Bash verification scripts.

---

## Scope and File Map

- Create `code/src/edge_compute.py`: deterministic sampling, immutable snapshots and state, per-device completion/epoch transitions, and the unchanged analytical edge-draft formula over a snapshot.
- Create `code/tests/test_dynamic_edge_compute.py`: focused configuration, sampling, state, snapshot, completion, routing, trace, and isolation tests.
- Modify `code/configs/default.yaml`: disabled-by-default global switch, fixed five-completion interval, and per-template rate ranges.
- Modify `code/src/config.py`: validate the global configuration and every present template range without changing `Device` construction.
- Modify `code/src/simulator.py`: own one model, centralize request finalization, route all edge planning/starts through it, and add dynamic trace provenance.
- Do not modify `code/src/entities.py`: `Device` stays frozen and its fixed rate remains the disabled fallback.
- Do not modify `code/src/communication.py`, `code/src/scheduler.py`, `code/src/verification_latency_profile.py`, `code/src/methods.py`, or target/server latency formulas.

## Pre-Implementation Call-Chain Confirmation

Run these read-only checks before Task 1. Stop if the results differ; update the plan against the branch rather than guessing.

- [ ] **Check 1: Confirm the three actual edge draft start locations**

```bash
rtk rg -n "def _start_draft|def _start_proactive_draft|draft_compute_ms = draft_latency_ms" code/src/simulator.py
```

Expected: ordinary async/SpecEdge work starts in `_start_draft`, proactive SpecEdge work starts in `_start_proactive_draft`, and DiP-SD computes each per-request task inside `_run_dip_sd`. `_server_only_draft_latency_ms` is separate server work and is not an edge entry point.

- [ ] **Check 2: Confirm every current edge-rate read**

```bash
rtk rg -n "draft_latency_ms\(|draft_token_rate_tok_s" code/src/simulator.py code/src/latency.py
```

Expected: actual reads occur in the three start paths; planning reads occur in `_select_gamma`, `_specedge_edge_cycle_ms`, and `_build_dip_sd_problem`; server-only reads its own configuration.

- [ ] **Check 3: Confirm the two final request-state write locations**

```bash
rtk rg -n 'request\.status = "finished"|request\.finish_time_ms =' code/src/simulator.py
```

Expected: exactly two pairs: the inline DiP-SD completion branch in `_run_dip_sd` and the event-driven `_on_request_finish` handler.

## TDD Rules for Every Task

Each behavior change follows RED -> GREEN -> focused regression -> commit. A RED command must fail because the named behavior is absent, not because of an import typo or broken fixture. Do not write production code until that task's failing test has been observed. If a later verification finds a defect, first add a focused failing regression test to the task that owns the behavior, then apply the minimum fix.

### Task 1: Define and validate dynamic edge configuration

**Files:**
- Modify: `code/configs/default.yaml:1-120`
- Modify: `code/src/config.py:59-243`
- Create: `code/tests/test_dynamic_edge_compute.py`
- Regression: `code/tests/test_config.py`

- [ ] **Step 1: Write failing configuration tests**

Create `code/tests/test_dynamic_edge_compute.py` with the shared helpers and configuration contract:

```python
from __future__ import annotations

import copy
import math
import unittest

from src.config import load_config, validate_config
from src.model_runner import FakeModelRunner
from src.simulator import Simulator
from tests.common import accepting_model_runner, small_config


def enable_dynamic_edge(config: dict) -> dict:
    config["dynamic_edge_compute"]["enabled"] = True
    return config


def force_one_device(config: dict) -> None:
    config["simulation"]["num_devices"] = 1
    for pool_name, pool in config["device_pools"].items():
        templates = pool["templates"]
        for template in templates.values():
            template["count"] = 0
        selected = "low_end" if pool_name == "heterogeneous" else "medium"
        templates[selected]["count"] = 1


class DynamicEdgeConfigurationTest(unittest.TestCase):
    def test_defaults_are_disabled_with_five_completion_interval_and_ranges(self) -> None:
        config = load_config("configs/default.yaml")

        self.assertEqual(
            config["dynamic_edge_compute"],
            {"enabled": False, "resample_every_completed_requests": 5},
        )
        templates = config["device_pools"]["heterogeneous"]["templates"]
        self.assertEqual(
            templates["low_end"]["dynamic_draft_token_rate_range_tok_s"],
            [20, 30],
        )
        self.assertEqual(
            templates["mid_end"]["dynamic_draft_token_rate_range_tok_s"],
            [48, 72],
        )
        self.assertEqual(
            templates["high_end"]["dynamic_draft_token_rate_range_tok_s"],
            [80, 120],
        )

    def test_enabled_mode_requires_valid_ranges_for_populated_templates(self) -> None:
        cases = (
            ("not-a-bool", "enabled"),
            ([0, 30], "range"),
            ([30, 20], "range"),
            ([20], "range"),
            ([20, math.inf], "range"),
        )
        for value, message in cases:
            with self.subTest(value=value):
                config = load_config("configs/default.yaml")
                if message == "enabled":
                    config["dynamic_edge_compute"]["enabled"] = value
                else:
                    config["dynamic_edge_compute"]["enabled"] = True
                    config["device_pools"]["heterogeneous"]["templates"][
                        "low_end"
                    ]["dynamic_draft_token_rate_range_tok_s"] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_config(config)

        config = enable_dynamic_edge(load_config("configs/default.yaml"))
        del config["device_pools"]["heterogeneous"]["templates"]["low_end"][
            "dynamic_draft_token_rate_range_tok_s"
        ]
        with self.assertRaisesRegex(ValueError, "low_end"):
            validate_config(config)

    def test_resample_interval_is_exactly_five(self) -> None:
        config = load_config("configs/default.yaml")
        config["dynamic_edge_compute"]["resample_every_completed_requests"] = 4
        with self.assertRaisesRegex(ValueError, "must be 5"):
            validate_config(config)

    def test_disabled_legacy_config_may_omit_dynamic_section_and_ranges(self) -> None:
        config = load_config("configs/default.yaml")
        del config["dynamic_edge_compute"]
        for pool in config["device_pools"].values():
            for template in pool["templates"].values():
                template.pop("dynamic_draft_token_rate_range_tok_s", None)

        validate_config(config)
```

- [ ] **Step 2: Run the tests and verify RED**

```bash
cd code && rtk pytest -q tests/test_dynamic_edge_compute.py -k configuration
```

Expected: FAIL because `dynamic_edge_compute` and template ranges are absent and `validate_config` has no dynamic validation.

- [ ] **Step 3: Add the minimal default configuration**

Add to `code/configs/default.yaml`:

```yaml
dynamic_edge_compute:
  enabled: false
  resample_every_completed_requests: 5
```

Add these fields to the matching templates without changing existing fixed rates:

```yaml
dynamic_draft_token_rate_range_tok_s: [20, 30]   # low_end
dynamic_draft_token_rate_range_tok_s: [48, 72]   # mid_end
dynamic_draft_token_rate_range_tok_s: [80, 120]  # high_end
dynamic_draft_token_rate_range_tok_s: [48, 72]   # medium_only.medium
```

- [ ] **Step 4: Add minimal validation**

Import `math` in `code/src/config.py` and add a helper used by `validate_config`:

```python
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
```

Call `_validate_dynamic_edge_compute(config)` after obtaining `simulation` and before method-specific validation. Do not change `build_devices`.

- [ ] **Step 5: Run focused configuration tests GREEN**

```bash
cd code && rtk pytest -q tests/test_dynamic_edge_compute.py -k configuration
cd code && rtk pytest -q tests/test_config.py
```

Expected: PASS. Existing fixed rates and scenario merge behavior remain unchanged.

- [ ] **Step 6: Commit the configuration contract**

```bash
rtk git add code/configs/default.yaml code/src/config.py code/tests/test_dynamic_edge_compute.py
rtk git commit -m "feat: configure dynamic edge compute"
```

### Task 2: Implement deterministic per-device compute state and snapshots

**Files:**
- Create: `code/src/edge_compute.py`
- Modify: `code/tests/test_dynamic_edge_compute.py`
- Regression: `code/tests/test_latency_estimator.py`

- [ ] **Step 1: Add failing model, sampling, transition, and snapshot tests**

Append imports and tests:

```python
from unittest.mock import patch

from src.config import build_devices
from src.edge_compute import EdgeComputeModel, deterministic_draft_rate


class EdgeComputeModelTest(unittest.TestCase):
    def make_model(self, *, seed: int = 42) -> EdgeComputeModel:
        config = enable_dynamic_edge(load_config("configs/default.yaml"))
        config["simulation"]["seed"] = seed
        return EdgeComputeModel(config, build_devices(config), "heterogeneous")

    def test_same_type_initial_rates_are_distinct_and_in_range(self) -> None:
        model = self.make_model()
        rates = [model.state(device_id).draft_token_rate_tok_s for device_id in (0, 1, 2)]

        self.assertEqual(len(set(rates)), 3)
        self.assertTrue(all(20.0 <= rate < 30.0 for rate in rates))

    def test_sampling_is_reproducible_and_call_order_independent(self) -> None:
        bounds = (20.0, 30.0)
        forward = {
            epoch: deterministic_draft_rate(7, 3, "low_end", epoch, bounds)
            for epoch in (0, 1, 2)
        }
        reverse = {
            epoch: deterministic_draft_rate(7, 3, "low_end", epoch, bounds)
            for epoch in (2, 1, 0)
        }

        self.assertEqual(forward, reverse)
        self.assertEqual(
            forward[1],
            deterministic_draft_rate(7, 3, "low_end", 1, bounds),
        )
        self.assertNotEqual(forward[0], forward[1])

    def test_fifth_completion_advances_only_that_device(self) -> None:
        model = self.make_model()
        old_zero = model.state(0)
        old_one = model.state(1)

        for _ in range(4):
            self.assertIsNone(model.record_request_completion(0))
        transition = model.record_request_completion(0)

        self.assertIsNotNone(transition)
        self.assertEqual(model.state(0).completed_requests, 5)
        self.assertEqual(model.state(0).epoch, 1)
        self.assertNotEqual(model.state(0).draft_token_rate_tok_s, old_zero.draft_token_rate_tok_s)
        self.assertEqual(model.state(1), old_one)

    def test_started_snapshot_keeps_old_rate_after_resample(self) -> None:
        model = self.make_model()
        started = model.snapshot(0)
        started_ms = model.latency_ms(started, 4)

        for _ in range(5):
            model.record_request_completion(0)

        next_started = model.snapshot(0)
        self.assertEqual(model.latency_ms(started, 4), started_ms)
        self.assertEqual(started.epoch, 0)
        self.assertEqual(next_started.epoch, 1)
        self.assertNotEqual(
            started.draft_token_rate_tok_s,
            next_started.draft_token_rate_tok_s,
        )

    def test_disabled_model_uses_fixed_device_rate_and_never_transitions(self) -> None:
        config = load_config("configs/default.yaml")
        devices = build_devices(config)
        model = EdgeComputeModel(config, devices, "heterogeneous")

        self.assertEqual(
            model.snapshot(0).draft_token_rate_tok_s,
            devices[0].draft_token_rate_tok_s,
        )
        for _ in range(5):
            self.assertIsNone(model.record_request_completion(0))
        self.assertEqual(model.state(0).epoch, 0)

    def test_initial_collision_is_resolved_deterministically(self) -> None:
        config = enable_dynamic_edge(load_config("configs/default.yaml"))
        devices = build_devices(config)
        values = iter((21.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0))
        with patch("src.edge_compute.deterministic_draft_rate", side_effect=values):
            model = EdgeComputeModel(config, devices, "heterogeneous")

        low_rates = [model.state(index).draft_token_rate_tok_s for index in (0, 1, 2)]
        self.assertEqual(low_rates, [21.0, 22.0, 23.0])
```

- [ ] **Step 2: Run model tests and verify RED**

```bash
cd code && rtk pytest -q tests/test_dynamic_edge_compute.py -k EdgeComputeModelTest
```

Expected: collection ERROR because `src.edge_compute` does not exist. This is the expected RED for the new focused component.

- [ ] **Step 3: Implement immutable state, snapshot, transition, and sampling**

Create `code/src/edge_compute.py` with these public types and signatures:

```python
from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any, Sequence

from src.entities import Device


@dataclass(frozen=True)
class EdgeComputeState:
    completed_requests: int
    epoch: int
    draft_token_rate_tok_s: float


@dataclass(frozen=True)
class EdgeComputeSnapshot:
    device_id: int
    device_type: str
    epoch: int
    draft_token_rate_tok_s: float
    draft_startup_ms: float


@dataclass(frozen=True)
class EdgeComputeTransition:
    device_id: int
    device_type: str
    completed_requests: int
    old_epoch: int
    new_epoch: int
    old_rate: float
    new_rate: float


def deterministic_draft_rate(
    seed: int,
    device_id: int,
    device_type: str,
    epoch: int,
    bounds: tuple[float, float],
    *,
    attempt: int = 0,
) -> float:
    suffix = f":{attempt}" if attempt else ""
    key = (
        f"edge-compute-v1:{seed}:{device_id}:{device_type}:{epoch}{suffix}"
    ).encode()
    ratio = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
    lower, upper = bounds
    return lower + ratio * (upper - lower)


class EdgeComputeModel:
    def __init__(
        self,
        config: dict[str, Any],
        devices: Sequence[Device],
        pool_name: str,
    ) -> None:
        dynamic = config.get("dynamic_edge_compute", {})
        self.enabled = bool(dynamic.get("enabled", False))
        self._seed = int(config["simulation"]["seed"])
        self._devices = {device.device_id: device for device in devices}
        templates = config["device_pools"][pool_name]["templates"]
        self._bounds = {
            device_type: tuple(float(value) for value in template["dynamic_draft_token_rate_range_tok_s"])
            for device_type, template in templates.items()
            if "dynamic_draft_token_rate_range_tok_s" in template
        }
        self._states: dict[int, EdgeComputeState] = {}
        used_initial_rates: dict[str, set[float]] = {}
        for device in sorted(devices, key=lambda item: item.device_id):
            rate = device.draft_token_rate_tok_s
            if self.enabled:
                attempt = 0
                used = used_initial_rates.setdefault(device.device_type, set())
                while True:
                    rate = deterministic_draft_rate(
                        self._seed,
                        device.device_id,
                        device.device_type,
                        0,
                        self._bounds[device.device_type],
                        attempt=attempt,
                    )
                    if rate not in used:
                        used.add(rate)
                        break
                    attempt += 1
            self._states[device.device_id] = EdgeComputeState(0, 0, rate)

    def state(self, device_id: int) -> EdgeComputeState:
        return self._states[device_id]

    def current_rate(self, device_id: int) -> float:
        return self._states[device_id].draft_token_rate_tok_s

    def snapshot(self, device_id: int) -> EdgeComputeSnapshot:
        device = self._devices[device_id]
        state = self._states[device_id]
        return EdgeComputeSnapshot(
            device_id=device_id,
            device_type=device.device_type,
            epoch=state.epoch,
            draft_token_rate_tok_s=state.draft_token_rate_tok_s,
            draft_startup_ms=device.draft_startup_ms,
        )

    def latency_ms(self, snapshot: EdgeComputeSnapshot, work_units: int) -> float:
        if work_units < 0:
            raise ValueError("work_units must be >= 0")
        return snapshot.draft_startup_ms + (
            1000.0 * work_units / snapshot.draft_token_rate_tok_s
        )

    def current_latency_ms(self, device_id: int, work_units: int) -> float:
        return self.latency_ms(self.snapshot(device_id), work_units)

    def record_request_completion(
        self,
        device_id: int,
    ) -> EdgeComputeTransition | None:
        if not self.enabled:
            return None
        device = self._devices[device_id]
        old = self._states[device_id]
        completed = old.completed_requests + 1
        if completed % 5:
            self._states[device_id] = replace(old, completed_requests=completed)
            return None
        epoch = old.epoch + 1
        rate = deterministic_draft_rate(
            self._seed,
            device_id,
            device.device_type,
            epoch,
            self._bounds[device.device_type],
        )
        self._states[device_id] = EdgeComputeState(completed, epoch, rate)
        return EdgeComputeTransition(
            device_id=device_id,
            device_type=device.device_type,
            completed_requests=completed,
            old_epoch=old.epoch,
            new_epoch=epoch,
            old_rate=old.draft_token_rate_tok_s,
            new_rate=rate,
        )
```

The exact implementation may factor private helpers, but keep these contracts and do not import simulator state into this module.

- [ ] **Step 4: Run focused model tests GREEN**

```bash
cd code && rtk pytest -q tests/test_dynamic_edge_compute.py -k EdgeComputeModelTest
cd code && rtk pytest -q tests/test_latency_estimator.py
```

Expected: PASS. The legacy `draft_latency_ms(Device, ...)` tests remain unchanged.

- [ ] **Step 5: Commit the isolated resource model**

```bash
rtk git add code/src/edge_compute.py code/tests/test_dynamic_edge_compute.py
rtk git commit -m "feat: add deterministic edge compute model"
```

### Task 3: Give each Simulator exactly one EdgeComputeModel

**Files:**
- Modify: `code/src/simulator.py:1-110`
- Modify: `code/tests/test_dynamic_edge_compute.py`
- Regression: `code/tests/test_target_only.py`

- [ ] **Step 1: Add a failing ownership and disabled-fallback test**

Append:

```python
class SimulatorEdgeComputeOwnershipTest(unittest.TestCase):
    def test_simulator_owns_one_model_with_selected_pool_devices(self) -> None:
        config, runner, workload = small_config(num_requests=1, output_len=2)
        simulator = Simulator(config, runner, workload, "test", "full")

        self.assertIsInstance(simulator.edge_compute, EdgeComputeModel)
        self.assertEqual(
            simulator.edge_compute.current_rate(0),
            simulator.devices[0].draft_token_rate_tok_s,
        )
        self.assertIs(simulator.edge_compute, simulator.edge_compute)

    def test_enabled_simulators_reproduce_the_same_initial_mapping(self) -> None:
        config, runner, workload = small_config(num_requests=3, output_len=2)
        enable_dynamic_edge(config)
        first = Simulator(config, runner, workload, "test", "full")
        second = Simulator(copy.deepcopy(config), runner, workload, "test", "dip_sd")

        self.assertEqual(
            [first.edge_compute.current_rate(index) for index in range(3)],
            [second.edge_compute.current_rate(index) for index in range(3)],
        )
```

- [ ] **Step 2: Run the ownership tests and verify RED**

```bash
cd code && rtk pytest -q tests/test_dynamic_edge_compute.py -k ownership
```

Expected: FAIL with `AttributeError: 'Simulator' object has no attribute 'edge_compute'`.

- [ ] **Step 3: Construct one model after devices are built**

In `code/src/simulator.py`, import `EdgeComputeModel` and initialize it exactly once:

```python
self.devices = build_devices(config, self.spec.device_pool)
self.edge_compute = EdgeComputeModel(config, self.devices, self.spec.device_pool)
self.device_runtimes = [DeviceRuntime(device) for device in self.devices]
```

Do not construct models in method branches, request handlers, or draft starts.

- [ ] **Step 4: Run ownership and target-only regressions GREEN**

```bash
cd code && rtk pytest -q tests/test_dynamic_edge_compute.py -k ownership
cd code && rtk pytest -q tests/test_target_only.py tests/test_config.py
```

Expected: PASS. At this commit no simulator behavior reads or mutates the new model.

- [ ] **Step 5: Commit simulator ownership**

```bash
rtk git add code/src/simulator.py code/tests/test_dynamic_edge_compute.py
rtk git commit -m "feat: attach edge compute model to simulator"
```

### Task 4: Centralize idempotent request finalization and completion counting

**Files:**
- Modify: `code/src/simulator.py:195-475, 2176-2240`
- Modify: `code/tests/test_dynamic_edge_compute.py`
- Regression: `code/tests/test_dip_sd.py`
- Regression: `code/tests/test_target_only.py`
- Regression: `code/tests/test_server_only_linear.py`
- Regression: `code/tests/test_server_only_tree.py`

- [ ] **Step 1: Add failing event-driven, DiP-SD, and idempotency tests**

Append helpers and tests:

```python
def dynamic_single_device_case(
    *,
    num_requests: int = 5,
    output_len: int = 2,
) -> tuple[dict, FakeModelRunner, list]:
    config, runner, workload = small_config(
        num_requests=num_requests,
        output_len=output_len,
    )
    force_one_device(config)
    enable_dynamic_edge(config)
    config["simulation"]["request_arrival"] = "burst"
    return config, runner, workload


class RequestFinalizationTest(unittest.TestCase):
    def assert_one_epoch_after_five(self, method: str) -> None:
        config, runner, workload = dynamic_single_device_case()
        simulator = Simulator(config, runner, workload, "test", method)
        simulator.run()

        state = simulator.edge_compute.state(0)
        self.assertEqual(state.completed_requests, 5)
        self.assertEqual(state.epoch, 1)
        transitions = [
            event
            for event in simulator._trace
            if event["event"] == "edge_compute_transition"
        ]
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0]["completed_requests"], 5)

    def test_event_driven_completion_advances_once(self) -> None:
        self.assert_one_epoch_after_five("target_only")

    def test_dip_sd_inline_completion_uses_the_same_counter(self) -> None:
        self.assert_one_epoch_after_five("dip_sd")

    def test_finalize_request_is_idempotent(self) -> None:
        config, runner, workload = dynamic_single_device_case(num_requests=1)
        simulator = Simulator(config, runner, workload, "test", "target_only")
        simulator._schedule_request_arrivals()
        request = simulator.requests[0]

        simulator._finalize_request(request, 10.0)
        simulator._finalize_request(request, 20.0)

        self.assertEqual(request.finish_time_ms, 10.0)
        self.assertEqual(simulator.edge_compute.state(0).completed_requests, 1)
        self.assertEqual(
            sum(event["event"] == "request_finish" for event in simulator._trace),
            1,
        )
```

- [ ] **Step 2: Run finalization tests and verify RED**

```bash
cd code && rtk pytest -q tests/test_dynamic_edge_compute.py -k finalization
```

Expected: FAIL because `_finalize_request` does not exist, target-only completion does not update model state, and DiP-SD still writes final state inline.

- [ ] **Step 3: Extract the sole final-state mutation operation**

Add `_finalize_request` next to `_on_request_finish`. Move the current handler body into it and insert one completion transition:

```python
def _finalize_request(self, request: Request, now_ms: float) -> None:
    if request.status == "finished":
        return
    request.status = "finished"
    request.finish_time_ms = now_ms
    request.draft_queued = False
    for segment_id in list(request.in_flight_segments):
        segment = self.segments[segment_id]
        if segment.status in ACTIVE_SEGMENT_STATUSES:
            self._discard_segment(segment)
    transition = self.edge_compute.record_request_completion(request.device_id)
    if transition is not None:
        self._trace.append(
            {
                "event": "edge_compute_transition",
                "method": self.spec.name,
                "device_id": transition.device_id,
                "device_type": transition.device_type,
                "completed_requests": transition.completed_requests,
                "old_epoch": transition.old_epoch,
                "new_epoch": transition.new_epoch,
                "old_draft_token_rate_tok_s": transition.old_rate,
                "new_draft_token_rate_tok_s": transition.new_rate,
                "time_ms": now_ms,
            }
        )
    self._trace.append(
        {
            "event": "request_finish",
            "method": self.spec.name,
            "request_id": request.request_id,
            "device_id": request.device_id,
            "finish_time_ms": now_ms,
        }
    )
    if self._progress_callback is not None:
        self._progress_callback(
            sum(item.status == "finished" for item in self.requests),
            len(self.requests),
        )
    if (
        self._is_server_only_runtime()
        and self._server_only_active_request_id == request.request_id
    ):
        self._server_only_active_request_id = None
        self._maybe_start_server_only_request(now_ms)


def _on_request_finish(self, now_ms: float, request_id: int) -> None:
    self._finalize_request(self.requests[request_id], now_ms)
```

In the DiP-SD completion condition, replace all inline status/time/trace/progress writes with:

```python
self._finalize_request(request, result_arrival_ms)
```

Do not call the model from any other completion check.

- [ ] **Step 4: Prove there is now one final-state write location**

```bash
rtk rg -n 'request\.status = "finished"|request\.finish_time_ms =' code/src/simulator.py
```

Expected: exactly one pair, both inside `_finalize_request`.

- [ ] **Step 5: Run finalization and affected runtime tests GREEN**

```bash
cd code && rtk pytest -q tests/test_dynamic_edge_compute.py -k finalization
cd code && rtk pytest -q tests/test_target_only.py tests/test_dip_sd.py tests/test_server_only_linear.py tests/test_server_only_tree.py tests/test_baseline_system_invariants.py
```

Expected: PASS. Every unique request increments its assigned device once; duplicate finish delivery is a no-op; server-only queue release and DiP-SD progress behavior are preserved.

- [ ] **Step 6: Commit unified completion semantics**

```bash
rtk git add code/src/simulator.py code/tests/test_dynamic_edge_compute.py
rtk git commit -m "refactor: centralize request finalization"
```

### Task 5: Route ordinary edge draft planning and starts through snapshots

**Files:**
- Modify: `code/src/simulator.py:1123-1544`
- Modify: `code/tests/test_dynamic_edge_compute.py`
- Regression: `code/tests/test_latency_estimator.py`
- Regression: `code/tests/test_runtime_prediction.py`
- Regression: `code/tests/test_linear_sd_core.py`

- [ ] **Step 1: Add failing ordinary-draft snapshot and provenance tests**

Append:

```python
class OrdinaryDraftSnapshotTest(unittest.TestCase):
    def test_full_draft_uses_snapshot_rate_and_records_provenance(self) -> None:
        config, _, workload = dynamic_single_device_case(
            num_requests=1,
            output_len=6,
        )
        simulator = Simulator(
            config,
            accepting_model_runner(),
            workload,
            "test",
            "full",
        )
        result = simulator.run()
        event = next(item for item in result.event_trace if item["event"] == "draft_compute")
        segment = result.segments[event["segment_id"]]

        expected = simulator.devices[0].draft_startup_ms + (
            1000.0
            * segment.processed_candidate_count
            / event["draft_token_rate_tok_s"]
        )
        self.assertEqual(event["edge_compute_epoch"], 0)
        self.assertAlmostEqual(segment.draft_analytical_ms, expected)

    def test_started_ordinary_draft_duration_is_not_recomputed_after_epoch_change(self) -> None:
        config, _, workload = dynamic_single_device_case(
            num_requests=1,
            output_len=6,
        )
        simulator = Simulator(
            config,
            accepting_model_runner(),
            workload,
            "test",
            "full",
        )
        simulator._schedule_request_arrivals()
        request = simulator.requests[0]
        runtime = simulator.device_runtimes[0]
        simulator._start_draft(runtime, request, 0.0, 0.0)
        segment = simulator.segments[0]
        old_duration = segment.draft_compute_ms
        old_rate = simulator._trace[-1]["draft_token_rate_tok_s"]

        for _ in range(5):
            simulator.edge_compute.record_request_completion(0)

        self.assertEqual(segment.draft_compute_ms, old_duration)
        self.assertEqual(simulator._trace[-1]["draft_token_rate_tok_s"], old_rate)
        self.assertNotEqual(
            old_rate,
            simulator.edge_compute.snapshot(0).draft_token_rate_tok_s,
        )
```

- [ ] **Step 2: Run ordinary-draft tests and verify RED**

```bash
cd code && rtk pytest -q tests/test_dynamic_edge_compute.py -k ordinary_draft
```

Expected: FAIL because `draft_compute` lacks `edge_compute_epoch` and `draft_token_rate_tok_s`, and `_start_draft` still calls `draft_latency_ms(Device, ...)`.

- [ ] **Step 3: Capture one snapshot at ordinary task start**

Extend the existing `src.edge_compute` import to include
`EdgeComputeSnapshot` for the helper and `_select_gamma` annotations.

At the beginning of `_start_draft`, before gamma selection, capture:

```python
compute = self.edge_compute.snapshot(request.device_id)
```

Pass that snapshot into `_select_gamma`. For edge methods, replace each ordinary task formula with:

```python
analytical_ms = self.edge_compute.latency_ms(compute, processed_candidates)
fresh_compute_ms = (
    self.edge_compute.latency_ms(compute, fresh_processed_candidates)
    if fresh_ids
    else 0.0
)
```

Change `_select_gamma` to accept `compute: EdgeComputeSnapshot | None = None`. Preserve the server-only branch exactly. Ordinary task starts pass their captured snapshot; direct planning callers that do not pass one take one current-state snapshot without reserving capacity:

```python
draft_ms = (
    self._server_only_draft_latency_ms(gamma)
    if self._is_server_only_runtime()
    else self.edge_compute.latency_ms(
        compute or self.edge_compute.snapshot(device.device_id),
        gamma,
    )
)
```

- [ ] **Step 4: Use the same snapshot for SpecEdge pipeline accounting**

Change `_specedge_edge_cycle_ms` to receive `compute: EdgeComputeSnapshot` and calculate draft work through `self.edge_compute.latency_ms(compute, tree_plan.draft_compute_nodes)`. In `_start_draft`, subtract `self.edge_compute.latency_ms(compute, processed_candidates)` when deriving network-only cycle time. Do not alter uplink/downlink calls or pipeline scheduling policy.

- [ ] **Step 5: Add dynamic-only trace provenance**

Add one simulator helper so every edge start uses the same disabled-mode rule:

```python
def _edge_compute_trace_fields(
    self,
    compute: EdgeComputeSnapshot,
) -> dict[str, Any]:
    if not self.edge_compute.enabled:
        return {}
    return {
        "edge_compute_epoch": compute.epoch,
        "draft_token_rate_tok_s": compute.draft_token_rate_tok_s,
    }
```

Insert this expansion as the final entry in both the existing `draft_compute`
and `pipeline_schedule` dictionaries, before each closing brace:

```python
**self._edge_compute_trace_fields(compute),
```

Do not add fields in disabled mode.

- [ ] **Step 6: Run focused and prediction regressions GREEN**

```bash
cd code && rtk pytest -q tests/test_dynamic_edge_compute.py -k ordinary_draft
cd code && rtk pytest -q tests/test_latency_estimator.py tests/test_runtime_prediction.py tests/test_linear_sd_core.py
```

Expected: PASS. Ordinary tasks retain scheduled durations after state changes, and scheduler/gamma algorithms are unchanged apart from reading current edge capacity.

- [ ] **Step 7: Audit remaining legacy edge formula calls**

```bash
rtk rg -n "draft_latency_ms\(" code/src/simulator.py
```

Expected: remaining edge calls are only DiP-SD and proactive paths scheduled for Tasks 6-7, plus the separate `_server_only_draft_latency_ms` name. No ordinary `_start_draft`, `_select_gamma`, or pipeline-cycle call remains.

- [ ] **Step 8: Commit ordinary snapshot routing**

```bash
rtk git add code/src/simulator.py code/tests/test_dynamic_edge_compute.py
rtk git commit -m "feat: snapshot ordinary edge draft capacity"
```

### Task 6: Route proactive SpecEdge tasks through independent snapshots

**Files:**
- Modify: `code/src/simulator.py:1967-2046`
- Modify: `code/tests/test_dynamic_edge_compute.py`
- Regression: `code/tests/test_specedge_linear.py`
- Regression: `code/tests/test_specedge_tree.py`
- Regression: `code/tests/test_specedge_methods.py`

- [ ] **Step 1: Add a failing proactive snapshot test**

Append:

```python
class ProactiveDraftSnapshotTest(unittest.TestCase):
    def test_proactive_draft_uses_its_own_start_snapshot(self) -> None:
        config, _, workload = dynamic_single_device_case(
            num_requests=1,
            output_len=12,
        )
        config["specedge"]["server_batch_size"] = 1
        simulator = Simulator(
            config,
            accepting_model_runner(),
            workload,
            "test",
            "specedge_linear",
        )
        result = simulator.run()
        event = next(item for item in result.event_trace if item["event"] == "proactive_draft")

        expected = simulator.devices[event["device_id"]].draft_startup_ms + (
            1000.0
            * event["processed_candidate_count"]
            / event["draft_token_rate_tok_s"]
        )
        self.assertEqual(event["edge_compute_epoch"], 0)
        self.assertAlmostEqual(event["compute_ms"], expected)
```

- [ ] **Step 2: Run the proactive test and verify RED**

```bash
cd code && rtk pytest -q tests/test_dynamic_edge_compute.py -k proactive_draft
```

Expected: FAIL because proactive trace provenance is absent and `_start_proactive_draft` still reads the static `Device` rate.

- [ ] **Step 3: Capture and use a proactive-task snapshot**

Inside `_start_proactive_draft`, after confirming the device is available and immediately before calculating task duration, capture a new snapshot and use it for the complete proactive task:

```python
compute = self.edge_compute.snapshot(segment.device_id)
proactive_compute_ms = self.edge_compute.latency_ms(
    compute,
    processed_candidates,
)
```

The snapshot is independent of the parent segment's ordinary draft snapshot. Do not reread current rate after scheduling `PROACTIVE_DRAFT_DONE`.

- [ ] **Step 4: Add dynamic-only proactive trace fields**

Insert the Task 5 helper as the final entry in the existing
`proactive_draft` event dictionary, using its own snapshot:

```python
**self._edge_compute_trace_fields(compute),
```

Do not change tree counts, proactive policy, resource occupancy, or finish scheduling.

- [ ] **Step 5: Run proactive and SpecEdge regressions GREEN**

```bash
cd code && rtk pytest -q tests/test_dynamic_edge_compute.py -k proactive_draft
cd code && rtk pytest -q tests/test_specedge_linear.py tests/test_specedge_tree.py tests/test_specedge_methods.py
```

Expected: PASS. Both linear and tree proactive behavior retain existing semantic and scheduling contracts.

- [ ] **Step 6: Commit proactive snapshot routing**

```bash
rtk git add code/src/simulator.py code/tests/test_dynamic_edge_compute.py
rtk git commit -m "feat: snapshot proactive edge draft capacity"
```

### Task 7: Route DiP-SD planning and task starts through the shared model

**Files:**
- Modify: `code/src/simulator.py:195-535`
- Modify: `code/tests/test_dynamic_edge_compute.py`
- Regression: `code/tests/test_dip_sd.py`

- [ ] **Step 1: Add failing DiP-SD planner and actual-task tests**

Append:

```python
class DipSDEdgeComputeTest(unittest.TestCase):
    def test_dip_sd_problem_reads_current_device_rate(self) -> None:
        config, runner, workload = dynamic_single_device_case(
            num_requests=1,
            output_len=4,
        )
        simulator = Simulator(config, runner, workload, "test", "dip_sd")
        simulator._schedule_request_arrivals()
        rate = simulator.edge_compute.current_rate(0)

        problem = simulator._build_dip_sd_problem([0], 0)

        self.assertEqual(problem.users[0].draft_latency_overhead_ms, 1000.0 / rate)

    def test_dip_sd_actual_draft_uses_start_snapshot_and_provenance(self) -> None:
        config, runner, workload = dynamic_single_device_case(
            num_requests=1,
            output_len=4,
        )
        simulator = Simulator(config, runner, workload, "test", "dip_sd")
        result = simulator.run()
        event = next(item for item in result.event_trace if item["event"] == "dip_sd_draft")

        expected = simulator.devices[0].draft_startup_ms + (
            1000.0
            * result.segments[event["segment_id"]].processed_candidate_count
            / event["draft_token_rate_tok_s"]
        )
        self.assertEqual(event["edge_compute_epoch"], 0)
        self.assertAlmostEqual(event["compute_ms"], expected)
```

- [ ] **Step 2: Run DiP-SD dynamic tests and verify RED**

```bash
cd code && rtk pytest -q tests/test_dynamic_edge_compute.py -k dip_sd
```

Expected: FAIL because `_build_dip_sd_problem` reads `Device.draft_token_rate_tok_s`, actual DiP-SD draft latency uses `draft_latency_ms(Device, ...)`, and trace provenance is absent.

- [ ] **Step 3: Route optimizer inputs through current capacity**

In `_build_dip_sd_problem`, replace only the edge rate input:

```python
draft_latency_overhead_ms=(
    1000.0 / self.edge_compute.current_rate(request.device_id)
),
```

Do not change DiP-SD batches, objective, draft lengths, epoch barriers, target latency inputs, communication latency, or optimizer calls. An already-created plan is not rebuilt when a later completion changes capacity.

- [ ] **Step 4: Capture one snapshot per actual DiP-SD draft start**

Inside the per-request draft block in `_run_dip_sd`, capture at `draft_start_ms` and compute duration from it:

```python
compute = self.edge_compute.snapshot(request.device_id)
draft_compute_ms = self.edge_compute.latency_ms(compute, len(draft_ids))
```

Use the same snapshot for that task's trace provenance. The scheduled `draft_done_ms` remains fixed after assignment.

- [ ] **Step 5: Add dynamic-only DiP-SD trace fields**

Insert the Task 5 helper as the final entry in the existing `dip_sd_draft`
event dictionary, using that task's snapshot:

```python
**self._edge_compute_trace_fields(compute),
```

Do not add these fields to `dip_sd_batch_verify` or other server events.

- [ ] **Step 6: Run DiP-SD and completion regressions GREEN**

```bash
cd code && rtk pytest -q tests/test_dynamic_edge_compute.py -k "dip_sd or finalization"
cd code && rtk pytest -q tests/test_dip_sd.py
```

Expected: PASS. Planner reads current state at plan construction, each task captures start state, and completion still advances through `_finalize_request` once.

- [ ] **Step 7: Prove every edge draft formula is routed**

```bash
rtk rg -n "draft_latency_ms\(" code/src/simulator.py
rtk rg -n "draft_token_rate_tok_s" code/src/simulator.py
```

Expected: no direct `draft_latency_ms(Device, ...)` edge calls remain in `Simulator`; matches are limited to `_server_only_draft_latency_ms`, dynamic trace field names, and the server-only configuration read. All edge capacity reads go through `self.edge_compute`.

- [ ] **Step 8: Commit DiP-SD routing**

```bash
rtk git add code/src/simulator.py code/tests/test_dynamic_edge_compute.py
rtk git commit -m "feat: route dip sd through dynamic edge capacity"
```

### Task 8: Lock disabled behavior and excluded subsystems

**Files:**
- Modify: `code/tests/test_dynamic_edge_compute.py`
- Inspect only: `code/src/communication.py`
- Inspect only: `code/src/scheduler.py`
- Inspect only: `code/src/latency.py`
- Inspect only: `code/src/verification_latency_profile.py`
- Regression: `code/tests/test_target_latency_profile_integration.py`
- Regression: `code/tests/test_baseline_system_invariants.py`

- [ ] **Step 1: Add failing cross-method disabled and isolation tests**

Append:

```python
class DynamicEdgeIsolationTest(unittest.TestCase):
    METHODS = (
        "target_only",
        "server_only_linear",
        "server_only_tree",
        "specedge_linear",
        "specedge_tree",
        "dip_sd",
        "full",
    )

    def test_disabled_and_legacy_missing_config_are_trace_identical(self) -> None:
        for method in self.METHODS:
            with self.subTest(method=method):
                config, runner, workload = small_config(num_requests=2, output_len=4)
                explicit = Simulator(
                    copy.deepcopy(config), runner, workload, "test", method
                ).run()
                legacy_config = copy.deepcopy(config)
                del legacy_config["dynamic_edge_compute"]
                for pool in legacy_config["device_pools"].values():
                    for template in pool["templates"].values():
                        template.pop("dynamic_draft_token_rate_range_tok_s", None)
                legacy = Simulator(
                    legacy_config, runner, workload, "test", method
                ).run()

                self.assertEqual(explicit.event_trace, legacy.event_trace)
                self.assertEqual(
                    [request.finish_time_ms for request in explicit.requests],
                    [request.finish_time_ms for request in legacy.requests],
                )

    def test_enabled_mode_does_not_change_target_or_server_compute(self) -> None:
        for method, event_name in (
            ("target_only", "target_only_service"),
            ("server_only_linear", "server_only_draft"),
            ("server_only_tree", "server_only_draft"),
        ):
            with self.subTest(method=method):
                config, runner, workload = small_config(num_requests=1, output_len=4)
                disabled = Simulator(
                    copy.deepcopy(config), runner, workload, "test", method
                ).run()
                enabled_config = enable_dynamic_edge(copy.deepcopy(config))
                enabled = Simulator(
                    enabled_config, runner, workload, "test", method
                ).run()
                disabled_events = [
                    event["compute_ms"]
                    for event in disabled.event_trace
                    if event["event"] == event_name
                ]
                enabled_events = [
                    event["compute_ms"]
                    for event in enabled.event_trace
                    if event["event"] == event_name
                ]

                self.assertEqual(enabled_events, disabled_events)
                self.assertFalse(
                    any(
                        "edge_compute_epoch" in event
                        for event in enabled.event_trace
                        if event["event"] == event_name
                    )
                )

    def test_dynamic_runs_preserve_greedy_outputs_for_all_methods(self) -> None:
        config, _, workload = small_config(num_requests=2, output_len=6)
        enable_dynamic_edge(config)
        target = Simulator(
            copy.deepcopy(config),
            accepting_model_runner(),
            workload,
            "test",
            "target_only",
        ).run()
        expected = [request.generated_ids for request in target.requests]

        for method in self.METHODS:
            with self.subTest(method=method):
                result = Simulator(
                    copy.deepcopy(config),
                    accepting_model_runner(),
                    workload,
                    "test",
                    method,
                ).run()
                self.assertEqual(
                    [request.generated_ids for request in result.requests],
                    expected,
                )
```

- [ ] **Step 2: Run isolation tests and verify RED if any routing leaked**

```bash
cd code && rtk pytest -q tests/test_dynamic_edge_compute.py -k isolation
```

Expected before final cleanup: any remaining disabled trace fields, server/target routing leak, or missing legacy fallback produces a focused FAIL. If all tests already pass, verify that each assertion exercises the newly added model and trace paths; do not manufacture production changes merely to force another failure.

- [ ] **Step 3: Apply only test-proven cleanup**

For each observed failure, make the minimum correction in `edge_compute.py` or `simulator.py`. Required end state:

```text
disabled: no sampling, no transitions, no dynamic trace fields, old rates
target_only: fixed target latency only
server_only_*: fixed server_only draft latency only
edge methods: dynamic snapshots only for edge draft work
all methods: unchanged semantic token output
```

Do not edit scheduler, communication, target latency, verification profile, batching, or method specifications.

- [ ] **Step 4: Run focused integration and invariant regressions GREEN**

```bash
cd code && rtk pytest -q tests/test_dynamic_edge_compute.py
cd code && rtk pytest -q tests/test_target_latency_profile_integration.py tests/test_baseline_system_invariants.py
```

Expected: PASS. Dynamic behavior is isolated to edge draft resource timing and trace provenance.

- [ ] **Step 5: Audit excluded files and call sites**

```bash
rtk git diff -- code/src/communication.py code/src/scheduler.py code/src/latency.py code/src/verification_latency_profile.py code/src/methods.py
rtk rg -n "EdgeComputeModel|edge_compute|dynamic_draft" code/src code/configs code/tests/test_dynamic_edge_compute.py
```

Expected: the first command has no output. The second shows one simulator-owned model, the three edge start integrations, planner reads, configuration, and tests; it shows no method-specific model construction.

- [ ] **Step 6: Commit regression contracts or narrowly scoped fixes**

```bash
rtk git add code/tests/test_dynamic_edge_compute.py code/src/edge_compute.py code/src/simulator.py
rtk git commit -m "test: lock dynamic edge compute boundaries"
```

If Step 3 required no production correction, this commit contains only the regression tests. Do not create an empty commit.

## Final Verification

Do not add features in this phase. If a command fails, return to the owning task, add or correct a focused failing test, apply the minimum fix, rerun that task's GREEN commands, and restart this sequence.

- [ ] **Step 1: Run the dedicated target tests**

```bash
cd code && rtk pytest -q tests/test_dynamic_edge_compute.py tests/test_config.py tests/test_latency_estimator.py
```

Expected: all configuration, deterministic sampling, state transition, snapshot, finalization, routing, disabled-mode, and analytical fallback tests pass.

- [ ] **Step 2: Run affected method and target-profile regressions**

```bash
cd code && rtk pytest -q \
  tests/test_target_only.py \
  tests/test_server_only_linear.py \
  tests/test_server_only_tree.py \
  tests/test_specedge_linear.py \
  tests/test_specedge_tree.py \
  tests/test_specedge_methods.py \
  tests/test_dip_sd.py \
  tests/test_linear_sd_core.py \
  tests/test_runtime_prediction.py \
  tests/test_target_latency_profile_integration.py \
  tests/test_baseline_system_invariants.py
```

Expected: zero failures and unchanged algorithm semantics.

- [ ] **Step 3: Run the complete pytest suite**

```bash
cd code && rtk pytest -q
```

Expected: zero failures.

- [ ] **Step 4: Run baseline reconstruction verification**

```bash
cd code && rtk bash scripts/verify_baseline_rebuild.sh
```

Expected: full and method-specific suites pass; static checks find no forbidden prefill or obsolete DiP-SD paths.

- [ ] **Step 5: Run the six-method baseline trace**

```bash
cd code && rtk bash scripts/run_baseline_trace.sh
```

Expected: target-only, both server-only variants, both SpecEdge variants, and DiP-SD succeed under default `dynamic_edge_compute.enabled: false`; generated baseline trace semantics remain unchanged.

- [ ] **Step 6: Run final diff and scope checks**

```bash
rtk git diff --check
rtk git status --short
rtk git diff -- \
  code/configs/default.yaml \
  code/src/config.py \
  code/src/edge_compute.py \
  code/src/simulator.py \
  code/tests/test_dynamic_edge_compute.py
rtk git diff -- \
  code/src/communication.py \
  code/src/scheduler.py \
  code/src/latency.py \
  code/src/verification_latency_profile.py \
  code/src/methods.py
rtk rg -n 'request\.status = "finished"|request\.finish_time_ms =' code/src/simulator.py
```

Expected: `git diff --check` is clean; only planned files changed; excluded subsystem diff is empty; final request status/time have exactly one write pair in `_finalize_request`.

- [ ] **Step 7: Commit verification-only corrections if needed**

If all previous commits pass unchanged, create no empty commit. If verification exposes a narrow regression, first add a failing test to `test_dynamic_edge_compute.py`, apply the minimum correction, rerun all final verification, then commit only that correction:

```bash
rtk git add code/tests/test_dynamic_edge_compute.py code/src/edge_compute.py code/src/config.py code/src/simulator.py code/configs/default.yaml
rtk git commit -m "fix: preserve dynamic edge compute contracts"
```

## Non-Goals Audit

The implementation must not change server-only draft capacity, target compute, target latency profile selection or queries, network bandwidth/RTT/jitter, scheduler policy, lane assignment, batching, timeout behavior, gamma candidates, proactive policy, verification semantics, request-to-device assignment, semantic token generation, or greedy output equivalence. It must not add time-based throttling, utilization coupling, failures, batteries, correlated device states, prefill, prompt transmission, TTFT, or a method-specific resource advantage.
