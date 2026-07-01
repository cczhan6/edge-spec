# Probabilistic Network Blocking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, device-level probabilistic short-blocking model shared by every networked simulator method while preserving exact default network timing and trace behavior.

**Architecture:** Extend immutable `Device` records and device templates with `block_probability`, defaulting to `1.0`. Keep `Simulator._network_delay_ms` as a pass-through facade and implement both the domain-separated block decision and conditional reuse of the legacy jitter-duration hash solely in `src/communication.py`; methods with no decode-stage communication continue to bypass the network model.

**Tech Stack:** Python 3, frozen dataclasses, SHA-256 from `hashlib`, YAML configuration, `unittest`-style pytest tests, fake model runner, Bash baseline verification and trace scripts.

---

## Scope and File Map

- Modify `code/src/entities.py`: append the immutable device-level `block_probability` field with compatibility default `1.0`.
- Modify `code/src/config.py`: validate device link fields and propagate `block_probability` through `build_devices`.
- Modify `code/src/communication.py`: add the independent block-decision hash domain and conditionally apply the unchanged jitter-duration sampler.
- Modify `code/configs/default.yaml`: declare `block_probability: 1.0` for every default device template.
- Modify `code/docs/experiment.md`: document the probabilistic blocking formula and default compatibility behavior.
- Modify `code/tests/test_config.py`: cover defaulting, propagation, scenario merge, and invalid device network configuration.
- Modify `code/tests/test_communication.py`: cover the block hash, probabilities zero and one, intermediate deterministic behavior, independence, bounds, and defensive validation.
- Create `code/tests/test_probabilistic_network_blocking.py`: cover shared simulator routing, current event keys, exact trace-row compatibility, reproducibility, and network-free methods.
- Do not modify `code/src/simulator.py`: its existing facade and event keys are already the shared integration boundary.
- Do not modify `code/src/edge_compute.py`, `code/src/latency.py`, `code/src/scheduler.py`, `code/src/verification_latency_profile.py`, `code/src/methods.py`, model execution, or any algorithm implementation.

## Pre-Implementation Call-Chain Confirmation

Run these read-only checks before Task 1. If the branch differs from the expected results, stop and update this plan against the actual branch instead of adding a second network path.

- [ ] **Check 1: Confirm `Device` owns the static network fields**

```bash
rtk rg -n -A 18 "class Device" code/src/entities.py
```

Expected: frozen `Device` contains `uplink_mbps`, `downlink_mbps`, `rtt_ms`, and `jitter_ms`; it does not yet contain `block_probability`.

- [ ] **Check 2: Confirm `build_devices` is the production construction path**

```bash
rtk rg -n -A 35 "def build_devices" code/src/config.py
rtk rg -n "build_devices\(" code/src code/tests --glob '*.py'
```

Expected: `src/config.py:build_devices` reads network values from `device_pools.<pool>.templates`, assigns sequential IDs, and `Simulator.__init__` calls it using `MethodSpec.device_pool`.

- [ ] **Check 3: Confirm the authoritative communication functions**

```bash
rtk rg -n -A 35 "def dssd_transmission_delay_ms|def deterministic_jitter_ms|def network_delay_ms" code/src/communication.py
```

Expected: `network_delay_ms` selects directional bandwidth, calls `dssd_transmission_delay_ms`, and unconditionally adds `deterministic_jitter_ms`; the jitter hash material is exactly `"{seed}:{device_id}:{direction}:{key}"`.

- [ ] **Check 4: Confirm the simulator facade supplies the experiment seed**

```bash
rtk rg -n -A 28 "def _speculative_network_delay_ms|def _network_delay_ms" code/src/simulator.py
```

Expected: `_speculative_network_delay_ms` converts tokens to payload bytes, `_network_delay_ms` passes `simulation.seed`, device, direction, key, and payload to `src.communication.network_delay_ms`, and neither wrapper samples randomness.

- [ ] **Check 5: Confirm every actual and predictive communication key**

```bash
rtk rg -n -C 4 "_network_delay_ms\(|_speculative_network_delay_ms\(" code/src/simulator.py
```

Expected keys:

| Consumer | Direction | Existing key |
| --- | --- | --- |
| Async/full and SpecEdge actual transfer | Uplink | `segment.segment_id` |
| Async/full and SpecEdge actual transfer | Downlink | `segment.segment_id` |
| DiP-SD actual transfer | Uplink | `dip-sd-up:{epoch_index}:{request_id}` |
| DiP-SD actual transfer | Downlink | `dip-sd-down:{epoch_index}:{request_id}` |
| Adaptive gamma estimate | Uplink | `estimate-up:{gamma}` |
| Adaptive gamma estimate | Downlink | `estimate-down:{gamma}` |
| SpecEdge pipeline estimate | Uplink | `pipeline-up:{device_id}:{gamma}` |
| SpecEdge pipeline estimate | Downlink | `pipeline-down:{device_id}:{gamma}` |
| DiP-SD planning estimate | Uplink | `dip-sd-plan:{epoch_index}:{request_id}` |

- [ ] **Check 6: Confirm the network-free method contract**

```bash
rtk rg -n "has_no_network|No network event|Network communication: none|Edge.server communication: none" code/tests code/docs/baseline_contract.md
```

Expected: Target-only and both server-only variants explicitly require no network events; networked canonical methods and `full` already share the simulator facade.

## TDD Discipline

Every production behavior change follows RED -> GREEN -> focused regression -> commit. Run each RED command and confirm it fails for the expected missing behavior before editing production code. A syntax error, import typo, or broken fixture is not a valid RED result. Implement only enough behavior for the current task, rerun its GREEN commands, and commit before continuing.

Configuration and documentation edits in Task 5 are driven by a failing configuration-surface test. Do not add production code in Task 5.

### Task 1: Add the device field and construction fallback

**Files:**
- Modify: `code/tests/test_config.py`
- Modify: `code/src/entities.py:20-32`
- Modify: `code/src/config.py:286-308`

- [ ] **Step 1: Write failing device-construction tests**

Add `build_devices` to the imports in `code/tests/test_config.py` and add these tests to `ConfigTest`:

```python
from src.config import build_devices, load_config, validate_config


class ConfigTest(unittest.TestCase):
    # Keep the existing tests above and below these methods.

    def test_build_devices_defaults_missing_block_probability_to_one(self) -> None:
        config = load_config("configs/default.yaml")
        for pool in config["device_pools"].values():
            for template in pool["templates"].values():
                template.pop("block_probability", None)

        for pool_name in ("heterogeneous", "medium_only"):
            with self.subTest(pool_name=pool_name):
                devices = build_devices(config, pool_name)
                self.assertTrue(devices)
                self.assertTrue(
                    all(device.block_probability == 1.0 for device in devices)
                )

    def test_build_devices_propagates_template_block_probability(self) -> None:
        config = load_config("configs/default.yaml")
        templates = config["device_pools"]["heterogeneous"]["templates"]
        templates["low_end"]["block_probability"] = 0.1
        templates["mid_end"]["block_probability"] = 0.4
        templates["high_end"]["block_probability"] = 0.9

        devices = build_devices(config, "heterogeneous")

        expected = {"low_end": 0.1, "mid_end": 0.4, "high_end": 0.9}
        self.assertTrue(devices)
        for device in devices:
            self.assertEqual(
                device.block_probability,
                expected[device.device_type],
            )
```

- [ ] **Step 2: Run the tests to verify RED**

```bash
cd code && rtk pytest -q tests/test_config.py -k "build_devices_defaults_missing_block_probability or build_devices_propagates_template_block_probability"
```

Expected: FAIL because current `Device` instances have no `block_probability` field.

- [ ] **Step 3: Add the compatibility-defaulted `Device` field**

Append the field after `jitter_ms` in `code/src/entities.py` so every existing positional constructor remains valid:

```python
@dataclass(frozen=True)
class Device:
    device_id: int
    device_type: str
    drafter_profile: str
    acceptance_prior: float
    draft_token_rate_tok_s: float
    draft_startup_ms: float
    uplink_mbps: float
    downlink_mbps: float
    rtt_ms: float
    jitter_ms: float
    block_probability: float = 1.0
```

- [ ] **Step 4: Propagate the template value in `build_devices`**

Add the final keyword argument to the existing constructor in `code/src/config.py`:

```python
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
```

- [ ] **Step 5: Run focused and constructor regressions to verify GREEN**

```bash
cd code && rtk pytest -q tests/test_config.py tests/test_communication.py tests/test_metrics_speedup.py
```

Expected: PASS. Existing direct `Device(...)` calls continue to work because the new final field defaults to `1.0`.

- [ ] **Step 6: Commit the device surface**

```bash
rtk git add code/src/entities.py code/src/config.py code/tests/test_config.py
rtk git commit -m "feat: add device network block probability"
```

### Task 2: Validate probabilities and device link bounds

**Files:**
- Modify: `code/tests/test_config.py`
- Modify: `code/src/config.py:220-245`

- [ ] **Step 1: Write failing probability validation tests**

Add these tests to `ConfigTest`:

```python
    def test_block_probability_accepts_closed_unit_interval(self) -> None:
        for value in (0, 0.25, 1):
            with self.subTest(value=value):
                config = load_config("configs/default.yaml")
                config["device_pools"]["heterogeneous"]["templates"]["low_end"][
                    "block_probability"
                ] = value
                validate_config(config)

    def test_block_probability_rejects_invalid_values_with_template_path(self) -> None:
        invalid_values = (
            True,
            "0.5",
            -0.01,
            1.01,
            float("nan"),
            float("inf"),
            float("-inf"),
        )
        for value in invalid_values:
            with self.subTest(value=value):
                config = load_config("configs/default.yaml")
                config["device_pools"]["heterogeneous"]["templates"]["low_end"][
                    "block_probability"
                ] = value
                with self.assertRaisesRegex(
                    ValueError,
                    r"device_pools\.heterogeneous\.templates\.low_end\."
                    r"block_probability must be a finite number in \[0, 1\]",
                ):
                    validate_config(config)

    def test_invalid_network_bounds_are_rejected_even_for_zero_count_template(self) -> None:
        invalid_fields = (
            ("jitter_ms", -1.0, "must be a finite non-negative number"),
            ("jitter_ms", float("nan"), "must be a finite non-negative number"),
            ("rtt_ms", -1.0, "must be a finite non-negative number"),
            ("rtt_ms", float("inf"), "must be a finite non-negative number"),
            ("uplink_mbps", float("nan"), "must be a finite positive number"),
            ("downlink_mbps", 0.0, "must be a finite positive number"),
        )
        for field, value, message in invalid_fields:
            with self.subTest(field=field, value=value):
                config = load_config("configs/default.yaml")
                template = config["device_pools"]["heterogeneous"]["templates"][
                    "high_end"
                ]
                template["count"] = 0
                template[field] = value
                with self.assertRaisesRegex(
                    ValueError,
                    rf"device_pools\.heterogeneous\.templates\.high_end\."
                    rf"{field} {message}",
                ):
                    validate_config(config)
```

- [ ] **Step 2: Run the validation tests to verify RED**

```bash
cd code && rtk pytest -q tests/test_config.py -k "block_probability or invalid_network_bounds"
```

Expected: FAIL because probability is not validated, strings and booleans can reach float conversion, and current finite/non-negative link checks are incomplete.

- [ ] **Step 3: Add one focused device-network validator**

Add this helper near `_validate_dynamic_edge_compute` in `code/src/config.py`:

```python
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
```

In the existing device-template validation loop, call the helper before counting or constructing devices and remove the old combined bandwidth-only condition:

```python
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
```

Do not introduce clipping or accept numeric strings. Missing probability remains valid and resolves to `1.0`.

- [ ] **Step 4: Run focused and full configuration tests to verify GREEN**

```bash
cd code && rtk pytest -q tests/test_config.py
```

Expected: PASS, including existing pool-count, drafter, and bandwidth validation tests.

- [ ] **Step 5: Commit configuration validation**

```bash
rtk git add code/src/config.py code/tests/test_config.py
rtk git commit -m "feat: validate probabilistic network configuration"
```

### Task 3: Add the independent deterministic block decision

**Files:**
- Modify: `code/tests/test_communication.py`
- Modify: `code/src/communication.py:1-27`

- [ ] **Step 1: Write failing block-domain tests**

Replace the narrow imports in `code/tests/test_communication.py` and add a reusable device helper plus these tests:

```python
from __future__ import annotations

import hashlib
import unittest

import src.communication as communication
from src.communication import (
    deterministic_jitter_ms,
    dssd_transmission_delay_ms,
    network_delay_ms,
)
from src.entities import Device


def make_device(
    *,
    device_id: int = 0,
    jitter_ms: float = 25.0,
    block_probability: float = 1.0,
) -> Device:
    return Device(
        device_id=device_id,
        device_type="small_device",
        drafter_profile="small",
        acceptance_prior=0.5,
        draft_token_rate_tok_s=500.0,
        draft_startup_ms=1.0,
        uplink_mbps=8.0,
        downlink_mbps=16.0,
        rtt_ms=20.0,
        jitter_ms=jitter_ms,
        block_probability=block_probability,
    )


def reference_block_ratio(
    seed: int,
    device_id: int,
    direction: str,
    key: object,
) -> float:
    material = (
        f"network-block-v1:{seed}:{device_id}:{direction}:{key}".encode()
    )
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") / 2**64


class CommunicationTest(unittest.TestCase):
    # Keep the existing DSSD formula test.

    def test_block_decision_uses_documented_independent_hash_domain(self) -> None:
        cases = (
            (make_device(device_id=0, block_probability=0.5), "uplink", "segment-1"),
            (make_device(device_id=1, block_probability=0.5), "uplink", "segment-1"),
            (make_device(device_id=0, block_probability=0.5), "downlink", "segment-1"),
            (make_device(device_id=0, block_probability=0.5), "uplink", "segment-2"),
        )

        actual = [
            communication.deterministic_network_blocked(7, device, direction, key)
            for device, direction, key in cases
        ]
        expected = [
            reference_block_ratio(7, device.device_id, direction, key)
            < device.block_probability
            for device, direction, key in cases
        ]

        self.assertEqual(actual, expected)
        self.assertEqual(actual, [True, False, True, False])

    def test_block_decision_is_reproducible_and_call_order_independent(self) -> None:
        device = make_device(block_probability=0.5)
        keys = ["segment-1", "segment-2", "segment-3"]

        forward = [
            communication.deterministic_network_blocked(7, device, "uplink", key)
            for key in keys
        ]
        reverse = {
            key: communication.deterministic_network_blocked(
                7, device, "uplink", key
            )
            for key in reversed(keys)
        }

        self.assertEqual(forward, [reverse[key] for key in keys])

    def test_block_decision_includes_seed(self) -> None:
        device = make_device(block_probability=0.5)

        self.assertTrue(
            communication.deterministic_network_blocked(
                7, device, "uplink", "segment-1"
            )
        )
        self.assertFalse(
            communication.deterministic_network_blocked(
                8, device, "uplink", "segment-1"
            )
        )

    def test_block_and_wait_materials_are_distinct(self) -> None:
        block_material = b"network-block-v1:7:0:uplink:segment-2"
        wait_material = b"7:0:uplink:segment-2"

        self.assertNotEqual(
            hashlib.sha256(block_material).digest()[:8],
            hashlib.sha256(wait_material).digest()[:8],
        )
```

- [ ] **Step 2: Run the new tests to verify RED**

```bash
cd code && rtk pytest -q tests/test_communication.py -k "block_decision or block_and_wait_materials"
```

Expected: FAIL with `AttributeError` because `src.communication` does not yet
define `deterministic_network_blocked`; test collection itself succeeds.

- [ ] **Step 3: Implement only the block-decision helper**

Add the namespace constant and helper to `code/src/communication.py` without changing `network_delay_ms` yet:

```python
NETWORK_BLOCK_DOMAIN = "network-block-v1"


def deterministic_network_blocked(
    seed: int,
    device: Device,
    direction: str,
    key: Any,
) -> bool:
    digest_key = (
        f"{NETWORK_BLOCK_DOMAIN}:{seed}:{device.device_id}:{direction}:{key}"
    ).encode()
    ratio = int.from_bytes(hashlib.sha256(digest_key).digest()[:8], "big") / 2**64
    return ratio < device.block_probability
```

Do not refactor `deterministic_jitter_ms`; preserving its exact material and arithmetic is the default trace-compatibility mechanism.

- [ ] **Step 4: Run the communication tests to verify GREEN**

```bash
cd code && rtk pytest -q tests/test_communication.py
```

Expected: PASS. At this commit the helper is tested but `network_delay_ms` still has legacy unconditional jitter behavior.

- [ ] **Step 5: Commit the independent sampling primitive**

```bash
rtk git add code/src/communication.py code/tests/test_communication.py
rtk git commit -m "feat: add deterministic network block decision"
```

### Task 4: Apply conditional jitter through the shared communication path

**Files:**
- Modify: `code/tests/test_communication.py`
- Create: `code/tests/test_probabilistic_network_blocking.py`
- Modify: `code/src/communication.py:8-60`
- Regression only: `code/src/simulator.py`

- [ ] **Step 1: Write failing probability-zero, probability-one, and intermediate tests**

Add these methods to `CommunicationTest` in `code/tests/test_communication.py`:

```python
    def test_probability_zero_always_omits_extra_jitter(self) -> None:
        device = make_device(jitter_ms=25.0, block_probability=0.0)
        for direction, payload_bytes, bandwidth in (
            ("uplink", 1000, device.uplink_mbps),
            ("downlink", 1000, device.downlink_mbps),
        ):
            with self.subTest(direction=direction):
                expected = dssd_transmission_delay_ms(
                    payload_bytes,
                    device.rtt_ms,
                    bandwidth,
                )
                self.assertEqual(
                    network_delay_ms(7, device, direction, "segment-1", payload_bytes),
                    expected,
                )

    def test_probability_one_is_exactly_legacy_network_delay(self) -> None:
        device = make_device(jitter_ms=25.0, block_probability=1.0)
        cases = (
            (1, "uplink", "x", 1000),
            (7, "downlink", "segment-1", 64),
            (99, "uplink", 17, 0),
        )
        for seed, direction, key, payload_bytes in cases:
            with self.subTest(
                seed=seed,
                direction=direction,
                key=key,
                payload_bytes=payload_bytes,
            ):
                bandwidth = (
                    device.uplink_mbps
                    if direction == "uplink"
                    else device.downlink_mbps
                )
                legacy = dssd_transmission_delay_ms(
                    payload_bytes,
                    device.rtt_ms,
                    bandwidth,
                ) + deterministic_jitter_ms(seed, device, direction, key)
                self.assertEqual(
                    network_delay_ms(seed, device, direction, key, payload_bytes),
                    legacy,
                )

    def test_intermediate_probability_uses_legacy_wait_only_when_blocked(self) -> None:
        device = make_device(jitter_ms=25.0, block_probability=0.5)
        payload_bytes = 1000
        base = dssd_transmission_delay_ms(
            payload_bytes,
            device.rtt_ms,
            device.uplink_mbps,
        )

        blocked = network_delay_ms(
            7, device, "uplink", "segment-6", payload_bytes
        )
        unblocked = network_delay_ms(
            7, device, "uplink", "segment-2", payload_bytes
        )

        self.assertEqual(
            blocked,
            base + deterministic_jitter_ms(7, device, "uplink", "segment-6"),
        )
        self.assertEqual(unblocked, base)
        self.assertGreaterEqual(blocked - base, 0.0)
        self.assertLessEqual(blocked - base, device.jitter_ms)
        self.assertLess(
            reference_block_ratio(7, 0, "uplink", "segment-6"),
            0.5,
        )
        self.assertGreater(
            deterministic_jitter_ms(7, device, "uplink", "segment-6")
            / device.jitter_ms,
            0.5,
        )

    def test_direct_device_network_validation_rejects_invalid_values(self) -> None:
        invalid_devices = (
            make_device(block_probability=-0.1),
            make_device(block_probability=1.1),
            make_device(block_probability=float("nan")),
            make_device(jitter_ms=-1.0),
        )
        for device in invalid_devices:
            with self.subTest(device=device):
                with self.assertRaises(ValueError):
                    network_delay_ms(7, device, "uplink", "x", 1000)
```

- [ ] **Step 2: Write the failing shared-runtime integration tests**

Create `code/tests/test_probabilistic_network_blocking.py`:

```python
from __future__ import annotations

import copy
import unittest
from unittest import mock

from scripts.baseline_trace import (
    _batch_trace_rows,
    _event_trace_rows,
    _request_trace_rows,
    _resource_timeline_rows,
    _token_trace_rows,
)
from src.communication import dssd_transmission_delay_ms, network_delay_ms
from src.simulator import Simulator
from tests.common import accepting_model_runner, small_config


NETWORKED_METHODS = (
    "full",
    "wo_async",
    "wo_scheduling",
    "conservative_rollback",
    "specedge_linear",
    "specedge_tree",
    "dip_sd",
)
NETWORK_FREE_METHODS = (
    "target_only",
    "server_only_linear",
    "server_only_tree",
)


def set_network_profile(config: dict, probability: float) -> None:
    for pool in config["device_pools"].values():
        for template in pool["templates"].values():
            template["rtt_ms"] = 20.0
            template["uplink_mbps"] = 8.0
            template["downlink_mbps"] = 16.0
            template["jitter_ms"] = 25.0
            template["block_probability"] = probability


def run_method(method: str, probability: float, seed: int = 42):
    config, _, workload = small_config(num_requests=2, output_len=6)
    config["simulation"]["seed"] = seed
    config["simulation"]["request_arrival"] = "burst"
    config["specedge"]["server_batch_size"] = 2
    config["dip_sd"]["batch_count"] = 2
    config["dip_sd"]["max_batch_size"] = 2
    set_network_profile(config, probability)
    result = Simulator(
        config,
        accepting_model_runner(),
        workload,
        "test",
        method,
    ).run()
    return config, result


def trace_rows(result) -> dict[str, list[dict]]:
    return {
        "requests": _request_trace_rows(result),
        "events": _event_trace_rows(result),
        "tokens": _token_trace_rows(result),
        "resources": _resource_timeline_rows(result),
        "batches": _batch_trace_rows(result),
    }


class ProbabilisticNetworkBlockingIntegrationTest(unittest.TestCase):
    def test_probability_zero_removes_jitter_for_every_networked_method(self) -> None:
        for method in NETWORKED_METHODS:
            with self.subTest(method=method):
                _, result = run_method(method, probability=0.0)
                self.assertTrue(result.segments)
                for segment in result.segments:
                    device = result.devices[segment.device_id].device
                    if segment.uplink_payload_bytes:
                        expected_uplink = dssd_transmission_delay_ms(
                            segment.uplink_payload_bytes,
                            device.rtt_ms,
                            device.uplink_mbps,
                        )
                        self.assertEqual(segment.uplink_delay_ms, expected_uplink)
                    if segment.downlink_payload_bytes:
                        expected_downlink = dssd_transmission_delay_ms(
                            segment.downlink_payload_bytes,
                            device.rtt_ms,
                            device.downlink_mbps,
                        )
                        self.assertEqual(segment.downlink_delay_ms, expected_downlink)

    def test_default_probability_one_and_missing_field_have_exact_trace_rows(self) -> None:
        for method in (*NETWORK_FREE_METHODS, *NETWORKED_METHODS):
            with self.subTest(method=method):
                config, _, workload = small_config(num_requests=2, output_len=6)
                config["simulation"]["request_arrival"] = "burst"
                set_network_profile(config, probability=1.0)
                explicit = Simulator(
                    copy.deepcopy(config),
                    accepting_model_runner(),
                    workload,
                    "test",
                    method,
                ).run()
                legacy = copy.deepcopy(config)
                for pool in legacy["device_pools"].values():
                    for template in pool["templates"].values():
                        template.pop("block_probability", None)
                missing = Simulator(
                    legacy,
                    accepting_model_runner(),
                    workload,
                    "test",
                    method,
                ).run()

                self.assertEqual(trace_rows(explicit), trace_rows(missing))
                self.assertEqual(explicit.requests, missing.requests)
                self.assertEqual(explicit.segments, missing.segments)

    def test_intermediate_probability_is_reproducible(self) -> None:
        _, first = run_method("full", probability=0.5, seed=42)
        _, second = run_method("full", probability=0.5, seed=42)

        first_delays = [
            (segment.uplink_delay_ms, segment.downlink_delay_ms)
            for segment in first.segments
        ]
        second_delays = [
            (segment.uplink_delay_ms, segment.downlink_delay_ms)
            for segment in second.segments
        ]
        self.assertEqual(first.event_trace, second.event_trace)
        self.assertEqual(first_delays, second_delays)

    def test_all_networked_methods_keep_current_keys_at_shared_function(self) -> None:
        expected_prefixes = {
            "full": ("estimate-up:", "estimate-down:"),
            "wo_async": ("estimate-up:", "estimate-down:"),
            "wo_scheduling": ("estimate-up:", "estimate-down:"),
            "conservative_rollback": ("estimate-up:", "estimate-down:"),
            "specedge_linear": ("pipeline-up:", "pipeline-down:"),
            "specedge_tree": ("pipeline-up:", "pipeline-down:"),
            "dip_sd": ("dip-sd-plan:", "dip-sd-up:", "dip-sd-down:"),
        }
        for method, prefixes in expected_prefixes.items():
            with self.subTest(method=method):
                with mock.patch(
                    "src.simulator.network_delay_ms",
                    wraps=network_delay_ms,
                ) as shared_delay:
                    run_method(method, probability=0.5)
                keys = [call.args[3] for call in shared_delay.call_args_list]
                self.assertTrue(keys)
                for prefix in prefixes:
                    self.assertTrue(
                        any(str(key).startswith(prefix) for key in keys),
                        (method, prefix, keys),
                    )
                if method != "dip_sd":
                    self.assertTrue(any(isinstance(key, int) for key in keys))

    def test_target_and_server_only_remain_network_free(self) -> None:
        for method in NETWORK_FREE_METHODS:
            with self.subTest(method=method):
                with mock.patch(
                    "src.simulator.network_delay_ms",
                    wraps=network_delay_ms,
                ) as shared_delay:
                    _, result = run_method(method, probability=0.5)
                shared_delay.assert_not_called()
                self.assertTrue(
                    all(segment.uplink_delay_ms == 0.0 for segment in result.segments)
                )
                self.assertTrue(
                    all(segment.downlink_delay_ms == 0.0 for segment in result.segments)
                )
                for event in result.event_trace:
                    self.assertNotIn("uplink_ms", event)
                    self.assertNotIn("downlink_ms", event)
```

- [ ] **Step 3: Run unit and integration tests to verify RED**

```bash
cd code && rtk pytest -q tests/test_communication.py tests/test_probabilistic_network_blocking.py
```

Expected: FAIL in the probability-zero and intermediate-blocking assertions because `network_delay_ms` still adds jitter unconditionally. The probability-one compatibility and pre-existing network-free invariants may already pass; the test command as a whole must be RED for the missing conditional behavior.

- [ ] **Step 4: Add defensive runtime link validation**

Add `math` to the imports in `code/src/communication.py` and add this helper:

```python
def _validate_device_network(device: Device) -> None:
    probability = device.block_probability
    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(float(probability))
        or not 0.0 <= float(probability) <= 1.0
    ):
        raise ValueError("block_probability must be a finite number in [0, 1]")
    if not math.isfinite(device.rtt_ms) or device.rtt_ms < 0.0:
        raise ValueError("rtt_ms must be finite and non-negative")
    if not math.isfinite(device.jitter_ms) or device.jitter_ms < 0.0:
        raise ValueError("jitter_ms must be finite and non-negative")
    for name, value in (
        ("uplink_mbps", device.uplink_mbps),
        ("downlink_mbps", device.downlink_mbps),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
```

- [ ] **Step 5: Apply conditional legacy jitter in `network_delay_ms`**

Keep direction validation first, validate the direct `Device`, preserve the current base-delay calculation, and add jitter only for a blocked transmission:

```python
def network_delay_ms(
    seed: int,
    device: Device,
    direction: str,
    key: Any,
    payload_bytes: int,
) -> float:
    if direction not in {"uplink", "downlink"}:
        raise ValueError(f"unsupported direction: {direction}")
    _validate_device_network(device)
    bandwidth_mbps = (
        device.uplink_mbps if direction == "uplink" else device.downlink_mbps
    )
    base_delay_ms = dssd_transmission_delay_ms(
        payload_bytes,
        device.rtt_ms,
        bandwidth_mbps,
    )
    if not deterministic_network_blocked(seed, device, direction, key):
        return base_delay_ms
    return base_delay_ms + deterministic_jitter_ms(seed, device, direction, key)
```

This preserves the exact existing arithmetic order for `block_probability=1.0`: calculate the same DSSD base, calculate the same legacy jitter, then add them. Do not change any existing simulator key or add a method branch.

- [ ] **Step 6: Run focused tests to verify GREEN**

```bash
cd code && rtk pytest -q tests/test_communication.py tests/test_probabilistic_network_blocking.py
```

Expected: PASS. Probability zero has no extra wait, probability one is exactly the legacy result, intermediate probability follows the domain-separated decision, all actual and planning paths keep their current keys, and network-free methods never call the shared function.

- [ ] **Step 7: Run affected method regressions**

```bash
cd code && rtk pytest -q \
  tests/test_target_only.py \
  tests/test_server_only_linear.py \
  tests/test_server_only_tree.py \
  tests/test_specedge_linear.py \
  tests/test_specedge_tree.py \
  tests/test_specedge_methods.py \
  tests/test_dip_sd.py \
  tests/test_runtime_prediction.py \
  tests/test_baseline_system_invariants.py
```

Expected: PASS with unchanged candidate, scheduler, batching, verification, and greedy-output semantics.

- [ ] **Step 8: Commit the shared behavior and integration guards**

```bash
rtk git add \
  code/src/communication.py \
  code/tests/test_communication.py \
  code/tests/test_probabilistic_network_blocking.py
rtk git commit -m "feat: apply probabilistic network blocking"
```

### Task 5: Declare the default configuration and update the formula documentation

**Files:**
- Modify: `code/tests/test_config.py`
- Modify: `code/configs/default.yaml:80-126`
- Modify: `code/docs/experiment.md:215-235`

- [ ] **Step 1: Write the failing explicit-default and scenario-merge test**

Add this method to `ConfigTest`:

```python
    def test_default_and_scenario_device_templates_declare_probability_one(self) -> None:
        for scenario in (None, "homogeneous", "combined_strong_heterogeneous"):
            with self.subTest(scenario=scenario):
                config = load_config("configs/default.yaml", scenario)
                for pool_name, pool in config["device_pools"].items():
                    for template_name, template in pool["templates"].items():
                        self.assertIn(
                            "block_probability",
                            template,
                            (scenario, pool_name, template_name),
                        )
                        self.assertEqual(template["block_probability"], 1.0)
```

- [ ] **Step 2: Run the test to verify RED**

```bash
cd code && rtk pytest -q tests/test_config.py -k default_and_scenario_device_templates_declare_probability_one
```

Expected: FAIL because the current default templates do not explicitly declare `block_probability`.

- [ ] **Step 3: Add the explicit compatibility default to every default template**

In `code/configs/default.yaml`, add the field beside `jitter_ms` for `low_end`, `mid_end`, `high_end`, and `medium_only.medium`:

```yaml
        uplink_mbps: 5
        downlink_mbps: 30
        rtt_ms: 90
        jitter_ms: 25
        block_probability: 1.0
```

Use each template's existing bandwidth, RTT, and jitter values; only the new probability line is common. Do not add a method-level or top-level probability.

- [ ] **Step 4: Update the experiment communication formula**

Replace the unconditional-jitter formula in `code/docs/experiment.md` with:

```text
u_block = deterministic_ratio("network-block-v1", seed, device_id, direction, event_key)
blocked = u_block < block_probability
blocking_wait_ms = deterministic_legacy_jitter_ms if blocked else 0
delay_ms = RTT_ms / 2
         + payload_bytes * 8 / (bandwidth_mbps * 1000)
         + blocking_wait_ms
```

Immediately state that `block_probability` is device-level, defaults to `1.0`, and therefore preserves the existing legacy jitter and trace timing exactly. State that formal dynamic-network scenarios opt in with a value below one.

- [ ] **Step 5: Run configuration and communication regressions to verify GREEN**

```bash
cd code && rtk pytest -q \
  tests/test_config.py \
  tests/test_communication.py \
  tests/test_probabilistic_network_blocking.py
```

Expected: PASS. Default and merged scenarios expose `1.0`, while omitted external configuration remains compatible through the construction fallback.

- [ ] **Step 6: Commit defaults and user-facing documentation**

```bash
rtk git add code/configs/default.yaml code/docs/experiment.md code/tests/test_config.py
rtk git commit -m "docs: declare probabilistic network defaults"
```

## Final Verification

Do not add features during this phase. If a command fails, return to the task that owns the behavior, add or correct a focused failing regression test, apply the minimum fix, rerun that task's GREEN commands, and restart this sequence.

- [ ] **Step 1: Reconfirm the final shared call graph and unchanged keys**

```bash
rtk rg -n "class Device|block_probability" code/src/entities.py code/src/config.py code/configs/default.yaml
rtk rg -n "def deterministic_jitter_ms|def deterministic_network_blocked|def network_delay_ms" code/src/communication.py
rtk rg -n -C 4 "_network_delay_ms\(|_speculative_network_delay_ms\(" code/src/simulator.py
```

Expected: one device field, one config propagation path, one block decision helper, one shared delay function, and the original actual/planning event keys.

- [ ] **Step 2: Run the dedicated target tests**

```bash
cd code && rtk pytest -q \
  tests/test_communication.py \
  tests/test_probabilistic_network_blocking.py \
  tests/test_config.py
```

Expected: zero failures across probability boundaries, intermediate determinism, device/direction/key independence, wait bounds, validation, trace compatibility, shared routing, and network-free methods.

- [ ] **Step 3: Run affected simulator and resource regressions**

```bash
cd code && rtk pytest -q \
  tests/test_target_only.py \
  tests/test_server_only_linear.py \
  tests/test_server_only_tree.py \
  tests/test_specedge_linear.py \
  tests/test_specedge_tree.py \
  tests/test_specedge_methods.py \
  tests/test_dip_sd.py \
  tests/test_runtime_prediction.py \
  tests/test_target_latency_profile_integration.py \
  tests/test_dynamic_edge_compute.py \
  tests/test_baseline_system_invariants.py
```

Expected: zero failures; edge capacity, target/server latency, scheduler behavior, verification profiles, and all algorithm semantics remain intact.

- [ ] **Step 4: Run the complete pytest suite**

```bash
cd code && rtk pytest -q
```

Expected: zero failures.

- [ ] **Step 5: Run baseline reconstruction verification**

```bash
cd code && rtk bash scripts/verify_baseline_rebuild.sh
```

Expected: the full and method-specific test suites pass, and static checks find no forbidden prefill or obsolete DiP-SD execution paths.

- [ ] **Step 6: Run the six-method baseline trace in an isolated output directory**

```bash
cd code && TRACE_ROOT="$(mktemp -d)/baseline_trace" rtk bash scripts/run_baseline_trace.sh
```

Expected: Target-only, both server-only variants, both SpecEdge variants, and DiP-SD complete successfully; baseline verification reports PASS under default `block_probability=1.0`. The isolated directory prevents validation artifacts from modifying the checked-in baseline trace bundle.

- [ ] **Step 7: Run final whitespace, scope, and forbidden-diff checks**

```bash
rtk git diff --check
rtk git status --short
rtk git diff --name-only
rtk git diff -- \
  code/src/simulator.py \
  code/src/edge_compute.py \
  code/src/latency.py \
  code/src/scheduler.py \
  code/src/verification_latency_profile.py \
  code/src/methods.py
```

Expected: `git diff --check` is clean; only planned files changed; the forbidden subsystem diff is empty. The existing design and plan documents may remain as their own documentation changes, but no generated output is committed.

- [ ] **Step 8: Commit a verification correction only if one was required**

If all prior commits pass unchanged, do not create an empty commit. If final verification exposes a narrow defect, first add a focused failing test in the owning test file, apply the minimum correction, rerun the full final sequence, and commit only that correction:

```bash
rtk git add \
  code/src/entities.py \
  code/src/config.py \
  code/src/communication.py \
  code/configs/default.yaml \
  code/docs/experiment.md \
  code/tests/test_config.py \
  code/tests/test_communication.py \
  code/tests/test_probabilistic_network_blocking.py
rtk git commit -m "fix: preserve probabilistic network contracts"
```

## Non-Goals Audit

The implementation must not change edge-drafter compute or dynamic capacity, server draft latency, target decode or verification latency, target latency profile loading or queries, scheduler policy, lane assignment, batching, timeout behavior, gamma selection, candidate/tree construction, proactive drafting, rollback, acceptance, semantic token generation, request-to-device assignment, or greedy output equivalence. It must not add packet loss, retry, failure, dynamic bandwidth or RTT, correlated blocking epochs, mutable network RNG state, direction-specific probabilities, prompt transmission, prefill, or TTFT.
