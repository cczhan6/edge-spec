# Canonical Baseline Performance Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible, decode-only `dynamic_heterogeneous` evaluation harness for the six canonical baselines, producing a complete 30-row `runs.csv` and per-method sample mean/std `summary.csv` without changing core simulation behavior.

**Architecture:** A scenario override fixes the formal resource configuration. A dedicated evaluation runner materializes one JSONL workload/arrival trace per seed, feeds that file to a thin `Simulator` subclass that only replaces request construction, and runs each seed/method cell in an isolated subprocess while reusing existing model, simulator, metric, and trace-bundle code. A separate validator preallocates all expected rows, validates immutable cross-method inputs and output equivalence, then aggregates only the ten approved performance metrics.

**Tech Stack:** Python 3.11, existing YAML configuration loader, `argparse`, `csv`, `json`, `hashlib`, `random`, `statistics`, `subprocess`, pytest, existing Hugging Face model runner and simulator APIs.

---

## File Structure

- Create `configs/dynamic_heterogeneous.yaml`: scenario-only overrides for the formal evaluation.
- Create `scripts/run_baseline_performance_eval.py`: shared trace schema/materialization, trace-driven simulator adapter, per-cell execution, and failure-tolerant matrix orchestration.
- Create `scripts/summarize_baseline_performance_eval.py`: fixed CSV schemas, 30-row initialization, trace/config validation, output-equivalence checks, and approved mean/std aggregation.
- Create `tests/test_baseline_performance_eval.py`: focused tests for configuration, shared trace consumption, matrix completeness, validation boundaries, failure propagation, and aggregation.

No existing production file is modified. In particular, do not edit `src/simulator.py`, `src/edge_compute.py`, `src/scheduler.py`, `src/methods.py`, `src/latency.py`, or any baseline implementation.

## Fixed Contracts

Use these constants in both scripts; import them from the summary script in the runner rather than duplicating them:

```python
SCENARIO = "dynamic_heterogeneous"
SEEDS = (0, 1, 2, 3, 4)
METHODS = (
    "target_only",
    "server_only_linear",
    "server_only_tree",
    "specedge_linear",
    "specedge_tree",
    "dip_sd",
)
METRIC_SCOPE = "decode_only"
PERFORMANCE_FIELDS = (
    "avg_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "avg_tpot_ms",
    "avg_tbt_ms",
    "makespan_ms",
    "goodput_tok_s",
    "avg_acceptance_rate",
    "avg_selected_gamma",
)
```

`runs.csv` contains exactly these columns:

```python
RUN_FIELDS = (
    "scenario", "seed", "method", "metric_scope",
    "num_requests", "committed_tokens", "success", "failure_reason",
    *PERFORMANCE_FIELDS,
)
```

`summary.csv` contains identity/integrity columns followed only by mean/std
pairs for `PERFORMANCE_FIELDS`:

```python
SUMMARY_FIELDS = (
    "scenario", "method", "metric_scope",
    "num_runs", "successful_runs", "success",
    *(field for metric in PERFORMANCE_FIELDS for field in (f"{metric}_mean", f"{metric}_std")),
)
```

Neither schema contains TTFT. Do not add mean/std columns for `seed`,
`success`, `num_requests`, or `committed_tokens`.

### Task 1: Lock the formal scenario configuration

**Files:**
- Create: `configs/dynamic_heterogeneous.yaml`
- Create: `tests/test_baseline_performance_eval.py`

- [ ] **Step 1: Write the failing configuration test**

```python
from src.config import load_config


def test_dynamic_heterogeneous_configuration_contract() -> None:
    config = load_config("configs/default.yaml", "dynamic_heterogeneous")

    assert config["simulation"]["num_requests"] == 80
    assert config["simulation"]["request_arrival"] == "poisson"
    assert config["dynamic_edge_compute"] == {
        "enabled": True,
        "resample_every_completed_requests": 5,
    }
    assert config["target_latency"] == {
        "mode": "profile",
        "profile_path": "outputs/profiling/target_verification_latency_full_merged.csv",
        "metric": "p50_ms",
    }
    templates = config["device_pools"]["heterogeneous"]["templates"]
    assert {name: values["count"] for name, values in templates.items()} == {
        "low_end": 3,
        "mid_end": 3,
        "high_end": 2,
    }
    assert all(values["block_probability"] == 0.2 for values in templates.values())
    assert all("dynamic_draft_token_rate_range_tok_s" in values for values in templates.values())
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
rtk pytest -q tests/test_baseline_performance_eval.py::test_dynamic_heterogeneous_configuration_contract
```

Expected: FAIL because the scenario file does not exist and the inherited default keeps 480 requests, burst arrivals, disabled dynamic compute, analytical target latency, and blocking probability `1.0`.

- [ ] **Step 3: Add the minimal scenario override**

Create `configs/dynamic_heterogeneous.yaml`:

```yaml
simulation:
  num_requests: 80
  request_arrival: poisson

dynamic_edge_compute:
  enabled: true
  resample_every_completed_requests: 5

target_latency:
  mode: profile
  profile_path: outputs/profiling/target_verification_latency_full_merged.csv
  metric: p50_ms

device_pools:
  heterogeneous:
    templates:
      low_end:
        block_probability: 0.2
      mid_end:
        block_probability: 0.2
      high_end:
        block_probability: 0.2
```

- [ ] **Step 4: Verify GREEN and configuration validation**

Run:

```bash
rtk pytest -q tests/test_baseline_performance_eval.py::test_dynamic_heterogeneous_configuration_contract
rtk python -c 'from src.config import load_config; load_config("configs/default.yaml", "dynamic_heterogeneous")'
```

Expected: both commands PASS without output.

- [ ] **Step 5: Commit the scenario contract**

```bash
rtk git add configs/dynamic_heterogeneous.yaml tests/test_baseline_performance_eval.py
rtk git commit -m "test: lock dynamic baseline evaluation config"
```

### Task 2: Preallocate the complete run matrix and constrain aggregation

**Files:**
- Create: `scripts/summarize_baseline_performance_eval.py`
- Modify: `tests/test_baseline_performance_eval.py`

- [ ] **Step 1: Write failing tests for 30-row initialization and explicit aggregation**

Append:

```python
import csv
import math
from pathlib import Path

from scripts.summarize_baseline_performance_eval import (
    METHODS,
    PERFORMANCE_FIELDS,
    SEEDS,
    aggregate_rows,
    initialize_runs_csv,
)


def test_initialize_runs_csv_contains_every_expected_cell(tmp_path: Path) -> None:
    path = initialize_runs_csv(tmp_path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 30
    assert {(int(row["seed"]), row["method"]) for row in rows} == {
        (seed, method) for seed in SEEDS for method in METHODS
    }
    assert all(row["metric_scope"] == "decode_only" for row in rows)
    assert all(row["success"] == "False" for row in rows)
    assert all(row["failure_reason"] == "not run" for row in rows)


def test_aggregate_rows_uses_sample_stats_only_for_performance_fields() -> None:
    rows = []
    for seed in SEEDS:
        row = {
            "scenario": "dynamic_heterogeneous",
            "seed": seed,
            "method": "target_only",
            "metric_scope": "decode_only",
            "num_requests": 80,
            "committed_tokens": 1000 + seed,
            "success": True,
            "failure_reason": "",
        }
        row.update({field: float(seed + 1) for field in PERFORMANCE_FIELDS})
        rows.append(row)

    summary = aggregate_rows(rows)[0]

    assert summary["num_runs"] == 5
    assert summary["successful_runs"] == 5
    assert summary["success"] is True
    assert summary["avg_latency_ms_mean"] == 3.0
    assert math.isclose(summary["avg_latency_ms_std"], math.sqrt(2.5))
    assert "seed_mean" not in summary
    assert "success_mean" not in summary
    assert "num_requests_mean" not in summary
    assert "committed_tokens_mean" not in summary
```

- [ ] **Step 2: Run the tests and verify RED**

```bash
rtk pytest -q tests/test_baseline_performance_eval.py -k 'initialize_runs_csv or aggregate_rows'
```

Expected: collection ERROR because `scripts.summarize_baseline_performance_eval` does not exist.

- [ ] **Step 3: Implement fixed schemas, initialization, and sample aggregation**

Create `scripts/summarize_baseline_performance_eval.py` with the fixed constants above and these functions:

```python
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any, Iterable

from src.metrics import write_csv

SCENARIO = "dynamic_heterogeneous"
SEEDS = (0, 1, 2, 3, 4)
METHODS = (
    "target_only",
    "server_only_linear",
    "server_only_tree",
    "specedge_linear",
    "specedge_tree",
    "dip_sd",
)
METRIC_SCOPE = "decode_only"
PERFORMANCE_FIELDS = (
    "avg_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "avg_tpot_ms",
    "avg_tbt_ms",
    "makespan_ms",
    "goodput_tok_s",
    "avg_acceptance_rate",
    "avg_selected_gamma",
)
RUN_FIELDS = (
    "scenario", "seed", "method", "metric_scope",
    "num_requests", "committed_tokens", "success", "failure_reason",
    *PERFORMANCE_FIELDS,
)
SUMMARY_FIELDS = (
    "scenario", "method", "metric_scope",
    "num_runs", "successful_runs", "success",
    *(
        field
        for metric in PERFORMANCE_FIELDS
        for field in (f"{metric}_mean", f"{metric}_std")
    ),
)


def initialize_runs_csv(root: str | Path) -> Path:
    output = Path(root) / "runs.csv"
    rows = []
    for seed in SEEDS:
        for method in METHODS:
            row = {field: "" for field in RUN_FIELDS}
            row.update(
                scenario=SCENARIO,
                seed=seed,
                method=method,
                metric_scope=METRIC_SCOPE,
                success=False,
                failure_reason="not run",
            )
            rows.append(row)
    write_csv(output, rows, list(RUN_FIELDS))
    return output


def aggregate_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    materialized = list(rows)
    summaries = []
    for method in METHODS:
        selected = [row for row in materialized if row["method"] == method]
        successful = [row for row in selected if _as_bool(row["success"])]
        complete = len(selected) == len(SEEDS) and len(successful) == len(SEEDS)
        summary = {
            "scenario": SCENARIO,
            "method": method,
            "metric_scope": METRIC_SCOPE,
            "num_runs": len(selected),
            "successful_runs": len(successful),
            "success": complete,
        }
        for metric in PERFORMANCE_FIELDS:
            values = [float(row[metric]) for row in successful] if complete else []
            summary[f"{metric}_mean"] = statistics.mean(values) if values else ""
            summary[f"{metric}_std"] = statistics.stdev(values) if values else ""
        summaries.append(summary)
    return summaries


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"
```

Generate `SUMMARY_FIELDS` with a small helper expression at module import, but materialize it as a tuple and assert in tests that no unapproved `_mean`/`_std` column exists.

- [ ] **Step 4: Verify GREEN**

```bash
rtk pytest -q tests/test_baseline_performance_eval.py -k 'initialize_runs_csv or aggregate_rows'
```

Expected: PASS.

- [ ] **Step 5: Commit matrix/schema behavior**

```bash
rtk git add scripts/summarize_baseline_performance_eval.py tests/test_baseline_performance_eval.py
rtk git commit -m "feat: define baseline evaluation result schemas"
```

### Task 3: Materialize one immutable workload/arrival trace per seed

**Files:**
- Create: `scripts/run_baseline_performance_eval.py`
- Modify: `tests/test_baseline_performance_eval.py`

- [ ] **Step 1: Write failing tests for deterministic materialization and immutable reuse**

Append:

```python
from dataclasses import replace

import pytest

from scripts.run_baseline_performance_eval import (
    load_shared_trace,
    materialize_shared_trace,
)
from src.workload import WorkloadItem


def _workload(count: int = 4) -> list[WorkloadItem]:
    rows = []
    for index in range(count):
        prompt = f"prompt text {index}"
        rows.append(
            WorkloadItem(
                prompt_id=f"prompt-{index}",
                prompt=prompt,
                prompt_token_count=len(prompt.encode("utf-8")),
                category="qa",
                category_group="QA",
            )
        )
    return rows


def test_materialized_trace_is_reused_byte_for_byte(tmp_path: Path) -> None:
    config = load_config("configs/default.yaml")
    config["simulation"].update(
        seed=3,
        num_requests=4,
        num_devices=4,
        output_len_choices=[8, 16],
        request_arrival="poisson",
        poisson_rate_per_s=20,
    )
    path = tmp_path / "seed_3.jsonl"

    first_hash = materialize_shared_trace(config, _workload(), path)
    first_bytes = path.read_bytes()
    second_hash = materialize_shared_trace(config, _workload(), path)
    rows = load_shared_trace(path, config)

    assert second_hash == first_hash
    assert path.read_bytes() == first_bytes
    assert [row.request_id for row in rows] == [0, 1, 2, 3]
    assert [row.device_id for row in rows] == [0, 1, 2, 3]
    assert rows[0].arrival_time_ms == 0.0
    assert [row.arrival_time_ms for row in rows] == sorted(
        row.arrival_time_ms for row in rows
    )


def test_existing_shared_trace_rejects_different_content(tmp_path: Path) -> None:
    config = load_config("configs/default.yaml")
    config["simulation"].update(seed=0, num_requests=4, num_devices=4)
    path = tmp_path / "seed_0.jsonl"
    materialize_shared_trace(config, _workload(), path)

    changed = [replace(item, prompt="changed") if index == 0 else item for index, item in enumerate(_workload())]
    with pytest.raises(ValueError, match="existing shared trace differs"):
        materialize_shared_trace(config, changed, path)
```

- [ ] **Step 2: Run the tests and verify RED**

```bash
rtk pytest -q tests/test_baseline_performance_eval.py -k 'materialized_trace or existing_shared_trace'
```

Expected: collection ERROR because the run script does not exist.

- [ ] **Step 3: Implement the trace record and materializer**

Create the run script with this public surface:

```python
from __future__ import annotations

import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from src.workload import WorkloadItem


@dataclass(frozen=True)
class SharedRequest:
    request_id: int
    device_id: int
    prompt_id: str
    prompt: str
    prompt_token_count: int
    category: str
    category_group: str
    output_len: int
    arrival_time_ms: float
    decode_ready_time_ms: float

    def workload_item(self) -> WorkloadItem:
        return WorkloadItem(
            prompt_id=self.prompt_id,
            prompt=self.prompt,
            prompt_token_count=self.prompt_token_count,
            category=self.category,
            category_group=self.category_group,
        )


def materialize_shared_trace(
    config: dict,
    workload: Sequence[WorkloadItem],
    path: str | Path,
) -> str:
    simulation = config["simulation"]
    if len(workload) != int(simulation["num_requests"]):
        raise ValueError("workload size does not match simulation.num_requests")
    rng = random.Random(int(simulation["seed"]))
    current_ms = 0.0
    rows = []
    for request_id, item in enumerate(workload):
        if request_id and simulation["request_arrival"] == "poisson":
            current_ms += rng.expovariate(float(simulation["poisson_rate_per_s"])) * 1000.0
        rows.append(
            SharedRequest(
                request_id=request_id,
                device_id=request_id % int(simulation["num_devices"]),
                prompt_id=item.prompt_id,
                prompt=item.prompt,
                prompt_token_count=item.prompt_token_count,
                category=item.category,
                category_group=item.category_group,
                output_len=int(rng.choice(simulation["output_len_choices"])),
                arrival_time_ms=current_ms,
                decode_ready_time_ms=current_ms,
            )
        )
    payload = b"".join(
        (json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in rows
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    try:
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() != payload:
                raise ValueError(f"existing shared trace differs: {destination}")
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def load_shared_trace(path: str | Path, config: dict) -> list[SharedRequest]:
    with Path(path).open(encoding="utf-8") as handle:
        rows = [SharedRequest(**json.loads(line)) for line in handle if line.strip()]
    expected = int(config["simulation"]["num_requests"])
    if len(rows) != expected or [row.request_id for row in rows] != list(range(expected)):
        raise ValueError("shared trace request IDs are incomplete or unordered")
    devices = int(config["simulation"]["num_devices"])
    if any(row.device_id != row.request_id % devices for row in rows):
        raise ValueError("shared trace device mapping is invalid")
    return rows
```

The temporary-file plus hard-link sequence makes first creation atomic without
overwriting an existing trace. Existing identical bytes are reusable, but
different bytes are an error.

- [ ] **Step 4: Verify GREEN**

```bash
rtk pytest -q tests/test_baseline_performance_eval.py -k 'materialized_trace or existing_shared_trace'
```

Expected: PASS.

- [ ] **Step 5: Commit shared trace materialization**

```bash
rtk git add scripts/run_baseline_performance_eval.py tests/test_baseline_performance_eval.py
rtk git commit -m "feat: materialize shared baseline workloads"
```

### Task 4: Make every method consume the shared trace without core changes

**Files:**
- Modify: `scripts/run_baseline_performance_eval.py`
- Modify: `tests/test_baseline_performance_eval.py`

- [ ] **Step 1: Write the failing trace-consumption test**

Append:

```python
from types import SimpleNamespace

from scripts.run_baseline_performance_eval import SharedTraceSimulator
from tests.common import small_config


def test_all_methods_consume_shared_arrivals_without_resampling(tmp_path: Path) -> None:
    config, runner, _ = small_config(num_requests=4, output_len=8)
    config["simulation"].update(
        seed=4,
        num_devices=4,
        request_arrival="poisson",
        poisson_rate_per_s=20,
    )
    path = tmp_path / "shared.jsonl"
    materialize_shared_trace(config, _workload(), path)
    shared = load_shared_trace(path, config)

    observed = []
    for method in ("target_only", "server_only_linear", "specedge_linear", "dip_sd"):
        simulator = SharedTraceSimulator(config, runner, shared, "dynamic_heterogeneous", method)
        simulator._rng = SimpleNamespace(
            expovariate=lambda *_: (_ for _ in ()).throw(AssertionError("resampled arrival")),
            choice=lambda *_: (_ for _ in ()).throw(AssertionError("resampled output length")),
        )
        simulator._schedule_request_arrivals()
        observed.append(
            [
                (request.request_id, request.arrival_time_ms, request.output_len, request.device_id)
                for request in simulator.requests
            ]
        )

    assert all(rows == observed[0] for rows in observed[1:])
```

- [ ] **Step 2: Run the test and verify RED**

```bash
rtk pytest -q tests/test_baseline_performance_eval.py::test_all_methods_consume_shared_arrivals_without_resampling
```

Expected: FAIL because `SharedTraceSimulator` is not defined.

- [ ] **Step 3: Implement the thin simulator adapter**

Add imports for `Request`, `EventType`, `Simulator`, and model-runner typing, then implement:

```python
class SharedTraceSimulator(Simulator):
    def __init__(self, config, model_runner, shared_requests, scenario, method, **kwargs):
        self._shared_requests = list(shared_requests)
        super().__init__(
            config,
            model_runner,
            [row.workload_item() for row in self._shared_requests],
            scenario,
            method,
            **kwargs,
        )

    def _schedule_request_arrivals(self) -> None:
        if self.requests:
            raise RuntimeError("shared requests were already scheduled")
        for row in self._shared_requests:
            prompt_ids = self.model_runner.encode_prompt(row.prompt)
            if len(prompt_ids) != row.prompt_token_count:
                raise ValueError(
                    f"shared prompt token count differs for request {row.request_id}"
                )
            request = Request(
                request_id=row.request_id,
                device_id=row.device_id,
                output_len=row.output_len,
                arrival_time_ms=row.arrival_time_ms,
                decode_ready_time_ms=row.decode_ready_time_ms,
                prompt_id=row.prompt_id,
                category=row.category,
                category_group=row.category_group,
                prompt=row.prompt,
                prompt_token_count=len(prompt_ids),
                prompt_ids=prompt_ids,
            )
            self.requests.append(request)
            self.device_runtimes[row.device_id].assigned_requests += 1
            self._schedule(row.arrival_time_ms, EventType.REQUEST_ARRIVE, row.request_id)
```

This override must not create draft, network, or edge events. It only performs the request construction that the base `_schedule_request_arrivals` performs, using values read from the shared file.

- [ ] **Step 4: Verify GREEN and relevant simulator regressions**

```bash
rtk pytest -q tests/test_baseline_performance_eval.py::test_all_methods_consume_shared_arrivals_without_resampling
rtk pytest -q tests/test_decode_only_initialization.py tests/test_determinism.py tests/test_workload.py
```

Expected: PASS.

- [ ] **Step 5: Commit trace-driven request construction**

```bash
rtk git add scripts/run_baseline_performance_eval.py tests/test_baseline_performance_eval.py
rtk git commit -m "feat: consume shared baseline arrival traces"
```

### Task 5: Run isolated cells, continue failures, and record immutable resource inputs

**Files:**
- Modify: `scripts/run_baseline_performance_eval.py`
- Modify: `tests/test_baseline_performance_eval.py`

- [ ] **Step 1: Write failing tests for continuation and resource fingerprints**

Append tests around dependency-injected subprocess execution:

```python
from scripts.run_baseline_performance_eval import (
    build_resource_fingerprint,
    run_matrix_cells,
)


def test_run_matrix_cells_continues_after_process_failure(tmp_path: Path) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append(tuple(command))
        return SimpleNamespace(returncode=7 if len(calls) == 2 else 0)

    initialize_runs_csv(tmp_path)
    statuses = run_matrix_cells(
        root=tmp_path,
        seeds=SEEDS,
        methods=METHODS,
        command_for=lambda seed, method: ["worker", str(seed), method],
        run_process=fake_run,
    )

    assert len(calls) == 30
    assert len(statuses) == 30
    assert statuses[1]["success"] is False
    assert "return code 7" in statuses[1]["failure_reason"]


def test_resource_fingerprint_ignores_runtime_event_order() -> None:
    config = load_config("configs/default.yaml", "dynamic_heterogeneous")
    config["simulation"]["seed"] = 2

    first = build_resource_fingerprint(config)
    second = build_resource_fingerprint(config)

    assert first == second
    assert len(first["epoch0_rates"]) == 8
    assert first["dynamic_edge_compute"] == {
        "enabled": True,
        "resample_every_completed_requests": 5,
    }
    assert first["block_probabilities"] == {
        "low_end": 0.2,
        "mid_end": 0.2,
        "high_end": 0.2,
    }
    assert "transition_times" not in first
    assert "network_events" not in first
```

- [ ] **Step 2: Run the tests and verify RED**

```bash
rtk pytest -q tests/test_baseline_performance_eval.py -k 'run_matrix_cells or resource_fingerprint'
```

Expected: FAIL because the orchestration and fingerprint functions do not exist.

- [ ] **Step 3: Implement resource fingerprints and one-cell execution**

`build_resource_fingerprint(config)` must:

1. call `build_devices(config, "heterogeneous")`;
2. construct `EdgeComputeModel(config, devices, "heterogeneous")`;
3. record device IDs/types, static device fields, all epoch-zero rates, dynamic ranges, seed, dynamic settings, block probabilities, and target profile configuration;
4. compute deterministic rates for epochs `0`, `1`, and `2` with `deterministic_draft_rate` for every device and include their SHA-256 digest as `epoch_rate_mapping_sha256`; and
5. contain no observed transition timestamps or network event data.

Implement `run_cell(config_path, trace_path, method, output_dir)` using existing components:

```python
config = load_config(config_path)
shared = load_shared_trace(trace_path, config)
runner = build_model_runner(config, use_fake_model_runner=False)
simulator = SharedTraceSimulator(config, runner, shared, SCENARIO, method)
result = simulator.run()
main, system = summarize(result, int(config["simulation"]["num_devices"]))
write_trace_bundle(output_dir, config, result, main, system)
```

On success atomically write `run_status.json` containing `success=true`, empty
`failure_reason`, the shared trace path and SHA-256, and the resource
fingerprint. The worker CLI must return nonzero on exceptions; the parent writes
or replaces the status with `success=false`, the subprocess return code, and a
concise failure reason.

- [ ] **Step 4: Implement failure-tolerant matrix orchestration**

Add `matrix` and `cell` argparse subcommands. The matrix command must:

1. require `--execute-formal-matrix` before any model loading or subprocess;
2. call `initialize_runs_csv(root)` before launching the first cell;
3. resolve and audit one config per seed;
4. call `load_workload` exactly once per seed and then
   `materialize_shared_trace` exactly once for that seed;
5. launch each of the 30 cells as a subprocess, redirecting output to
   `<root>/<scenario>/<seed>/<method>/stdout.log`;
6. continue after every nonzero return code; and
7. invoke `summarize_results(root)` only after all cells have been attempted,
   finally returning nonzero if any cell or validation failed.

The `cell` command accepts one seed, one canonical method, one resolved config,
one shared trace, and one output directory. Reject methods outside `METHODS`.
Do not expose fake-runner flags.

- [ ] **Step 5: Verify GREEN without launching the formal matrix**

```bash
rtk pytest -q tests/test_baseline_performance_eval.py -k 'run_matrix_cells or resource_fingerprint'
rtk python -m scripts.run_baseline_performance_eval matrix --help
rtk python -m scripts.run_baseline_performance_eval cell --help
```

Expected: tests PASS and both help commands exit zero. Do not pass
`--execute-formal-matrix`.

- [ ] **Step 6: Commit isolated orchestration**

```bash
rtk git add scripts/run_baseline_performance_eval.py tests/test_baseline_performance_eval.py
rtk git commit -m "feat: orchestrate isolated baseline evaluation cells"
```

### Task 6: Validate all 30 cells and committed-output equivalence

**Files:**
- Modify: `scripts/summarize_baseline_performance_eval.py`
- Modify: `tests/test_baseline_performance_eval.py`

- [ ] **Step 1: Write failing tests for missing runs, input mapping, and output mismatch**

Add a test helper that writes minimal CSV/JSON bundles under the formal layout.
The helper must always write the shared input JSONL and may leave output
`device_id` empty for target/server methods. Implement it with the production
schemas so malformed fixtures cannot pass accidentally:

```python
import hashlib
import json

from scripts.baseline_trace import (
    BATCH_TRACE_FIELDS,
    EVENT_TRACE_FIELDS,
    REQUEST_TRACE_FIELDS,
    RESOURCE_TIMELINE_FIELDS,
    TOKEN_TRACE_FIELDS,
)
from scripts.run_baseline_performance_eval import build_resource_fingerprint
from src.metrics import MAIN_FIELDS, SYSTEM_FIELDS, write_csv


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_complete_synthetic_matrix(
    root: Path,
    *,
    blank_device_ids_for: set[str] | None = None,
) -> None:
    blank = blank_device_ids_for or set()
    initialize_runs_csv(root)
    for seed in SEEDS:
        config = load_config("configs/default.yaml", SCENARIO)
        config["simulation"]["seed"] = seed
        workload = [
            WorkloadItem(
                prompt_id=f"prompt-{index}",
                prompt=f"prompt text {index}",
                prompt_token_count=3,
                category="qa",
                category_group="QA",
            )
            for index in range(80)
        ]
        trace_path = root / "_workloads" / f"{SCENARIO}_seed_{seed}.jsonl"
        trace_hash = materialize_shared_trace(config, workload, trace_path)
        shared = load_shared_trace(trace_path, config)
        fingerprint = build_resource_fingerprint(config)
        for method in METHODS:
            directory = root / SCENARIO / str(seed) / method
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "run_status.json").write_text(
                json.dumps(
                    {
                        "scenario": SCENARIO,
                        "seed": seed,
                        "method": method,
                        "success": True,
                        "return_code": 0,
                        "failure_reason": "",
                        "shared_trace_path": str(trace_path),
                        "shared_trace_sha256": trace_hash,
                        "resource_fingerprint": fingerprint,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            (directory / "resolved_config.json").write_text(
                json.dumps({"method": method, "scenario": SCENARIO, "config": config}),
                encoding="utf-8",
            )
            metric = {field: 1.0 for field in MAIN_FIELDS}
            metric.update(method=method, scenario=SCENARIO, num_requests=80)
            write_csv(directory / "metrics.csv", [metric], MAIN_FIELDS)
            requests = []
            tokens = []
            for row in shared:
                requests.append(
                    {
                        "method": method,
                        "scenario": SCENARIO,
                        "request_id": row.request_id,
                        "device_id": "" if method in blank else row.device_id,
                        "prompt_id": row.prompt_id,
                        "arrival_time_ms": row.arrival_time_ms,
                        "decode_ready_time_ms": row.decode_ready_time_ms,
                        "finish_time_ms": row.arrival_time_ms + 1.0,
                        "latency_ms": 1.0,
                        "output_len": row.output_len,
                        "generated_tokens": row.output_len,
                        "generated_ids": list(range(row.output_len)),
                        "status": "finished",
                        "in_flight_segment_count": 0,
                        "pending_segment_count": 0,
                        "completed_result_count": 0,
                        "draft_queued": False,
                        "proactive_pending_count": 0,
                    }
                )
                tokens.extend(
                    {
                        "method": method,
                        "scenario": SCENARIO,
                        "token_type": "committed",
                        "request_id": row.request_id,
                        "device_id": row.device_id,
                        "position": position,
                        "token_index": position,
                        "token_id": position,
                        "commit_time_ms": row.arrival_time_ms + position + 1.0,
                        "count": 1,
                        "status": "finished",
                        "source_event": "final_output",
                    }
                    for position in range(row.output_len)
                )
            write_csv(directory / "request_trace.csv", requests, REQUEST_TRACE_FIELDS)
            write_csv(directory / "event_trace.csv", [], EVENT_TRACE_FIELDS)
            write_csv(directory / "token_trace.csv", tokens, TOKEN_TRACE_FIELDS)
            write_csv(directory / "resource_timeline.csv", [], RESOURCE_TIMELINE_FIELDS)
            write_csv(directory / "batch_trace.csv", [], BATCH_TRACE_FIELDS)
            system = {field: 0.0 for field in SYSTEM_FIELDS}
            system.update(method=method, scenario=SCENARIO)
            write_csv(directory / "system_metrics.csv", [system], SYSTEM_FIELDS)
            (directory / "stdout.log").write_text("synthetic success\n", encoding="utf-8")


def _replace_one_committed_token(path: Path) -> None:
    rows = _read_csv(path)
    rows[0]["token_id"] = str(int(rows[0]["token_id"]) + 1000)
    write_csv(path, rows, TOKEN_TRACE_FIELDS)
```

Then add:

```python
from scripts.summarize_baseline_performance_eval import summarize_results


def test_missing_trace_retains_complete_matrix_and_fails(tmp_path: Path) -> None:
    initialize_runs_csv(tmp_path)
    failures = summarize_results(tmp_path)
    with (tmp_path / "runs.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 30
    assert all(row["success"] == "False" for row in rows)
    assert any("missing" in row["failure_reason"] for row in rows)
    assert failures


def test_missing_output_device_id_does_not_fail_input_mapping(tmp_path: Path) -> None:
    _write_complete_synthetic_matrix(tmp_path, blank_device_ids_for={
        "target_only", "server_only_linear", "server_only_tree"
    })

    failures = summarize_results(tmp_path)

    assert failures == []


def test_committed_token_mismatch_marks_cell_failed(tmp_path: Path) -> None:
    _write_complete_synthetic_matrix(tmp_path)
    _replace_one_committed_token(
        tmp_path / SCENARIO / "0" / "specedge_linear" / "token_trace.csv"
    )

    failures = summarize_results(tmp_path)
    rows = _read_csv(tmp_path / "runs.csv")
    failed = next(
        row for row in rows
        if row["seed"] == "0" and row["method"] == "specedge_linear"
    )

    assert failed["success"] == "False"
    assert "committed token trace differs from target_only" in failed["failure_reason"]
    assert failures
```

The synthetic matrix may give methods different `edge_compute_transition` and
network event rows. The successful test proves these events are intentionally
not compared across methods.

- [ ] **Step 2: Run the tests and verify RED**

```bash
rtk pytest -q tests/test_baseline_performance_eval.py -k 'missing_trace or output_device_id or committed_token_mismatch'
```

Expected: FAIL because `summarize_results` and the validation pipeline are not implemented.

- [ ] **Step 3: Implement per-cell validation**

Implement loaders for `run_status.json`, `resolved_config.json`, `metrics.csv`,
`request_trace.csv`, `token_trace.csv`, and the shared JSONL. For each expected
cell, accumulate failures instead of raising immediately. Validate:

- all required files exist and contain exactly one metrics/config record where applicable;
- status return code/success is valid;
- resolved seed, method, scenario, 80 requests, dynamic settings, all three
  blocking probabilities, and target profile path/metric match the contract;
- status shared-trace SHA-256 equals the actual seed file and is the same for
  all six methods;
- resource fingerprints are identical across methods for the seed;
- request ID, prompt ID, output length, arrival time, and decode-ready time
  exactly match the shared trace;
- requests are finished, generated token count equals output length, and no
  pending/draft/proactive state remains;
- required metrics are finite and non-negative; and
- committed token rows reconstruct the same per-request token-ID sequence as
  `target_only` for the seed.

Do not compare output `device_id` for any method. The authoritative assignment
is `SharedRequest.device_id` in the input trace. Do not compare
`edge_compute_transition` event times or network-blocking event rows.

Reuse `_check_no_pending_state`, `_check_event_monotonicity`, and
`_check_resource_overlap` from `scripts.baseline_trace`; do not reuse its
hard-coded four-request `_check_all_requests_finished` helper.

- [ ] **Step 4: Materialize rows and summary before returning failure**

`summarize_results(root)` must always:

1. start from all 30 expected identities, regardless of directory contents;
2. populate available integrity/performance fields;
3. set `success=false` and join deterministic failure messages with `; `;
4. write `runs.csv` in seed-major, method-minor order;
5. call `aggregate_rows` and write `summary.csv` in canonical method order; and
6. return the list of failures without raising.

CLI `main()` calls this function, prints failures, and exits `1` only after both
files have been written. A clean matrix exits `0`.

- [ ] **Step 5: Verify GREEN**

```bash
rtk pytest -q tests/test_baseline_performance_eval.py -k 'missing_trace or output_device_id or committed_token_mismatch'
rtk pytest -q tests/test_baseline_performance_eval.py
```

Expected: PASS.

- [ ] **Step 6: Commit validation and aggregation**

```bash
rtk git add scripts/summarize_baseline_performance_eval.py tests/test_baseline_performance_eval.py
rtk git commit -m "feat: validate and summarize baseline evaluation runs"
```

### Task 7: Complete lightweight verification without running the formal matrix

**Files:**
- Verify only; modify the new test/scripts/config files only if a failing test exposes a defect.

- [ ] **Step 1: Record the implementation boundary and inspect changed files**

```bash
rtk git status --short
rtk git log --oneline --decorate -8
rtk git diff --name-only 0242159..HEAD
```

Expected: implementation changes are limited to:

```text
code/configs/dynamic_heterogeneous.yaml
code/scripts/run_baseline_performance_eval.py
code/scripts/summarize_baseline_performance_eval.py
code/tests/test_baseline_performance_eval.py
```

- [ ] **Step 2: Run focused experiment-tool regressions**

```bash
rtk pytest -q \
  tests/test_baseline_performance_eval.py \
  tests/test_experiment_tools.py \
  tests/test_baseline_trace_runner.py \
  tests/test_config.py \
  tests/test_dynamic_edge_compute.py \
  tests/test_probabilistic_network_blocking.py \
  tests/test_target_latency_profile_integration.py
```

Expected: PASS.

- [ ] **Step 3: Run the repository-required full test suite**

```bash
rtk pytest -q
```

Expected: PASS.

- [ ] **Step 4: Verify scope, formatting, and forbidden output behavior**

```bash
rtk git diff --check 0242159..HEAD
rtk git diff --exit-code 0242159..HEAD -- \
  src/simulator.py src/edge_compute.py src/scheduler.py src/methods.py src/latency.py src/dip_sd.py
rtk rg -n "ttft|TTFT|first.token" \
  scripts/run_baseline_performance_eval.py \
  scripts/summarize_baseline_performance_eval.py \
  configs/dynamic_heterogeneous.yaml
rtk git status --short --branch
```

Expected: diff checks pass; the core-file diff is empty; the TTFT search has no
matches; the branch is clean after task commits.

- [ ] **Step 5: Verify CLI discovery only**

```bash
rtk python -m scripts.run_baseline_performance_eval matrix --help
rtk python -m scripts.run_baseline_performance_eval cell --help
rtk python -m scripts.summarize_baseline_performance_eval --help
```

Expected: all commands exit zero and describe their arguments. Do not invoke
the matrix command with `--execute-formal-matrix`; the formal 30-run execution
remains separately gated by user approval.

- [ ] **Step 6: Commit only if verification required a correction**

If all prior commits pass unchanged, do not create an empty commit. If a defect
is found, first add or tighten a failing test, observe RED, apply the minimum
fix, rerun Steps 2–5, then commit only the correction:

```bash
rtk git add \
  configs/dynamic_heterogeneous.yaml \
  scripts/run_baseline_performance_eval.py \
  scripts/summarize_baseline_performance_eval.py \
  tests/test_baseline_performance_eval.py
rtk git commit -m "fix: preserve baseline evaluation contracts"
```

## Formal Execution Gate

This implementation plan ends after code and lightweight verification. It does
not authorize the following formal execution:

```bash
python -m scripts.run_baseline_performance_eval matrix --execute-formal-matrix
```

Run that command only after separate explicit user approval. It must write
under `outputs/baseline_performance_eval/` and must never modify or overwrite
`outputs/baseline_trace/`.
