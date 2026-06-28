# Decode-Only Baseline Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a real-runner, decode-only 24-cell baseline preflight with exact output/invariant verification and drafter residency evidence.

**Architecture:** Add observation-only committed-token timestamps to the shared simulator result, then materialize a narrow formal metric schema from trace bundles. A dedicated preflight verifier prepares reduced configs, validates each scenario/seed cell, and generates summaries; a shell driver invokes the existing formal CLI once per scenario/seed for all six methods. A standalone CUDA residency script records individual and simultaneous drafter memory behavior.

**Tech Stack:** Python 3.11, pytest/unittest, PyYAML, PyTorch CUDA, Hugging Face Transformers, Bash, existing simulator/CLI/trace utilities.

---

### Task 1: Observation-only committed-token timestamps

**Files:**
- Modify: `src/entities.py`
- Modify: `src/simulator.py`
- Modify: `scripts/baseline_trace.py`
- Test: `tests/test_baseline_preflight.py`

- [ ] **Step 1: Write failing timestamp tests**

Add tests that run `target_only` and a speculative method with the existing deterministic model runner, then assert:

```python
assert len(request.committed_token_times_ms) == len(request.generated_ids)
assert request.committed_token_times_ms == sorted(request.committed_token_times_ms)
assert request.committed_token_times_ms[-1] == request.finish_time_ms
assert all(row["commit_time_ms"] != "" for row in committed_trace_rows)
```

- [ ] **Step 2: Verify RED**

Run: `rtk pytest -q tests/test_baseline_preflight.py -k committed_token_times`

Expected: fail because `Request.committed_token_times_ms` and the trace column do not exist.

- [ ] **Step 3: Implement minimal shared observation**

Add `committed_token_times_ms: list[float]` to `Request`. At the three existing commit sites, append one timestamp per emitted token. Target-only timestamps are evenly spaced over its configured autoregressive compute interval and end at service finish; DiP-SD and shared speculative commit sites use the result-visible time for every token in that commit. Add `commit_time_ms` to committed rows in `token_trace.csv`; non-committed aggregate rows leave it empty.

- [ ] **Step 4: Verify GREEN and regression scope**

Run: `rtk pytest -q tests/test_baseline_preflight.py -k committed_token_times tests/test_target_only.py tests/test_dip_sd.py tests/test_baseline_trace_runner.py`

Expected: all selected tests pass without output-token changes.

### Task 2: Formal preflight metrics and invariant verifier

**Files:**
- Create: `scripts/verify_baseline_preflight.py`
- Test: `tests/test_baseline_preflight.py`

- [ ] **Step 1: Write failing metric/config tests**

Define fixture trace bundles for one scenario and seed. Tests require the exact metric fields:

```python
PREFLIGHT_METRIC_FIELDS = [
    "decode_makespan", "request_decode_latency",
    "mean_inter_token_latency", "p50_inter_token_latency",
    "p95_inter_token_latency", "effective_throughput_tokens_per_s",
    "speedup_vs_target_only", "acceptance_ratio", "drafted_tokens",
    "verified_tokens", "accepted_tokens", "committed_tokens",
    "wasted_tokens", "target_utilization", "draft_utilization",
    "verification_queue_wait",
]
```

Assert 16 requests, `[32]` output lengths, Poisson mode, fixed seed, ms units, real runner, same-cell target speedup, finite nonnegative metrics, and no field whose lowercase name contains `ttft` or `first_token`.

- [ ] **Step 2: Verify RED**

Run: `rtk pytest -q tests/test_baseline_preflight.py -k 'metric or config or invariant'`

Expected: import failure for the missing preflight verifier.

- [ ] **Step 3: Implement preparation and materialization APIs**

Implement focused functions:

```python
def prepare_preflight_config(config_path, scenario, seed, output_path) -> Path: ...
def materialize_run(directory, *, scenario, seed, method, environment_path, command) -> dict: ...
def verify_preflight(root: Path) -> list[dict]: ...
```

`prepare_preflight_config` changes only `num_requests=16`, `output_len_choices=[32]`, `seed`, and `request_arrival="poisson"` after scenario merge. `materialize_run` writes `resolved_config.yaml`, `request_metrics.csv`, the narrow `metrics.csv`, and a real-runner manifest. `verify_preflight` groups by `(scenario, seed)`, uses only that group's `target_only`, and checks all 16 required semantic conditions.

- [ ] **Step 4: Verify GREEN**

Run: `rtk pytest -q tests/test_baseline_preflight.py -k 'metric or config or invariant'`

Expected: all selected tests pass.

### Task 3: Drafter residency manifest

**Files:**
- Create: `scripts/check_drafter_residency.py`
- Test: `tests/test_drafter_residency.py`

- [ ] **Step 1: Write failing residency tests**

Use injected fake CUDA/model/tokenizer adapters to assert individual records, simultaneous peak/free memory, OOM capture, dtype recording, vocabulary compatibility, atomic JSON output, and the policy transition:

```python
assert manifest["residency_policy"] == (
    "all_configured_models_simultaneous"
    if manifest["simultaneous"]["success"]
    else "sequential_lazy_model_loading"
)
assert manifest["model_loading_in_decode_latency"] is False
```

- [ ] **Step 2: Verify RED**

Run: `rtk pytest -q tests/test_drafter_residency.py`

Expected: import failure for the missing script.

- [ ] **Step 3: Implement CUDA/Hugging Face checker**

Read the three configured profile references (allow explicit per-profile environment overrides), require CUDA, load each model/tokenizer individually on `cuda:0`, release and empty cache, then load all three together. Reset/read peak CUDA statistics around each phase, compare tokenizer vocabularies and special-token IDs exactly, catch only CUDA OOM as a measured result, and fail explicitly for unavailable model paths or unrelated exceptions. Write `outputs/drafter_residency_manifest.json` with `allow_nan=False`.

- [ ] **Step 4: Verify GREEN**

Run: `rtk pytest -q tests/test_drafter_residency.py`

Expected: all residency unit tests pass without allocating real models.

### Task 4: Multi-method formal CLI trace output and shell orchestration

**Files:**
- Modify: `scripts/run_all.py`
- Modify: `scripts/baseline_trace.py`
- Create: `scripts/run_baseline_preflight.sh`
- Test: `tests/test_baseline_preflight.py`

- [ ] **Step 1: Write failing CLI/layout tests**

Test that a multi-method `--trace-bundle-root` writes one trace bundle per canonical method, while the existing single-method `--trace-bundle-dir` contract remains unchanged. Inspect the shell script and a fake-command execution harness to require four scenario/seed CLI invocations, all six methods, no fake-runner flag, and the required hierarchy/files.

- [ ] **Step 2: Verify RED**

Run: `rtk pytest -q tests/test_baseline_preflight.py -k 'cli or layout or shell'`

Expected: parser rejects `--trace-bundle-root` and the shell script is missing.

- [ ] **Step 3: Implement trace root and driver**

Add an additive `--trace-bundle-root` option whose per-method destination is `<root>/<method>`, and write auxiliary `system_metrics.csv` so formal utilization metrics use the simulator's existing definitions. The shell script uses fixed seeds `20260628` and `20260629`, audits every reduced config, invokes `python -m scripts.run_all` through the configured real Python environment, copies the shared group log/environment manifest into every method directory, materializes each run, and finally invokes the verifier. It never passes a fake-runner option and never runs the 480-request config.

- [ ] **Step 4: Verify GREEN**

Run: `rtk pytest -q tests/test_baseline_preflight.py`

Expected: all preflight tests pass.

### Task 5: Documentation and live preflight

**Files:**
- Modify: `docs/experiment_resource_contract.md`
- Modify: `docs/baseline_status.md`
- Modify: `docs/experiment.md`
- Generated: `outputs/drafter_residency_manifest.json`
- Generated: `outputs/baseline_preflight/**`

- [ ] **Step 1: Run drafter residency check**

Run: `rtk /root/miniforge3/envs/edge-spec/bin/python scripts/check_drafter_residency.py`

Expected: JSON manifest records all three individual loads and the simultaneous result; no fake/model substitution occurs.

- [ ] **Step 2: Run the 24-cell preflight**

Run: `rtk env PYTHON_BIN=/root/miniforge3/envs/edge-spec/bin/python bash scripts/run_baseline_preflight.sh`

Expected: 2 scenarios × 2 seeds × 6 methods finish and the verifier writes PASS summaries. Any resource/model failure remains explicit and stops the run.

- [ ] **Step 3: Update documentation from measured evidence**

Document the decode clock boundary, exact metric formulas, residency result/policy, preflight command/layout, non-paper-result warning, and freeze-readiness status. Remove the obsolete statement that token-level inter-arrival instrumentation is absent. Do not add TTFT or first-token metrics.

- [ ] **Step 4: Run full verification**

Run, in order:

```bash
rtk /root/miniforge3/envs/edge-spec/bin/python scripts/check_drafter_residency.py
rtk env PYTHON_BIN=/root/miniforge3/envs/edge-spec/bin/python bash scripts/run_baseline_preflight.sh
rtk /root/miniforge3/envs/edge-spec/bin/python scripts/verify_baseline_preflight.py
rtk pytest -q
rtk bash scripts/verify_baseline_rebuild.sh
rtk bash scripts/run_baseline_trace.sh
rtk git diff --check
```

Expected: every command exits zero; summaries contain no TTFT column.

- [ ] **Step 5: Commit the completed milestone**

Run:

```bash
rtk git add src scripts tests docs outputs/drafter_residency_manifest.json outputs/baseline_preflight
rtk git commit -m "test: add decode-only baseline experiment preflight"
```

Expected: a standalone implementation commit with the requested message and a clean worktree.
