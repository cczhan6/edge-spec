# Target Verification Latency Profiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone CUDA profiler for KV-cached target decode, linear verification, and explicitly approximate fixed-forward tree verification, then run only the approved smoke matrix.

**Architecture:** `scripts/profile_target_verification_latency.py` contains a pure matrix/CSV orchestration layer and a CUDA/Hugging Face backend. The backend instantiates `HuggingFaceModelRunner` with an empty drafter map, builds deterministic prefix KV outside event timing, isolates every sample from prior returned cache state, and records CUDA device elapsed time. Tree node rows share one physical measurement per batch/context pair.

**Tech Stack:** Python 3, PyTorch CUDA events, Hugging Face Transformers, pytest, CSV.

---

## File Structure

- Create `scripts/profile_target_verification_latency.py`: CLI, workload expansion, target-only model loading, cache preparation/isolation, CUDA timing, OOM handling, statistics, and CSV output.
- Create `tests/test_target_verification_profiler.py`: pure orchestration tests plus fake CUDA/model/cache tests; no real model required.
- Modify `docs/superpowers/specs/2026-06-29-target-verification-latency-profiling-design.md`: capture approved cache-position, synchronization, tree sample reuse, OOM fan-out, and CSV-field requirements.
- Create `outputs/profiling/target_verification_latency_smoke.csv` only when the approved smoke command runs; this generated output is evidence, not source code.

### Task 1: Matrix, statistics, and CSV contract

**Files:**
- Create: `tests/test_target_verification_profiler.py`
- Create: `scripts/profile_target_verification_latency.py`

- [x] **Step 1: Write failing tests for matrix expansion and row metadata**

Define tests that import the new script, assert the default constants, and verify that smoke expansion yields two decode specs, four linear specs, and two physical tree groups expanded to two `tree_nodes=8` rows. Assert irrelevant dimensions are empty and tree rows carry `fixed_forward_approx`.

- [x] **Step 2: Run the tests and verify RED**

Run: `rtk pytest -q tests/test_target_verification_profiler.py`

Expected: collection fails because `scripts.profile_target_verification_latency` does not exist.

- [x] **Step 3: Add the pure data contract**

Implement immutable `ProfileSpec` objects, full-grid defaults, positive integer CSV-list parsing, deterministic spec expansion, the complete ordered CSV field list, population standard deviation, and linearly interpolated percentiles.

- [x] **Step 4: Run the focused tests and verify GREEN**

Run: `rtk pytest -q tests/test_target_verification_profiler.py`

Expected: matrix/statistics/schema tests pass.

### Task 2: Cached-forward construction and sample isolation

**Files:**
- Modify: `tests/test_target_verification_profiler.py`
- Modify: `scripts/profile_target_verification_latency.py`

- [x] **Step 1: Write failing tests for forward inputs and cache isolation**

Use CPU torch tensors and fake cache/model objects to assert:

```python
attention_mask.shape == (batch_size, context_length + timed_input_length)
position_ids.tolist() == [list(range(context_length, context_length + timed_input_length))] * batch_size
cache_position.tolist() == list(range(context_length, context_length + timed_input_length))
```

Assert `cache_position` is passed only when supported, every measured sample obtains prefix state at exactly `context_length`, no returned KV is reused, and the canonical cache fingerprint remains unchanged.

- [x] **Step 2: Run the new tests and verify RED**

Run: `rtk pytest -q tests/test_target_verification_profiler.py`

Expected: failures name the missing forward-input and cache-isolation helpers.

- [x] **Step 3: Implement target-only loading and CUDA measurement**

Create `CudaTargetProfiler` that:

```python
runner_config["model_runner"]["drafter_models"] = {}
runner = HuggingFaceModelRunner(runner_config)
runner.target_model.eval()
```

Build deterministic prefix/timed IDs without tokenization. Construct prefix KV with `torch.inference_mode()` and `use_cache=True` before timing. Clone a legacy cache only when available and safe; otherwise rebuild an equivalent prefix cache outside timing. Fingerprint the canonical cache before and after samples. For each sample, create fresh CUDA events, synchronize before `start.record()`, invoke the model with `attention_mask`, `position_ids`, `past_key_values`, `use_cache=True`, and supported `cache_position`, record the end event, synchronize, and read `start.elapsed_time(end)`.

- [x] **Step 4: Run the focused tests and verify GREEN**

Run: `rtk pytest -q tests/test_target_verification_profiler.py`

Expected: cached-forward and isolation tests pass without CUDA.

### Task 3: OOM continuation, shared tree samples, and durable CSV

**Files:**
- Modify: `tests/test_target_verification_profiler.py`
- Modify: `scripts/profile_target_verification_latency.py`

- [x] **Step 1: Write failing orchestration tests**

Use a fake backend to prove:

- prefix OOM emits OOM rows for decode, every gamma, and every tree node in one batch/context group;
- an OOM in one timed combination does not suppress later combinations;
- tree measurement is called once per batch/context and all node rows have byte-for-byte equal timing statistics;
- every appended row causes the complete CSV to be safely rewritten;
- peak memory, versions, dtype, device, attention implementation, revision, past length, timed input length, warmup, repeat, and latency scope are populated.

- [x] **Step 2: Run the new tests and verify RED**

Run: `rtk pytest -q tests/test_target_verification_profiler.py`

Expected: failures identify missing orchestration and OOM behavior.

- [x] **Step 3: Implement grouped profiling and CLI**

Group work by `(batch_size, context_length)`. Prepare a prefix-cache probe first; on OOM emit every group row and clean up. Otherwise measure decode once, each gamma once, and tree once; replicate tree statistics across node rows. Catch only CUDA OOM as recoverable, compact its message, run `gc.collect()` and `torch.cuda.empty_cache()`, and continue. Rewrite CSV after every emitted row. Add CLI options for config/model/device/revision/cache/local-only, matrix values, warmup, repeat, and output.

- [x] **Step 4: Run focused tests and static checks**

Run:

```bash
rtk pytest -q tests/test_target_verification_profiler.py
rtk python -m py_compile scripts/profile_target_verification_latency.py
rtk git diff --check
```

Expected: all focused tests pass, compilation exits 0, and diff check reports no whitespace errors.

### Task 4: Approved real-model smoke only

**Files:**
- Create: `outputs/profiling/target_verification_latency_smoke.csv`

- [x] **Step 1: Inspect available CUDA devices and cached target configuration**

Run read-only environment checks to select the configured target device or an explicit available CUDA device. Do not download or commit model files.

- [x] **Step 2: Run exactly the approved smoke matrix**

Run:

```bash
rtk python scripts/profile_target_verification_latency.py \
  --config configs/default.yaml \
  --batch-sizes 1,2 \
  --context-lengths 128 \
  --gammas 1,4 \
  --tree-nodes 8 \
  --warmup 10 \
  --repeat 30 \
  --output outputs/profiling/target_verification_latency_smoke.csv
```

Expected: eight rows total (two decode, four linear, two approximate tree), or explicit OOM rows where applicable; no full matrix is run.

- [x] **Step 3: Verify smoke CSV and source scope**

Run a targeted CSV validation command that checks row count, required fields, sample statistics for success rows, OOM encoding for failure rows, shared tree semantics, and `tree_mode=fixed_forward_approx`. Confirm `scheduler.py`, `simulator.py`, `latency.py`, and baseline algorithm files are absent from `git diff --name-only`.

- [ ] **Step 4: Report only requested handoff data**

Report modified files, smoke result, and one suggested commit message. Do not report full preflight or full-matrix results because neither is run.
