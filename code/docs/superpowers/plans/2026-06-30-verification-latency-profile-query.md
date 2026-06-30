# Verification Latency Profile Query Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pure Python, immutable, indexed query layer over the merged target verification latency profile.

**Architecture:** `VerificationLatencyProfile` parses and validates the CSV once, stores typed immutable rows plus method-specific indexes, and answers queries without filesystem access. Query planning rounds context/gamma/batch conservatively, splits infeasible batches over serial GPU subbatches, and returns frozen result/provenance dataclasses.

**Tech Stack:** Python standard-library `csv`, dataclasses, pytest.

---

## File Structure

- Create `src/verification_latency_profile.py`: CSV validation, immutable row/result types, indexes, tier selection, splitting, and query API.
- Create `tests/test_verification_latency_profile.py`: temporary mock-profile fixtures and all query/validation regression tests.
- Modify `docs/superpowers/specs/2026-06-30-verification-latency-profile-query-design.md`: record strict arguments, exact index keys, and immutability.
- Create `docs/superpowers/plans/2026-06-30-verification-latency-profile-query.md`: implementation and verification steps.

### Task 1: Immutable loading and indexes

**Files:**
- Create: `tests/test_verification_latency_profile.py`
- Create: `src/verification_latency_profile.py`

- [x] **Step 1: Write failing load/index tests**

Create a temporary CSV writer using the real profiling columns. Assert default
metric selection, unique original keys, success-only runtime indexes,
diagnostic OOM retention, tree fixed-forward validation, exact tree-statistic
consistency, smallest-node canonical selection, and successful query after the
CSV file is deleted.

Core expected API:

```python
profile = VerificationLatencyProfile(csv_path)
assert profile.metric == "p50_ms"
assert profile.oom_rows[0].status == "oom"
csv_path.unlink()
result = profile.query("target_decode", batch_size=1, context_length=128)
assert result.total_latency_ms == 10.0
```

- [x] **Step 2: Run the test file and verify RED**

Run: `rtk pytest -q tests/test_verification_latency_profile.py`

Expected: import failure because `src.verification_latency_profile` does not
exist.

- [x] **Step 3: Implement typed parsing and indexes**

Add `ProfileValidationError`, `ProfileQueryError`, frozen `_ProfileRow`,
`ProfileSourceRow`, and `VerificationLatencyQueryResult`. Parse integral CSV
values such as `1` and `1.0`, reject malformed method-specific fields, validate
finite success statistics, enforce the five-field raw uniqueness key, and
build target, linear, canonical-tree, feasible-batch, legal-tier, and OOM
structures during construction.

- [x] **Step 4: Run target tests and verify GREEN**

Run: `rtk pytest -q tests/test_verification_latency_profile.py`

Expected: loading, indexing, validation, and immutable provenance tests pass.

### Task 2: Conservative query and serial splitting

**Files:**
- Modify: `tests/test_verification_latency_profile.py`
- Modify: `src/verification_latency_profile.py`

- [x] **Step 1: Write failing query-planning tests**

Cover exact P50/mean/P95 queries; B=3/L=900/gamma=3 rounding to B=4/L=1024/
gamma=4; `[120,500,900]` max-context padding; B16/L2048 -> `[8,8]`;
B9/L2048 -> `[8,1]`; B20/L2048 -> `[8,8,4]`; and B17/L1024 -> `[16,1]`.
Assert actual sizes differ from conservative profile tiers without adding
requests, and total latency equals the source-row metric sum.

- [x] **Step 2: Run new tests and verify RED**

Run: `rtk pytest -q tests/test_verification_latency_profile.py`

Expected: query planning and split assertions fail before implementation.

- [x] **Step 3: Implement query planning**

Resolve the global padded context and linear gamma through a binary-search
ceiling helper. Obtain feasible batches from the prebuilt condition index.
Use one direct subbatch when the globally rounded batch tier is feasible;
otherwise chunk by the largest feasible tier and independently ceiling each
remainder within feasible tiers. Look up only success indexes and sum the
selected metric serially.

- [x] **Step 4: Run target tests and verify GREEN**

Run: `rtk pytest -q tests/test_verification_latency_profile.py`

Expected: exact, rounded, mixed-context, OOM split, and B>16 tests pass.

### Task 3: Strict errors and final scope verification

**Files:**
- Modify: `tests/test_verification_latency_profile.py`
- Modify: `src/verification_latency_profile.py`

- [x] **Step 1: Write failing strict-validation tests**

Parametrize invalid target/linear/tree argument combinations, nonpositive
dimensions, conflicting context arguments, context-list size mismatch,
context above 2048, gamma above 8, unsupported metrics/methods, and profiles
with no feasible success batch.

- [x] **Step 2: Run new tests and verify RED**

Run: `rtk pytest -q tests/test_verification_latency_profile.py`

Expected: each unsupported input fails until its explicit validation exists.

- [x] **Step 3: Implement explicit query errors**

Validate method-specific arguments before tier selection and raise
`ProfileQueryError` messages naming the invalid dimension or missing feasible
condition. Do not fall back to OOM rows or interpolate absent success rows.

- [x] **Step 4: Run only target verification commands**

Run:

```bash
rtk pytest -q tests/test_verification_latency_profile.py
rtk python -m py_compile src/verification_latency_profile.py
rtk git diff --check
rtk git diff --name-only
```

Expected: target tests pass, compilation and whitespace checks exit zero, and
the diff contains only the four requested files. No real profile, simulator,
or full-suite command is run.

- [ ] **Step 5: Report requested handoff**

Report modified files, target-test result, and a suggested commit message. Do
not commit, modify simulator code, or run unrelated tests.
