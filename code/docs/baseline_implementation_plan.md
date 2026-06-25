# Baseline Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use executing-plans task-by-task. Steps use checkbox (`- [ ]`) syntax for milestone tracking.

**Goal:** Rebuild the decode-only baselines required by `docs/baseline_contract.md` without changing the proposed method or reintroducing prefill simulation.

**Architecture:** Keep the existing event simulator for target-only, linear SD, SpecEdge, and server-only paths, but split method names and candidate strategy selection explicitly. Add DiP-SD as a separate ordered-batch pipeline module that returns the same `SimulationResult` shape. Keep legacy behavior until replacement tests pass, then migrate docs and CLI defaults.

**Tech Stack:** Python 3, unittest/pytest-compatible tests, deterministic `FakeModelRunner`, analytical latency helpers, git milestone commits.

---

## File Map

- `src/methods.py`: canonical method names, legacy aliases, runtime metadata.
- `src/simulator.py`: target-only communication removal, method-specific strategy overrides, server-only no-network semantics, SpecEdge linear/tree routing, DiP-SD delegation.
- `src/model_runner.py`: shared verification result contract compatibility.
- `src/dip_sd.py`: DiP-SD fixed pipeline and optimizer implementation.
- `src/config.py`: DiP-SD config validation and method-specific tree strategy validation.
- `src/metrics.py`: comparison baselines for new method names and resource accounting.
- `configs/default.yaml`: explicit baseline config sections and DiP-SD capacity/optimizer knobs.
- `scripts/run.sh` and `scripts/run_all.py`: canonical method defaults and CLI choices.
- `scripts/verify_baseline_rebuild.sh`: full pytest, milestone-specific pytest, static no-prefill check.
- `tests/test_target_only.py`: target-only contract tests.
- `tests/test_linear_sd_core.py`: shared linear verification and greedy equivalence tests.
- `tests/test_server_only_linear.py`: server-only linear tests.
- `tests/test_specedge_linear.py`: SpecEdge linear tests.
- `tests/test_dip_sd.py`: DiP-SD pipeline and optimizer tests.
- `tests/test_server_only_tree.py`: server-only tree tests.
- `tests/test_specedge_tree.py`: SpecEdge tree tests.
- `README.md`, `docs/experiment.md`, `docs/metric.md`: baseline naming and decode-only documentation updates.
- `docs/baseline_status.md`: milestone status, changed files, commands, results, deviations, commits.

## Milestones

### M0: Audit Existing Code vs Contract

**Completion conditions**

- [ ] Read `AGENTS.md`, `docs/baseline_contract.md`, README, configs, tests, source, and git status.
- [ ] Record current supported methods and current semantic gaps.
- [ ] Inspect official `kaist-ina/specedge` repository and DiP-SD paper source.
- [ ] Create this plan with fixed M0-M10 order.
- [ ] Update `docs/baseline_status.md` with changed files, commands, test results, and deviations.
- [ ] Run `pytest -q`.
- [ ] Commit M0 documentation only.

**Verification commands**

```bash
pytest -q
grep -Rni --exclude-dir=.git --exclude='*.md' -E 'include_prefill|draft_prefill|target_prefill|prefill_latency' src configs tests
```

The grep command is expected to fail in M0 because existing tests contain forbidden literal names; M10 must make the final verification script pass.

### M1: Target-Only Cleanup and Correctness Tests

**Completion conditions**

- [ ] Add `tests/test_target_only.py`.
- [ ] Ensure `target_only` creates only request-arrival, target service, and finish events.
- [ ] Remove target-only decode downlink latency/payload from execution and metrics.
- [ ] Preserve FCFS single target resource serialization.
- [ ] Prove output equals `FakeModelRunner.target_only()`.
- [ ] Update status and commit.

**Verification commands**

```bash
pytest -q tests/test_target_only.py
pytest -q
```

### M2: Shared Linear SD Semantics

**Completion conditions**

- [ ] Add `tests/test_linear_sd_core.py`.
- [ ] Expose contract-compatible verification fields: `accepted_count`, `committed_tokens`, `correction_token`, and `bonus_token`.
- [ ] Keep greedy correction/bonus behavior lossless.
- [ ] Add shared tests that linear speculative output equals target-only greedy output.
- [ ] Ensure no unverified draft token is committed before verification result arrival.
- [ ] Update status and commit.

**Verification commands**

```bash
pytest -q tests/test_linear_sd_core.py tests/test_target_only.py
pytest -q
```

### M3: Server-Only-Linear

**Completion conditions**

- [ ] Register `server_only_linear`.
- [ ] Force linear candidate drafting for this method independent of tree config.
- [ ] Preserve separate server draft and target logical resources.
- [ ] Remove all network events and final response downlink for server-only baselines.
- [ ] Enforce round order: draft, verify, state update, next draft.
- [ ] Add `tests/test_server_only_linear.py`.
- [ ] Update status and commit.

**Verification commands**

```bash
pytest -q tests/test_server_only_linear.py tests/test_linear_sd_core.py tests/test_target_only.py
pytest -q
```

### M4: SpecEdge-Linear

**Completion conditions**

- [ ] Register `specedge_linear`.
- [ ] Force linear initial and proactive candidates while keeping edge/server deployment, network, batching, and proactive scheduling.
- [ ] Test dynamic batching takes ready requests without waiting for a full batch.
- [ ] Test static batching waits for a full batch unless a declared timeout is configured.
- [ ] Test proactive work is never committed before target verification.
- [ ] Add `tests/test_specedge_linear.py`.
- [ ] Update status and commit.

**Verification commands**

```bash
pytest -q tests/test_specedge_linear.py tests/test_server_only_linear.py tests/test_linear_sd_core.py tests/test_target_only.py
pytest -q
```

### M5: DiP-SD Fixed Pipeline

**Completion conditions**

- [ ] Add `src/dip_sd.py` with fixed grouping and fixed draft length support.
- [ ] Add temporary method `dip_sd_greedy` for the fixed pipeline before the optimizer is complete.
- [ ] Model one origin edge device per request, local linear draft, upload, ordered batch verification, download, sync, and next draft.
- [ ] Enforce one unverified draft per request.
- [ ] Enforce non-empty ordered batches visited cyclically.
- [ ] Enforce epoch-barrier admission for new arrivals.
- [ ] Add `tests/test_dip_sd.py` covering fixed pipeline invariants.
- [ ] Update status and commit.

**Verification commands**

```bash
pytest -q tests/test_dip_sd.py tests/test_linear_sd_core.py tests/test_target_only.py
pytest -q
```

### M6: DiP-SD Optimizer

**Completion conditions**

- [ ] Implement deterministic optimizer for `dip_sd`.
- [ ] Enumerate feasible batch counts.
- [ ] For fixed draft lengths, deterministically update assignment to reduce pipeline span.
- [ ] For fixed assignment, solve integer draft lengths with an exact bounded enumeration/Dinkelbach-equivalent objective for configured active cohorts.
- [ ] Use only configured/calibrated/causal estimated acceptance, never future realized acceptance.
- [ ] Switch canonical method name from `dip_sd_greedy` to `dip_sd` only after optimizer tests pass.
- [ ] Extend `tests/test_dip_sd.py` for determinism, estimated acceptance, partition constraints, and online epoch barrier.
- [ ] Update status and commit.

**Verification commands**

```bash
pytest -q tests/test_dip_sd.py tests/test_linear_sd_core.py tests/test_target_only.py
pytest -q
```

### M7: Shared Tree Drafting and Verification

**Completion conditions**

- [ ] Add `tests/test_server_only_tree.py` and `tests/test_specedge_tree.py` skeletons with shared helpers.
- [ ] Validate `DraftCandidateTree` construction, accepted path, tree-node accounting, and tree greedy equivalence.
- [ ] Record official SpecEdge source commit and behavior decisions in status.
- [ ] Ensure tree verification still obeys shared target semantics.
- [ ] Update status and commit.

**Verification commands**

```bash
pytest -q tests/test_server_only_tree.py tests/test_specedge_tree.py tests/test_linear_sd_core.py
pytest -q
```

### M8: Server-Only-Tree

**Completion conditions**

- [ ] Register `server_only_tree`.
- [ ] Force SpecExec-style tree strategy for this method.
- [ ] Use server-only tree config with official placement: target `cuda:0`, draft `cuda:1` as source attribution, represented as separate logical resources in the simulator.
- [ ] Preserve no proactive drafting, no network, no overlapping cross-round draft.
- [ ] Test one unverified candidate tree per request and target-only greedy equivalence.
- [ ] Update status and commit.

**Verification commands**

```bash
pytest -q tests/test_server_only_tree.py tests/test_target_only.py
pytest -q
```

### M9: SpecEdge-Tree

**Completion conditions**

- [ ] Register `specedge_tree`.
- [ ] Force SpecExec-style initial and proactive tree strategies.
- [ ] Keep server batching modes and proactive edge drafting.
- [ ] Enforce exact proactive reuse alignment: accepted path reaches proactive source leaf and bonus token equals proactive root token.
- [ ] Test alignment success reuses and failure discards.
- [ ] Test final output equals target-only greedy.
- [ ] Update status and commit.

**Verification commands**

```bash
pytest -q tests/test_specedge_tree.py tests/test_server_only_tree.py tests/test_target_only.py
pytest -q
```

### M10: Global Regression, Legacy Cleanup, and Docs

**Completion conditions**

- [ ] Create `scripts/verify_baseline_rebuild.sh`.
- [ ] Update README, docs, CLI defaults, and metrics comparisons to canonical baseline names.
- [ ] Remove or migrate obsolete old-baseline tests after replacement tests pass.
- [ ] Keep proposed methods (`full`, `wo_async`, `wo_scheduling`, `conservative_rollback`) behavior intact.
- [ ] Make the no-prefill static check pass without excluding `src`, `configs`, or `tests`.
- [ ] Run the verification script and full pytest.
- [ ] Update status and commit.

**Verification commands**

```bash
bash scripts/verify_baseline_rebuild.sh
pytest -q
```

### M15: DiP-SD Paper-to-Code Reproduction Specification

**Completion conditions**

- [x] Re-read the DiP-SD paper and current `src/dip_sd.py`, simulator path,
      method registry, config, tests, and M5/M6 commits.
- [x] Create `docs/dip_sd_reproduction_spec.md`.
- [x] Map paper symbols, variables, formulas, constraints, and Algorithm 1 to
      planned code functions.
- [x] Mark current implementation status for every paper requirement.
- [x] Record acceptance-estimate, fixed-cohort, and no-future-oracle rules.
- [x] Update status and commit documentation only.

**Verification commands**

```bash
git diff --check
pytest -q tests/test_dip_sd.py
```

### M16: Full DiP-SD Paper Optimizer

**Completion conditions**

- [x] Replace heuristic `dip_sd` planning with the paper optimizer.
- [x] Implement batch-count scan, exact assignment subproblem, exact/equivalent
      draft-length subproblem, Dinkelbach/equivalent fractional objective, and
      deterministic tie-breaking.
- [x] Add feasibility diagnostics and explicit errors for infeasible input.
- [x] Ensure the optimizer cannot read future realized acceptance.
- [x] Add optimizer tests, including tiny-case brute-force oracle tests.
- [x] Update status and commit.

**Verification commands**

```bash
pytest -q tests/test_dip_sd.py
pytest -q
```

### M17: DiP-SD Optimizer Integration Into Event Simulation

**Completion conditions**

- [x] Ensure optimizer assignment controls batch membership.
- [x] Ensure per-user optimized draft lengths control drafting.
- [x] Preserve batch readiness, slow-member blocking, batch order, real batch
      verification, and per-request verification barriers.
- [x] Validate trace span against the optimizer model or record bounded modeling
      error.
- [x] Add event-trace tests for optimizer-controlled execution.
- [x] Update status and commit.

**Verification commands**

```bash
pytest -q tests/test_dip_sd.py
pytest -q
```

### M18: DiP-SD Public Interface Cleanup and Final Acceptance

**Completion conditions**

- [x] Public method registry exposes `dip_sd` only for the original paper method.
- [x] Remove `dip_sd_greedy`, `dip_sd_static`, and `dip_sd_heuristic` from
      default/public method paths.
- [x] Update README, default config, contract, status, semantic audit,
      experiment docs, run scripts, and verification script.
- [x] Run the baseline verification script, full pytest, and diff check.
- [x] Record remaining paper deviations, if any.
- [x] Update status and commit.

**Verification commands**

```bash
bash scripts/verify_baseline_rebuild.sh
pytest -q
git diff --check
```

## Cross-Milestone Rules

- Each milestone must update `docs/baseline_status.md` before its commit.
- Each milestone must include method-specific tests created for that milestone.
- Do not weaken `docs/baseline_contract.md` to make tests pass.
- Do not use future acceptance outcomes, future arrivals, or future completion information.
- All lossless methods must output exactly the same tokens as `target_only` greedy decoding.
- Keep legacy implementations until replacement tests pass.
- Record any contract/source conflict explicitly in `docs/baseline_status.md`.
