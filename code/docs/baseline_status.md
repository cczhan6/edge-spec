# Baseline Reconstruction Status

Current milestone: M17 complete

| Milestone | Status | Commit | Tests |
|---|---|---|---|
| M0 Audit existing code vs contract | complete | `d71648d` | `pytest -q` -> 87 passed |
| M1 Target-only cleanup and correctness tests | complete | `84a873c` | `pytest -q tests/test_target_only.py tests/test_target_only_capacity.py` -> 7 passed; `pytest -q` -> 93 passed |
| M2 Shared linear SD semantics | complete | `1c4a9b6` | `pytest -q tests/test_linear_sd_core.py tests/test_target_only.py` -> 13 passed; `pytest -q` -> 100 passed |
| M3 Server-only-Linear | complete | `e007e74` | `pytest -q tests/test_server_only_linear.py tests/test_linear_sd_core.py tests/test_target_only.py` -> 18 passed; `pytest -q` -> 105 passed |
| M4 SpecEdge-Linear | complete | `bdaa859` | `pytest -q tests/test_specedge_linear.py tests/test_server_only_linear.py tests/test_linear_sd_core.py tests/test_target_only.py` -> 25 passed; `pytest -q` -> 112 passed |
| M5 DiP-SD fixed pipeline | complete | `a7191db` | `pytest -q tests/test_dip_sd.py tests/test_linear_sd_core.py tests/test_target_only.py` -> 20 passed; `pytest -q` -> 119 passed |
| M6 DiP-SD optimizer | complete | `8e34dd5` | `pytest -q tests/test_dip_sd.py tests/test_linear_sd_core.py tests/test_target_only.py` -> 22 passed; `pytest -q` -> 121 passed |
| M7 Shared tree drafting and verification | complete | `a99b92a` | `pytest -q tests/test_server_only_tree.py tests/test_specedge_tree.py tests/test_linear_sd_core.py` -> 13 passed; `pytest -q` -> 127 passed |
| M8 Server-only-Tree | complete | `2605beb` | `pytest -q tests/test_server_only_tree.py tests/test_target_only.py` -> 13 passed; `pytest -q` -> 131 passed |
| M9 SpecEdge-Tree | complete | `ab0ab57` | `pytest -q tests/test_specedge_tree.py tests/test_server_only_tree.py tests/test_target_only.py` -> 21 passed; `pytest -q` -> 136 passed |
| M10 Regression and cleanup | complete | `46065c6` | `bash scripts/verify_baseline_rebuild.sh` -> `pytest -q` 137 passed; method-specific pytest 49 passed; no prefill grep matches |
| M15 DiP-SD paper-to-code reproduction spec | complete | `f052b0e` | `git diff --check` -> passed; `pytest -q tests/test_dip_sd.py` -> 9 passed |
| M16 Full DiP-SD paper optimizer | complete | `07bc84c` | `pytest -q tests/test_dip_sd.py` -> 16 passed; `pytest -q` -> 144 passed |
| M17 DiP-SD optimizer simulation integration | complete | this commit | `pytest -q tests/test_dip_sd.py` -> 24 passed; `pytest -q` -> 152 passed |

## M15 DiP-SD Paper-To-Code Reproduction Spec

### Completion Conditions

- Re-read the DiP-SD paper through arXiv HTML and extracted formulas/algorithm
  steps from arXiv:2604.20919v1.
- Re-audited `src/dip_sd.py`, `src/simulator.py` DiP-SD path, `src/methods.py`,
  `configs/default.yaml`, `tests/test_dip_sd.py`, and M5/M6 commits.
- Created `docs/dip_sd_reproduction_spec.md`.
- Added M15-M18 continuation milestones to `docs/baseline_implementation_plan.md`.
- Confirmed canonical `dip_sd` must mean the original paper method; static,
  greedy, or heuristic substitutes are not accepted public baselines.
- Re-stated current implementation gaps before M16.

### Changed Files

- Added `docs/baseline_semantic_audit.md` to version control as the completed
  semantic audit prerequisite for M15.
- Added `docs/dip_sd_reproduction_spec.md`.
- Updated `docs/baseline_implementation_plan.md`.
- Updated `docs/baseline_status.md`.

### Commands And Results

- `curl --max-time 20 -L -sS https://arxiv.org/html/2604.20919 > /tmp/dip.html`
  -> success.
- `curl --max-time 30 -L -sS https://arxiv.org/pdf/2604.20919 -o dip.pdf`
  -> timed out with 0 bytes; not used because arXiv HTML exposed formula
  `alttext` and Algorithm 1 text.
- `git show --stat --oneline a7191db` -> inspected M5 fixed-pipeline commit.
- `git show --stat --oneline 8e34dd5` -> inspected M6 optimizer commit.
- `git diff --check` -> passed.
- `pytest -q tests/test_dip_sd.py` -> 9 passed.

### Paper-To-Code Decisions

- `dip_sd` will not use the current static/heuristic planner as a public
  substitute.
- `dip_sd` must implement the paper variables `N`, `x_mn`, `l_m`, `b_n`,
  `L_n`, `I_n`, `t_n^d`, `t_n^v`, `T_n`, and `S`.
- `u_m(l_m)=(1-alpha_m^(l_m+1))/(1-alpha_m)` can be retained from current code.
- Current `_assign_for_lengths`, `_pipeline_span`, and `_batch_span` are not
  paper-equivalent and must be replaced for canonical `dip_sd`.
- The optimizer must accept explicit profiled draft, communication, verify,
  memory, prefix-length, and acceptance-estimate inputs.
- Acceptance estimates must be offline/profile/past-only and the optimizer must
  not receive future realized target acceptance.
- Canonical `dip_sd` starts from a fixed cohort assumption. Any online wrapper
  must be separately named `dip_sd_online`.

### Deviations Remaining At M15 Exit

- Full DiP-SD optimizer is not implemented yet; planned for M16.
- DiP-SD optimizer output is not yet connected to event simulation; planned for
  M17.
- Public method cleanup has not happened yet; planned for M18.

## M16 Full DiP-SD Paper Optimizer

### Completion Conditions

- Added paper-level DiP-SD optimizer data model in `src/dip_sd.py`.
- Implemented paper variables and objective terms:
  `N`, `x_mn`, `l_m`, `b_n`, `L_n`, `I_n`, `tau_m^d`, `tau_m^c`,
  `t_n^d`, `t_n^v`, `T_n`, `S`, `U(l)`, and `R=U/S`.
- Replaced canonical `optimize_epoch_plan` wrapper with a call into the new
  paper-equivalent optimizer instead of the old greedy assignment heuristic.
- Implemented exact assignment subproblem for fixed draft lengths.
- Implemented exact finite-domain Dinkelbach draft-length subproblem for fixed
  assignment.
- Implemented deterministic batch-count scan and tie-breaking.
- Added explicit feasibility errors for invalid input and excessive optimizer
  state spaces.
- Added optimizer tests for feasibility, complete/disjoint assignment,
  non-empty batches, draft-length bounds, determinism, manual objective,
  brute-force tiny-case agreement, and no future-acceptance access.

### Changed Files

- Updated `src/dip_sd.py`.
- Updated `tests/test_dip_sd.py`.
- Updated `docs/baseline_status.md`.

### Commands And Results

- `pytest -q tests/test_dip_sd.py` -> 16 passed.
- `pytest -q` -> 144 passed.

### Deviations Remaining

- The simulator still builds simplified optimizer inputs through the
  compatibility wrapper; M17 must pass real prefix length, communication latency,
  draft timing profile, verify timing profile, and memory cap inputs.
- `dip_sd_greedy` remains registered until M18 public interface cleanup.
- Event trace does not yet prove optimizer assignment/draft lengths control
  execution; planned for M17.

## M17 DiP-SD Optimizer Simulation Integration

### Completion Conditions

- Canonical `dip_sd` now builds a `DipSDProblem` per active epoch and calls
  `optimize_dip_sd` directly.
- Optimizer assignment controls runtime batch membership.
- Optimizer per-request draft lengths control actual local drafting.
- Batch verification trace records optimizer batch, request ids, ready-time
  model, verify-time model, objective, and diagnostics.
- Per-request redraft is blocked until the previous verification result arrives
  and the next epoch barrier is reached.
- Batch readiness uses the slowest member's local draft/upload arrival.
- Other batches can draft while an earlier ordered batch is verifying.
- Verification follows optimizer batch order and supports true multi-request
  batch verification.
- Causal acceptance estimates are updated after verification and clamped into
  the paper optimizer's open interval `(0, 1)`; the optimizer still receives no
  future target outcomes.
- Added trace-level tests for assignment, draft lengths, slow-member blocking,
  cross-batch overlap, ordered verification, redraft barriers, multi-request
  verification, and optimizer span.

### Changed Files

- Updated `src/simulator.py`.
- Updated `tests/test_dip_sd.py`.
- Updated `docs/baseline_implementation_plan.md`.
- Updated `docs/baseline_status.md`.

### Commands And Results

- `pytest -q tests/test_dip_sd.py` before final fixes -> 15 passed, 1 failed;
  the failure showed the old test still compared canonical simulation against
  the M16 simplified compatibility wrapper.
- `pytest -q tests/test_dip_sd.py` after adding trace tests and acceptance
  estimator feedback -> 14 passed, 10 failed; failures showed all-accept or
  all-reject causal estimates could become `1.0` or `0.0`, outside the paper
  optimizer's required open interval.
- `pytest -q tests/test_dip_sd.py` after epsilon-clamping causal estimates ->
  24 passed.
- `pytest -q` -> 152 passed.

### Deviations Remaining

- `dip_sd_greedy` remains registered as a temporary fixed-pipeline method until
  M18 public interface cleanup.
- The event trace validates optimizer `S` against the ordered verification-stage
  span. Full epoch wall-clock additionally includes warm-up/drain and epoch
  barrier time, which is intentionally treated as bounded online-adaptation
  overhead rather than a paper optimizer objective term.

## M0 Audit Existing Code vs Contract

### Completion Conditions

- Read `AGENTS.md`, `docs/baseline_contract.md`, README, configs, tests, source, and git status.
- Audited current method registration and simulator semantics.
- Inspected official SpecEdge repository.
- Inspected DiP-SD paper source through an explorer subagent.
- Created `docs/baseline_implementation_plan.md` with fixed M0-M10 order.
- Ran current baseline tests.

### Changed Files

- Added `AGENTS.md`.
- Added `docs/baseline_contract.md`.
- Updated `docs/baseline_implementation_plan.md`.
- Updated `docs/baseline_status.md`.

### Commands And Results

- `git worktree add /edge-spec-baseline-rebuild -b baseline-rebuild main` -> success.
- `python3 -m unittest discover -s tests -v` -> 87 tests, OK.
- `pytest -q` -> 87 passed.
- `git clone --depth 1 https://github.com/kaist-ina/specedge.git /tmp/specedge-official` -> success.
- `git -C /tmp/specedge-official rev-parse HEAD` -> `1edcaf02ffc41a7b57726450c5357ed216a3b9bc`.
- `grep -Rni --exclude-dir=.git --exclude='*.md' -E 'include_prefill|draft_prefill|target_prefill|prefill_latency' src configs tests` -> found only existing tests asserting prefill fields are absent, plus generated `__pycache__` matches. Final M10 verification must avoid generated bytecode and remove forbidden literals from non-markdown test code.

### Current Method Registry

Current registered methods in `src/methods.py`:

```text
target_only
sync_batch_sd
SpecEdge
server_only
wo_async
wo_scheduling
conservative_rollback
full
```

Required baseline names from the contract:

```text
target_only
server_only_linear
server_only_tree
specedge_linear
specedge_tree
dip_sd
```

Only `target_only` currently exists under the required name. Current `SpecEdge` and `server_only` multiplex linear and tree behavior through `tree_draft_strategy` instead of method names. `dip_sd` is not implemented.

### Contract Deviations Found

- `target_only` currently models final response downlink latency/payload, but the contract requires no decode-stage communication for target-only.
- Current `VerificationResult` has `emitted_ids` and `rejected`; the contract requires `committed_tokens`, `correction_token`, and `bonus_token`.
- `server_only` currently runs one request lifecycle at a time with batch size 1 and no explicit `server_only.batch_size`.
- `server_only` still models final response downlink, while the contract says server-only has no decode-stage network.
- `server_only` and `SpecEdge` are old names; linear/tree variants are not separated.
- `SpecEdge` default config uses static batch size 1; contract identifies dynamic batching as canonical for online-arrival experiments while retaining static mode for sensitivity.
- `DiP-SD` has only helper remnants (`dssd_transmission_delay_ms`) and no ordered-batch pipeline or optimizer.
- README, docs, metrics, scripts, and tests still reference old baseline names.

### Source Attribution Notes

- Official SpecEdge repository: `https://github.com/kaist-ina/specedge`, branch `main`, inspected commit `1edcaf02ffc41a7b57726450c5357ed216a3b9bc`.
- Official SpecEdge key files inspected:
  - `config/server_only.example.yaml`: target `cuda:0`, draft `cuda:1`, `max_n_beams=32`, `max_beam_len=4`, `max_branch_width=16`, `max_budget=64`.
  - `config/specedge.example.yaml`: target `cuda:0`, clients on `cuda:0/cuda:1`, server `batch_type`, tree limits, proactive limits.
  - `src/script/server_only.py`: round loop drafts, verifies, updates state in order.
  - `src/specedge/client/specexec.py`: SpecExec tree growth and validation loop.
  - `src/specedge/client/proactive.py`: proactive expansion from best bonus-token candidate.
  - `src/strategy/server_verify/specexec/server_only.py` and `src/strategy/edge_verify/specexec.py`: target verification and accepted-token update.
- DiP-SD source: arXiv `2604.20919v1`, submitted April 22, 2026. No official implementation found or used.

### Official Source Conflicts And Decisions

- Official SpecEdge examples use `temperature: 0.7`; this repository keeps greedy decoding because `docs/baseline_contract.md` requires lossless equality with `target_only` greedy decoding.
- Official example defaults do not share tree budget between server-only and SpecEdge (`server_only.max_budget=64`, `specedge.max_budget=32`). The contract requires shared tree policy for controlled comparison, so implementation will use the configured shared policy for canonical `server_only_tree`/`specedge_tree` comparisons while recording source defaults in manifests/docs.
- Official SpecEdge has prefill and cached-prefill runtime paths. This repository remains decode-only and does not model prompt prefill or prompt transmission.
- Official server-only batch knob is `client.max_batch_size`; this repository will expose an explicit `server_only.batch_size`/manifest field mapped to the simulator's server-only active-batch setting.
- Official proactive reuse checks both source leaf and bonus/root token alignment. Current local implementation only fully checks bonus/root token for linear proactive and prefix equality for tree state; M9 must strengthen this to the exact alignment rule.

### Implementation Decisions

- Treat existing `full`, `wo_async`, `wo_scheduling`, and `conservative_rollback` as proposed/ablation methods and avoid modifying their semantics except where shared infrastructure changes require compatibility.
- Keep old `SpecEdge`/`server_only` behavior until replacement tests for canonical names pass.
- Implement M5 as `dip_sd_greedy` until M6 optimizer passes; only then expose canonical `dip_sd`.
- Record official SpecEdge behavior as the source of truth for tree variants when it conflicts with local approximations.

## Current Blockers

None for M1-M4. DiP-SD M6 needs deterministic solver choices because the paper does not specify all tie-breakers; the planned choices are documented in `docs/baseline_implementation_plan.md`.

## Decisions Requiring Human Review

None at M0.

## M1 Target-Only Cleanup And Correctness Tests

### Completion Conditions

- Added target-only contract tests.
- Confirmed `target_only` is registered as a non-speculative runtime.
- Confirmed target-only output equals direct greedy target output from the model runner.
- Removed target-only decode-stage downlink latency and payload from execution trace and request state.
- Preserved single logical target resource serialization and FCFS service order.
- Confirmed target-only creates no draft, verify, batch, segment, or network trace fields.
- Confirmed target-only speculative counters remain zero.

### Changed Files

- Added `tests/test_target_only.py`.
- Updated `tests/test_target_only_capacity.py`.
- Updated `tests/test_specedge_methods.py` to stop coupling legacy server-only response payload to target-only network fields.
- Updated `src/simulator.py`.
- Updated `docs/baseline_status.md`.

### Commands And Results

- `pytest -q tests/test_target_only.py tests/test_target_only_capacity.py` before implementation -> 3 failed, 4 passed; failures were target-only downlink fields.
- `pytest -q tests/test_target_only.py tests/test_target_only_capacity.py` after implementation -> 7 passed.
- `pytest -q` after implementation -> 92 passed, 1 failed; the failure was a legacy server-only test expecting target-only downlink payload.
- `pytest -q tests/test_target_only.py tests/test_target_only_capacity.py` after legacy test update -> 7 passed.
- `pytest -q` after legacy test update -> 93 passed.

### Contract Deviations Remaining

- `VerificationResult` still lacks the contract field names `committed_tokens` and `correction_token`; planned for M2.
- Canonical method names for server-only, SpecEdge, and DiP-SD are still absent; planned for M3-M9.
- Legacy `server_only` still models final response downlink; planned for M3.
- DiP-SD remains unimplemented; planned for M5-M6.

### Decisions

- Target-only request fields `target_only_downlink_ms` and `target_only_downlink_payload_bytes` remain in the data model for CSV compatibility, but target-only now leaves them at zero.

## M2 Shared Linear SD Semantics

### Completion Conditions

- Added shared linear speculative decoding core tests.
- Exposed `VerificationResult` contract fields: `accepted_count`, `committed_tokens`, `correction_token`, and `bonus_token`.
- Preserved legacy compatibility through `emitted_ids` and `rejected` properties.
- Confirmed all-accept linear verification returns a bonus token.
- Confirmed first-mismatch verification returns a correction token and no bonus token.
- Confirmed batched linear verification matches individual verification.
- Confirmed linear speculative output equals target-only greedy output.
- Confirmed rejected draft tokens are not committed.
- Confirmed max-output truncation does not commit an extra bonus token.

### Changed Files

- Added `tests/test_linear_sd_core.py`.
- Updated `src/model_runner.py`.
- Updated `docs/baseline_status.md`.

### Commands And Results

- `pytest -q tests/test_linear_sd_core.py tests/test_target_only.py` before implementation -> 3 failed, 10 passed; failures were missing `committed_tokens` and `correction_token` fields.
- `pytest -q tests/test_linear_sd_core.py tests/test_target_only.py` after implementation -> 13 passed.
- `pytest -q` after implementation -> 100 passed.

### Contract Deviations Remaining

- Canonical method names for server-only, SpecEdge, and DiP-SD are still absent; planned for M3-M9.
- Legacy `server_only` still models final response downlink; planned for M3.
- DiP-SD remains unimplemented; planned for M5-M6.

### Decisions

- `VerificationResult` is now contract-first while retaining read-only compatibility properties for existing simulator and tests.

## M3 Server-Only-Linear

### Completion Conditions

- Registered `server_only_linear`.
- Forced `server_only_linear` to use linear candidates independent of tree config.
- Added explicit `server_only.batch_size` config validation with default `1`.
- Added server draft and target logical resource labels to server-only trace events.
- Removed decode-stage network fields and final response downlink from server-only runtime completion.
- Confirmed draft/verify round order for server-only-linear.
- Confirmed server-only-linear output equals target-only greedy output for all-accept and all-reject model runners.
- Confirmed no proactive events are generated.

### Changed Files

- Added `tests/test_server_only_linear.py`.
- Updated `src/methods.py`.
- Updated `src/simulator.py`.
- Updated `src/config.py`.
- Updated `configs/default.yaml`.
- Updated `tests/test_decode_only_initialization.py`.
- Updated `tests/test_specedge_methods.py`.
- Updated `docs/baseline_status.md`.

### Commands And Results

- `pytest -q tests/test_server_only_linear.py tests/test_linear_sd_core.py tests/test_target_only.py` before implementation -> 5 failed, 13 passed; failures were unsupported method and missing server-only-linear semantics.
- `pytest -q tests/test_server_only_linear.py tests/test_linear_sd_core.py tests/test_target_only.py` after implementation -> 18 passed.
- `pytest -q` after implementation -> 102 passed, 3 failed; failures were legacy tests expecting server-only upload/final downlink timing.
- `pytest -q tests/test_server_only_linear.py tests/test_linear_sd_core.py tests/test_target_only.py` after legacy test updates -> 18 passed.
- `pytest -q` after legacy test updates -> 105 passed.

### Contract Deviations Remaining

- Server-only batch sizes greater than one are explicit in config but the current event loop still executes one active server-only request lifecycle at a time. This is a known gap for tree/batched server-only work in M8/M10.
- `specedge_linear`, `specedge_tree`, `server_only_tree`, and `dip_sd` are still absent.
- DiP-SD remains unimplemented; planned for M5-M6.

### Decisions

- The legacy `server_only` runtime now follows the decode-only no-network boundary as well. This keeps old tests compatible with the contract while the canonical `server_only_linear` and later `server_only_tree` names are introduced.

## M4 SpecEdge-Linear

### Completion Conditions

- Registered `specedge_linear`.
- Forced linear initial candidate drafting for `specedge_linear`.
- Forced linear proactive drafting for `specedge_linear`.
- Preserved edge-origin drafting, upload/download communication, server batching, and proactive scheduling.
- Confirmed dynamic batching takes currently ready requests.
- Confirmed static batching waits for full batch in the tested no-timeout case.
- Confirmed proactive drafting runs while waiting and final output is still target-only greedy.
- Confirmed alignment failure discards proactive state.

### Changed Files

- Added `tests/test_specedge_linear.py`.
- Updated `src/methods.py`.
- Updated `src/simulator.py`.
- Updated `docs/baseline_status.md`.

### Commands And Results

- `pytest -q tests/test_specedge_linear.py tests/test_server_only_linear.py tests/test_linear_sd_core.py tests/test_target_only.py` before implementation -> 7 failed, 18 passed; failures were unsupported `specedge_linear`.
- `pytest -q tests/test_specedge_linear.py tests/test_server_only_linear.py tests/test_linear_sd_core.py tests/test_target_only.py` after implementation -> 25 passed.
- `pytest -q` after implementation -> 112 passed.

### Contract Deviations Remaining

- `dip_sd`, `server_only_tree`, and `specedge_tree` are still absent.
- Server-only batch sizes greater than one remain a known gap.
- Exact SpecEdge tree proactive reuse source-leaf check remains for M9.

### Decisions

- `specedge_linear` keeps SpecEdge deployment, network, server batching, and proactive state machine, but replaces both initial and proactive tree strategies with linear strategies at simulator initialization.

## M5 DiP-SD Fixed Pipeline

### Completion Conditions

- Added fixed DiP-SD epoch planning in `src/dip_sd.py`.
- Registered temporary method `dip_sd_greedy`.
- Kept canonical `dip_sd` unsupported until optimizer completion.
- Added DiP-SD config section with fixed batch count, fixed draft length, and capacity limits.
- Modeled local linear draft, upload, ordered batch verification, download, synchronization, and next-round draft.
- Enforced one unverified draft per request through synchronization-before-redraft tests.
- Enforced complete, disjoint, non-empty ordered batches.
- Enforced epoch-barrier admission for newly arrived requests.
- Confirmed DiP-SD fixed pipeline output equals target-only greedy output.

### Changed Files

- Added `src/dip_sd.py`.
- Added `tests/test_dip_sd.py`.
- Updated `src/simulator.py`.
- Updated `src/methods.py`.
- Updated `src/config.py`.
- Updated `configs/default.yaml`.
- Updated `docs/baseline_status.md`.

### Commands And Results

- `pytest -q tests/test_dip_sd.py tests/test_linear_sd_core.py tests/test_target_only.py` after first implementation -> 19 passed, 1 failed; failure showed new arrivals were admitted at an epoch barrier but draft start still used original arrival time.
- `pytest -q tests/test_dip_sd.py tests/test_linear_sd_core.py tests/test_target_only.py` after admission ready-time fix -> 20 passed.
- `pytest -q` after implementation -> 119 passed.

### Contract Deviations Remaining

- `dip_sd` canonical method is still intentionally unsupported because the optimizer is not complete. The available method is `dip_sd_greedy`.
- The M5 pipeline uses fixed grouping and fixed draft length only; M6 must add deterministic optimizer and then expose `dip_sd`.
- Server-only batch sizes greater than one remain a known gap.
- Tree baselines remain absent.

### Decisions

- New arrivals are admitted only at epoch barriers by setting request ready time to the barrier time when admitted.
- M5 uses configured profile acceptance only for future optimizer plumbing; semantic verification still determines actual committed tokens.

## M6 DiP-SD Optimizer

### Completion Conditions

- Registered canonical `dip_sd`.
- Kept `dip_sd_greedy` available for the fixed pipeline.
- Added deterministic bounded optimizer over feasible batch counts and integer draft lengths.
- Added deterministic assignment for fixed draft lengths.
- Used configured device/profile acceptance estimates for scheduling.
- Confirmed optimizer determinism.
- Confirmed optimizer choices respond to estimated acceptance and do not inspect realized future acceptance.
- Confirmed `dip_sd` output equals target-only greedy output.

### Changed Files

- Updated `src/dip_sd.py`.
- Updated `src/methods.py`.
- Updated `src/simulator.py`.
- Updated `configs/default.yaml`.
- Updated `tests/test_dip_sd.py`.
- Updated `docs/baseline_status.md`.

### Commands And Results

- `pytest -q tests/test_dip_sd.py tests/test_linear_sd_core.py tests/test_target_only.py` -> 22 passed.
- `pytest -q` -> 121 passed.

### Contract Deviations Remaining

- The optimizer uses deterministic bounded enumeration for draft lengths and deterministic assignment updates rather than an external MILP solver. This is treated as an exact bounded implementation for configured small active cohorts; larger equal-resource experiments should revisit solver scalability.
- Server-only batch sizes greater than one remain a known gap.
- Tree baselines remain absent.

### Decisions

- Tie-breakers are deterministic: higher objective, lower span, fewer batches, lexicographically smaller batches, then lexicographically smaller draft lengths.
- Scheduling estimates come from configured device acceptance profiles; target verification remains the sole authority for committed tokens.

## M7 Shared Tree Drafting And Verification

### Completion Conditions

- Added tree-core tests in `tests/test_server_only_tree.py`.
- Added SpecEdge tree-core tests in `tests/test_specedge_tree.py`.
- Validated server-only tree strategy uses configured SpecExec-style budget.
- Validated tree verification can accept a non-primary target path.
- Validated tree rejection exposes the shared correction-token contract.
- Validated SpecEdge initial and proactive tree limits are configured.
- Validated proactive bonus tree starts from the bonus continuation.
- Validated batched tree verification matches single tree verification.

### Changed Files

- Added `tests/test_server_only_tree.py`.
- Added `tests/test_specedge_tree.py`.
- Updated `docs/baseline_status.md`.

### Commands And Results

- `pytest -q tests/test_server_only_tree.py tests/test_specedge_tree.py tests/test_linear_sd_core.py` -> 13 passed.
- `pytest -q` -> 127 passed.

### Contract Deviations Remaining

- `server_only_tree` and `specedge_tree` method names are still absent; planned for M8-M9.
- Server-only batch sizes greater than one remain a known gap.
- Exact SpecEdge proactive reuse source-leaf check remains for M9.

### Decisions

- Official SpecEdge source attribution from M0 remains the basis for tree variants: `kaist-ina/specedge` commit `1edcaf02ffc41a7b57726450c5357ed216a3b9bc`.

## M8 Server-Only-Tree

### Completion Conditions

- Registered `server_only_tree`.
- Forced SpecExec-style tree strategy for `server_only_tree` even when legacy config is set to linear.
- Preserved server-only no-network semantics.
- Preserved no proactive drafting for server-only tree.
- Added server draft and target resource assertions.
- Confirmed server-only tree output equals target-only greedy output.

### Changed Files

- Updated `src/methods.py`.
- Updated `src/simulator.py`.
- Updated `tests/test_server_only_tree.py`.
- Updated `tests/test_decode_only_initialization.py`.
- Updated `docs/baseline_status.md`.

### Commands And Results

- `pytest -q tests/test_server_only_tree.py tests/test_target_only.py` -> 13 passed.
- `pytest -q` -> 131 passed.

### Contract Deviations Remaining

- Server-only batch sizes greater than one remain a known gap.
- `specedge_tree` is still absent.
- Exact SpecEdge proactive reuse source-leaf check remains for M9.

### Decisions

- `server_only_tree` uses the local SpecExec approximation class with official-style limits from `server_only` config and source attribution from M0.

## M9 SpecEdge-Tree

### Completion Conditions

- Registered `specedge_tree`.
- Forced SpecExec-style initial tree strategy for `specedge_tree`.
- Forced SpecExec-style proactive tree strategy for `specedge_tree`.
- Preserved SpecEdge edge/server deployment, network, server batching, and proactive drafting.
- Confirmed proactive tree runs and is reused on exact alignment.
- Confirmed proactive alignment failure discards state.
- Confirmed `specedge_tree` output equals target-only greedy output.

### Changed Files

- Updated `src/methods.py`.
- Updated `src/simulator.py`.
- Updated `tests/test_specedge_tree.py`.
- Updated `docs/baseline_status.md`.

### Commands And Results

- `pytest -q tests/test_specedge_tree.py tests/test_server_only_tree.py tests/test_target_only.py` -> 21 passed.
- `pytest -q` -> 136 passed.

### Contract Deviations Remaining

- Server-only batch sizes greater than one remain a known gap.
- M10 still needs global verification script, docs cleanup, old-name cleanup, and final no-prefill static check.

### Decisions

- Tree proactive reuse requires the retained proactive tree prefix to match the accepted target-verified path before retaining state; bonus/root token equality is also required by the existing shared reuse path.

## M10 Global Regression, Legacy Cleanup, And Docs

### Completion Conditions

- Created `scripts/verify_baseline_rebuild.sh`.
- Updated README, experiment docs, metric docs, CLI defaults, and metric comparisons to canonical baseline names.
- Kept legacy method aliases (`sync_batch_sd`, `SpecEdge`, `server_only`) available for historical compatibility after canonical replacement tests passed.
- Removed obsolete server-only response downlink scheduling helper.
- Kept proposed methods (`full`, `wo_async`, `wo_scheduling`, `conservative_rollback`) behavior intact.
- Made the no-prefill static check pass without excluding `src`, `configs`, or `tests`.
- Ran full regression and method-specific baseline tests.

### Changed Files

- Updated `README.md`.
- Updated `docs/experiment.md`.
- Updated `docs/metric.md`.
- Updated `docs/baseline_status.md`.
- Updated `scripts/run.sh`.
- Added `scripts/verify_baseline_rebuild.sh`.
- Updated `src/methods.py`.
- Updated `src/metrics.py`.
- Updated `src/simulator.py`.
- Updated `tests/test_decode_only_initialization.py`.
- Updated `tests/test_metrics_speedup.py`.

### Commands And Results

- `chmod +x scripts/verify_baseline_rebuild.sh` -> success.
- `pytest -q tests/test_metrics_speedup.py tests/test_target_only.py tests/test_linear_sd_core.py` -> 16 passed.
- `rg -n 'include_prefill|draft_prefill|target_prefill|prefill_latency' src configs tests --glob '!**/__pycache__/**'` -> no matches.
- `bash scripts/verify_baseline_rebuild.sh` -> full `pytest -q` 137 passed; method-specific pytest 49 passed; final static prefill grep had no matches.

### Contract Deviations Remaining

- Server-only batch sizes greater than one are accepted in config but the current server-only event loop still executes one active server-only request lifecycle at a time. This is retained as a documented resource-model limitation rather than silently claiming equal-resource batched server-only behavior.
- Legacy aliases remain in code and tests for backward compatibility. Canonical baseline names are the default CLI methods and the names used by the rebuild tests.

### Decisions

- Canonical metric comparison fields were added for `dip_sd`, `specedge_linear`, `specedge_tree`, `server_only_linear`, and `server_only_tree`; legacy comparison fields still fall back to old method names for historical run compatibility.
- Final no-prefill verification excludes generated bytecode/cache directories but does not exclude source, config, or test files.
