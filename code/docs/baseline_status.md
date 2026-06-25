# Baseline Reconstruction Status

Current milestone: M22 mixed-length real-model batch verification correctness

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
| M17 DiP-SD optimizer simulation integration | complete | `a1fbd02` | `pytest -q tests/test_dip_sd.py` -> 24 passed; `pytest -q` -> 152 passed |
| M18 DiP-SD public interface cleanup | complete | `b7410ec` | `bash scripts/verify_baseline_rebuild.sh` -> `pytest -q` 151 passed; method-specific pytest 63 passed; static checks passed; `pytest -q` -> 151 passed; `git diff --check` -> passed |
| M19 Final baseline display/default alignment | complete | `c85f9eb` | `pytest -q` -> 154 passed; `bash scripts/verify_baseline_rebuild.sh` -> full pytest 154 passed; method-specific pytest 66 passed; static checks passed; `git diff --check` -> passed |
| M20 Baseline event semantics validation | complete | this commit | `pytest -q` -> 176 passed; `bash scripts/verify_baseline_rebuild.sh` -> full pytest 176 passed; method-specific pytest 79 passed; static checks passed |
| M21 Real-model baseline smoke harness | complete / live run blocked | this commit | `pytest -q tests/test_real_model_smoke.py tests/test_baseline_trace_runner.py tests/test_cli_smoke.py` -> 11 passed; `bash scripts/run_real_model_smoke.sh` -> blocked with explicit missing `TARGET_MODEL_PATH`/`DRAFT_MODEL_PATH`; `python3 -c "import torch"` -> `ModuleNotFoundError: No module named 'torch'`; `pytest -q` -> 180 passed |
| M22 Mixed-length real-model batch verification | complete | this commit | real Qwen smoke -> `target_only`, `server_only_linear`, `specedge_linear`, `dip_sd` all `success=True`; `pytest -q` -> 187 passed; `bash scripts/verify_baseline_rebuild.sh` -> full pytest 187 passed, method-specific pytest 79 passed; `bash scripts/run_baseline_trace.sh` -> all trace methods `success=True`; `git diff --check` -> passed |

## M22 Mixed-Length Real-Model Batch Verification

### Semantic Boundary

- `HuggingFaceModelRunner` owns real token semantics: accepted length,
  accepted tokens, correction token, bonus token, EOS handling, and committed
  tokens.
- `HuggingFaceModelRunner.verify_batch` is still the simulator-facing logical
  batch verification API, but mixed effective lengths are no longer sent through
  one right-padded decoder-only target forward.
- Real-model verification requests are stably grouped by
  `(prefix_length, draft_length)`. Each group uses an equal-length physical
  target forward, and results are restored to the original request order by
  original input index.
- Simulator latency remains separate from physical Hugging Face execution.
  `global_batch_verify` and DiP-SD continue to use the analytical verification
  latency model for one logical batch; the number of safe equal-length physical
  forwards is not used as virtual simulation time.
- DiP-SD optimizer, scheduling semantics, simulator commit rules, and proposed
  methods are unchanged.

### Regression Coverage

- Added mixed-length HF runner tests requiring
  `verify(request) == verify_batch(batch)[request]` for every request.
- Covered same prefix with different draft lengths, different prefix lengths
  with the same draft length, both prefix and draft length variation, shuffled
  input order, all-accepted bonus, middle/first-token rejection correction,
  EOS accepted/correction/bonus outcomes, completed EOS requests inside a
  mixed-length logical batch, and the real smoke failure shape where
  `[330, 3838]` became `[330, 2610]` under mixed-length right padding.

### Commands And Results

- `PYTHON_BIN=/root/miniforge3/envs/edge-spec/bin/python TARGET_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct DRAFT_MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct LOCAL_FILES_ONLY=true bash scripts/run_real_model_smoke.sh`
  -> `target_only`, `server_only_linear`, `specedge_linear`, and `dip_sd` all
  `success=True`; automatic checks passed, including speculative committed
  token traces equal to `target_only`.
- `pytest -q tests/test_dssd_oracle.py` -> 17 passed.
- `pytest -q` -> 187 passed.
- `bash scripts/verify_baseline_rebuild.sh` -> full pytest 187 passed;
  method-specific pytest 79 passed.
- `bash scripts/run_baseline_trace.sh` -> `target_only`,
  `server_only_linear`, `server_only_tree`, `specedge_linear`,
  `specedge_tree`, and `dip_sd` all `success=True`.
- `git diff --check` -> passed.

### Compatibility Notes

- Real-runner compatibility remains limited to causal LM target/drafter pairs
  whose tokenizer vocabulary and mapping are compatible with the existing
  `HuggingFaceModelRunner` checks.
- Mixed `(prefix_length, draft_length)` logical batches may require multiple
  physical Hugging Face target forwards. This is intentional for decoder-only
  correctness and must not be interpreted as simulator latency.

## M21 Real-Model Baseline Smoke Harness

### Completion Conditions

- Added `scripts/run_real_model_smoke.sh` for the canonical smoke methods:
  `target_only`, `server_only_linear`, `specedge_linear`, and `dip_sd`.
- Added `scripts/real_model_smoke.py` to prepare a fixed small decode-only
  config/dataset subset, write per-method run manifests, and verify trace
  outputs.
- The runner requires explicit `TARGET_MODEL_PATH` and `DRAFT_MODEL_PATH` or
  matching CLI flags. It does not pass `--use-fake-model-runner` and exits
  before running if model paths are absent.
- The generated smoke config uses 4 requests, 8 output tokens per request,
  fixed seed `20260625`, virtual heterogeneous devices, nonzero virtual
  communication latency, `server_only.batch_size=1`, SpecEdge proactive
  drafting enabled, and DiP-SD `paper_exact` optimizer settings.
- The verifier checks required output files, request completion, no pending
  state, no OOM/NaN/traceback patterns, greedy equivalence against
  `target_only`, real-runner manifests, target verification events, event-time
  monotonicity, resource non-overlap, server-only batch size, SpecEdge
  proactive drafting, DiP-SD optimizer assignment/per-request draft lengths, and
  same-epoch DiP-SD plans preceding realized acceptance results.
- The summary is written to `outputs/real_model_smoke/summary.md` on real runs
  and includes model names, request count, committed/drafted/verified/accepted/
  wasted tokens, acceptance ratio, finish time, GPU peak memory if available,
  and caveats.

### Live Run Status

- `bash scripts/run_real_model_smoke.sh` was executed without model paths and
  failed intentionally with a usage message requiring `TARGET_MODEL_PATH` and
  `DRAFT_MODEL_PATH`; no fake runner fallback occurred.
- A CPU probe with `sshleifer/tiny-gpt2` could prepare the smoke config, but the
  current environment lacks `torch`, so the real `HuggingFaceModelRunner` cannot
  load any model here.
- No correctness check was weakened to bypass the missing runtime dependency.

### Commands And Results

- `pytest -q tests/test_real_model_smoke.py` -> 3 passed.
- `pytest -q tests/test_real_model_smoke.py tests/test_baseline_trace_runner.py tests/test_cli_smoke.py` -> 11 passed.
- `bash scripts/run_real_model_smoke.sh` -> exit 2 with explicit missing
  `TARGET_MODEL_PATH`/`DRAFT_MODEL_PATH`.
- `python3 -c "import torch, transformers"` -> `ModuleNotFoundError: No module
  named 'torch'`.
- `pytest -q` -> 180 passed.

### Deviations Remaining

- Baseline smoke readiness is not yet achieved in this environment because a
  real target model, real drafter model, and installed `torch`/`transformers`
  runtime are required to execute `scripts/run_real_model_smoke.sh` end to end.
- GPU peak memory is reported as `n/a` unless the run environment supplies or
  records a value in the generated run manifest.

## M20 Baseline Event Semantics Validation

### Completion Conditions

- Added deterministic DiP-SD online trace tests for optimizer assignment,
  per-request draft lengths, batch-local slow-device blocking, cross-batch
  drafting progress, ordered server verification, post-verify/state-update
  redraft gating, epoch-barrier admission, and one unverified segment per
  request.
- Added SpecEdge proactive alignment tests for `specedge_linear` and
  `specedge_tree`, including success, failure, retained suffix/subtree reuse,
  wasted invalid proactive tokens, and no direct commit of unverified proactive
  tokens.
- Added canonical invariant tests across `target_only`,
  `server_only_linear`, `server_only_tree`, `specedge_linear`,
  `specedge_tree`, and `dip_sd` for token accounting, target/draft resource
  non-overlap, event time well-formedness, target-greedy equality, and no
  pending unverified state at request finish.
- Audited legacy aliases and changed them into strict canonical redirects:
  `sync_batch_sd -> dip_sd`, `SpecEdge -> specedge_tree`, and
  `server_only -> server_only_tree`.
- Kept proposed methods unchanged, did not restore prefill, did not change
  Server-only default `batch_size=1`, and did not add a new scheduling
  mechanism.

### Exposed Issues And Fixes

- Legacy aliases still entered old simulator semantics. Fixed
  `src/methods.py` so aliases return canonical `MethodSpec`s and emit visible
  `FutureWarning`s.
- Explicit alias CLI runs produced canonical metric rows but legacy-named
  detail CSV artifacts. Fixed `scripts/run_all.py` to name request, segment,
  event, device, and round-trace detail files with `result.method`.
- Existing legacy tests assumed `SpecEdge` and `server_only` could still be
  config-multiplexed into linear/tree behavior. Updated those tests to assert
  canonical tree aliasing instead.

### New Tests

- `test_dip_sd_trace_uses_optimizer_assignment`
- `test_dip_sd_trace_uses_per_request_draft_length`
- `test_dip_sd_slow_member_blocks_own_batch`
- `test_dip_sd_other_batch_can_continue_drafting`
- `test_dip_sd_verification_follows_batch_order`
- `test_dip_sd_request_waits_for_verify_and_kv_update`
- `test_dip_sd_online_arrival_waits_for_epoch_boundary`
- `test_dip_sd_one_unverified_segment_per_request`
- `test_specedge_linear_proactive_alignment_success`
- `test_specedge_linear_proactive_alignment_failure`
- `test_specedge_tree_proactive_alignment_success`
- `test_specedge_tree_proactive_alignment_failure`
- `test_specedge_never_commits_unverified_proactive_tokens`
- `test_token_accounting_conservation`
- `test_no_illegal_target_resource_overlap`
- `test_no_illegal_draft_resource_overlap`
- `test_event_time_monotonicity_all_methods`
- `test_all_lossless_methods_equal_target_greedy`
- `test_no_request_finishes_with_pending_unverified_state`
- `tests/test_legacy_aliases.py`

### Trace Readiness

| Method | Deterministic trace readiness | Caveat |
|---|---|---|
| `target_only` | ready | Single serialized target service, decode-only. |
| `server_only_linear` | ready | Main contract remains `server_only.batch_size=1`. |
| `server_only_tree` | ready | `specexec_approx`; main contract remains `server_only.batch_size=1`. |
| `specedge_linear` | ready | Proactive alignment covered for deterministic fake-runner traces. |
| `specedge_tree` | ready | `specexec_approx`; not exact official tree-kernel replay. |
| `dip_sd` | ready | Online epoch/barrier adaptation of the paper optimizer. |

### Commands And Results

- `pytest -q` -> 176 passed.
- `bash scripts/verify_baseline_rebuild.sh` -> full pytest 176 passed;
  method-specific pytest 79 passed; no-prefill and public DiP-SD static checks
  passed.

### Deviations Remaining

- Server-only `batch_size > 1` is still an optional extension and remains
  rejected until real multi-request server-only verification is implemented.
- Tree baselines using `specexec_approx` must not be described as exact
  official SpecEdge tree-kernel replay.

## M19 Final Baseline Display And Default Alignment

### Completion Conditions

- Standardized the public DiP-SD display name to `DiP-SD (Online Adaptation)`.
- Documented `dip_sd` as the paper optimizer and synchronized batch pipeline
  adapted to this project's online epoch/barrier request admission framework.
- Kept proposed-method behavior unchanged and did not restore prefill.
- Fixed `server_only_linear` and `server_only_tree` main experiments to the
  same server-only system setting: independent server draft and target GPUs, no
  edge-server communication, no proactive drafting, synchronous draft -> verify
  -> state update, and `server_only.batch_size = 1`.
- Documented official SpecEdge server-only multi-request tree verification as an
  optional extension; this repository currently rejects `server_only.batch_size
  > 1` rather than accepting it and executing single-request service.
- Marked server-only and SpecEdge tree paths as `specexec_approx` where they use
  the local approximation.

### Changed Files

- Updated `README.md`.
- Updated `configs/default.yaml`.
- Updated `docs/baseline_contract.md`.
- Updated `docs/baseline_semantic_audit.md`.
- Updated `docs/baseline_status.md`.
- Updated `docs/experiment.md`.
- Updated `src/config.py`.
- Updated `src/simulator.py`.
- Updated `tests/test_server_only_linear.py`.
- Updated `tests/test_server_only_tree.py`.

### Commands And Results

- `pytest -q` -> 154 passed.
- `bash scripts/verify_baseline_rebuild.sh` -> full pytest 154 passed;
  method-specific pytest 66 passed; no-prefill and public DiP-SD static checks
  passed.
- `git diff --check` -> initially reported one trailing whitespace in
  `docs/baseline_contract.md`; after removing it, `git diff --check` passed.

### Deviations Remaining

- `server_only.batch_size > 1` is an optional extension and is currently
  rejected by config/runtime validation.
- Legacy aliases remain compatibility paths and should not be used for final
  paper result labels.

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
- Superseded by M19: canonical `dip_sd` is reported as
  `DiP-SD (Online Adaptation)`, using the paper optimizer within the project's
  online epoch/barrier framework.

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

## M18 DiP-SD Public Interface Cleanup

### Completion Conditions

- Public method registry exposes canonical `dip_sd` only for the DiP-SD paper
  method.
- Removed public `dip_sd_greedy` method registration and simulator fixed-path
  branch.
- Removed `build_fixed_epoch_plan` and `optimize_epoch_plan` static/compatibility
  public helpers from `src/dip_sd.py`.
- Changed `configs/default.yaml` to `dip_sd.optimizer: paper_exact`.
- Added config validation rejecting non-`paper_exact` DiP-SD optimizers.
- Removed `dip_sd_greedy` metrics fallback.
- Updated README, default config, baseline contract, implementation plan,
  semantic audit, DiP-SD reproduction spec, experiment docs, and verification
  script.
- Extended the verification script to reject public static/greedy DiP-SD method
  names in `src`, `configs`, and `scripts`.
- Updated DiP-SD tests to assert `dip_sd_greedy` is unsupported and static
  optimizer names are rejected.

### Changed Files

- Updated `README.md`.
- Updated `configs/default.yaml`.
- Updated `docs/baseline_contract.md`.
- Updated `docs/baseline_implementation_plan.md`.
- Updated `docs/baseline_semantic_audit.md`.
- Updated `docs/dip_sd_reproduction_spec.md`.
- Updated `docs/experiment.md`.
- Updated `scripts/verify_baseline_rebuild.sh`.
- Updated `src/config.py`.
- Updated `src/dip_sd.py`.
- Updated `src/methods.py`.
- Updated `src/metrics.py`.
- Updated `src/simulator.py`.
- Updated `tests/test_dip_sd.py`.

### Commands And Results

- `pytest -q tests/test_dip_sd.py` -> 23 passed.
- Static public DiP-SD method grep over `src configs scripts` with
  `verify_baseline_rebuild.sh` excluded -> no matches.
- `bash scripts/verify_baseline_rebuild.sh` -> full pytest 151 passed;
  method-specific pytest 63 passed; no-prefill static check passed; public
  static/greedy DiP-SD method-name static check passed.
- `pytest -q` -> 151 passed.
- `git diff --check` -> passed.

### Deviations Remaining

- The `dip_sd` simulator is the documented `DiP-SD (Online Adaptation)` method:
  the paper optimizer and synchronized batch pipeline inside the project's
  online epoch/barrier framework.
- DiP-SD trace-span validation compares optimizer `S` to ordered
  verification-stage span; full epoch wall-clock also includes warm-up/drain and
  barrier overhead.
- Server-only `batch_size > 1` is an optional extension and is rejected by the
  current runtime until true multi-request verification is implemented.
- Legacy aliases remain compatibility paths and should not be used for final
  paper result labels.

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
- At M0, `server_only` ran one request lifecycle at a time with batch size 1
  and no explicit `server_only.batch_size`.
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

- Superseded by M19: server-only batch sizes greater than one are an optional
  extension and are rejected by current config/runtime validation.
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
- Server-only batch sizes greater than one remain an optional extension and are
  rejected by current config/runtime validation.
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
- Server-only batch sizes greater than one remain an optional extension and are
  rejected by current config/runtime validation.
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
- Server-only batch sizes greater than one remain an optional extension and are
  rejected by current config/runtime validation.
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
- Server-only batch sizes greater than one remain an optional extension and are
  rejected by current config/runtime validation.
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

- Server-only batch sizes greater than one remain an optional extension and are
  rejected by current config/runtime validation.
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

- Server-only batch sizes greater than one remain an optional extension and are
  rejected by current config/runtime validation.
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

- Superseded by M19: server-only batch sizes greater than one are rejected until
  true multi-request server-only verification is implemented.
- Legacy aliases remain in code and tests for backward compatibility. Canonical baseline names are the default CLI methods and the names used by the rebuild tests.

### Decisions

- Canonical metric comparison fields were added for `dip_sd`, `specedge_linear`, `specedge_tree`, `server_only_linear`, and `server_only_tree`; legacy comparison fields still fall back to old method names for historical run compatibility.
- Final no-prefill verification excludes generated bytecode/cache directories but does not exclude source, config, or test files.
