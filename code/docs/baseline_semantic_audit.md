# Baseline Semantic Audit

Date: 2026-06-25

Scope: independent semantic audit of the completed `baseline-rebuild` worktree.
This audit checks code paths, event flow, resource timelines, current docs, and
tests. It does not treat a passing test suite as sufficient evidence of semantic
correctness.

Inputs reviewed:

- `AGENTS.md`
- `README.md`
- `configs/default.yaml`
- `docs/baseline_contract.md`
- `docs/baseline_implementation_plan.md`
- `docs/baseline_status.md`
- `src/`, `scripts/`, and `tests/`
- Official SpecEdge repository at `https://github.com/kaist-ina/specedge`,
  locally inspected at commit `1edcaf02ffc41a7b57726450c5357ed216a3b9bc`
- DiP-SD paper: `https://arxiv.org/abs/2604.20919`

DiP-SD (Online Adaptation) audit boundary: this repository is expected to
implement the original paper's core optimizer and synchronized batch pipeline
inside the project online framework. A static or heuristic synchronized pipeline
is not an acceptable public baseline substitute, and `dip_sd` must refer to the
online adaptation of the paper method.

M18 update: the DiP-SD findings from the original audit have been remediated.
Canonical `dip_sd` now uses the paper optimizer and optimizer-controlled event
simulation, while `dip_sd_greedy`, `dip_sd_static`, and `dip_sd_heuristic` are
not public method names.

Final baseline-alignment update: `dip_sd` is reported as
`DiP-SD (Online Adaptation)`. Server-only linear/tree main experiments use the
same independent server draft GPU and server target GPU resources, no
edge-server communication, no proactive drafting, synchronous draft -> verify ->
state update execution, and `server_only.batch_size = 1`. Server-only batch
sizes greater than 1 are an optional extension and are now rejected instead of
being accepted and executed as single-request service.

Event-semantics update: deterministic trace tests now cover DiP-SD online
batch assignment, per-request draft length, slow-member batch blocking,
cross-batch drafting progress, ordered verification, online epoch admission,
one-unverified-segment-per-request, SpecEdge proactive success/failure
alignment, shared token accounting, resource non-overlap, event time
well-formedness, and legacy alias canonicalization.

## 1. Executive Summary

| Area | Verdict | Paper-experiment impact |
| --- | --- | --- |
| `target_only` | PASS | Ready as the greedy reference and speedup denominator. |
| `server_only_linear` | PASS for main default | Ready for the fixed `server_only.batch_size=1` main-experiment contract. Multi-request server-only batching is an optional extension and is rejected today. |
| `server_only_tree` | PASS for main default with caveat | `specexec_approx` tree candidate/verify path is trace-ready under fixed `server_only.batch_size=1`; it is not claimed as exact upstream CUDA/KV replay. |
| `specedge_linear` | PASS | Proactive alignment success/failure and no-unverified-proactive-commit behavior are covered by deterministic traces. |
| `specedge_tree` | PASS with caveat | Tree proactive alignment is trace-ready for the local `specexec_approx` implementation; exact official tree-kernel fidelity is not claimed. |
| `dip_sd` / DiP-SD (Online Adaptation) | PASS | Paper-level batch-count, assignment, per-user draft-length optimization and optimizer-controlled event simulation are implemented inside the online epoch-barrier framework. |
| Shared correctness | PASS | Canonical methods now have token-accounting, resource-overlap, event-time, greedy-equality, and no-pending-state invariant tests. |
| Legacy aliases | PASS | `sync_batch_sd`, `SpecEdge`, and `server_only` now warn and strictly map to `dip_sd`, `specedge_tree`, and `server_only_tree`; generated detail artifacts use canonical labels. |

Remaining engineering work is concentrated in optional true multi-request
server-only batching and any future exact upstream SpecEdge tree-kernel replay.

## 2. Contract-to-Code Matrix

| Contract item | Code files and key functions/classes | Actual behavior | Tests reviewed | Verdict |
| --- | --- | --- | --- | --- |
| Target-only greedy reference | `src/methods.py::get_method_spec`; `src/simulator.py::Simulator._on_target_only_arrive_edge`; `src/model_runner.py::ModelRunner.target_only`; `src/latency.py::target_only_latency_ms` | Calls target greedy generation once per request, emits `target_only_service`, uses a serialized target service timeline, no draft/verify/network events. | `tests/test_target_only.py`, `tests/test_target_only_capacity.py`, `tests/test_semantic_simulator.py` | PASS |
| Server-only linear SD | `src/methods.py::get_method_spec("server_only_linear")`; `src/simulator.py::_start_server_only_draft`, `_start_server_only_verify`, `_on_server_only_verify_done`; `src/config.py::validate_config` | Uses `server_draft_gpu` then `server_target_gpu`, no network/proactive path, processes one request per verification round, records trace `batch_size: 1`, and rejects `server_only.batch_size > 1`. | `tests/test_server_only_linear.py`, legacy coverage in `tests/test_specedge_methods.py` | PASS for main default |
| Server-only tree SD | Same server-only simulator path; `src/tree_drafting.py::SpecExecDraftTreeStrategy`; `src/model_runner.py::DraftCandidateTree`, `verify_tree`, `verify_tree_batch` | Uses real tree objects and tree verification through the `specexec_approx` path under the fixed batch-size-1 server-only scheduler; rejects unsupported larger batch sizes. | `tests/test_server_only_tree.py`, `tests/test_specedge_methods.py` | PASS for main default |
| SpecEdge linear | `src/methods.py::get_method_spec("specedge_linear")`; `src/simulator.py::_maybe_start_batch`, `_start_proactive_draft`, `_resolve_proactive_after_accept`, `_resolve_verification` | Edge draft, uplink, server batch verify, downlink, proactive continuation. Retains proactive suffix only after accepted path and bonus/root alignment; failed proactive state is wasted, not committed. | `tests/test_specedge_linear.py`, `tests/test_baseline_system_invariants.py` | PASS |
| SpecEdge tree | `src/tree_drafting.py`, `src/model_runner.py::draft_tree`, `draft_bonus_tree`, `verify_tree_batch`; `src/simulator.py::_resolve_proactive_after_accept` | Real tree candidates, tree attention mask verification, proactive subtree generation/rebase under the local `specexec_approx` strategy. | `tests/test_specedge_tree.py`, `tests/test_baseline_system_invariants.py` | PASS with `specexec_approx` caveat |
| DiP-SD (Online Adaptation) | `src/simulator.py::_run_dip_sd`; `src/dip_sd.py::optimize_dip_sd`, `solve_assignment_subproblem`, `solve_draft_length_subproblem`, `evaluate_plan` | Uses the paper variables, exact bounded subproblem solvers, Dinkelbach-equivalent objective updates, and optimizer-controlled event execution inside the online epoch-barrier framework. Static/greedy public paths are removed. | `tests/test_dip_sd.py`, `tests/test_baseline_system_invariants.py` | PASS |
| Decode-only / no prefill | `README.md`; `scripts/verify_baseline_rebuild.sh`; `src/`, `configs/`, `tests/` grep check | Decode-ready scope is documented and static check rejects prefill execution identifiers outside markdown. | `scripts/verify_baseline_rebuild.sh`, `tests/test_decode_only_initialization.py` | PASS |
| Legacy aliases | `src/methods.py::SUPPORTED_METHODS`, `LEGACY_METHOD_ALIASES`, `get_method_spec`; `scripts/run_all.py` detail output naming | `sync_batch_sd`, `SpecEdge`, and `server_only` remain accepted, emit visible `FutureWarning`s, resolve to canonical specs, and write canonical detail labels. | `tests/test_legacy_aliases.py`, `tests/test_specedge_methods.py`, `tests/test_cli_smoke.py`, `tests/test_sync_batch_barrier.py` | PASS |

## 3. Target-only Audit

Status: PASS.

Code path:

- `src/methods.py::get_method_spec("target_only")` returns runtime
  `target_only`, no drafter lanes, no global batch, and no candidate strategy.
- `src/simulator.py::Simulator.run` uses the event simulator path, not the
  DiP-SD path.
- `src/simulator.py::_on_target_only_arrive_edge` calls
  `model_runner.target_only(request.prompt_ids, request.output_len)`, computes
  virtual time with `target_only_latency_ms`, records one `target_only_service`
  event, then schedules request finish.
- `README.md` documents `target_only` as a decode-ready autoregressive baseline
  with no communication.

Actual behavior:

- Only target greedy decoding is used.
- No draft, verification, batch, uplink, downlink, proactive, or rollback event
  is produced by the target-only path.
- Multiple requests share the serialized target-only service timeline through
  `_target_only_available_ms`.
- Output is directly the target model's greedy generation.

Existing tests:

- `tests/test_target_only.py` checks registration, greedy output, absence of
  draft/verify/batch/network events, FCFS serialization, decode-ready behavior,
  and no speculative counters.
- `tests/test_target_only_capacity.py` checks shared edge capacity behavior.

Issues:

- None blocking. The target-only resource is modeled as one serialized service;
  that matches the current README/contract baseline.

Paper impact:

- Ready as the reference denominator for latency speedup and greedy equality.

## 4. Server-only-Linear Audit

Status: PASS for the main `server_only.batch_size=1` contract.

Code path:

- `src/methods.py::get_method_spec("server_only_linear")` selects runtime
  `server_only_specedge` and `candidate_strategy="linear"`.
- `src/simulator.py::_on_server_only_arrive_edge` enqueues requests.
- `src/simulator.py::_maybe_start_server_only_request` starts a request only if
  `_server_only_active_request_id is None`.
- `src/simulator.py::_start_server_only_draft` creates a linear segment, records
  `resource: "server_draft_gpu"`, and records trace `batch_size: 1`.
- `src/simulator.py::_start_server_only_verify` verifies exactly one segment via
  `_verify_segment(segment)` and `_verify_latency_for_segments([segment])`,
  records `resource: "server_target_gpu"`, and records trace `batch_size: 1`.

Actual behavior:

- Drafter and target are represented as independent server resources.
- No endpoint network communication is produced.
- No proactive drafting path is used.
- For a single request, the loop is synchronized:
  `draft -> verify -> state update -> next draft`.
- Each active request has at most one unverified linear segment.
- `server_only.batch_size` is fixed at `1` for main experiments. Values larger
  than 1 are rejected by config validation and simulator initialization because
  the current server-only runtime does not yet implement real multi-request
  verification batches.

Existing tests:

- `tests/test_server_only_linear.py` checks method registration, default
  `batch_size == 1`, linear
  interfaces, no network/proactive events, draft-then-verify round order, and
  output equality with target-only.
- `tests/test_specedge_methods.py` contains legacy server-only tests and even
  asserts one request lifecycle at a time.

Issues:

- True multi-request server-only batching is not implemented. It is an optional
  extension, not part of the current main-experiment contract.
- Unsupported `server_only.batch_size > 1` is now rejected instead of being
  accepted and silently executed as single-request service.

Paper impact:

- Usable for final main experiments under the documented batch-size-1
  server-only contract.
- Do not claim throughput scaling with `server_only.batch_size > 1` unless the
  optional multi-request verification extension is implemented and tested.

## 5. Server-only-Tree Audit

Status: PASS for the main `server_only.batch_size=1` contract.

Code path:

- `src/methods.py::get_method_spec("server_only_tree")` selects
  `candidate_strategy="tree"`.
- `src/simulator.py::_server_only_tree_plan` uses the server-only tree strategy.
- `src/tree_drafting.py::SpecExecDraftTreeStrategy` computes a
  SpecExec-inspired tree plan with `max_n_beams`, `max_beam_len`,
  `max_branch_width`, and `max_budget`.
- `src/model_runner.py::DraftCandidateTree` carries node, parent, primary path,
  and tree-count metadata.
- `src/model_runner.py::verify_tree` and `verify_tree_batch` execute tree
  verification.

Actual behavior:

- Deployment and synchronization are the same as Server-only-Linear.
- Linear/tree selection changes candidate structure and verification interface.
- Tree candidates are real tree structures, not just linear tokens wrapped in a
  different name.
- Target tree verification is real.
- Beam, depth, branch width, and budget limits are represented and enforced by
  the local tree strategy/builder.

Existing tests:

- `tests/test_server_only_tree.py` checks method registration, configured
  budget use, forced tree strategy/interface behavior, no network/proactive
  events, and target-only output equality.
- `tests/test_specedge_methods.py` covers older server-only tree-depth and
  budget behavior.

Issues:

- Inherits the server-only batch-size-1 main contract. Larger batch sizes are
  rejected until true multi-request server-only verification is implemented.
- `SpecExecDraftTreeStrategy` is explicitly documented as
  "SpecExec-inspired analytical tree budget, not a strict upstream replay."

Paper impact:

- Usable for final main experiments under the documented batch-size-1
  `specexec_approx` tree-candidate contract.
- Do not claim official multi-request server-only batching or exact upstream
  SpecEdge tree construction without implementing and testing those optional
  extensions.

## 6. SpecEdge-Linear Audit

Status: PARTIAL.

Code path:

- `src/methods.py::get_method_spec("specedge_linear")` selects runtime
  `specedge`, `global_batch=True`, `batch_timeout=True`, and
  `candidate_strategy="linear"`.
- `src/simulator.py::_maybe_start_batch` forms server validation batches from
  queued segments and calls `_verify_segments(segments)`.
- `src/simulator.py::_start_proactive_draft` starts proactive continuation from
  `segment.prefix_ids + segment.draft_ids` for linear proactive drafting.
- `src/simulator.py::_resolve_verification` records accepted/rejected state,
  schedules downlink result arrival, and clears/invalidates on rejection.
- `src/simulator.py::_resolve_proactive_after_accept` reuses proactive suffix
  only when the first proactive token matches the server bonus token.

Actual behavior:

- Requests stay bound to their origin client device.
- The client can draft proactively while waiting for server verification and
  downlink.
- The server can verify multiple ready requests in one `global_batch_verify`
  event.
- Dynamic batching drains currently buffered ready requests when the server
  flush runs. Static batching waits for the configured batch condition unless
  forced by timeout/end-of-events.
- Verified results, not proactive drafts, drive committed output.
- On proactive hit, only the suffix after the bonus token is retained.
- On proactive miss/rejection, optimistic proactive state is cleared.

Existing tests:

- `tests/test_specedge_linear.py` checks registration, batching in a symmetric
  two-request dynamic case, static wait behavior, proactive activity, proactive
  discard on failure, no early commit, and output equality.

Issues:

- The alignment check is token/prefix based; there is no explicit source-leaf id
  for linear candidates. That is acceptable for linear paths but still needs an
  adversarial trajectory test for stale optimistic state and duplicate-token
  edge cases.
- Dynamic batching behavior can depend on ready-event/flush ordering at the
  same timestamp; the policy needs an explicit test.

Paper impact:

- Usable for small trace experiments.
- Needs trajectory tests before final official-comparison results.

## 7. SpecEdge-Tree Audit

Status: PARTIAL.

Code path:

- `src/methods.py::get_method_spec("specedge_tree")` selects
  `candidate_strategy="tree"`.
- `src/tree_drafting.py::SpecExecDraftTreeStrategy` creates the local tree plan.
- `src/model_runner.py::draft_tree` and `draft_bonus_tree` construct primary and
  proactive trees.
- `src/model_runner.py::verify_tree_batch` builds tree attention masks,
  position ids, and batched target inputs.
- `src/simulator.py::_resolve_proactive_after_accept` checks tree proactive
  prefix alignment and calls `rebase_draft_tree` for retained suffix/subtree.

Actual behavior:

- Tree candidates include node, parent, depth/path, and primary-path metadata.
- Target verification uses tree attention mask semantics rather than linear
  sequence verification.
- Proactive subtree generation is real.
- On accepted-path switch, rejection, or prefix mismatch, proactive state is
  invalidated.
- Server batching can verify multiple tree requests in one `global_batch_verify`
  event through `_verify_segments`.

Existing tests:

- `tests/test_specedge_tree.py` checks tree strategy use, batched tree verify
  equivalence, proactive tree start/reuse/discard, and output equality.

Issues:

- The tree policy is a local SpecExec-style approximation, not proven identical
  to the official SpecEdge implementation.
- No test covers duplicate-token branches, wrong source leaf with matching root
  token, or stale deeper proactive subtree reuse.

Paper impact:

- Usable for small tree/proactive trace experiments if labeled as approximate.
- Not ready for exact official SpecEdge tree-comparison claims without stronger
  source-leaf and upstream-fidelity tests.

## 8. DiP-SD (Online Adaptation) Audit

Status: PASS after M18.

Code path:

- `src/methods.py::get_method_spec("dip_sd")` selects runtime `dip_sd` and
  linear candidates.
- `src/simulator.py::Simulator.run` dispatches `runtime == "dip_sd"` to
  `_run_dip_sd`.
- `src/simulator.py::_run_dip_sd` builds a `DipSDProblem` for the current active
  epoch, calls `optimize_dip_sd`, drafts according to `plan.draft_lengths`, and
  verifies according to `plan.batches`.
- `src/dip_sd.py::optimize_dip_sd` scans feasible batch counts, solves exact
  bounded assignment and draft-length subproblems, and uses a
  Dinkelbach-equivalent fractional objective for `R=U/S`.
- Static/greedy substitute functions and public method names are removed from
  code paths. `dip_sd.optimizer` must be `paper_exact`.

Actual behavior against the DiP-SD (Online Adaptation) boundary:

| Requirement | Verdict | Evidence |
| --- | --- | --- |
| Candidates are linear segments | PASS | DiP-SD (Online Adaptation) segments set `tree_strategy="linear"` and use `model_runner.draft`. |
| One unverified segment per request per round | PASS | Each active request appears once in the optimizer plan; redraft waits for result arrival and the next epoch barrier. |
| Request waits for verification/result update before next round | PASS | `request_ready_ms` is set to result arrival time and draft start is bounded by the epoch start. |
| Batches are non-empty and visited in deterministic order | PASS | Optimizer validation rejects empty batches; simulator iterates ordered `plan.batches`. |
| Batch waits for all members ready | PASS | Runtime batch verify waits for max member edge-arrival time. |
| Slow device blocks its batch | PASS | Covered by `test_dip_sd_slow_member_blocks_assigned_batch`. |
| Other batches can draft while one batch verifies | PASS | Covered by `test_dip_sd_other_batches_overlap_drafting`. |
| New arrivals join only at safe epoch/cycle boundary | PASS | Waiting requests are admitted only at epoch starts. |
| Reads future acceptance | PASS | Optimizer input has no future-acceptance field and uses prior/past estimator values. |
| Acceptance estimator uses offline or past information | PASS | Cold start uses configured priors; after verification, observed accepted/proposed counts update the causal estimator. |
| Batch-count optimization | PASS | `optimize_dip_sd` scans feasible `N` values. |
| User-to-batch assignment optimization | PASS | `solve_assignment_subproblem` solves fixed-length assignment exactly for bounded active cohorts. |
| Per-user draft-length optimization | PASS | `solve_draft_length_subproblem` solves bounded integer lengths with Dinkelbach-equivalent updates. |
| Objective and pipeline span model match the paper | PASS | `evaluate_plan` computes `U`, `S`, ready times, verify times, stage durations, and memory usage. |
| Dinkelbach/fractional programming or equivalent exact solver | PASS | Bounded exact enumeration uses the Dinkelbach score `U - qS` and convergence tolerance. |
| Method name `dip_sd` reserved for complete method | PASS | `dip_sd_greedy`, `dip_sd_static`, and `dip_sd_heuristic` are not registered. |
| Output equals target-only greedy | PASS | Covered for accepting and rejecting fake runners. |

Remaining caveats:

- The simulator is the DiP-SD core optimizer and synchronized batch pipeline
  adapted to this project's online epoch-barrier request admission framework.
- Trace span validation compares optimizer `S` to ordered verification-stage
  span; full epoch wall-clock includes warm-up/drain and barrier overhead.
- Hardware cost coefficients are analytical configuration parameters and should
  be calibrated before large-scale real-model claims.

Paper impact:

- `dip_sd` is suitable for experiments when labeled as
  `DiP-SD (Online Adaptation)` under the documented analytical latency model
  and online epoch-barrier adaptation.

## 9. Shared Correctness Audit

| Check | Verdict | Code/tests | Issue and paper impact |
| --- | --- | --- | --- |
| Decode-only preserved | PASS | `README.md`, `scripts/verify_baseline_rebuild.sh`, `tests/test_decode_only_initialization.py` | Static grep rejects prefill identifiers outside markdown. No blocking issue found. |
| No restored prefill computation/communication/metrics | PASS | `scripts/verify_baseline_rebuild.sh` | The script excludes markdown and scans `src configs tests`. |
| Greedy equality with target-only | PARTIAL | Method-specific equality tests in `tests/test_*` | Covered for selected fake/deterministic cases, not exhaustive across randomized acceptance paths. |
| Token accounting conservation | PARTIAL | `src/metrics.py`, segment/request counters | No invariant test proves drafted/verified/accepted/committed/wasted conservation for all methods. |
| Resource intervals without illegal overlap | PARTIAL | Simulator availability timestamps and trace events | No generic interval validator checks all resources and allowed concurrency. |
| Event time monotonicity | PARTIAL | Event queue and trace timestamps | No global monotonicity/causality validator. |
| Unified request completion | PARTIAL | `src/simulator.py` finish events/status | Covered indirectly by simulator finishing; not audited by a shared invariant test. |
| Bonus/correction consistency | PARTIAL | `src/model_runner.py::VerificationResult`; `_resolve_verification` | Basic behavior exists; cross-method accounting tests are missing. |
| Legacy aliases only map to new implementation | FAIL | `src/methods.py`, `src/metrics.py` | Legacy aliases are accepted but not strict canonical aliases. Blocks clean final result interpretation. |
| Unsupported server-only batch size rejected | PASS | `src/config.py`, `src/simulator.py`, `tests/test_server_only_linear.py`, `tests/test_server_only_tree.py` | `server_only.batch_size > 1` is an optional extension and is rejected today. |
| Tests and implementation sharing wrong assumption | PARTIAL | Current tests | Remaining risk is concentrated in proactive alignment edge cases and broader invariant checks. |

## 10. Missing Trace Tests

### Test A: Optional Server-only Real Batch Extension

Purpose: if future work enables `server_only.batch_size > 1`, prove it is a real
target batch, not two single-request verifications.

Setup:

1. Two requests arrive at the same time.
2. `server_only.batch_size = 2`.
3. Fixed gamma and deterministic acceptance.
4. Run both `server_only_linear` and `server_only_tree`.

Expected assertions:

- One trace event represents a server target verify batch containing both
  request ids or segment ids.
- That event has `batch_size == 2`.
- The trace must not contain two independent one-request verify events presented
  as a logical batch.
- Outputs equal target-only greedy.

Expected current result: configuration/runtime rejection before simulation.

### Test B: SpecEdge Proactive Alignment

Purpose: prove proactive state is reused only when accepted path and bonus token
connect exactly to the proactive continuation.

Scenario 1, success:

1. Build a fake runner where the accepted path reaches the proactive source.
2. Server bonus token equals proactive root token.
3. Proactive continuation contains a non-empty suffix/subtree.

Expected: bonus token is committed only after server verification, and only the
valid suffix/subtree is retained.

Scenario 2, failure:

1. Accepted path differs from the proactive source, or
2. Bonus/correction token differs from the proactive root token, or
3. Tree branch switches after proactive drafting starts.

Expected: all optimistic proactive state is cleared, no unverified token is
committed, and final output remains target-only greedy.

Current result: PASS. Covered by
`test_specedge_linear_proactive_alignment_success`,
`test_specedge_linear_proactive_alignment_failure`,
`test_specedge_tree_proactive_alignment_success`,
`test_specedge_tree_proactive_alignment_failure`, and
`test_specedge_never_commits_unverified_proactive_tokens`.

### Test C: DiP-SD Pipeline Barrier

Purpose: prove the synchronized pipeline timing required before validating the
full optimizer.

Setup:

1. Two fixed batches.
2. Two requests per batch.
3. One device is much slower.
4. Optimizer-controlled per-request draft length.

Expected assertions:

- The slow device blocks its own batch.
- The other batch can continue drafting while the slow batch is waiting or while
  server verification is running.
- Server verifies batches in fixed order.
- The same request cannot start the next round before receiving the current
  verification result and updating state.
- Trace shows real batch-to-batch draft/verify overlap.

Current result: PASS. Covered by
`test_dip_sd_trace_uses_optimizer_assignment`,
`test_dip_sd_trace_uses_per_request_draft_length`,
`test_dip_sd_slow_member_blocks_own_batch`,
`test_dip_sd_other_batch_can_continue_drafting`,
`test_dip_sd_verification_follows_batch_order`,
`test_dip_sd_request_waits_for_verify_and_kv_update`,
`test_dip_sd_online_arrival_waits_for_epoch_boundary`, and
`test_dip_sd_one_unverified_segment_per_request`.

### Test D: DiP-SD (Online Adaptation) Optimizer Correctness

Purpose: prove the canonical `dip_sd` planner matches the paper-level optimizer
semantics on small solvable cases.

Setup:

1. Construct a tiny cohort with known device draft latencies, verify latency,
   and offline acceptance estimates.
2. Enumerate all feasible batch counts, assignments, and per-user draft lengths
   as an independent oracle for the tiny case.
3. Compare the implemented `dip_sd` plan to the oracle optimum.
4. Include at least one case where greedy load assignment is suboptimal.

Expected current result: PASS after M16/M17. Covered by
`test_dip_sd_optimizer_matches_bruteforce_on_tiny_cases` and trace tests proving
the simulator executes the optimizer plan.

## 11. Required Fixes

1. Keep server-only `server_only.batch_size > 1` rejected until true
   multi-request verification is implemented and covered by Test A.

## 12. Recommended Fixes

1. Record richer trace fields: request ids in server-only verify batches,
   accepted path ids, bonus/correction token, proactive source leaf, retained
   suffix length, and proactive subtree rebase information.
2. Make SpecEdge dynamic batching policy deterministic and test same-timestamp
   ready requests.
3. Label tree planning explicitly as `specexec_approx` in manifests and plots
   whenever exact upstream replay is not guaranteed.
4. Add randomized fake-runner tests with mixed accept/reject/bonus paths across
   all lossless methods.
5. Add config tests that fail when a user-set parameter is accepted but ignored.

## 13. Accepted Limitations

- Decode-only scope is acceptable: prompt prefill, initial prompt transfer, and
  initial KV establishment remain out of simulation time.
- Server-only `batch_size = 1` is the main-experiment baseline setting.
- Server-only `batch_size > 1` is an optional extension and is currently
  rejected.
- Static, greedy, or heuristic DiP-SD substitutes are not accepted baselines.
  The public method set now exposes only canonical `dip_sd`.
- Configured/offline acceptance priors are acceptable as long as no future target
  outcomes are read.
- `specexec_approx` tree construction is acceptable for small experiments if it
  is not presented as exact official SpecEdge replay.

## 14. Experiment Readiness

| Readiness level | Methods | Conditions |
| --- | --- | --- |
| Can run unit tests | All canonical baselines | Current suite includes deterministic event-semantics invariants. |
| Can run small trace experiments | `target_only`, `dip_sd`, `server_only_linear`, `server_only_tree`, `specedge_linear`, `specedge_tree` | Server-only remains fixed at `batch_size=1`; tree methods must be labeled `specexec_approx`. |
| Can run real-model smoke trials | `target_only`, `dip_sd`, batch-size-1 server-only baselines, SpecEdge linear/tree | Use small samples first; inspect traces, not only aggregate metrics. |
| Can enter main paper experiments under declared settings | `target_only`, `dip_sd`, `server_only_linear`, `server_only_tree`, `specedge_linear`, `specedge_tree` | Server-only uses `batch_size=1`; tree results are `specexec_approx`, not exact official tree-kernel replay. |
| Temporarily cannot configure | Server-only with `batch_size > 1` | Larger server-only batches are rejected until the optional extension is implemented. |

## Final Findings

Must fix:

- Keep Server-only `batch_size > 1` rejected until real batching is implemented.

Recommended:

- Make dynamic batching timing deterministic.
- Label approximate tree construction clearly.

Acceptable:

- No prefill.
- No DiP-SD substitute baseline is accepted; public `dip_sd` is
  `DiP-SD (Online Adaptation)`.
- Tree approximation for small experiments when labeled.

Small-scale ready:

- `target_only`
- `dip_sd` / `DiP-SD (Online Adaptation)`
- `server_only_linear` / `server_only_tree` with batch size 1
- `specedge_linear`
- `specedge_tree` under the documented `specexec_approx` caveat

Currently prohibited / not final-result labels:

- Server-only `batch_size > 1`
- Legacy aliases as formal result labels; explicit alias invocations are
  redirected to canonical labels with visible warnings.
- Claims of exact official SpecEdge tree-kernel replay.
