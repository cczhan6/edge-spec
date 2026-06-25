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

DiP-SD audit boundary: this repository is expected to implement the original
paper's complete DiP-SD method. A static or heuristic synchronized pipeline is
not an acceptable public baseline substitute, and `dip_sd` must refer to the
paper method.

## 1. Executive Summary

| Area | Verdict | Paper-experiment impact |
| --- | --- | --- |
| `target_only` | PASS | Ready as the greedy reference and speedup denominator. |
| `server_only_linear` | PARTIAL | Ready only for batch-size-1 traces. Blocks any final result claiming server-only multi-request batching. |
| `server_only_tree` | PARTIAL | Real tree candidate/verify path exists, but server-only batching gap blocks final batched claims. |
| `specedge_linear` | PARTIAL | Main proactive and server batch mechanics exist. Needs stronger trajectory tests before final official-comparison claims. |
| `specedge_tree` | PARTIAL | Real tree construction and verification exist. Upstream tree/proactive alignment fidelity remains approximate. |
| Full `dip_sd` | FAIL | Current code implements a static/heuristic synchronized pipeline, not the paper's full joint batch-count, grouping, and per-user draft-length optimization. |
| Shared correctness | PARTIAL | Greedy equality is tested on selected paths, but token-accounting and resource-interval invariants are not independently validated. |
| Legacy aliases | FAIL | Legacy aliases still carry old semantics/config paths instead of being thin mappings to canonical baselines. |

Must-fix items are concentrated in three places: true server-only batching,
complete DiP-SD implementation, and legacy alias behavior.

## 2. Contract-to-Code Matrix

| Contract item | Code files and key functions/classes | Actual behavior | Tests reviewed | Verdict |
| --- | --- | --- | --- | --- |
| Target-only greedy reference | `src/methods.py::get_method_spec`; `src/simulator.py::Simulator._on_target_only_arrive_edge`; `src/model_runner.py::ModelRunner.target_only`; `src/latency.py::target_only_latency_ms` | Calls target greedy generation once per request, emits `target_only_service`, uses a serialized target service timeline, no draft/verify/network events. | `tests/test_target_only.py`, `tests/test_target_only_capacity.py`, `tests/test_semantic_simulator.py` | PASS |
| Server-only linear SD | `src/methods.py::get_method_spec("server_only_linear")`; `src/simulator.py::_start_server_only_draft`, `_start_server_only_verify`, `_on_server_only_verify_done` | Uses `server_draft_gpu` then `server_target_gpu`, no network/proactive path, but processes one active request lifecycle and hardcodes trace `batch_size: 1`. | `tests/test_server_only_linear.py`, legacy coverage in `tests/test_specedge_methods.py` | PARTIAL |
| Server-only tree SD | Same server-only simulator path; `src/tree_drafting.py::SpecExecDraftTreeStrategy`; `src/model_runner.py::DraftCandidateTree`, `verify_tree`, `verify_tree_batch` | Uses real tree objects and tree verification, but inherits the single-active-request server-only scheduler. | `tests/test_server_only_tree.py`, `tests/test_specedge_methods.py` | PARTIAL |
| SpecEdge linear | `src/methods.py::get_method_spec("specedge_linear")`; `src/simulator.py::_maybe_start_batch`, `_start_proactive_draft`, `_resolve_proactive_after_accept`, `_resolve_verification` | Edge draft, uplink, server batch verify, downlink, proactive continuation. Conservative commit path, but exact proactive source identity is not exhaustively tested. | `tests/test_specedge_linear.py` | PARTIAL |
| SpecEdge tree | `src/tree_drafting.py`, `src/model_runner.py::draft_tree`, `draft_bonus_tree`, `verify_tree_batch`; `src/simulator.py::_resolve_proactive_after_accept` | Real tree candidates, tree attention mask verification, proactive subtree generation/rebase. Strategy is documented as SpecExec-inspired approximation. | `tests/test_specedge_tree.py` | PARTIAL |
| Full DiP-SD | `src/simulator.py::_run_dip_sd_greedy`; `src/dip_sd.py::build_fixed_epoch_plan`, `optimize_epoch_plan` | Current code still exposes/uses static and heuristic planning paths. Neither is the original paper method. | `tests/test_dip_sd.py` | FAIL |
| Decode-only / no prefill | `README.md`; `scripts/verify_baseline_rebuild.sh`; `src/`, `configs/`, `tests/` grep check | Decode-ready scope is documented and static check rejects prefill execution identifiers outside markdown. | `scripts/verify_baseline_rebuild.sh`, `tests/test_decode_only_initialization.py` | PASS |
| Legacy aliases | `src/methods.py::SUPPORTED_METHODS`, `get_method_spec`; `src/metrics.py` baseline fallback logic | `sync_batch_sd`, `SpecEdge`, and `server_only` are accepted, but not proven to be thin aliases to canonical rebuilt methods. | `tests/test_specedge_methods.py`, `tests/test_cli_smoke.py` | FAIL |

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

Status: PARTIAL.

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
- `server_only.batch_size` exists in `configs/default.yaml`, but values larger
  than 1 do not create a multi-request verify batch. The simulator serializes
  one request lifecycle at a time.

Existing tests:

- `tests/test_server_only_linear.py` checks method registration, linear
  interfaces, no network/proactive events, draft-then-verify round order, and
  output equality with target-only.
- `tests/test_specedge_methods.py` contains legacy server-only tests and even
  asserts one request lifecycle at a time.

Issues:

- FAIL for true batching: `batch_size > 1` is accepted by configuration but is
  operationally ineffective for server-only verification.
- Existing tests do not include the required two-request batch trajectory.

Paper impact:

- Usable for batch-size-1 server-only traces.
- Not usable for paper claims involving server-only batched verification,
  throughput scaling with `batch_size`, or official server-only batching
  fidelity.

## 5. Server-only-Tree Audit

Status: PARTIAL.

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

- Inherits the server-only `batch_size > 1` failure.
- `SpecExecDraftTreeStrategy` is explicitly documented as
  "SpecExec-inspired analytical tree budget, not a strict upstream replay."

Paper impact:

- Usable for batch-size-1 tree-candidate experiments.
- Not ready for final claims about official server-only batched tree
  verification or exact upstream SpecEdge tree construction.

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

## 8. Full DiP-SD Audit

Status: FAIL.

Code path:

- `src/methods.py::get_method_spec("dip_sd_greedy")` selects runtime `dip_sd`
  and linear candidates.
- `src/methods.py::get_method_spec("dip_sd")` also selects runtime `dip_sd`.
- `src/simulator.py::Simulator.run` dispatches all `runtime == "dip_sd"` methods
  to `_run_dip_sd_greedy`.
- `src/simulator.py::_run_dip_sd_greedy` uses `build_fixed_epoch_plan` for
  `dip_sd_greedy`, but uses `optimize_epoch_plan` when `self.spec.name ==
  "dip_sd"`.
- `src/dip_sd.py::build_fixed_epoch_plan` builds fixed cyclic batches and fixed
  draft lengths from explicit config.
- `src/dip_sd.py::optimize_epoch_plan` is a deterministic heuristic extension
  using configured acceptance priors. It is not a reproduction of the paper's
  optimizer.

Actual behavior against the full DiP-SD boundary:

| Requirement | Verdict | Evidence |
| --- | --- | --- |
| Candidates are linear segments | PASS | DiP-SD segments set `tree_strategy="linear"` and use `model_runner.draft`. |
| One unverified segment per request per round | PASS | Each active request appears once in the epoch plan; next draft uses `request_ready_ms`. |
| Request waits for verification/result update before next round | PASS | `request_ready_ms` is set to result arrival time; epoch end waits for result arrivals. |
| Batches are non-empty and visited in deterministic order | PASS | The simulator iterates `for batch_index, batch in enumerate(plan.batches)`. |
| Batch waits for all members ready | PASS | `batch_ready_ms = max(segment.edge_arrival_time_ms for segment in segments)`. |
| Slow device blocks its batch | PASS | The max edge-arrival time gates the batch verify start. |
| Other batches can draft while one batch verifies | PARTIAL | Draft timestamps are computed independently from request readiness, so traces can overlap with server verification. A dedicated slow-device test is still missing. |
| New arrivals join only at safe epoch/cycle boundary | PASS | Waiting requests are admitted at the start of epochs, not mid-batch. |
| Reads future acceptance | PASS | The heuristic uses configured profile priors, not future target outcomes. |
| Acceptance estimator uses offline or past information | PARTIAL | Configured profile priors are oracle-free, but there is no implemented estimator from offline samples or past observed acceptance. |
| Batch-count optimization | PARTIAL | `optimize_epoch_plan` scans up to configured `batch_count`, but it is a bounded heuristic search, not the paper's full optimization loop. |
| User-to-batch assignment optimization | FAIL | Assignment is a greedy load-balancing heuristic, not the paper's association optimization. |
| Per-user draft-length optimization | PARTIAL | Draft lengths are searched within configured bounds, but the subproblem is not the paper's Dinkelbach/MILP-equivalent formulation. |
| Objective and pipeline span model match the paper | FAIL | The span model in `src/dip_sd.py` is simplified and does not model the paper's complete latency/objective terms. |
| Dinkelbach/fractional programming or equivalent exact solver | FAIL | No such solver or equivalent exact subproblem exists. |
| Method name `dip_sd` reserved for complete method | FAIL | The canonical `dip_sd` name currently executes the heuristic path. |
| Output equals target-only greedy | PARTIAL | Covered by selected tests, but not by broad trajectory/randomized invariants. |

Existing tests:

- `tests/test_dip_sd.py` checks registration, fixed cyclic order, optimizer
  deterministic behavior, sync before redraft, epoch-barrier admission, one
  unverified segment per request, and output equality.
- The tests do not check optimizer optimality, association correctness, objective
  fidelity, Dinkelbach/MILP-equivalent behavior, or paper-level convergence.

Issues:

- The current implementation is not full DiP-SD.
- The existing static/heuristic path should be replaced by the original paper
  method and should not remain a public baseline.
- The canonical `dip_sd` name must mean the complete paper method.
- The required full implementation needs batch-count search, user-to-batch
  association optimization, per-user draft-length optimization, a paper-faithful
  objective/span model, and offline/past acceptance estimation.
- Missing trace test for two batches, one slow device, and visible batch-to-batch
  pipeline overlap.

Paper impact:

- Current DiP-SD-related methods are not ready for paper formal results.
- No static/heuristic DiP-SD substitute should be reported as a baseline.

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
| Config accepted but operationally ineffective | FAIL for server-only batch | `configs/default.yaml`, `src/simulator.py::_start_server_only_verify` | `server_only.batch_size > 1` does not produce true batch verification. |
| Tests and implementation sharing wrong assumption | PARTIAL | Current tests | Risk is real for server-only batching, proactive alignment edge cases, and DiP-SD optimizer fidelity. |

## 10. Missing Trace Tests

### Test A: Server-only Real Batch

Purpose: prove `server_only.batch_size = 2` is a real target batch, not two
single-request verifications.

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

Expected current result: FAIL.

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

Expected current result: basic cases likely pass; duplicate-token/wrong-source
cases are not currently proven.

### Test C: DiP-SD Pipeline Barrier

Purpose: prove the synchronized pipeline timing required before validating the
full optimizer.

Setup:

1. Two fixed batches.
2. Two requests per batch.
3. One device is much slower.
4. Fixed draft length.

Expected assertions:

- The slow device blocks its own batch.
- The other batch can continue drafting while the slow batch is waiting or while
  server verification is running.
- Server verifies batches in fixed order.
- The same request cannot start the next round before receiving the current
  verification result and updating state.
- Trace shows real batch-to-batch draft/verify overlap.

Expected current result: likely PARTIAL/PASS, but not currently proven by a
dedicated trace test.

### Test D: Full DiP-SD Optimizer Correctness

Purpose: prove the canonical `dip_sd` planner matches the paper-level optimizer
semantics on small solvable cases.

Setup:

1. Construct a tiny cohort with known device draft latencies, verify latency,
   and offline acceptance estimates.
2. Enumerate all feasible batch counts, assignments, and per-user draft lengths
   as an independent oracle for the tiny case.
3. Compare the implemented `dip_sd` plan to the oracle optimum.
4. Include at least one case where greedy load assignment is suboptimal.

Expected current result: FAIL.

## 11. Required Fixes

1. Implement true multi-request server-only batch verification for
   `server_only_linear` and `server_only_tree`, or document `batch_size` as
   currently limited to 1 and remove larger values from valid final experiments.
2. Implement the complete DiP-SD planner for canonical `dip_sd`: batch-count
   search, user-to-batch association optimization, per-user draft-length
   optimization, paper-faithful objective/span model, and offline/past acceptance
   estimation.
3. Remove the public `dip_sd_greedy` / static substitute path from final method
   registration, or make it internal-only while developing the original paper
   method.
4. Make legacy aliases strict mappings to canonical implementations, or exclude
   them from final experiment method sets.
5. Add the missing trajectory/optimizer tests above.
6. Add shared invariant validators for token accounting, resource interval
   overlap, event causality, and target-only greedy equality.

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
- Server-only `batch_size = 1` is acceptable as a correctness-only baseline.
- A static DiP-SD-style pipeline is not an accepted baseline for this project.
  Replace it with the original paper method.
- Configured/offline acceptance priors are acceptable as long as no future target
  outcomes are read.
- `specexec_approx` tree construction is acceptable for small experiments if it
  is not presented as exact official SpecEdge replay.

## 14. Experiment Readiness

| Readiness level | Methods | Conditions |
| --- | --- | --- |
| Can run unit tests | All canonical baselines | Current suite can be run, but passing tests are not sufficient for semantic closure. |
| Can run small trace experiments | `target_only`, `server_only_linear` with batch size 1, `server_only_tree` with batch size 1, `specedge_linear`, `specedge_tree` | Report caveats in manifests and analysis. |
| Can run real-model smoke trials | `target_only`, batch-size-1 server-only baselines, SpecEdge linear/tree | Use small samples first; inspect traces, not only aggregate metrics. |
| Can enter paper formal experiments | `target_only` now; other baselines only after required fixes/tests | Server-only batching, SpecEdge proactive alignment, and full DiP-SD optimizer must be resolved first. |
| Temporarily cannot use | Server-only with `batch_size > 1`; legacy aliases for final results; current `dip_sd` as full DiP-SD | These would misstate the implemented semantics. |

## Final Findings

Must fix:

- Server-only real batching is missing.
- Full DiP-SD is not implemented; current `dip_sd` is heuristic/static-like.
- Legacy aliases are not clean canonical aliases.
- Required trajectory and shared invariant tests are missing.

Recommended:

- Strengthen SpecEdge proactive alignment trace fields and tests.
- Make dynamic batching timing deterministic.
- Label approximate tree construction clearly.

Acceptable:

- No prefill.
- No DiP-SD substitute baseline is accepted; implement the original paper method.
- Tree approximation for small experiments when labeled.

Small-scale ready:

- `target_only`
- `server_only_linear` / `server_only_tree` only with batch size 1
- `specedge_linear` / `specedge_tree` with alignment caveats
- No DiP-SD variant is currently small-scale ready under the required original
  paper-method scope

Not ready for final paper results:

- Server-only `batch_size > 1`
- Legacy aliases
- Current `dip_sd` as the full DiP-SD paper baseline
- SpecEdge final official-comparison claims before missing trajectory tests pass
