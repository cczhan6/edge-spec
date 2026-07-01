# Proposed Async Speculative Decoding Design

## Scope

This design adds one independent proposed method named `async_speculative`,
without changing the semantics of the six canonical
baselines. It follows `docs/method/async_speculative_decoding_method.md` in two
stages:

- Stage A implements fixed-gamma, fixed-lookahead multi-channel asynchronous
  verification and establishes lossless output semantics.
- Stage B replaces only the fixed drafting policy with heterogeneity-aware
  dynamic drafting.

The implementation remains decode-only. It does not introduce an in-flight
token limit, rollback controller, task preemption, verifier microbatching,
channel optimizer, or physical target-KV synchronization. Existing legacy
methods named `full`, `wo_async`, `wo_scheduling`, and
`conservative_rollback` are not the proposed method and are not reused as its
semantic definition.

## Audit Findings and Reusable Code

### Directly reusable

- `methods.py`: `MethodSpec` and `get_method_spec` are the method-selection
  entry point. The proposed method needs a new canonical name and runtime, but
  the six baseline branches remain unchanged.
- `events.py`: the ordered event heap and existing request, draft, packet,
  verification-completion, result-arrival, and request-finish events provide
  the outer simulation clock. Proposed-specific payloads may be added, but a
  second event loop is unnecessary.
- `simulator.py`: workload arrival, fixed request-to-device mapping,
  per-device non-concurrent drafting, analytical edge compute, network delay,
  trace collection, request finalization, and `SimulationResult` assembly are
  reusable through a thin runtime adapter.
- `model_runner.py`: linear `draft`, `verify`, `target_only`,
  `VerificationResult`, and `_verify_linear_candidate` already implement the
  required greedy target semantics for accepted tokens, correction, bonus,
  and EOS. Stage A must use the single-request `verify` interface, never
  `verify_batch`.
- `latency.py`: `draft_latency_ms`, `TargetLatencyModel` and the existing
  network/edge-compute models remain authoritative. A proposed verification
  job queries linear verification latency with its complete contiguous
  verification input length; it must not use the current
  `Simulator.predict_verify_latency_ms`, which ignores `gamma` and always
  passes one analytical work unit.
- `metrics.py`: existing main, request, segment, device, and event writers are
  retained. Proposed-only fields should be additive so baseline schemas and
  calculations do not change.

### Reusable only after isolation

`Request`, `Segment`, `VerifierLane`, and the existing async helper methods mix
logical commitment, transport, lane assignment, proactive drafting, token
budgeting, and baseline behavior. In particular, the current async path:

- assigns each arriving segment to a lane-local FIFO through
  `LaneScheduler`, so it cannot enforce one global current-first queue;
- permits out-of-order verification but resolves results against
  `edge_frontier_pos`, with bonus retargeting specialized to one immediate
  successor;
- uses `unconfirmed_token_budget`, which is explicitly outside this design;
- updates a token acceptance-ratio window rather than continuous accepted
  length EWMA; and
- stores enough state in `Simulator` that extending it directly would further
  couple the proposed method to baseline code.

The new runtime may reuse common entity fields for output compatibility, but
its dependency graph, queue ownership, cached verification result, and
controller statistics must be separate proposed-method state.

## Modules and State

### `async_verification.py`

This module owns Stage A server semantics. `AsyncVerificationCoordinator`
maintains:

- one global waiting set, not per-lane queues;
- `M` identical `VerificationChannelState` records containing only channel
  identity, active job, busy-until time, and accounting;
- one `AsyncRequestState` per request, containing the confirmed target prefix,
  current segment index `k`, ordered segment records, completed-result buffer,
  path generation, and terminal flag; and
- immutable `VerificationJob` records containing request/segment identity,
  path generation, dependency interval, arrival sequence, contiguous verify
  prefix, segment-local token range, and status.

A path generation is an opaque monotonically increasing integer. Every job
captures it at creation. Rejection, a non-reusable bonus, or EOS increments
the generation and invalidates every descendant on the old path in constant
logical time; queued and completed jobs are removed, while active jobs keep
their channel until completion and then discard their result. This is path
invalidation, not a rollback controller.

The coordinator exposes narrow operations: register a drafted segment, mark
its packet arrived, dispatch into all idle channels, consume a verification
completion, drain newly committable results, and invalidate a request path.
It returns declarative actions to the simulator adapter (schedule completion,
send result, request another draft, finish request) rather than mutating the
global event heap itself.

### `dynamic_drafting.py`

Stage A uses `FixedDraftingPolicy(gamma, lookahead_depth)`. Stage B swaps in
`DynamicDraftingPolicy` without changing the verification coordinator. The
module owns:

- candidate set `Gamma`, including `1`, and `gamma_start`;
- `L_max_ver`;
- ready-time EWMA keyed by device and candidate gamma;
- continuous accepted-length EWMA keyed by request;
- server-level normalized successor completion-rate estimator `mu_suc`; and
- pure decisions for candidate construction, gamma selection, target depth,
  and whether drafting may start.

The device still generates at most one segment at a time. Drafting policies
receive snapshots of request, device, network, and verification state; they do
not inspect future acceptance outcomes.

### Entity integration

Keep baseline `Request` and `Segment` fields intact. Add proposed-only state as
composition owned by the two modules. A proposed segment needs stable
`segment_index`, `path_generation`, dependency start/end positions,
`verify_input_ids`, and segment-local offsets. These fields must not be
overloaded onto `base_pos` after bonus trimming: `base_pos` remains the
original trace identity, while the coordinator records the current unconsumed
range separately.

`Simulator` should only select the proposed runtime, translate common events
to coordinator calls, apply returned actions, and mirror finalized values into
existing trace entities. Baseline dispatch paths do not call either new
module.

## Events and State Transitions

### Draft and arrival

For a request's first segment, Stage A uses fixed `gamma`; Stage B uses
`gamma_start`. A successor is drafted from the prompt, confirmed output, and
all earlier tokens on the current predicted path. On draft completion it
enters transport; on packet arrival it becomes globally waiting if its path
generation is still current.

The fixed Stage A depth is the number of unfinished successor segments in
`drafting`, `in_transit`, `waiting`, or `verifying`. It excludes the current
segment and completed results waiting for ordered commit. The policy starts a
new successor whenever this count is below `lookahead_depth`, subject only to
remaining output length, EOS already present on the speculative path, device
availability, and the contiguous verification limit. No token-budget cap is
applied.

### Dispatch

Whenever a packet arrives or a channel becomes idle, repeatedly select the
minimum priority job until no idle channel or waiting job remains. Priority is

```text
(is_successor, dependency_distance, arrival_time, arrival_sequence)
```

where `is_successor=0` exactly when the segment is request `k`. Thus all
waiting current jobs strictly precede every successor job. Successors are
ordered by `segment_index-k`, then arrival time; a monotonic arrival sequence
makes equal-time behavior deterministic. Channel identity is not optimized:
the lowest-id idle channel receives the selected job. An active job is never
preempted and each channel executes exactly one job with batch size one.

Before dispatch, the coordinator constructs a contiguous verification call
from the request's confirmed target prefix through the end of the selected
segment. The total unconfirmed portion must be at most `L_max_ver`. The
returned target decisions are projected onto the selected segment's local
range and cached together with enough leading overlap to validate later bonus
trimming. A job that no longer fits after the frontier changes is invalidated
and re-created from the current path; cached target decisions are never
silently interpreted under a different prefix.

### Completion, ordered commit, and transport

Verification may complete in any order. A valid completion moves to the
request result buffer. Only the result for `k` may change the confirmed
frontier; after it commits, drain consecutive buffered successors while their
generation and dependency prefix remain valid.

For each committed segment:

1. commit its consecutive matching draft tokens;
2. on the first mismatch, commit exactly the target correction token and
   invalidate all descendants;
3. on full acceptance, commit the target bonus token unless it is beyond the
   requested output length;
4. if that bonus equals the first unconsumed token of the immediate successor,
   consume that token from the successor and validate the cached result after
   the trim; otherwise invalidate that successor and all descendants; and
5. stop immediately when committed output reaches the requested length or
   contains target EOS, then invalidate every remaining job and segment.

The coordinator advances the logical server-confirmed prefix at this point.
Existing downlink delay remains user-visible: newly committed tokens are sent
as one ordered result action, and `Request.generated_ids` plus token timestamps
advance only when that result arrives at the device. Result-arrival events for
one request must retain commit order. Device drafting decisions that require a
new confirmed prefix are re-evaluated after that ordered arrival.

If an invalidated job is waiting or completed, discard it immediately. If it
is active, mark it invalid, let its scheduled completion release the channel,
record wasted verification time, and do not expose its result to commitment or
EWMA updates.

## Configuration

Use a dedicated top-level block so existing `speculation` fields retain their
baseline meaning:

```yaml
async_speculative:
  mode: fixed
  num_channels: 4
  gamma_fixed: 4
  lookahead_depth_fixed: 2
  gamma_candidates: [1, 2, 4, 6, 8]
  gamma_start: 1
  l_max_ver: 16
  ready_time_ewma_beta: 0.8
  accepted_length_ewma_beta: 0.8
  mu_suc_ewma_beta: 0.8
  mu_suc_window_ms: 1000
  mu_suc_min_completions: 1
```

`num_channels` is the proposed method's `M`; it is deliberately not inferred
from a baseline batching setting. `gamma_fixed` and
`lookahead_depth_fixed` are Stage A controls. Stage B requires a unique,
positive, sorted `gamma_candidates` containing `1`; `gamma_start` must be a
member; and `l_max_ver >= 1`. EWMA betas are in `[0,1)`, the observation window
is positive, and the minimum completion count is positive. Stage selection is
an explicit proposed-method mode (`fixed` or `dynamic`), not a different
method name.

### Stage B decisions

`T_ready(request, gamma)` is measured from actual draft start until the packet
becomes verification-queue eligible, so it includes device compute and the
realized uplink delay/blocking for that segment. Per-device/per-gamma EWMAs are
initialized from analytical draft plus expected configured uplink latency;
missing candidate observations may use that analytical estimate.

The accepted-length sample is the number of consecutive draft tokens accepted
from the segment's unconsumed start. Update it only for a segment committed on
the valid path; invalidated speculative work contributes no sample. Initialize
the request EWMA from an offline drafter statistic when one is explicitly
configured, otherwise from `gamma_start`.

Build the candidate set exactly as the method document specifies:

```text
gamma_cap = min(max(Gamma), max(1, floor(accepted_length_ewma) + 1))
candidates = {g in Gamma |
              g <= gamma_cap and existing_unconfirmed + g <= L_max_ver}
```

Before `mu_suc` has `mu_suc_min_completions` valid samples, choose
`min(gamma_start, largest feasible candidate)`; if no candidate exists, pause.
After warm-up, let `service_interval = 1 / mu_suc`. Among candidates with
`T_ready <= service_interval`, choose the one with largest `T_ready`, breaking
ties toward larger gamma. If none is on time, choose the smallest `T_ready`,
breaking ties toward smaller gamma.

The rate observation counts valid successor verification completions in the
sliding window and divides by the integral of the number of requests having at
least one unfinished successor. The EWMA is updated only when that exposure is
positive; otherwise it is held. Units are completions per millisecond per
active request. For the selected gamma,

```text
d_star = max(1, ceil(mu_suc * T_ready(selected_gamma)))
```

and drafting continues while the unfinished-successor count is below
`d_star`. Completed results waiting for ordered commit are excluded from the
count, matching the method definition.

## Stage A Acceptance Criteria

Stage A is accepted only when all of the following hold:

1. `async_speculative` is independently selectable and adding it changes no
   canonical baseline `MethodSpec`, trace, metric, or output tokens.
2. Configured fixed gamma and fixed successor depth are honored without an
   in-flight token cap; every device has at most one active draft.
3. Across all waiting jobs, every current segment is dispatched before every
   successor; successor ordering is dependency distance, arrival time, then a
   deterministic tie-breaker.
4. Exactly `M` non-preemptive channels exist, every verification event has
   batch size one, and invalidated active jobs occupy their channels until
   their original completion event.
5. Tests force successors to finish before predecessors and demonstrate that
   output commits strictly in dependency order.
6. Dedicated cases cover rejection at every draft position, correction token,
   full acceptance plus bonus, reusable and non-reusable bonus, EOS as an
   accepted token/correction/bonus, output-length truncation, waiting/completed
   invalidation, and active-job late discard.
7. Every case compares the complete per-request token-id sequence with
   `target_only`; randomized deterministic-runner cases cover multiple
   requests, channels, gammas, and network delays.
8. At request finish there is no committable result or live proposed state;
   active invalidated work may only remain as already scheduled cleanup events
   and cannot alter output or statistics after completion.
9. Trace/accounting distinguishes useful and invalidated verification time,
   records channel occupancy and queue priority inputs, and all existing
   metric writers can serialize the result without special-casing baselines.

Stage B is not a prerequisite for Stage A acceptance. Its controller tests
must later establish candidate filtering, cold start, one-sided on-time
selection, EWMA update eligibility, normalized-rate units, and the exact
`d_star` formula without using future outcomes.

## Method-Document Ambiguities Resolved by This Design

- **Fixed lookahead depth:** the method document defines dynamic depth but not
  Stage A's fixed-depth counting. This design counts unfinished successors,
  excludes the current segment and completed buffered results, and imposes no
  token cap.
- **Global current-job tie-breaking:** the document defines successor ordering
  but leaves ties among current jobs implicit. Arrival time and monotonic
  arrival sequence apply to both classes.
- **Continuous verification result reuse:** segment-local results alone are
  insufficient to prove validity after predecessor decisions. Jobs therefore
  retain dependency identity and overlap evidence; otherwise they are
  re-created rather than reused.
- **Commit location versus result transport:** the method says results enter a
  request buffer but does not distinguish server confirmation from device
  visibility. Logical ordering is resolved at the server; existing downlink
  events determine user-visible token times and device controller updates.
- **Bonus and EOS:** bonus reuse is allowed only for the immediate successor's
  first unconsumed token and only when cached dependencies still match. EOS in
  any committed role terminates the request and invalidates descendants.
- **`L_max_ver` near the frontier:** it limits the contiguous unconfirmed
  verification input, not prompt or confirmed-prefix length. No new segment is
  drafted when even one token would exceed it.
- **Ready time:** it is start-to-queue-eligibility and includes realized
  drafting and uplink behavior. Downlink is excluded because it does not make
  a verification job ready.
- **`mu_suc` completion eligibility:** only completed successor jobs valid at
  completion count; current jobs and invalidated work do not. Active-request
  exposure uses the same unfinished-successor definition as depth.
- **Cold start and tie-breaking:** the method leaves stable-observation and
  equal-time choices open. This design makes both configurable/deterministic
  and uses `gamma_start` until the minimum valid sample count is reached.
- **Acceptance initialization:** device `acceptance_prior` is a token ratio,
  not a continuous accepted-length estimate, so it cannot initialize the new
  EWMA without an explicit conversion. The default is `gamma_start`.

These resolutions preserve the method's two intended control variables while
avoiding the excluded controllers and resource mechanisms.
