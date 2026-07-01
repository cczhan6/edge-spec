# Proposed Async Speculative Decoding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the independent `async_speculative` method in two gated stages while preserving greedy target output and all six canonical baseline semantics.

**Architecture:** `src/async_verification.py` owns the global verification queue, channels, dependency state, ordered confirmation, and invalidation. `src/dynamic_drafting.py` owns fixed and dynamic drafting policies. `Simulator` remains the event-loop adapter; existing model, latency, communication, edge-compute, and metric interfaces are reused.

**Tech Stack:** Python 3, dataclasses, heap-based discrete events, pytest/unittest, existing `FakeModelRunner` and CSV trace writers.

---

## File Map

- Create `src/async_verification.py`: proposed-method verification state and pure transitions.
- Create `src/dynamic_drafting.py`: Stage A fixed policy and Stage B dynamic controller/statistics.
- Modify `src/methods.py`, `src/config.py`, `configs/default.yaml`: independent method and configuration contract.
- Modify `src/events.py`, `src/simulator.py`: proposed-only event adaptation and lifecycle.
- Modify `src/entities.py`, `src/metrics.py`: additive proposed trace/accounting fields only.
- Create focused `tests/test_async_*.py` and `tests/test_dynamic_drafting.py`; do not repurpose legacy async tests as the proposed contract.

Every task follows RED → verify RED → minimal GREEN → targeted regression → commit. Run commands from `/edge-spec-proposed/code`.

**Execution checklist:**

- [ ] A1 method/config contract
- [ ] A2 global priority queue
- [ ] A3 channel semantics
- [ ] A4 contiguous verification input
- [ ] A5 ordered confirmation
- [ ] A6 correction/bonus/invalidation
- [ ] A7 EOS and terminal drain
- [ ] A8 fixed-runtime integration
- [ ] A9 server/device lifecycle
- [ ] A10 Stage A accounting/equivalence gate
- [ ] B1 dynamic config contract
- [ ] B2 candidate filtering
- [ ] B3 ready/accepted EWMAs
- [ ] B4 successor service rate
- [ ] B5 gamma/depth decision
- [ ] B6 dynamic integration/final gate

## Stage A — Fixed Gamma and Fixed Lookahead

### Task A1: Register the method and fixed-mode configuration

**Files:**
- Modify: `src/methods.py`
- Modify: `src/config.py`
- Modify: `configs/default.yaml`
- Create: `tests/test_async_speculative_config.py`

**First failing test:**

```python
def test_async_speculative_is_independent_and_fixed_config_is_valid():
    config = load_config("configs/default.yaml")
    spec = get_method_spec("async_speculative", config)
    assert spec.runtime == "async_speculative"
    assert spec.name == "async_speculative"
    assert "async_speculative" not in DEFAULT_METHODS
    validate_config(config)
```

**Minimal implementation:** Add `async_speculative` to `SUPPORTED_METHODS`, return a distinct runtime with no batching or baseline alias, and add/validate only Stage A keys: `mode=fixed`, positive `num_channels`, `gamma_fixed`, `lookahead_depth_fixed`, and `l_max_ver >= gamma_fixed`. Preserve all existing method branches byte-for-byte apart from the new branch.

```python
if name == "async_speculative":
    return MethodSpec(name, "async_speculative", "heterogeneous", 0,
                      int(config["async_speculative"]["num_channels"]),
                      False, "global_priority", "generation")
```

**Test command:** `pytest -q tests/test_async_speculative_config.py tests/test_config.py tests/test_legacy_aliases.py`

**Acceptance:** Invalid fixed values fail with key-specific messages; canonical method specs and `DEFAULT_METHODS` are unchanged.

**Commit:** `feat: register fixed async speculative method`

### Task A2: Define proposed state and global priority ordering

**Files:**
- Create: `src/async_verification.py`
- Create: `tests/test_async_verification_queue.py`

**First failing test:**

```python
def test_current_jobs_strictly_precede_successors_then_distance_and_arrival():
    coordinator = coordinator_with_requests(frontiers={0: 2, 1: 4})
    coordinator.enqueue(job(0, 4, arrival=1.0))
    coordinator.enqueue(job(1, 4, arrival=9.0))
    coordinator.enqueue(job(0, 3, arrival=2.0))
    assert [j.key for j in coordinator.pop_waiting(3)] == [(1, 4), (0, 3), (0, 4)]
```

**Minimal implementation:** Define `JobStatus`, `VerificationJob`, `VerificationChannelState`, `AsyncSegmentState`, `AsyncRequestState`, and `AsyncVerificationCoordinator`. Compute priority on pop as `(segment_index != k, segment_index-k, arrival_time_ms, arrival_sequence)` so a job becoming current is re-ranked without lane-local queues.

```python
def priority(self, job: VerificationJob) -> tuple[int, int, float, int]:
    k = self.requests[job.request_id].current_segment_index
    return (int(job.segment_index != k), job.segment_index - k,
            job.arrival_time_ms, job.arrival_sequence)
```

**Test command:** `pytest -q tests/test_async_verification_queue.py`

**Acceptance:** Current-first is global across requests; successor ties are deterministic; no `LaneScheduler` dependency exists.

**Commit:** `feat: add async verification priority queue`

### Task A3: Add mutually exclusive non-preemptive channels

**Files:**
- Modify: `src/async_verification.py`
- Create: `tests/test_async_verification_channels.py`

**First failing test:**

```python
def test_busy_channel_is_not_preempted_by_later_current_job():
    c = coordinator(num_channels=1)
    first = c.dispatch_one(now_ms=0.0)
    c.enqueue(current_job(arrival=1.0))
    assert c.dispatch_all(now_ms=1.0) == []
    assert c.channels[0].active_job_id == first.job_id
```

**Minimal implementation:** Dispatch repeatedly to the lowest-id idle channel, mark one active job per channel, reject duplicate completion, and release only from the matching completion event. Invalidation changes the active job to stale but never clears the channel early.

```python
def complete_channel(self, channel_id: int, job_id: int, now_ms: float):
    channel = self.channels[channel_id]
    if channel.active_job_id != job_id:
        raise ValueError("verification completion does not match active channel")
    channel.active_job_id = None
    channel.busy_until_ms = now_ms
```

**Test command:** `pytest -q tests/test_async_verification_channels.py tests/test_async_verification_queue.py`

**Acceptance:** At most `M` jobs run, each channel has one job, current arrivals cannot preempt, and invalid active jobs release only at original completion.

**Commit:** `feat: enforce async verification channel semantics`

### Task A4: Build bounded contiguous verification inputs

**Files:**
- Modify: `src/async_verification.py`
- Create: `tests/test_async_verification_input.py`

**First failing test:**

```python
def test_successor_verifies_contiguous_path_from_server_prefix():
    state = request_state(server_confirmed=[10], segments=[[11, 12], [13, 14]])
    verify = state.build_verify_input(segment_index=1, l_max_ver=4)
    assert verify.prefix_ids == [10]
    assert verify.draft_ids == [11, 12, 13, 14]
    assert verify.local_slice == slice(2, 4)
```

**Minimal implementation:** Add immutable `ContiguousVerifyInput` carrying confirmed prefix, all unconfirmed tokens through the selected segment, dependency fingerprint, and local slice. Return no dispatchable input when unconfirmed length exceeds `l_max_ver`; never truncate dependencies.

```python
@dataclass(frozen=True)
class ContiguousVerifyInput:
    prefix_ids: tuple[int, ...]
    draft_ids: tuple[int, ...]
    local_start: int
    local_end: int
    dependency_fingerprint: tuple[int, ...]
```

**Test command:** `pytest -q tests/test_async_verification_input.py`

**Acceptance:** Current and successor inputs are contiguous, local offsets are stable, and over-limit jobs remain undispatched rather than partially verified.

**Commit:** `feat: build contiguous async verification inputs`

### Task A5: Buffer out-of-order completions and confirm in order

**Files:**
- Modify: `src/async_verification.py`
- Create: `tests/test_async_verification_commit.py`

**First failing test:**

```python
def test_successor_completion_waits_for_current_then_drains_in_order():
    c = accepting_two_segment_coordinator()
    assert c.complete(job_id=2, result=accepted([3], bonus=4)) == []
    actions = c.complete(job_id=1, result=accepted([1], bonus=2))
    assert flatten_confirmed(actions) == [1, 2, 3, 4]
    assert c.request(0).server_confirmed_ids == [1, 2, 3, 4]
```

**Minimal implementation:** Store completed results by segment index, drain only from `k`, validate generation and dependency fingerprint, and emit ordered `SendResultAction` objects. Keep `server_confirmed_ids` internal; do not mutate `Request.generated_ids` here.

```python
while state.current_segment_index in state.completed_results:
    result = state.completed_results.pop(state.current_segment_index)
    actions.extend(self._confirm_current(state, result))
```

**Test command:** `pytest -q tests/test_async_verification_commit.py tests/test_async_verification_input.py`

**Acceptance:** Completion order cannot change token order; buffered successors drain only after every dependency is confirmed.

**Commit:** `feat: confirm async verification results in order`

### Task A6: Handle rejection, correction, bonus trimming, and path invalidation

**Files:**
- Modify: `src/async_verification.py`
- Modify: `tests/test_async_verification_commit.py`
- Create: `tests/test_async_verification_invalidation.py`

**First failing test:**

```python
@pytest.mark.parametrize("status", ["waiting", "completed", "active"])
def test_rejection_commits_correction_and_invalidates_descendants(status):
    c = coordinator_with_descendants(status=status)
    actions = c.complete_current(rejected(accepted_count=1, correction=99))
    assert c.request(0).server_confirmed_ids[-2:] == [accepted_token(), 99]
    assert descendants(c, 0).all_invalid
    assert active_channel_remains_busy_if_applicable(c, status)
```

**Minimal implementation:** On rejection increment path generation, commit accepted prefix plus exactly one correction, delete waiting/completed descendants, and mark active descendants invalid. On full acceptance, trim the immediate successor only when bonus equals its first unconsumed token and its cached dependency evidence remains valid; otherwise invalidate all descendants.

```python
def invalidate_descendants(self, request_id: int, after_index: int) -> None:
    state = self.requests[request_id]
    state.path_generation += 1
    # remove waiting/completed; mark active jobs invalid without releasing channels
```

**Test command:** `pytest -q tests/test_async_verification_commit.py tests/test_async_verification_invalidation.py`

**Acceptance:** Every rejection position, correction, reusable/non-reusable bonus, empty absorbed successor, and late stale completion has a focused passing test.

**Commit:** `feat: handle async correction and path invalidation`

### Task A7: Add EOS and terminal drain semantics

**Files:**
- Modify: `src/async_verification.py`
- Create: `tests/test_async_verification_terminal.py`

**First failing test:**

```python
@pytest.mark.parametrize("role", ["accepted", "correction", "bonus"])
def test_eos_makes_request_terminal_but_active_jobs_finish(role):
    c = coordinator_with_eos(role=role, active_successor=True)
    actions = c.complete_current_with_eos()
    assert c.request(0).terminal
    assert not c.waiting_jobs_for(0)
    assert c.has_active_invalid_job(0)
    c.complete_active_invalid_job()
    assert c.all_channels_idle()
```

**Minimal implementation:** Terminalize on EOS or output length, cancel local/waiting/buffered work, forbid registration/arrival/dispatch for terminal requests, but retain active invalid jobs until completion. Count those completions as successor service and wasted verification; do not update accepted-length observations.

```python
if reached_limit or eos_token_id in newly_confirmed:
    state.terminal = True
    actions.extend(self._cancel_non_active_work(state))
```

**Test command:** `pytest -q tests/test_async_verification_terminal.py tests/test_async_verification_invalidation.py`

**Acceptance:** EOS in all three roles and length truncation terminate scheduling immediately; active cleanup still releases channels and emits no output.

**Commit:** `feat: drain active verification after async termination`

### Task A8: Integrate fixed drafting and coordinator events into Simulator

**Files:**
- Create: `src/dynamic_drafting.py`
- Modify: `src/events.py`
- Modify: `src/simulator.py`
- Modify: `src/entities.py`
- Create: `tests/test_async_speculative_simulator.py`

**First failing test:**

```python
def test_fixed_runtime_drafts_to_depth_and_uses_single_job_channels():
    config, runner, workload = async_config(gamma=2, depth=2, channels=2)
    result = Simulator(config, runner, workload, "homogeneous", "async_speculative").run()
    verify = [e for e in result.event_trace if e["event"] == "async_verify"]
    assert verify and all(e["batch_size"] == 1 for e in verify)
    assert max_successor_depth(result) == 2
```

**Minimal implementation:** Add `FixedDraftingPolicy`, proposed completion payloads, and a narrow Simulator adapter that translates draft/packet/verify/result events into coordinator actions. Use existing device serialization, network delay, `ModelRunner.verify`, and `TargetLatencyModel.linear_verification_latency_ms`; do not call legacy `_allows_out_of_order_verify`, token-budget, batch, proactive, or lane-scheduler paths.

```python
class FixedDraftingPolicy:
    def __init__(self, gamma: int, lookahead_depth: int):
        self.gamma = int(gamma)
        self.lookahead_depth = int(lookahead_depth)
    def select_gamma(self, snapshot) -> int: return min(self.gamma, snapshot.remaining)
    def should_draft(self, snapshot) -> bool:
        return snapshot.unfinished_successors < self.lookahead_depth
```

**Test command:** `pytest -q tests/test_async_speculative_simulator.py tests/test_event_order.py tests/test_lane_no_microbatch.py`

**Acceptance:** Fixed gamma/depth run end-to-end; Simulator only adapts actions; no in-flight token cap, batch, preemption, or fake KV behavior appears.

**Commit:** `feat: integrate fixed async speculative runtime`

### Task A9: Separate server confirmation, device visibility, and lifecycle completion

**Files:**
- Modify: `src/simulator.py`
- Modify: `tests/test_async_speculative_simulator.py`
- Create: `tests/test_async_speculative_lifecycle.py`

**First failing test:**

```python
def test_correction_is_not_a_draft_prefix_before_ordered_result_arrival():
    simulator = instrumented_rejecting_simulator(downlink_ms=50.0)
    result = simulator.run()
    correction = first_correction(result)
    assert all(correction not in e["prefix_ids"] for e in drafts_before_result(result))
    assert correction in first_draft_after_result(result)["prefix_ids"]
```

**Minimal implementation:** Keep coordinator `server_confirmed_ids` separate from `Request.generated_ids`; advance the latter only at ordered `RESULT_ARRIVE_DEVICE`. Set request finish time at final device arrival, but keep the event loop alive while any verification channel is active. Suppress draft uplink and packet enqueue after server terminalization.

```python
def _simulation_has_work(self) -> bool:
    return bool(self.events or self._batch_buffer or self._async.active_job_count)
```

**Test command:** `pytest -q tests/test_async_speculative_lifecycle.py tests/test_async_speculative_simulator.py`

**Acceptance:** Correction/bonus/EOS cannot enter device prefixes early; latency ends at final ordered downlink; resource time includes late active verification cleanup.

**Commit:** `feat: separate async confirmation and device visibility`

### Task A10: Add trace/accounting and Stage A equivalence gate

**Files:**
- Modify: `src/entities.py`
- Modify: `src/metrics.py`
- Modify: `src/simulator.py`
- Create: `tests/test_async_speculative_metrics.py`
- Create: `tests/test_async_speculative_equivalence.py`

**First failing test:**

```python
def test_invalid_active_successor_counts_service_and_waste_not_acceptance():
    result = run_invalid_active_successor_case()
    system = summarize(result, 1)[1]
    assert system["successor_verify_completions"] == 1
    assert system["wasted_verify_time_ms"] > 0
    assert committed_acceptance_samples(result) == []

@pytest.mark.parametrize("seed", range(10))
def test_async_tokens_equal_target_only(seed):
    assert token_ids(run("async_speculative", seed)) == token_ids(run("target_only", seed))
```

**Minimal implementation:** Add trace fields for segment index, path generation, priority inputs, channel, useful/invalid completion, and server/device confirmation times. Aggregate useful/wasted verification time and successor completions additively. Build deterministic equivalence cases for rejection positions, bonus, EOS, multiple requests/channels, and delayed networks; assert the six canonical method specs and deterministic traces remain unchanged.

```python
successor_verify_completions = sum(
    e["event"] == "async_verify_complete" and e["is_successor"]
    for e in result.event_trace
)
```

**Test command:** `pytest -q tests/test_async_speculative_metrics.py tests/test_async_speculative_equivalence.py tests/test_baseline_system_invariants.py tests/test_determinism.py`

**Acceptance:** All Stage A design criteria pass; complete token sequences equal `target_only`; baseline regression tests pass; traces distinguish useful, canceled-before-dispatch, and invalid-after-dispatch work.

**Commit:** `test: establish fixed async speculative acceptance gate`

## Stage A Gate

Do not begin Stage B until this command passes without warnings or skips:

```bash
pytest -q \
  tests/test_async_speculative_config.py \
  tests/test_async_verification_queue.py \
  tests/test_async_verification_channels.py \
  tests/test_async_verification_input.py \
  tests/test_async_verification_commit.py \
  tests/test_async_verification_invalidation.py \
  tests/test_async_verification_terminal.py \
  tests/test_async_speculative_simulator.py \
  tests/test_async_speculative_lifecycle.py \
  tests/test_async_speculative_metrics.py \
  tests/test_async_speculative_equivalence.py \
  tests/test_baseline_system_invariants.py \
  tests/test_determinism.py
```

Then run `pytest -q`. Acceptance requires both commands to pass and `git diff --exit-code` after committing Task A10.

## Stage B — Heterogeneity-Aware Dynamic Drafting

### Task B1: Add dynamic-mode configuration and candidate contract

**Files:**
- Modify: `src/config.py`
- Modify: `configs/default.yaml`
- Modify: `tests/test_async_speculative_config.py`
- Create: `tests/test_dynamic_drafting.py`

**First failing test:**

```python
def test_dynamic_candidates_are_sorted_unique_positive_and_include_one():
    config = dynamic_config(gamma_candidates=[2, 4])
    with pytest.raises(ValueError, match="must contain 1"):
        validate_config(config)
```

**Minimal implementation:** Validate `mode in {fixed,dynamic}`, sorted unique positive `gamma_candidates` containing `1`, member `gamma_start`, positive `l_max_ver`, betas in `[0,1)`, and positive `mu_suc_window_ms`/`mu_suc_min_completions`. Fixed-mode behavior remains unchanged.

```python
required_dynamic = ("gamma_candidates", "gamma_start", "ready_time_ewma_beta",
                    "accepted_length_ewma_beta", "mu_suc_ewma_beta",
                    "mu_suc_window_ms", "mu_suc_min_completions")
```

**Test command:** `pytest -q tests/test_async_speculative_config.py tests/test_dynamic_drafting.py tests/test_config.py`

**Acceptance:** Every invalid field has a focused failure; switching to `dynamic` does not create a second method name.

**Commit:** `feat: validate dynamic drafting configuration`

### Task B2: Implement gamma cap and verification-length candidate filtering

**Files:**
- Modify: `src/dynamic_drafting.py`
- Modify: `tests/test_dynamic_drafting.py`

**First failing test:**

```python
def test_candidates_apply_acceptance_cap_and_contiguous_limit():
    policy = dynamic_policy(gammas=[1, 2, 4, 6], accepted_ewma=2.2, l_max_ver=5)
    assert policy.candidates(existing_unconfirmed=2) == (1, 2)
    assert policy.candidates(existing_unconfirmed=5) == ()
```

**Minimal implementation:** Compute `gamma_cap=min(max_gamma,max(1,floor(a_bar)+1))`; filter by membership, cap, and `existing_unconfirmed + gamma <= l_max_ver`. Return an immutable ordered tuple and pause on empty candidates.

```python
cap = min(self.max_gamma, max(1, math.floor(accepted_ewma) + 1))
return tuple(g for g in self.gammas if g <= cap and existing + g <= self.l_max_ver)
```

**Test command:** `pytest -q tests/test_dynamic_drafting.py -k 'candidate or cap'`

**Acceptance:** `1` is available whenever any token fits; no hidden in-flight token limit is introduced.

**Commit:** `feat: filter dynamic drafting candidates`

### Task B3: Add ready-time and accepted-length EWMAs

**Files:**
- Modify: `src/dynamic_drafting.py`
- Modify: `tests/test_dynamic_drafting.py`

**First failing test:**

```python
def test_ready_and_acceptance_ewmas_use_distinct_eligible_samples():
    policy = dynamic_policy(ready_beta=.5, accepted_beta=.5)
    policy.observe_ready(device_id=2, gamma=4, elapsed_ms=20)
    policy.observe_commit(request_id=7, consecutive_accepted=2, valid_path=True)
    policy.observe_commit(request_id=7, consecutive_accepted=4, valid_path=False)
    assert policy.ready_ms(2, 4) == 20
    assert policy.accepted_length(7) == 2
```

**Minimal implementation:** Key ready EWMA by `(device_id,gamma)`, measuring draft-start through queue eligibility; initialize missing values from analytical draft plus expected uplink. Key accepted-length EWMA by request, initialize from configured offline length or `gamma_start`, and reject invalid/uncommitted samples.

```python
def ewma(old: float | None, sample: float, beta: float) -> float:
    return sample if old is None else beta * old + (1.0 - beta) * sample
```

**Test command:** `pytest -q tests/test_dynamic_drafting.py -k 'ready or accepted or ewma'`

**Acceptance:** Realized blocking affects ready time; invalid paths never affect accepted length; no acceptance-ratio prior is silently reused.

**Commit:** `feat: track dynamic drafting ewmas`

### Task B4: Implement normalized successor service-rate estimation

**Files:**
- Modify: `src/dynamic_drafting.py`
- Create: `tests/test_successor_service_rate.py`

**First failing test:**

```python
def test_mu_counts_valid_and_invalid_active_completions_not_canceled_waiters():
    rate = SuccessorServiceRate(window_ms=100, beta=0.0, min_completions=1)
    rate.observe_exposure(0, 100, active_requests=2)
    rate.observe_completion(40, dispatched=True, invalidated=False)
    rate.observe_completion(60, dispatched=True, invalidated=True)
    rate.observe_completion(70, dispatched=False, invalidated=True)
    assert rate.value == pytest.approx(2 / 200)
```

**Minimal implementation:** Maintain timestamped dispatched successor completions and piecewise active-request exposure. Count every completion event after channel occupancy, including active-invalidated jobs; exclude queue cancellations and current jobs. Hold the prior value on zero exposure and gate warm-up by observed dispatched completions.

```python
if event.is_successor and event.was_dispatched:
    self.completions.append(event.time_ms)
raw = completion_count / active_request_time_ms
```

**Test command:** `pytest -q tests/test_successor_service_rate.py`

**Acceptance:** Units are completions/ms/active-request; active invalid jobs remain in exposure until completion; window expiry and zero exposure are deterministic.

**Commit:** `feat: estimate normalized successor service rate`

### Task B5: Implement cold start, on-time gamma choice, and target depth

**Files:**
- Modify: `src/dynamic_drafting.py`
- Modify: `tests/test_dynamic_drafting.py`

**First failing test:**

```python
def test_selects_latest_on_time_candidate_and_computes_depth():
    policy = observed_policy(mu_suc=.02, ready={1: 10, 2: 40, 4: 70})
    decision = policy.decide(candidates=(1, 2, 4))
    assert decision.gamma == 2       # service interval is 50 ms
    assert decision.target_depth == 1
```

**Minimal implementation:** Before warm-up choose `min(gamma_start, largest feasible candidate)`. Afterwards choose maximum ready time not exceeding `1/mu_suc`, tie toward larger gamma; if none is on time choose minimum ready time, tie toward smaller gamma. Compute `d_star=max(1,ceil(mu_suc*T_ready))`.

```python
on_time = [g for g in candidates if ready[g] <= 1.0 / mu_suc]
gamma = max(on_time, key=lambda g: (ready[g], g)) if on_time else min(
    candidates, key=lambda g: (ready[g], g))
```

**Test command:** `pytest -q tests/test_dynamic_drafting.py -k 'cold or on_time or target_depth'`

**Acceptance:** Cold start, both selection branches, deterministic ties, `mu_suc=0`, empty candidates, and exact ceiling behavior pass.

**Commit:** `feat: select dynamic gamma and lookahead depth`

### Task B6: Integrate dynamic decisions and establish Stage B gate

**Files:**
- Modify: `src/simulator.py`
- Modify: `src/async_verification.py`
- Modify: `src/dynamic_drafting.py`
- Create: `tests/test_async_dynamic_integration.py`
- Modify: `tests/test_async_speculative_equivalence.py`

**First failing test:**

```python
def test_dynamic_depth_counts_only_unfinished_successors():
    result = run_dynamic_case_with_completed_buffer_and_active_invalid_job()
    decisions = dynamic_decisions(result)
    assert decisions[-1]["unfinished_successors"] == count_statuses(
        result, {"drafting", "in_transit", "waiting", "verifying"})
    assert decisions[-1]["completed_buffered"] not in decisions[-1]["depth_members"]
```

**Minimal implementation:** Select `DynamicDraftingPolicy` only for proposed `mode=dynamic`; feed queue-eligibility, valid commit, exposure, and every dispatched successor completion into the appropriate estimator. Re-evaluate after draft arrival, verification completion, confirmation/result arrival, and invalidation. Count drafting/in-transit/waiting/verifying successors, including active invalid work until completion, but exclude completed buffers.

```python
policy = build_drafting_policy(config["async_speculative"])
if snapshot.unfinished_successors < decision.target_depth:
    self._queue_async_draft(request, decision.gamma, now_ms)
```

**Test command:** `pytest -q tests/test_async_dynamic_integration.py tests/test_dynamic_drafting.py tests/test_successor_service_rate.py tests/test_async_speculative_equivalence.py`

**Acceptance:** Dynamic traces expose gamma, ready estimate, `mu_suc`, and `d_star`; invalid samples follow the design; fixed-mode Stage A tests remain green; dynamic token sequences equal `target_only`.

**Commit:** `feat: integrate heterogeneity aware dynamic drafting`

## Stage B and Final Gate

Run:

```bash
pytest -q \
  tests/test_dynamic_drafting.py \
  tests/test_successor_service_rate.py \
  tests/test_async_dynamic_integration.py \
  tests/test_async_speculative_equivalence.py
pytest -q
```

Acceptance requires both commands to pass, the Stage A gate to remain green,
and a review confirming no canonical baseline branch, batching semantics,
token budget, rollback controller, preemption, channel optimization, or KV
physical synchronization was added to the proposed runtime.
