# Dynamic Edge Heterogeneity Design

## Scope

Introduce a shared, deterministic resource model for dynamic edge-device
draft compute. Only the edge drafter token rate changes. Device startup
latency, drafter model and acceptance prior, server compute, target latency
profiles, network parameters, batching, and scheduler policy remain unchanged.

The model gives devices of the same configured type distinct initial draft
rates within that type's configured range. Each device maintains its own
completed-request count. After every fifth completion on that device, its
compute epoch advances and a new rate is sampled. The new rate applies only to
draft tasks that start after the completion transition; tasks already started
retain the rate captured at their start.

This document specifies the design only. It does not change production code,
tests, experiment results, or provide an implementation plan.

## Current Call-Chain Findings

### Device fields and construction

`src/entities.py` defines immutable `Device` records. Edge draft capacity is
currently represented by:

- `draft_token_rate_tok_s`;
- `draft_startup_ms`.

`src/config.py:build_devices` constructs every virtual device from the selected
`device_pools.<pool>.templates` mapping. It iterates templates in configuration
order and assigns `device_id=len(devices)`. The template key becomes
`device_type`; the template also supplies the drafter profile, fixed draft
rate, startup latency, and network fields.

`Simulator.__init__` selects the method's device pool through `MethodSpec`,
calls `build_devices`, and creates one `DeviceRuntime` per `Device`. Requests
are assigned in `_schedule_request_arrivals` by
`request_id % len(self.devices)`. Consequently:

- `device_id` comes from sequential device construction;
- the device category is the `device_type` copied from the template key;
- request-to-device assignment is fixed round-robin;
- the authoritative experiment seed is `simulation.seed`.

The same seed currently initializes the simulator RNG used for arrivals and
output lengths, is passed independently to workload sampling, and participates
in deterministic network jitter. Dynamic compute must not consume any of
those RNG streams.

### Draft latency and task starts

`src/latency.py:draft_latency_ms` is the current edge formula:

```text
draft_startup_ms + 1000 * work_units / draft_token_rate_tok_s
```

The work units are linear draft tokens or processed tree candidates depending
on the caller. Actual edge draft tasks start in three distinct paths:

| Runtime path | Task start | Current latency read |
| --- | --- | --- |
| async/full and ordinary SpecEdge draft | `Simulator._start_draft` | `draft_latency_ms(runtime.device, ...)` |
| SpecEdge proactive continuation | `Simulator._start_proactive_draft` | `draft_latency_ms(self.devices[device_id], ...)` |
| DiP-SD | the per-request draft block in `Simulator._run_dip_sd` | `draft_latency_ms(device, ...)` |

There are also non-task reads that must observe the current rate:

- `_select_gamma` estimates edge draft latency;
- `_specedge_edge_cycle_ms` estimates the pipeline edge cycle;
- `_build_dip_sd_problem` converts device rate into DiP-SD optimizer latency.

`_server_only_draft_latency_ms` is intentionally separate. It reads
`server_only.draft_token_rate_tok_s` and models a server draft resource, not an
edge device. Target-only has no draft task.

### Final request completion

Most runtimes schedule `REQUEST_FINISH`, whose handler
`Simulator._on_request_finish` writes `request.status="finished"` and
`finish_time_ms`, emits the finish trace, and reports progress. DiP-SD bypasses
that event loop and performs the same final state mutation inline in
`_run_dip_sd`.

The current code therefore has two final-state write locations, not one. A
completion-count resource transition cannot safely be attached only to
`_on_request_finish`, because DiP-SD would be omitted. The design must first
make one idempotent completion-commit operation authoritative for every
runtime.

## Considered Approaches

### Recommended: one simulator-owned edge compute model

Construct one `EdgeComputeModel` per `Simulator`. It owns immutable
configuration and mutable per-device state, exposes current capacity for
planning, produces immutable snapshots for task starts, and observes committed
request completions. All edge-draft methods share this object. Server-only and
target-only still construct it for a uniform lifecycle but do not request edge
draft snapshots.

This centralizes the resource semantics and makes it difficult for later
methods to introduce a private dynamic-compute interpretation.

### Rejected: mutate or replace `Device`

`Device` is frozen and is referenced by both `Simulator.devices` and
`DeviceRuntime.device`. Reconstructing it at each epoch risks stale references,
while making it mutable would mix static identity/network configuration with
runtime resource state. Either choice also makes an in-flight task's capacity
ambiguous.

### Rejected: pass a dynamic multiplier through simulator branches

Adding local multipliers around individual `draft_latency_ms` calls is a small
initial diff but scatters state and sampling across ordinary, proactive, and
DiP-SD paths. It is likely to miss planner reads or future baselines and gives
no single completion or snapshot contract.

## Recommended Components

### Static `Device`

Keep the existing `Device` fields and meanings unchanged. In particular,
`draft_token_rate_tok_s` remains the legacy fixed rate and the disabled-mode
fallback. Network fields and `draft_startup_ms` remain immutable.

### `EdgeComputeModel`

Add a focused resource component, owned by one simulator, with one state entry
per `device_id`:

```text
completed_requests: int = 0
epoch: int = 0
current_draft_token_rate_tok_s: float
```

Its conceptual interface is:

- `current_rate(device_id)`: rate used by planning and resource-aware latency
  estimation at the time of the call;
- `snapshot(device_id)`: immutable `(device_id, device_type, epoch,
  draft_token_rate_tok_s, draft_startup_ms)` captured when an actual draft task
  starts;
- `latency_ms(snapshot, work_units)`: apply the existing analytical formula to
  the captured values;
- `record_request_completion(device_id)`: idempotency is handled by the
  request completion layer; increment this device's count once and resample
  when the count is a multiple of the configured interval.

The model does not own queues, task timing, scheduling, requests, network
state, or server state. `DeviceRuntime` continues to own queue and utilization
statistics.

### Unified completion commit

Extract one idempotent simulator operation, conceptually
`_finalize_request(request, finish_time_ms)`. It is the only operation allowed
to change a request from `running` to `finished`. It performs, in order:

1. return without side effects if the request is already finished;
2. write final request status and time;
3. discard or clear outstanding request state as required by the existing
   runtime;
4. call `EdgeComputeModel.record_request_completion(request.device_id)`;
5. emit the existing finish trace and progress callback;
6. perform existing runtime-specific release behavior, such as advancing the
   server-only request queue.

`_on_request_finish` delegates to this operation. The DiP-SD inline completion
branch delegates to the same operation instead of writing final state itself.
This makes the completion counter method-independent and prevents duplicate
finish events from advancing an epoch twice.

## Configuration Design

Dynamic behavior is opt-in and global, while ranges are colocated with the
device templates whose categories they describe:

```yaml
dynamic_edge_compute:
  enabled: false
  resample_every_completed_requests: 5

device_pools:
  heterogeneous:
    templates:
      low_end:
        draft_token_rate_tok_s: 25
        dynamic_draft_token_rate_range_tok_s: [20, 30]
      mid_end:
        draft_token_rate_tok_s: 60
        dynamic_draft_token_rate_range_tok_s: [48, 72]
      high_end:
        draft_token_rate_tok_s: 100
        dynamic_draft_token_rate_range_tok_s: [80, 120]
  medium_only:
    templates:
      medium:
        draft_token_rate_tok_s: 60
        dynamic_draft_token_rate_range_tok_s: [48, 72]
```

The range is closed for validation purposes; the deterministic mapping may
produce the lower bound and remains below the upper bound. The proposed
defaults use a symmetric 20% interval around the existing fixed rates. A
scenario may override the ranges through the existing deep-merge mechanism.

Validation rules are:

- `enabled` must be a boolean;
- `resample_every_completed_requests` must be the integer `5`; other intervals
  are outside this design and fail validation;
- when enabled, every template with a positive count in the selected validated
  pools must define exactly two finite positive numbers with `min < max`;
- the legacy fixed rate remains required and positive in every mode;
- when disabled, dynamic ranges are not used to construct state or calculate
  latency. Invalid present ranges should still fail validation so configuration
  mistakes are not latent, but absent ranges are allowed.

The enabled flag belongs outside scenario or method configuration so a run
cannot accidentally give one baseline a different resource model. Method
specifications receive no dynamic-compute option.

## Deterministic Sampling

Sampling is a pure function of the experiment seed and device state. It does
not use Python's process-randomized `hash`, the simulator's `_rng`, workload
sampling RNG, or network jitter state.

For namespace version `edge-compute-v1`, form:

```text
key = "edge-compute-v1:{seed}:{device_id}:{device_type}:{epoch}"
ratio = uint64_be(SHA-256(key)[0:8]) / 2**64
rate = lower + ratio * (upper - lower)
```

Epoch zero supplies the initial rate during simulator construction. Devices
are initialized in ascending `device_id` order. If an epoch-zero value exactly
matches an earlier device of the same type, the sampler deterministically adds
an `attempt` suffix to that device's key and hashes again until the value is
unique. This collision rule guarantees distinct initial rates rather than
depending only on collision probability; it remains reproducible because the
ordering, seed, ID, type, and epoch are fixed. Later epochs do not require
cross-device uniqueness. Every device advances only its own epoch, so another
device's completion order cannot perturb its samples. Including an explicit
namespace version prevents an unrelated deterministic hash use from sharing
the same sample stream.

Disabled mode does not hash or sample. Every capacity query returns the
existing `Device.draft_token_rate_tok_s`, and the old latency formula receives
the same values as before.

## State Transition

For each device, the transition is:

```text
initialization:
  completed_requests = 0
  epoch = 0
  current_rate = sample(seed, device_id, device_type, 0)

on one unique request completion:
  completed_requests += 1
  if completed_requests % 5 != 0:
      keep epoch and current_rate
  else:
      epoch += 1
      current_rate = sample(seed, device_id, device_type, epoch)
```

Thus completions 1-4 use epoch 0 for future starts, completion 5 changes the
device to epoch 1, completions 6-9 retain epoch 1, and completion 10 changes it
to epoch 2. Counts and epochs are per simulator and per device; they are not
global and are not persisted across runs.

All methods use the same initial mapping. Because each method is simulated in
its own `Simulator`, its completion-driven transitions occur according to its
own request execution. This is intended: the resource process is coupled to
completed work, while its sampled values remain reproducible.

## Data Flow and Effective Boundary

The data flow is:

```text
simulation.seed + Device identity + template range
                    |
                    v
             EdgeComputeModel state
              /                  \
 planning reads current rate      task start captures snapshot
              |                              |
              v                              v
 gamma/pipeline/DiP planning       fixed duration and trace provenance
                                             |
                                             v
 request completion -> unified commit -> per-device count/epoch transition
```

The snapshot boundary is the actual task-start operation, not queue admission,
segment planning, or verification start:

- `_start_draft` captures one snapshot before computing all latency terms for
  that ordinary draft task;
- `_start_proactive_draft` captures its own snapshot because proactive work is
  a separate task, even though it is associated with an existing segment;
- each DiP-SD per-request draft captures a snapshot at `draft_start_ms`;
- a task queued before the fifth completion but started after it uses the new
  epoch;
- a task started before the fifth completion keeps its scheduled duration even
  if it finishes afterward;
- all latency terms for one task, including analytical/proactive overlap
  accounting and pipeline diagnostics, use the same snapshot rather than
  rereading current state midway.

At a completion timestamp, finalization and its epoch change are atomic within
the current event handler. A draft start already processed at that timestamp
keeps its old snapshot; the next start processed after finalization sees the
new epoch. Existing event ordering is unchanged.

Planning reads do not reserve capacity. `_select_gamma` and
`_specedge_edge_cycle_ms` use the current rate when invoked. DiP-SD optimization
uses the current rates when building an epoch plan; a completion-driven change
does not trigger replanning of an already-created plan, but later actual draft
starts still capture the then-current rate. This preserves scheduler and
optimizer control flow while allowing the resource input to evolve.

## Method Routing

| Method/runtime | Dynamic edge model effect |
| --- | --- |
| `full` and async ablations | ordinary edge drafts and their estimates use it |
| `specedge_linear`, `specedge_tree` | ordinary and proactive edge drafts use it |
| `dip_sd` | optimizer input and actual per-device edge drafts use it |
| `server_only_linear`, `server_only_tree` | completion count is recorded, but server draft latency remains fixed and bypasses edge snapshots |
| `target_only` | completion count is recorded, but there is no draft task and target latency is unchanged |

This is one shared resource contract, not identical physical effects for
methods that do not execute on edge drafters. Server-only must continue to use
`server_only.draft_token_rate_tok_s`; target decode and every verification
operation must continue through the existing fixed server/target latency
paths. Network delay continues to use the existing device bandwidth, RTT,
jitter, and deterministic jitter key.

## Trace and Observability

Every actual edge draft trace (`draft_compute`, `proactive_draft`, and
`dip_sd_draft`) should record the snapshot's `edge_compute_epoch` and
`draft_token_rate_tok_s`. This is provenance only and must not create new
scheduling behavior. A completion that changes capacity should emit one
separate transition trace containing `device_id`, `device_type`, completed
count, old/new epoch, and old/new rate. No transition event is emitted for
non-boundary completions or when disabled.

Existing duration fields retain their meanings. Server-only draft traces must
not claim an edge compute epoch.

## Test Design

### Configuration and sampling

- disabled configuration accepts missing dynamic ranges and preserves all
  current built-device and latency values;
- enabled configuration rejects malformed flags, intervals, non-positive
  bounds, missing ranges for populated categories, and a resample interval
  other than `5`;
- every initial rate lies within its device type's range;
- same-type devices have different initial rates;
- the same seed, device ID, type, and epoch produce bit-for-bit identical
  rates across model instances and independent runs;
- changing seed, device ID, type, or epoch changes the deterministic sample;
- sampling does not advance or otherwise change the simulator RNG sequence or
  deterministic network jitter.

### Per-device state transitions

- completions 1-4 leave epoch and rate unchanged;
- completion 5 advances only that device to epoch 1 and resamples once;
- completion 10 advances it to epoch 2;
- interleaved completions on two devices maintain independent counts and
  epochs;
- duplicate finalization of one request does not increment twice;
- DiP-SD and event-driven completion both pass through the same finalization
  operation.

### Snapshot boundary

- a task started before completion 5 retains its epoch-0 rate and duration
  after the transition;
- a queued task that starts after completion 5 uses epoch 1;
- ordinary, proactive, and DiP-SD tasks each capture a start-time snapshot;
- all duration and pipeline-accounting terms for one task use that same
  snapshot;
- traces report the exact epoch and rate used by the task.

### Cross-method isolation and regression

- async/full, SpecEdge linear/tree, and DiP-SD use the shared edge model;
- target-only decode, server-only draft, linear/tree verification, target
  latency profile queries, and network delay are unchanged when dynamic edge
  compute is enabled;
- server-only and target-only completion accounting cannot alter their fixed
  server latency;
- disabled runs are regression-equal to current runs in request outputs,
  timestamps, event ordering, traces, metrics, and RNG-dependent choices;
  dynamic trace fields and transition events are absent;
- canonical and later method specifications cannot select a private range or
  bypass the shared edge model when starting an edge draft.

Tests should use fake model execution and small deterministic workloads. They
should assert boundaries from trace and state rather than rely on wall-clock
time or CUDA.

## Explicit Non-Goals

This design does not add or change:

- dynamic server draft capacity, target capacity, server multipliers, or GPU
  contention;
- target latency profile contents, lookup, fallback, or metric selection;
- network bandwidth, RTT, jitter, packet loss, blocking, or network sampling;
- drafter model assignment, acceptance priors, semantic token generation, or
  verification results;
- scheduler policy, lane assignment, batching, timeout, queue discipline,
  gamma candidates, or proactive policy;
- request-to-device assignment or device migration;
- correlated devices, thermal models, time-based changes, utilization-based
  throttling, failures, batteries, or persistence across simulations;
- prefill, prompt transmission, TTFT, or any metric outside the existing
  decode-only scope;
- a new baseline, a proposed-method-only resource advantage, or an
  implementation plan.
