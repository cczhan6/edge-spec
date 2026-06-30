# Target Latency Profile Integration Design

## Scope

Add a shared `TargetLatencyModel` facade in `src/latency.py` that selects
between the existing analytical target-latency formulas and the measured
`VerificationLatencyProfile`. The integration changes only the target latency
charged by canonical baseline execution. It does not change semantic model
execution, batching, scheduling, gamma selection, network behavior, or any
analytical formula.

The recommended design is one facade owned by each `Simulator`. This keeps
configuration, profile lifetime, and method routing in the shared latency
layer. Mode branches must not be scattered throughout `Simulator`, and the
current milestone does not introduce separate analytical/profile strategy
class hierarchies.

## Architectural Boundary

`TargetLatencyModel` preserves the current analytical functions in
`src/latency.py`. Those functions are neither deleted nor rewritten. The
facade selects one of two modes:

- `analytical`: call the existing formulas and preserve their current
  behavior;
- `profile`: construct one `VerificationLatencyProfile` and query its
  in-memory indexes.

`Simulator.__init__` constructs exactly one `TargetLatencyModel`. In profile
mode, the facade constructs exactly one `VerificationLatencyProfile` during
that initialization. It must not reopen or reload the CSV for individual
decode steps, verification tasks, or verification batches. In analytical
mode, it constructs no profile object and performs no profile-file access.

All actual canonical target decode, linear verification, and tree
verification execution latency is obtained through this single facade. For
each logical profile operation, the facade calls
`VerificationLatencyProfile.query` once and reads only
`result.total_latency_ms`. Neither the facade nor `Simulator` inspects
`subbatch_sizes`, repeats the profile's split, or sums source rows again.

## Configuration

The default configuration is planned as:

```yaml
target_latency:
  mode: analytical
  profile_path: outputs/profiling/target_verification_latency_full_merged.csv
  metric: p50_ms
```

Supported modes are exactly `analytical` and `profile`. Supported metrics are
exactly `p50_ms`, `mean_ms`, and `p95_ms`.

Configuration validation applies these rules:

- In every mode, `mode` is required and validated.
- In every mode, a present `metric` value is validated against the supported
  metric set.
- In profile mode, `profile_path` must be a non-empty string.
- In profile mode, `validate_config` resolves the path and verifies that it
  names an existing regular file. The sole
  `VerificationLatencyProfile` construction in `TargetLatencyModel` then
  performs the complete CSV schema and data validation. This division avoids
  loading the CSV once during configuration validation and again during
  simulator initialization while still rejecting an unloadable profile
  before execution begins. Malformed profile data remains a profile
  validation error rather than being silently converted to analytical mode.
- In analytical mode, the profile path is not resolved, opened, or checked.
  A configured nonexistent profile path therefore does not prevent simulator
  initialization.

Relative profile paths are resolved against the project `code/` directory,
derived from the installed source location rather than the process working
directory. The configured default consequently resolves to
`code/outputs/profiling/target_verification_latency_full_merged.csv` whether
the process starts in the repository root, `code/`, or another directory.
Absolute paths, including temporary test fixtures, are used unchanged.

## `TargetLatencyModel` API

The shared facade exposes three explicit operations. Exact argument names may
follow the established Python style during implementation, but their semantic
contracts are fixed:

- `target_decode_latency_ms(...)` returns the latency of one cached,
  single-token target decode step for the supplied logical request batch;
- `linear_verification_latency_ms(...)` returns the latency of one linear
  verification forward for the supplied logical segment batch;
- `tree_verification_latency_ms(...)` returns the fixed-forward tree
  approximation latency for the supplied logical segment batch.

In analytical mode, these operations delegate to the current analytical
functions with the same inputs used by existing execution paths. In profile
mode, they map their inputs to the corresponding profile method and return
only `query(...).total_latency_ms`.

The facade must not add a server compute factor, periodic capacity changes, or
resampling based on completed request count. Server capacity is fixed. Profile
latency may still vary deterministically with method, actual batch size,
context, gamma, and selected metric.

## Target-Only Decode

The target-only execution contract remains:

- FCFS service on the existing single target resource;
- logical batch size one, because the current path has no target decode
  batching;
- one `target_only_service` event per request;
- no new grouping or batch events;
- one semantic `model_runner.target_only` call that produces the request's
  complete output.

Analytical mode preserves the current total service-latency calculation and
the current token commit-timestamp behavior exactly. Existing analytical
tests and results must not change.

Profile mode accounts for the same output as a sequence of single-token
cached decode steps. For output token index `i`, before the token is committed,
the facade queries:

```python
profile.query(
    method="target_decode",
    batch_size=1,
    context_lengths=[prompt_token_count + i],
)
```

Here `i` is the number of output tokens already formally committed. The
context includes the original prompt because its KV state exists at decode
start, even though prefill time is outside the simulation. It excludes the
token being generated by the current step.

Each output token causes one `target_decode` query. The path must not emulate
decode with `linear_verification(gamma=1)` and does not involve draft tokens,
acceptance, rejection, bonus tokens, or tree verification. The per-token
latencies are accumulated serially to obtain request service time, and the
same cumulative offsets produce the token commit timestamps. The existing
single request-level service event records the accumulated total.

The current path therefore always queries `batch_size=1`. If a separate
milestone later adds target decode batching, the same API can accept the
actual active request count and each request's pre-step context, but this
integration does not add that batching.

Context values beyond the largest measured profile tier fail through the
query layer; they are not extrapolated. Formal profile-mode experiments must
satisfy:

```text
prompt_token_count + max_output_length <= maximum profile context
```

The current profile's maximum context tier is 2048.

## Linear Verification

Linear profile routing applies to:

- `server_only_linear`;
- `specedge_linear`;
- the linear verification path in `dip_sd`.

Each existing logical verification batch makes exactly one query with
`method="linear_verification"`. The query receives:

- `batch_size`: the number of real segments in the logical verification
  batch;
- `context_lengths`: one verifier KV prefix length per segment, measured
  immediately before the draft segment being verified;
- `gamma`: the maximum actual draft-segment length in that logical batch.

The context excludes the current draft tokens. If `segment.prefix_ids`
represents exactly the verifier prefix already resident in KV cache before
the segment, `len(segment.prefix_ids)` is the correct value. Implementation
tests must establish this invariant from segment construction and verification
behavior; the integration must not assume it solely from the field name.

Gamma is derived from the batch's actual segment lengths, not from the maximum
configured candidate gamma. For actual lengths `[2, 4, 3]`, the query receives
`gamma=4`. Shorter segments are right-padded to the longest physical forward
length, matching the current batched linear verification semantics. The query
layer then rounds that actual maximum upward to a measured gamma tier.

## Tree Verification

Tree profile routing applies to:

- `server_only_tree`;
- `specedge_tree`.

Each existing logical tree verification batch makes exactly one query with
`method="tree_verification"`. It receives the real segment count, each
segment's verifier KV prefix length before the tree input, and the maximum
`target_verify_tree_nodes` value in the batch.

`tree_nodes` is provenance metadata only. It does not select a latency row or
alter the returned latency. The result must carry
`tree_mode="fixed_forward_approx"`; any other tree mode is invalid.

This integration does not implement or claim a measured tree-attention
kernel. It reuses the profiler's explicitly approximate cached single-forward
measurement. Code comments, traces, and later paper text must continue to
label it as `fixed_forward_approx`, never as a real tree verification kernel.

## Context Padding and OOM Splitting

The integration passes real logical requests and segments without creating
virtual entries. `VerificationLatencyProfile` remains solely responsible for:

- applying max-context padding to mixed-context logical batches;
- rounding batch, context, and gamma to measured tiers;
- excluding OOM rows;
- splitting an infeasible logical batch into serial physical subbatches;
- summing selected physical row latencies.

For example, logical contexts `[120, 500, 900]` remain an actual batch of
three. The query layer may map them to profile batch tier four and context tier
1024 without introducing a fourth request.

Likewise, if the `B=16, L=2048` row is OOM while `B=8` succeeds, the query
layer may return two physical subbatches and the sum of two `B=8` row
latencies. The caller consumes that `total_latency_ms` once. The logical batch
still contains sixteen operations in one decode or verification round.

## Scheduling Boundary

This milestone changes only target latency charged by actual canonical
baseline execution events:

```text
scheduler-side prediction          analytical
actual canonical execution latency target_latency.mode
```

`predict_verify_latency_ms` remains analytical. Scheduler lane selection,
adaptive prediction, proposed methods, and legacy scheduling behavior are not
modified. Consequently a scheduler prediction may differ from the profile
latency later charged by the actual event. That mismatch is accepted and
explicit for this milestone. Switching scheduler-side prediction to profile
data would require a separate design and evaluation.

Canonical execution routing is:

| Baseline | Target latency operation |
| --- | --- |
| `target_only` | `target_decode` |
| `server_only_linear` | `linear_verification` |
| `specedge_linear` | `linear_verification` |
| `dip_sd` | `linear_verification` |
| `server_only_tree` | `tree_verification` |
| `specedge_tree` | `tree_verification` |

## Error Handling

Configuration errors must identify the invalid target-latency field. Invalid
mode and metric values fail during configuration validation. Profile mode with
an empty path, missing file, unreadable file, or malformed CSV fails explicitly
during initialization. There is no silent fallback to analytical mode.

Query-time dimension errors, including contexts or linear gamma beyond the
measured tiers and conditions with no feasible success row, propagate as the
query layer's explicit `ProfileQueryError`. This prevents an unmeasured
condition from being hidden by interpolation or an analytical fallback.

Analytical mode is isolated from all profile I/O errors because it never
resolves or opens the configured profile path.

## Test Design

A dedicated integration test module will use temporary mock CSV fixtures and
fake model execution. It will cover:

- analytical mode produces the current decode and verification results;
- analytical initialization succeeds when the configured profile path does
  not exist;
- profile mode routes target-only steps to `target_decode`;
- profile mode routes server-only, SpecEdge, and DiP-SD linear verification to
  `linear_verification`;
- profile mode routes server-only and SpecEdge tree verification to
  `tree_verification`;
- one `Simulator` constructs `VerificationLatencyProfile` exactly once across
  multiple decode or verification operations;
- mixed per-segment contexts reach one query unchanged;
- a logical `B=16, L=2048` operation consumes the query layer's split
  `total_latency_ms` once, with no caller-side repeat sum;
- invalid mode and invalid metric fail explicitly;
- profile mode rejects an empty path, missing path, and unloadable CSV;
- relative path resolution is unchanged when the shell working directory
  changes;
- target-only context grows as prompt length plus committed token count;
- target-only commit timestamps use cumulative per-token profile latencies;
- linear context excludes the current draft segment;
- linear gamma is the current batch's longest actual segment, including a
  mixed-length example;
- tree queries accept only `fixed_forward_approx`;
- context beyond the maximum measured tier fails explicitly.

Implementation follows test-first red-green-refactor cycles. Each behavior is
first represented by a failing test, then receives the minimum production
change needed to pass. Verification runs the dedicated integration test first
and then the complete `pytest -q` suite.

## Explicit Non-Goals

This design does not implement:

- dynamic edge-device compute;
- dynamic server compute or server compute multipliers;
- periodic or completion-count-based capacity resampling;
- probabilistic network blocking;
- scheduler changes;
- batching changes;
- gamma-selection changes;
- proposed-method changes;
- legacy scheduling changes;
- a real tree-attention verification kernel.

## Decision

Use one shared `TargetLatencyModel` facade in `src/latency.py`. It is the
smallest design that centralizes mode selection, enforces one profile load per
simulator, keeps the analytical implementation intact, and gives every
canonical baseline one explicit target-latency route.

Do not place repeated mode branches in `Simulator`; that would distribute
configuration and profile-lifetime rules across execution paths. Do not add
parallel analytical/profile strategy class hierarchies in this milestone;
with only two fixed modes and three operations, that abstraction would add
structure without reducing current complexity.
