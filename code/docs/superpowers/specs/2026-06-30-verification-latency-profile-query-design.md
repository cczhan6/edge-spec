# Verification Latency Profile Query Design

## Scope

Add a pure Python query layer in `src/verification_latency_profile.py`. It
loads `outputs/profiling/target_verification_latency_full_merged.csv` by
default and returns conservative target decode or verification latency without
changing the simulator, analytical latency functions, scheduler, or baseline
algorithms. Tests use temporary CSV fixtures and require neither CUDA nor model
files.

## Public API

`VerificationLatencyProfile` accepts a CSV path and metric name at
construction. Supported metrics are `p50_ms` (default), `mean_ms`, and
`p95_ms`.

Its `query` method accepts:

- `method`: `target_decode`, `linear_verification`, or `tree_verification`;
- `batch_size`: actual request count;
- exactly one of `context_length` or `context_lengths`;
- `gamma` for linear verification;
- `tree_nodes` as tree-query metadata.

Method arguments are strict: target decode accepts neither gamma nor tree
nodes; linear verification requires a positive integer gamma and rejects tree
nodes; tree verification requires positive integer tree nodes and rejects
gamma.

When `context_lengths` is provided, its length must equal `batch_size`. The
batch uses max-context padding: `actual_max_context_length` is the maximum
request context, and every physical subbatch queries the same rounded-up
context tier. No virtual requests are created.

The returned immutable result contains:

- `actual_batch_size`;
- `profile_batch_size`, the largest selected profile batch tier;
- `profile_batch_sizes`, one selected profile batch tier per subbatch;
- `subbatch_count`;
- `subbatch_sizes`, the actual request counts in physical subbatches;
- `profile_context_length`;
- `profile_gamma`, or `None` for non-linear methods;
- `total_latency_ms`, the serial sum of selected source-row metric values;
- `tree_mode`, `None` for non-tree methods and `fixed_forward_approx` for tree;
- `source_rows`, one provenance record per subbatch.

The result and each source-row provenance record are frozen dataclasses.
`profile_batch_sizes`, `subbatch_sizes`, and `source_rows` are tuples. Each
source record contains the selected row's typed dimensions, an immutable tuple
of the original CSV key/value pairs, the actual subbatch size, selected metric
name/value, and requested `tree_nodes`. Repeated use of one profile row
produces repeated provenance records so their metric values sum directly to
`total_latency_ms`.

## Load-Time Validation and Indexes

The constructor reads the CSV exactly once. Query operations use only in-memory
indexes and never scan the CSV.

Loading validates:

- the required schema and supported method/status values;
- positive integer batch/context tiers and method-appropriate gamma/tree data;
- finite, nonnegative timing statistics on success rows;
- unique query keys;
- `tree_mode=fixed_forward_approx` on every tree row;
- identical `mean_ms`, `p50_ms`, `p95_ms`, and `std_ms` across all successful
  tree rows sharing one `(batch_size, context_length)` pair.

The original-row uniqueness key is
`(method, batch_size, context_length, gamma, tree_nodes)`. Runtime indexes use
`(method, batch_size, context_length)` for target decode and canonical tree
rows, and `(method, batch_size, context_length, gamma)` for linear
verification.

Success rows populate the query index. OOM rows are excluded from that index
and retained in `oom_rows` for diagnostics. Legal batch, context, gamma, and
tree-node tiers are derived from both success and OOM rows, ensuring a measured
OOM tier remains known but unusable.

Tree rows are indexed only by `(batch_size, context_length)`. For each pair,
the successful row with the smallest `tree_nodes` is the canonical source row.
`tree_nodes` supplied to a query is metadata and never changes latency.

## Tier Selection

The global maximum context is rounded up to the smallest legal context tier.
Linear gamma is rounded up to the smallest legal gamma tier. Values above the
largest measured context tier or linear gamma tier fail explicitly; with the
current profile this rejects context greater than 2048 and gamma greater than
8.

For a requested batch:

1. Collect all feasible success batch tiers for the method, rounded context,
   and rounded gamma. Target and tree queries omit gamma from this key.
2. If no feasible success batch tier exists, raise a query error rather than
   interpolate or use an OOM row.
3. If actual batch size does not exceed the largest globally legal batch tier,
   round it up to the smallest globally legal tier. If that exact tier is in
   the feasible success set, use one physical subbatch with the actual request
   count.
4. If that rounded tier is infeasible, or actual batch size exceeds the largest
   globally legal tier, split using the largest feasible success batch tier for
   the current method/context/gamma condition.
5. Split actual requests serially into chunks no larger than that feasible
   maximum. For each remainder, independently select the smallest feasible
   success tier greater than or equal to the remainder's actual size. If no
   such tier exists, raise a query error.

This yields, for example:

- B=16,L=2048 -> actual subbatches `[8, 8]`, profile tiers `[8, 8]`;
- B=9,L=2048 -> `[8, 1]`, profile tiers `[8, 1]`;
- B=20,L=2048 -> `[8, 8, 4]`, profile tiers `[8, 8, 4]`;
- B=17,L=1024 -> `[16, 1]`, profile tiers `[16, 1]`.

Subbatches execute serially on one GPU, so total latency is the sum of the
selected metric values. OOM rows are never interpolated, substituted, or
included in the sum.

## Errors

Explicit `ValueError` subclasses distinguish invalid profile files from
invalid queries. Errors identify the failing method/dimension and cover:

- unsupported method or metric;
- missing or conflicting context arguments;
- nonpositive batch/context/gamma;
- context-list length different from `batch_size`;
- context or linear gamma beyond measured tiers;
- missing linear gamma;
- tree rows with invalid mode or inconsistent statistics;
- no feasible success row for a requested method/context/gamma.

## Tests

`tests/test_verification_latency_profile.py` creates temporary mock CSV files
and covers:

- exact row lookup and default P50 selection;
- explicit mean/P95 selection;
- upward rounding of batch, context, and gamma;
- mixed-context max padding without virtual requests;
- B=16,L=2048 OOM splitting into two B=8 subbatches;
- B greater than 16 and remainder-tier selection;
- OOM rows excluded from lookup and preserved for diagnostics;
- tree canonical-row selection, fixed-forward labeling, node metadata, and
  load-time statistic consistency validation;
- context/gamma bounds, missing feasible rows, and batch/context mismatch;
- proof that queries use the loaded index after the source CSV is removed.

Only this target test file is run. No real profiling matrix or simulator test
suite is part of this change.
