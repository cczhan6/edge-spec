# Canonical Baseline Performance Evaluation Design

## Scope

Establish one formal `dynamic_heterogeneous` performance evaluation for the six
canonical baselines:

- `target_only`
- `server_only_linear`
- `server_only_tree`
- `specedge_linear`
- `specedge_tree`
- `dip_sd`

This work adds only evaluation configuration, an orchestration script, and a
summary/validation script. It does not modify the simulator, edge compute
resource model, scheduler, server verification profile, or any baseline
algorithm. Full experiments are not run as part of this design stage.

The repository is decode-only. New outputs therefore set
`metric_scope=decode_only` and do not contain a TTFT metric. If an external
fixed schema ever requires a TTFT column, it must be empty, accompanied by
`ttft_supported=false`, and excluded from aggregation, plots, and reports.

## Existing Interfaces Reused

The evaluation reuses the model-runner, simulator, metric summarization, and
trace-bundle writing path already composed by `scripts.run_all`. A thin
evaluation adapter is necessary because the general CLI has no input for a
materialized arrival trace; the adapter supplies that trace while retaining
the existing simulation and output contracts. The actual configuration fields
are:

```yaml
dynamic_edge_compute:
  enabled: true
  resample_every_completed_requests: 5

target_latency:
  mode: profile
  profile_path: outputs/profiling/target_verification_latency_full_merged.csv
  metric: p50_ms
```

The design intentionally does not introduce the aliases
`edge_compute_dynamics` or `requests_per_update`. Device-level network blocking
uses the existing `device_pools.heterogeneous.templates.*.block_probability`
field.

The existing runner path already produces request latency percentiles,
TPOT/TBT, makespan, goodput, acceptance rate, and selected gamma. Trace bundles
provide request completion state and committed token sequences, which are
required to derive `committed_tokens` and `success` without changing
production metrics.

## Evaluation Configuration

Add a scenario override for `dynamic_heterogeneous` on top of
`configs/default.yaml`. It fixes:

- the existing heterogeneous pool composition: three low-end, three mid-end,
  and two high-end devices;
- the existing per-class dynamic compute ranges, which deterministically give
  devices of the same class distinct epoch-zero rates;
- dynamic edge compute enabled, with a per-device update after every five
  completed requests;
- `block_probability: 0.2` on every populated heterogeneous device template;
- target verification profile mode with `p50_ms`;
- 80 total requests and the existing Poisson arrival process.

The orchestration script runs seeds `0, 1, 2, 3, 4`. For each seed it first
materializes exactly one immutable workload/arrival trace. That trace contains
the selected workload order, requested output lengths, arrival/decode-ready
times, and input request-to-device mapping. All six methods read this same
file; no method independently samples the dataset, output-length choices, or
Poisson arrival process. Consequently the dataset sample, arrival sequence,
input request-to-device mapping, device identities and initial rates, network
fields, and target profile are common within the seed.

The script records a resolved per-seed configuration under the new evaluation
output root before running. These generated files are provenance artifacts,
not additional checked-in scenario definitions.

## Components and Data Flow

### Scenario configuration

One checked-in YAML override defines `dynamic_heterogeneous`. It contains only
evaluation-specific overrides and inherits model bindings, server-only draft
capacity, SpecEdge settings, DiP-SD settings, device ranges, and the target
profile path from the existing defaults unless an explicit evaluation value is
listed above.

### Orchestration script

A new run script:

1. creates a dedicated `outputs/baseline_performance_eval/` tree;
2. resolves the scenario once per seed and writes the seed into that resolved
   configuration;
3. materializes and validates the seed's unique shared workload/arrival trace;
4. audits the resolved configuration before execution;
5. pre-creates the complete 30-row run-status matrix;
6. invokes the existing simulation and trace-writing runner path once for each
   method, supplying the shared trace instead of allowing arrival resampling;
7. preserves per-run stdout and return status, records failures, and continues
   with all remaining cells; and
8. invokes the summary/validation script after the requested runs finish,
   returning nonzero after all cells have been attempted if any cell failed.

No output is written beneath `outputs/baseline_trace/`.

The formal matrix contains 30 runs: one scenario, five seeds, and six methods.
The script does not enable the fake model runner and does not change method
order semantics, scheduling, or resource parameters.

### Summary and validation script

The summary script reads existing main metrics and trace bundles. It does not
rerun simulations. It first materializes `runs.csv`, containing one row for
each `(seed, method)` pair, then produces `summary.csv`, containing one row per
method.

`runs.csv` fields are:

- identity: `scenario`, `seed`, `method`, `metric_scope`;
- integrity: `num_requests`, `committed_tokens`, `success`, `failure_reason`;
- latency: `avg_latency_ms`, `p50_latency_ms`, `p95_latency_ms`,
  `p99_latency_ms`;
- decode pacing: `avg_tpot_ms`, `avg_tbt_ms`;
- system outcome: `makespan_ms`, `goodput_tok_s`;
- speculative outcome: `avg_acceptance_rate`, `avg_selected_gamma`.

TPOT and TBT retain the definitions emitted by the current runner. This
evaluation does not redefine them or infer a prefill/first-token metric.

`runs.csv` is initialized with all 30 expected `(seed, method)` rows before any
simulation starts. A process failure or missing trace updates its existing row
to `success=false` with a concrete `failure_reason`; rows are never omitted.

`summary.csv` computes `<metric>_mean` and `<metric>_std`, grouped by method
across the five seeds, only for this explicit performance metric list:

- `avg_latency_ms`, `p50_latency_ms`, `p95_latency_ms`, `p99_latency_ms`;
- `avg_tpot_ms`, `avg_tbt_ms`;
- `makespan_ms`, `goodput_tok_s`;
- `avg_acceptance_rate`, `avg_selected_gamma`.

Standard deviation uses the sample definition. `seed`, `success`,
`num_requests`, and `committed_tokens` are integrity fields and are never
aggregated. The summary also records `num_runs`, `successful_runs`,
`metric_scope`, and `success`. Aggregated performance metrics are emitted only
when all five runs for the method pass validation; otherwise they remain empty
so failed runs cannot silently bias the reported result.

## Correctness Gates

For each seed, validation requires:

1. exactly the six canonical methods and exactly 80 finished requests per
   method;
2. every method consumes the seed's single materialized workload/arrival trace,
   with identical request IDs, prompt IDs, requested output lengths, arrival
   times, and decode-ready times;
3. the shared input trace contains one request-to-device mapping, and methods
   use one common resolved device configuration, epoch-zero rate vector,
   deterministic epoch-to-rate mapping, and target profile configuration;
4. dynamic compute enabled, update interval equal to five, and blocking
   probability equal to `0.2` for every populated heterogeneous template;
5. profile mode using the expected merged CSV and `metric=p50_ms`;
6. finite, non-negative required metrics;
7. identical per-request committed token sequences across all methods, using
   `target_only` as the reference, and therefore identical final committed
   token totals;
8. all requests finished with no pending speculative state; and
9. successful process return status and complete trace files.

Device assignment consistency is checked against the mapping in the shared
input trace. A missing or non-applicable `device_id` in `target_only` or a
server-only output trace is not a failure, and the evaluation must not create
fake edge events to make those methods appear device-backed.

Only immutable resource inputs are cross-method invariants. Different methods
may complete requests in different orders, so dynamic edge-compute transition
times and network-blocking event outcomes are not required to match event by
event. Their common seed, device configuration, epoch-zero rates, deterministic
epoch-to-rate function, and blocking probability are still validated.

A failed gate sets `success=false` and a concrete `failure_reason` in
`runs.csv`. The affected method is also unsuccessful in `summary.csv`, and the
summary command exits nonzero after writing both CSV files. Missing or failed
runs are never dropped from the integrity result.

## Output Layout

```text
outputs/baseline_performance_eval/
  _configs/
    dynamic_heterogeneous_seed_<seed>.yaml
  _workloads/
    dynamic_heterogeneous_seed_<seed>.jsonl
  dynamic_heterogeneous/
    <seed>/
      _raw/
      <method>/
        stdout.log
        run_status.json
        metrics.csv
        request_trace.csv
        event_trace.csv
        token_trace.csv
        resource_timeline.csv
        batch_trace.csv
        system_metrics.csv
        resolved_config.json
  runs.csv
  summary.csv
```

## Verification Before Full Execution

Implementation verification should remain lightweight until explicitly
approved for the formal run:

- validate the new YAML through the existing configuration loader and audit
  script;
- unit-test summary mean/std and failure propagation with temporary synthetic
  trace bundles;
- unit-test request/workload equality and committed-token mismatch detection;
- run existing experiment-tool and configuration tests; and
- perform at most a deliberately small smoke run using a temporary request
  count, outside the formal output tree.

The five-seed, six-method, 80-request matrix is a separate explicit execution
step and is not part of implementation or design verification.
