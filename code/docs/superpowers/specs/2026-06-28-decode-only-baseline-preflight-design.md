# Decode-Only Baseline Preflight Design

## Scope

Add a 16-request, 32-output-token, two-seed preflight for the six frozen
canonical baselines in `homogeneous` and `combined_strong_heterogeneous`.
The preflight uses the formal CLI and `HuggingFaceModelRunner`; it does not
change method scheduling, verification, model binding, or acceptance semantics.
It is pipeline validation and is not a paper performance result.

The decode clock begins after prompt processing and initial prefix/KV state are
ready. Prompt tokenization/transmission, prefill, initial KV construction,
model loading, TTFT, and first-token latency are excluded. TTFT remains `NA`
and no decode-first-token substitute is introduced.

## Architecture

`scripts/check_drafter_residency.py` loads the three configured Qwen drafters
individually and together on `cuda:0`, records CUDA peak/remaining memory,
dtype, OOM status, and exact tokenizer-vocabulary compatibility, and writes an
atomic JSON manifest. If simultaneous residency fails, the manifest records a
sequential/lazy loading requirement without changing the configured virtual
devices or model/profile bindings.

`scripts/run_baseline_preflight.sh` prepares scenario/seed configs by changing
only request count, fixed output length, seed, and Poisson arrival mode. It
audits each resolved config, invokes `scripts.run_all` once per canonical
method through the real runner, captures logs/manifests/environment metadata,
and writes the required `scenario/seed/method` hierarchy.

`scripts/verify_baseline_preflight.py` owns the formal preflight schema,
derived decode-only metrics, invariant checks, and summary generation. It
reuses trace concepts from `scripts.baseline_trace` but validates every
scenario/seed group independently so target-only references cannot cross
workloads.

## Token-Time Observation

The shared request entity gains one committed-token timestamp per generated
token. Existing method commit points append timestamps alongside the existing
token IDs; this is observation only and cannot affect event ordering,
scheduling, acceptance, or output values. Speculative tokens committed by one
visible verification result share its result-visible timestamp. Target-only
tokens receive monotonically spaced timestamps from the configured
autoregressive decode latency model. The first timestamp is retained only as
trace evidence; no first-token metric is produced.

Inter-token latency is computed only from adjacent committed timestamps within
the same request. Decode makespan is `max(finish) - min(decode_ready)`, request
decode latency is `finish - decode_ready`, effective throughput is committed
tokens divided by decode makespan, and speedup is the matching scenario/seed
target-only request-latency aggregate divided by the method aggregate.

## Outputs and Failure Behavior

Every run directory contains `resolved_config.yaml`,
`environment_manifest.json`, `run_manifest.json`, `metrics.csv`,
`request_metrics.csv`, `event_trace.csv`, `token_trace.csv`,
`resource_timeline.csv`, and `stdout.log`. Auxiliary trace files may remain.
The root contains `summary.csv` and `summary.md`; neither contains TTFT.

Missing model references, unavailable CUDA, fake-runner evidence, failed runs,
invalid/non-finite metrics, mismatched greedy tokens, pending state, resource
overlap, non-monotonic time, burst arrivals, unit mismatch, invalid speedup
reference, Server-only batches above one, absent SpecEdge proactive work,
insufficient DiP-SD batches/optimizer evidence, or missing tree approximation
markers fail verification explicitly. No fallback runner or semantic tuning is
allowed.

## Test Strategy

Unit tests cover residency manifest construction and OOM policy without
requiring GPU allocation, preflight config preparation, required output names,
metric formulas, same-scenario/seed reference selection, and each formal
invariant. Existing smoke/trace and full repository suites guard frozen method
semantics. Final verification runs the user-specified residency check,
preflight, verifier, pytest suite, baseline rebuild verifier, deterministic
trace runner, and whitespace check.
