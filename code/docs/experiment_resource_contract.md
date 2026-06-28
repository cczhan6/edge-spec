# Baseline Experiment Resource Contract

Status: frozen for canonical-baseline formal experiments. This contract fixes resources,
configuration, labels, and measurement semantics only. It does not change any baseline or
proposed-method implementation, and it does not claim that a 480-request formal performance
run has been completed.

## 1. Canonical methods and result labels

Formal result tables, CSV files, plots, and captions must use these exact pairs:

| Canonical method | Display name |
|---|---|
| `target_only` | Target-only |
| `server_only_linear` | Server-only Linear |
| `server_only_tree` | Server-only Tree (SpecExec-style approximation) |
| `specedge_linear` | SpecEdge Linear |
| `specedge_tree` | SpecEdge Tree (SpecExec-style approximation) |
| `dip_sd` | DiP-SD (Online Adaptation) |

Legacy aliases such as `server_only`, `SpecEdge`, and `sync_batch_sd` are compatibility inputs
only. They must never appear as formal result labels. Component ablations and `full` are outside
this canonical-baseline contract.

All six canonical methods have passed deterministic trace validation and a short live real-model
greedy-equivalence smoke. The 16-request formal-configuration preflight found longer-trace greedy
divergence and DiP-SD resource overlap, so baseline freeze readiness is currently blocked.
Formal-scale 480-request performance measurements have not been run.

## 2. Physical GPU duties and environment manifest

The semantic runner has two fixed physical roles:

| Physical device | Duty |
|---|---|
| `cuda:0` | Execute real drafter-model semantics. |
| `cuda:1` | Execute real target-model semantics. |

The physical GPUs execute token semantics; their wall-clock forward duration is not simulator
time. Before every formal run, execute:

```bash
python scripts/collect_experiment_environment.py \
  --output outputs/environment_manifest.json
```

The manifest records the full Git commit and dirty state; host/platform; Python executable and
version; PyTorch and Transformers versions; PyTorch CUDA runtime version and CUDA availability;
the NVIDIA driver version; and, for every visible GPU, index, model name, total memory in MiB,
and driver version. Missing packages, Git metadata, or `nvidia-smi` are recorded in
`collection_errors`; an unavailable field is never silently omitted. A formal run is invalid if
either required GPU or any required software/version field is unavailable.

The preflight host requires `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Without it, the
models remain resident but the first large tree-candidate forward can fail because reserved CUDA
segments fragment the remaining memory. This allocator setting changes physical memory management
only; it does not change virtual latency, method parameters, or model bindings. It is recorded in
the environment and per-run manifests.

## 3. Real semantics versus virtual time

The responsibilities are deliberately separate:

- The real Hugging Face models determine draft/target token IDs, acceptance, rejection
  correction, all-accepted bonus tokens, and EOS termination.
- The configured virtual devices determine draft latency, uplink/downlink serialization and
  propagation latency, and target verification latency.
- The simulator determines request arrivals, resource queues and competition, batch barriers,
  pipeline overlap, event order, and the order in which verified tokens become committed.
- The number or wall time of physical Hugging Face forwards is not charged to simulated time.
- A mixed-length logical verification batch may require multiple padded/grouped physical model
  forwards. It remains one logical batch verification in simulator time, with one readiness
  barrier and one configured virtual verification cost.

No measured host/GPU forward latency may be added to analytical simulator time. Conversely,
real model outputs may not be replaced by configured acceptance priors during a formal run.

## 4. Model and virtual-device binding

The immutable model/tokenizer bindings are recorded in `configs/default.yaml` under
`model_bindings`:

| Role/profile | Model id | Model and tokenizer revision | Physical device |
|---|---|---|---|
| target | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` | `cuda:1` |
| `small` | `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` | `cuda:0` |
| `medium` | `Qwen/Qwen2.5-1.5B-Instruct` | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | `cuda:0` |
| `large` | `Qwen/Qwen2.5-3B-Instruct` | `aa8e72537993ba99e69dfaafa59ed015b17504d1` | `cuda:0` |

All tokenizers must have exactly the target tokenizer's token-to-id mapping, and every model
vocabulary must cover that mapping. The real runner checks these properties on load.

The declared residency policy is `all_configured_models_simultaneous`: the target remains on
`cuda:1`, and all drafter profiles required by the scenario remain on `cuda:0`. The measured
`outputs/drafter_residency_manifest.json` result on the experiment host is:

| Drafter load | dtype | peak allocated MiB | free after load MiB | OOM |
|---|---|---:|---:|---:|
| Qwen2.5-0.5B | bfloat16 | 950.17 | 22680.06 | no |
| Qwen2.5-1.5B | bfloat16 | 2945.28 | 20538.06 | no |
| Qwen2.5-3B | bfloat16 | 5994.46 | 17532.06 | no |
| all three simultaneous | bfloat16 | 9889.91 | 13408.06 | no |

All three drafter tokenizers have the exact target token-to-id mapping and matching special token
IDs. Sequential/lazy loading is therefore not active on this host. If a future host cannot hold
all three drafters, it must keep the virtual-device/profile bindings unchanged, record
`sequential_lazy_model_loading` in both residency and environment manifests, and exclude model
load wall time from simulator decode latency. Profiles may not be merged or substituted.

Normal method/scenario isolation is process-level: load the pinned target and required drafters at
process start, run that one resolved cell, then terminate the process to release GPU memory before
loading the next cell. A combined heterogeneous cell still requires all three drafter profiles at
once and has no sequential-loading exemption.

Each virtual device template has one fixed `drafter_profile`. Requests are assigned round-robin
at arrival and retain that `device_id` and profile for their lifetime. The simulator must not
migrate a request to another virtual device unless a separately named method contract explicitly
allows migration; none of the six methods in this contract does.

## 5. Method resource configurations

### Target-only

- Uses only the target semantic model; no drafter and no edge/server network transfer.
- Performs target greedy autoregressive decoding through EOS or the output-length cap.
- Uses one serial target service resource with logical `batch_size=1`; requests queue FCFS.

### Server-only Linear and Tree

- The server drafter (`medium`, `cuda:0`) and target (`cuda:1`) are distinct resources.
- Decode-stage draft/verify has no edge-network charge.
- Every request follows synchronous `draft -> verify -> state/KV update` rounds. There is no
  proactive drafting and no cross-round overlap for one request.
- Both variants fix `server_only.batch_size=1`.
- Linear uses a single linear candidate path. Tree uses
  `server_only.tree_draft_strategy=specexec_approx` and must carry the
  “SpecExec-style approximation” label.

### SpecEdge Linear and Tree

- Every virtual edge device owns its assigned requests and serializes its own draft work.
- The device drafts proactively after a validation is launched, subject to the configured
  proactive policy, while the server performs logical batch verification.
- The formal default is `server_batch_type=static`, `server_batch_size=1`, and no batch timeout.
  Any sensitivity run that changes these values is a separately identified configuration.
- Linear candidates are one token sequence. Tree candidates use
  `tree_draft_strategy=proactive_tree_draft_strategy=specexec_approx` and retain the explicit
  approximation label.
- A proactive candidate is **retained** only when its base position/prefix version remains valid
  and it is consumed as the next candidate. It is **invalidated** when a correction, different
  bonus path, EOS, or prefix-version change makes that base unreachable. All generated proactive
  candidates that are invalidated or never consumed are **waste**.

### DiP-SD

- Formal display name is `DiP-SD (Online Adaptation)` and `optimizer=paper_exact`.
- Members of one batch share a readiness barrier; verification begins only when every member is
  ready.
- Different ordered batches may overlap edge drafting with server verification.
- A request may redraft only after its own verification result and KV/state update are complete.
- New requests enter only through the next online epoch/barrier; they never alter an in-flight
  optimizer plan.
- Fixed bounds are `batch_count=2`, `max_batch_size=4`, and per-request draft length in `[1, 4]`
  (`draft_length=2` is the configured initial/default value).

## 6. Formal scenarios

Common settings are 480 requests, output length sampled uniformly from `{64, 128, 256}`, seed
42, Poisson arrivals at 20 requests/s, four analytical verifier lanes, target rate 80 token/s,
target verification startup 8 ms, and server-only drafter rate 504 token/s. `burst` is forbidden
for these online scenarios.

Network delay is `RTT/2 + payload_bits/bandwidth + jitter`. Jitter is a deterministic hash of
`(seed, device_id, direction, transfer_key)` mapped uniformly to `[0, jitter_ms)`; therefore the
same resolved configuration has the same arrival/communication trace.

| Scenario/template | Count/profile | Draft compute | Uplink/downlink | RTT | Jitter |
|---|---:|---:|---:|---:|---:|
| `homogeneous` / medium | 8 / `medium` | 60 token/s, 2 ms startup | 25/100 Mbps | 40 ms | 10 ms |
| `combined_strong_heterogeneous` / low-end | 3 / `small` | 25 token/s, 1 ms startup | 5/30 Mbps | 90 ms | 25 ms |
| `combined_strong_heterogeneous` / mid-end | 3 / `medium` | 60 token/s, 2 ms startup | 25/100 Mbps | 40 ms | 10 ms |
| `combined_strong_heterogeneous` / high-end | 2 / `large` | 100 token/s, 3 ms startup | 100/300 Mbps | 10 ms | 2 ms |

The request-arrival inter-arrival time is exponentially distributed with lambda=20/s using the
single configured seed. Request-to-device assignment is fixed round-robin, not load-based
migration. The resolved config must record the exact scenario override after deep merge.

## 7. Metrics and counting rules

Simulator timestamps and durations are floating-point milliseconds. CSV time fields retain
milliseconds and the `_ms` suffix; no implicit conversion to seconds is allowed. Throughput uses
token/s, bandwidth uses Mbps, token counters are integers, and acceptance/speedup are unitless.

| Metric | Fixed definition |
|---|---|
| TTFT | Prefill/request-to-first-token TTFT is outside this decode-only repository and is not exported. If a downstream table requires the field, emit `NA`; no decode-first-token surrogate is permitted. |
| Inter-token latency | Within one request, subtract adjacent committed-token timestamps. Tokens made visible by one verification result share its result-visible time and therefore may have a zero interval. Aggregate mean/P50/P95 over all adjacent intervals in one cell. The first committed-token timestamp is trace evidence only and is not reported as a first-token metric. |
| Request completion latency | `finish_time_ms - decode_ready_time_ms`, in ms. |
| Throughput | Total committed output tokens divided by `(max finish - min decode-ready)/1000`, in token/s. Current CSV field is `goodput_tok_s`. |
| Accepted token ratio | Accepted draft tokens / verified proposed draft positions. Target-only has no proposals and reports 0/NA according to the output schema. |
| Drafted tokens | Candidate tokens produced by the drafter: linear path length; tree `processed_candidate_count`; proactive candidates included. |
| Verified tokens | Candidate positions presented to target verification: linear proposed positions; tree `target_verify_tree_nodes`; do not multiply by physical forward count. |
| Committed tokens | Tokens appended to the user-visible target-greedy trace, including accepted draft tokens and target-produced correction/bonus tokens. |
| Wasted tokens | Drafted candidates never committed because of rejection, pruning/discard, stale prefix, EOS truncation, or proactive invalidation/miss. |
| Speedup versus `target_only` | `target_only.avg_request_completion_latency_ms / method.avg_request_completion_latency_ms` for the identical scenario, workload sample/order, seed, model bindings, and output caps. |

A correction token is target-produced: it contributes one committed token (unless EOS handling
terminates without appending it), zero accepted draft tokens, and rejected suffix candidates to
waste. A bonus token produced after all draft positions accept contributes one committed token
and zero accepted/drafted tokens. EOS counts as committed only if it is present in the stored
committed trace. Every generated proactive token invalidated by a prefix/version change is added
once to wasted tokens; it may not be counted both as proactive waste and generic waste when totals
are combined.

Means, P50, P95, and P99 for latency/TPOT/TBT are computed over completed requests within one
`scenario x method` cell after correctness filtering, never over batches, segments, categories,
or pooled scenarios. Means are arithmetic. Percentiles sort request values and use linear
interpolation at `(N-1)*p/100`. Token ratios are ratios of cell-level sums, not means of
per-request ratios. No failed/incomplete request may simply be dropped; it invalidates the cell.

## 8. Pre-run audit and post-run correctness gates

Run the preflight audit separately for each scenario:

```bash
python scripts/audit_experiment_config.py \
  --config configs/default.yaml \
  --scenario homogeneous \
  --methods target_only server_only_linear server_only_tree \
            specedge_linear specedge_tree dip_sd \
  --resolved-config-out outputs/resolved_config_homogeneous.json
```

The audit rejects non-canonical methods, server-only batch sizes other than one, missing
`specexec_approx` tree labels, burst/non-Poisson formal online scenarios, missing local or cached
models, shared target/drafter physical devices, fake runners, absent/invalid seeds, invalid request
counts/output lengths/arrival rates, absent time units, fixed-device-assignment violations, and
non-JSON-exportable resolved configs. It aggregates errors and exits nonzero; it never repairs or
falls back from an invalid setting.

Every formal cell must additionally satisfy all of the following post-run checks:

1. Every committed token trace equals the paired `target_only` greedy trace token by token.
2. No request ends with pending, in-flight, unverified, or uncommitted state.
3. No physical or virtual resource has an illegal overlap.
4. Event time and per-resource intervals are monotonic.
5. Scheduling uses no future acceptance result or future arrival oracle.
6. The Hugging Face runner is real; fake/test runners are forbidden.
7. The fully merged configuration is written as the resolved config with canonical methods,
   scenario, full Git commit, and `use_fake_model_runner=false`.
8. `outputs/environment_manifest.json` is retained with the result directory, and its Git commit
   matches the resolved config/run manifest.

Failure of any gate invalidates the whole `scenario x method` cell. There is no silent fallback,
partial-result reporting, or relabeling of an invalid run.

## 9. Small formal-configuration preflight

Run the residency check and 24-cell preflight with:

```bash
python scripts/check_drafter_residency.py
PYTHON_BIN=/root/miniforge3/envs/edge-spec/bin/python \
  LOCAL_FILES_ONLY=true bash scripts/run_baseline_preflight.sh
python scripts/verify_baseline_preflight.py
```

The preflight fixes two scenarios, seeds `20260628`/`20260629`, 16 requests, 32 output tokens,
and Poisson arrivals. All other virtual devices, drafter profiles, network/verification settings,
Server-only batch size, SpecEdge proactive configuration, DiP-SD optimizer/barriers, and tree
labels come from the formal configuration unchanged. The shell invokes the real runner once per
scenario/seed and runs all six methods in that process; model-loading wall time is excluded from
virtual decode time.

Each `outputs/baseline_preflight/<scenario>/<seed>/<method>/` directory contains the resolved
config, environment/run manifests, narrow decode-only metrics, request metrics, event/token traces,
resource timeline, and stdout log. `summary.csv` and `summary.md` contain no TTFT or first-token
column. This preflight validates the pipeline and definitions only; its values are not paper
performance results.

The 2026-06-28 run completed all 24 real-runner cells and all expected output tokens, with finite
nonnegative metrics and valid time units/config markers. Verification failed because longer
bfloat16 traces are not greedily identical across the KV-cache, equal-length batch, and tree-mask
target kernels, and because DiP-SD starts simultaneous drafts for two requests assigned to one
virtual device. No method parameter or algorithm was changed to hide either failure.

## 10. Values requiring operator confirmation before the first formal run

The logical experiment values above are fixed. The following facts cannot be frozen from source
control and must be confirmed from the intended experiment host:

- actual `cuda:0`/`cuda:1` GPU models and memory and driver/runtime compatibility on any new host;
- the exact Python/PyTorch/Transformers/CUDA versions to retain as the publication environment;
- local cache/path availability for every pinned model/tokenizer revision;
- any downstream schema that requests TTFT; this repository must emit `NA` rather than introduce
  a decode-first-token replacement.

These are operator decisions or observations, not values the simulator may infer or replace.
