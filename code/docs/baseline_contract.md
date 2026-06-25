# Baseline Contract

**Status:** Draft v0.1  
**Scope:** Decode-only evaluation  
**Baselines:** `target_only`, `server_only`, `specedge`, `dip_sd`  
**Project:** Edge–server collaborative speculative decoding

---

## 1. Purpose

This document fixes the execution semantics, deployment assumptions, scheduling rules, and correctness requirements of all baselines before implementation and performance evaluation.

The contract serves four purposes:

1. prevent baseline behavior from changing implicitly during development;
2. separate algorithmic differences from implementation accidents;
3. make all adaptations to the original papers or repositories explicit;
4. provide testable invariants for each baseline.

A code change that alters any behavior specified here must update this document and the resolved experiment manifest.

---

## 2. Global evaluation scope

### 2.1 Decode-only boundary

All experiments start from the autoregressive decoding stage.

For every request:

- the edge drafter, when present, already has the prompt prefix and its local prefix KV cache;
- the server target model already has the prompt prefix and its target prefix KV cache;
- prompt prefill computation is not simulated;
- initial prompt transmission is not simulated;
- prefix KV-cache construction is not simulated;
- TTFT that includes prefill is not reported.

The prompt is retained only as semantic context for real-model token generation and verification.

The simulated request arrival time is therefore also its decode-ready time.

### 2.2 Shared semantic rules

All speculative baselines must use the same target-model verification semantics.

Only target-verified tokens may be committed. For greedy decoding, every completed output must be identical to the output of `target_only` under the same prompt, target model, stopping rule, and maximum output length.

The shared verification layer must return, at minimum:

```python
@dataclass
class VerificationResult:
    accepted_count: int
    committed_tokens: list[int]
    correction_token: int | None
    bonus_token: int | None
```

Tree-based methods may additionally return the accepted path and tree-node indices, but they must obey the same final target-model semantics.

### 2.3 Shared workload rules

For one experiment group, all methods must use the same:

- prompts and request identifiers;
- request arrival trace;
- maximum output lengths and stopping rules;
- target model and tokenizer;
- random seeds;
- virtual edge-device profiles;
- network profiles, where communication exists;
- measurement window and metric definitions.

A method must not use future acceptance outcomes, future request arrivals, or future completion information for scheduling.

### 2.4 Current resource-comparison status

Server resources are **not yet normalized across methods** in the current development stage.

This means initial results are implementation and mechanism-validation results, not equal-resource performance claims. Every result manifest must record the exact number, placement, and sharing mode of draft and target model resources.

Equal-resource and cost-normalized experiments will be added later as a separate evaluation stage.

### 2.5 Common metrics

The main decode-stage metrics are:

- effective throughput / goodput;
- per-request decode completion latency;
- TPOT or average inter-token latency, if measured consistently;
- target GPU utilization;
- draft-compute utilization;
- communication time and bytes for distributed methods;
- drafted, verified, accepted, committed, and wasted token counts;
- speedup relative to `target_only`.

TTFT including prefill must not be reported.

---

## 3. Baseline naming

Use the following internal method names:

```text
target_only
server_only_linear
server_only_tree
specedge_linear
specedge_tree
dip_sd
```

Recommended display names:

```text
Target-only
Server-only SD-Linear
Server-only SD-Tree
SpecEdge-Linear
SpecEdge-Tree
DiP-SD (online adaptation)
```

`server_only` must not be shortened to “server” in plots because it is a speculative-decoding method, not target-only autoregressive decoding.

---

# 4. Target-only contract

## 4.1 Role

`target_only` is the non-speculative autoregressive reference and the denominator of latency speedup.

It answers:

> How does each speculative system compare with direct decoding by the target model?

## 4.2 Deployment

- Target model: server.
- Drafter: none.
- Edge computation: none.
- Edge–server communication: none in the decode-only model.
- Server target resource: one logical target service unless a later experiment explicitly changes it.

## 4.3 Execution flow

```text
REQUEST_ARRIVAL
    ↓
enqueue request at target service
    ↓
target model generates the next token
    ↓
commit token immediately
    ↓
repeat until stop condition
```

The default implementation uses FCFS request admission and no target-side decode batching unless a separate stronger baseline is explicitly introduced.

The scheduler must not generate draft, verification, tree, batch-verification, or network events.

## 4.4 Required invariants

- Every generated token is produced by the target model.
- No speculative token counters are incremented.
- No network event is created.
- The target resource cannot execute overlapping operations.
- The final sequence is the canonical greedy target output.

---

# 5. Server-only SD contract

## 5.1 Role

`server_only` is the centralized speculative-decoding baseline.

It answers:

> What is the effect of moving drafting from the server to edge devices and changing the distributed scheduling design?

## 5.2 Source basis and declared adaptation

The implementation follows the official SpecEdge repository’s server-only execution pattern:

- a draft model constructs a SpecExec-style candidate tree;
- the target model verifies the candidate tree;
- accepted tokens and a bonus/correction token update the sequence and KV state;
- the next iteration starts after the previous verification completes.

The official example configuration places the draft and target models on separate server GPUs. This project intentionally changes that deployment:

> The draft model and target model are deployed on two separate server GPUs, following the official repository configuration.

The default official example places the target model on `cuda:0` and the draft model on `cuda:1`. This resource placement is retained for both the linear and tree variants.

## 5.3 Deployment

- Draft model: server.
- Target model: server.
- Draft model: one dedicated server draft GPU.
- Target model: one dedicated server target GPU.
- The two models use separate logical GPU resources.
- Edge device: not involved.
- Network communication: none.

Although the models occupy separate GPUs, the official server-only execution loop is round-synchronous: the active batch completes drafting before target verification starts, and the next draft round begins only after verification and state update. The canonical baseline must not introduce proactive drafting or cross-round overlap that is absent from the official implementation.

## 5.4 Candidate structure

Two candidate variants are supported:

- `server_only_linear`: conventional linear speculative decoding;
- `server_only_tree`: the official repository's SpecExec-style tree speculative decoding.

Both variants use the same official two-server-GPU placement: the drafter uses the server draft GPU and the target uses the server target GPU. The default main comparison may use the linear variant, while the tree variant is retained for faithful comparison with native SpecEdge.

For the tree variant, `server_only_tree` uses the same SpecExec-style tree construction parameters as `specedge_tree`:

```text
max_n_beams
max_beam_len
max_branch_width
max_budget
```

Using the same candidate-tree policy prevents candidate quality from becoming an uncontrolled difference between `server_only` and `specedge`.

## 5.5 Execution flow

For each active iteration:

```text
select active request batch
    ↓
server draft GPU constructs candidate tree(s)
    ↓
server target GPU batch-verifies candidate tree(s)
    ↓
commit accepted target-verified path and bonus/correction token
    ↓
update draft and target KV state
    ↓
refill finished batch slots if applicable
    ↓
next iteration
```

There is no proactive drafting. The next draft operation for a request cannot begin until its current target verification and state update have completed.

There is no edge–server communication latency.

## 5.6 Batching rule

`server_only.batch_size` must be explicit in the configuration and result manifest.

The default batching contract is:

- requests are admitted in FCFS order;
- up to `batch_size` active requests are processed together;
- tree drafting for the active batch finishes before target verification begins;
- drafting and target verification use separate GPU resources;
- the canonical official loop still executes the two phases in round order rather than proactively overlapping consecutive rounds;
- completed slots may be refilled at the next iteration boundary.

A batch size of one is valid for correctness testing but must not be silently used as the only performance configuration.

## 5.7 Required invariants

- Draft computation is assigned only to the server draft GPU.
- Target verification is assigned only to the server target GPU.
- For the canonical official implementation, each round's target verification begins after that round's drafting finishes.
- The next round's drafting begins after verification and state update.
- No network event is generated.
- No proactive tree is created.
- A request has at most one unverified candidate tree.
- All committed tokens are target verified.
- With identical tree parameters and model semantics, the final output equals `target_only`.

---

# 6. SpecEdge contract

## 6.1 Role

`specedge` is the distributed tree-based speculative-decoding baseline with proactive edge drafting and server-side batching.

It answers:

> Can the proposed method outperform a representative edge–server system that overlaps edge drafting with server verification and batches target verification?

## 6.2 Source basis

The baseline follows the mechanism exposed by the SpecEdge paper and official repository:

1. each edge client owns its request and runs a lightweight draft model;
2. the client constructs an initial SpecExec-style candidate tree;
3. the tree is uploaded to the server;
4. the server batch-verifies candidate trees with the target model;
5. while waiting, the edge client performs proactive drafting;
6. proactive work is reused only when the target result aligns with the predicted continuation;
7. otherwise, invalid proactive work is discarded;
8. the client updates sequence and KV state and continues.

This is a native tree-based SpecEdge baseline, not a linearized approximation.

## 6.3 Deployment

- One origin edge device per request.
- Draft model: request’s edge device.
- Target model: centralized edge server.
- Each edge device processes only its own request.
- The server stores or logically maintains the target KV state required for every active request.
- Candidate-tree upload and verification-result download are included in communication time.

The prompt is not transmitted because prefix state is assumed to exist before decode simulation starts.

## 6.4 Initial tree drafting

The initial candidate tree uses the official repository’s SpecExec-style logic and the configured limits:

```text
max_n_beams
max_beam_len
max_branch_width
max_budget
```

Candidate expansion is based on cumulative draft-model score subject to the branch, depth, and total-budget limits.

The exact tree configuration must be shared with `server_only` when those two baselines are compared.

## 6.5 Server verification and batching

The server accepts candidate trees from multiple clients and verifies a batch with one target-model forward execution.

Supported batching modes:

### Dynamic batching

- when the server becomes available, it takes currently queued trees up to `max_batch_size`;
- it does not wait for a full batch;
- this is the canonical mode for online-arrival experiments.

### Static batching

- the server waits until the configured batch is full before verification;
- this mode is retained for sensitivity analysis or faithful reproduction of a selected repository configuration;
- timeout behavior, if added, must be declared because it is not equivalent to strict static batching.

The selected mode and `max_batch_size` must be written to every result manifest.

Requests with different tree shapes may be verified together using the required masks and padding. Padding must affect simulated verification cost if the latency model includes padded dimensions.

## 6.6 Proactive drafting

After uploading the initial tree, the client may continue drafting while waiting for server verification.

The proactive root is selected from the client’s predicted best continuation of the initial tree. The client expands a proactive subtree according to the configured proactive depth and budget.

Proactive tokens are speculative only. They must not be exposed as committed output before target verification.

## 6.7 Alignment and reuse rule

When the server result returns, proactive work is reusable only if the verified result exactly connects to the proactive root.

The implementation must check both:

1. the accepted path ends at the initial-tree leaf from which proactive drafting started; and
2. the target-generated bonus/next token equals the proactive root token.

If both conditions hold:

- retain the valid proactive subtree;
- reorder or re-root it relative to the new committed prefix;
- continue from the retained state.

Otherwise:

- discard the proactive subtree;
- update the request to the target-verified prefix;
- start a fresh initial tree.

Partial prefix similarity is not sufficient unless a separate, explicitly documented extension is implemented.

## 6.8 Draft-depth configuration

The official system motivates choosing draft depth according to draft time, network RTT, and server verification time. For reproducibility, the first implementation uses fixed, explicitly configured tree and proactive depths per experiment.

Offline parameter sweeps may select these values, but online access to future verification completion or future acceptance is forbidden.

## 6.9 Required invariants

- Each request originates from exactly one edge device.
- The edge may draft while its earlier tree is awaiting verification.
- No proactive token is committed before target verification.
- Reuse occurs only after exact alignment checks.
- Failed alignment discards invalid proactive state.
- The server never concurrently executes two target verification batches on one logical target resource.
- Within a request, committed output order is preserved.
- Final output equals `target_only` greedy decoding.

---

# 7. DiP-SD contract

## 7.1 Role

`dip_sd` is the distributed pipelined speculative-decoding baseline with joint batch assignment and per-user draft-length optimization.

It answers:

> How does the proposed method compare with a distributed system that pipelines drafting and batched verification across request groups, while retaining a synchronization point for each request after every verification round?

## 7.2 Source basis and implementation status

No official DiP-SD implementation is used. This baseline is a paper-based reimplementation.

The paper defines:

- lightweight draft models on user devices;
- a centralized target model at the server;
- several ordered request batches;
- parallel local drafting;
- batch target verification;
- result synchronization and KV-state update;
- optimization of batch count, user-to-batch assignment, and per-user draft length.

The repository and paper version used for implementation must be recorded in the source-attribution document.

## 7.3 Deployment

- One origin edge device per request.
- Draft model: request’s edge device.
- Target model: centralized server.
- Candidate sequence upload and result download are included.
- Candidate structure: linear draft sequence, not a tree.
- The target verifies all linear candidates in a selected batch in one batched forward execution.

## 7.4 Per-request round

For request \(m\), one round is strictly:

```text
local draft of l_m tokens
    ↓
upload linear candidate sequence
    ↓
wait for request's assigned batch turn
    ↓
batch target verification
    ↓
download accepted tokens and bonus/correction token
    ↓
update request sequence and draft/target KV state
    ↓
start next local draft round
```

A request cannot begin its next draft round before receiving and applying the current round’s verification result.

Therefore DiP-SD does **not** create multiple dependent, unverified segments for the same request. Its pipeline comes from interleaving different batches, not from speculative advancement of one request across several verification rounds.

## 7.5 Ordered batch pipeline

At one optimization epoch, the active requests are partitioned into \(N\) non-empty ordered batches:

```text
B1, B2, ..., BN
```

Every active request belongs to exactly one batch.

The server visits batches in fixed cyclic order. For batch \(B_n\):

- all member requests draft in parallel on their own devices;
- the batch is ready only when every member’s candidate has arrived;
- target verification starts when both the batch is ready and the server reaches its turn;
- after verification, members update state and begin their next round;
- meanwhile, requests in other batches may continue local drafting.

The implementation must model bubbles caused by slow members, communication, or target-server availability.

The first faithful implementation must not silently skip an unready batch. A skip-ready scheduling extension would be a different method.

## 7.6 Draft-length and acceptance model

Every request \(m\) receives an integer draft length \(l_m\) within configured limits.

The scheduler uses an estimated acceptance parameter \(\hat{\alpha}_m\), not the future realized acceptance.

Allowed sources of \(\hat{\alpha}_m\):

1. a calibration split measured before the test run;
2. a configured profile indexed by task and draft-model type;
3. a causal moving estimate using only previously completed verification rounds.

The selected estimator must be fixed for one experiment and written to the manifest.

Real-model verification determines actual accepted tokens. The scheduling estimate must never override semantic verification.

## 7.7 Optimization objective

The paper models the expected useful tokens of request \(m\) under a geometric acceptance approximation as:

\[
u_m(l_m) =
\frac{1-\hat{\alpha}_m^{\,l_m+1}}
     {1-\hat{\alpha}_m}.
\]

For batch \(B_n\), draft readiness is governed by its slowest member, including local draft and communication time. Verification time depends on the batch configuration and draft lengths.

The pipeline span \(S\) must include all ordered verification stages and satisfy the per-stage feasibility constraints. The optimizer maximizes expected useful tokens per pipeline span:

\[
R = \frac{U}{S}.
\]

## 7.8 Solver contract

For an active cohort of \(M\) requests:

1. enumerate feasible batch counts \(N\);
2. initialize user assignment and draft lengths deterministically;
3. for fixed draft lengths, update user-to-batch assignment to reduce pipeline span;
4. for fixed assignment, update integer draft lengths to improve \(U/S\);
5. use the paper’s fractional-programming/Dinkelbach procedure or an equivalent exact implementation for the draft-length subproblem;
6. alternate until convergence or a declared iteration limit;
7. choose the feasible \(N\) with the highest objective.

All tie-breakers and iteration limits must be deterministic and recorded.

If the full solver is not yet implemented, the method must be named `dip_sd_greedy` rather than `dip_sd`.

## 7.9 Online-arrival adaptation

The paper’s optimization assumes an approximately stationary active cohort during an optimization horizon. This project uses online request arrivals, so the following explicit wrapper is adopted:

### Epoch-barrier adaptation

- batch membership, order, and draft lengths remain fixed during one epoch;
- one epoch ends after every current batch completes one verification turn;
- newly arrived requests wait in an admission queue until the next epoch barrier;
- at the barrier, completed requests are removed;
- waiting requests are admitted subject to the configured active-request limit;
- if membership changed, the optimizer recomputes \(N\), assignment, and draft lengths;
- if membership did not change, the previous solution may be reused.

Results must display the method as:

```text
DiP-SD (online adaptation)
```

and the manifest must contain:

```yaml
adaptation: online_epoch_barrier
```

This wrapper prevents mid-cycle reassignment and preserves the ordered pipeline semantics.

## 7.10 Memory and batch constraints

The optimizer must respect configured limits such as:

```text
max_active_requests
max_batch_size
min_draft_length
max_draft_length
```

If an explicit GPU-memory model has not yet been implemented, `max_batch_size` and `max_active_requests` are treated as declared capacity constraints rather than claiming exact reproduction of the paper’s memory constraint.

## 7.11 Required invariants

- Each request belongs to exactly one batch in an epoch.
- Batches are non-empty and visited in fixed cyclic order.
- A request has at most one unverified draft sequence.
- A request starts its next draft only after synchronization and state update.
- New arrivals do not alter the current epoch’s batch assignment.
- The optimizer never reads future realized acceptance.
- All committed tokens are target verified.
- Final output equals `target_only` greedy decoding.

---

# 8. Cross-baseline comparison table

| Property | Target-only | Server-only SD | SpecEdge | DiP-SD |
|---|---|---|---|---|
| Draft location | None | Server | Edge device | Edge device |
| Target location | Server | Server | Server | Server |
| Candidate form | None | SpecExec tree | SpecExec tree | Linear sequence |
| Draft/target resource | Target only | Separate server draft/target GPUs | Separate edge/server resources | Separate edge/server resources |
| Network during decode | No | No | Yes | Yes |
| Target verification batching | No by default | Configurable | Static or dynamic | Ordered fixed batches |
| Proactive drafting while waiting | No | No | Yes | No |
| Multiple unverified rounds/request | No | No | Proactive subtree only | No |
| Dynamic per-user draft length | No | Tree parameters | Fixed configured tree/proactive limits | Yes, optimizer selected |
| Reconfiguration for arrivals | FCFS queue | Batch refill | Server queue/batcher | Epoch barrier |
| Source status | Project implementation | Official-repo mechanism + shared-GPU adaptation | Official paper/repository | Paper reimplementation |

---

# 9. Required result-manifest fields

Every run must save at least:

```yaml
method:
  name:
  display_name:
  implementation_source:
  source_version:
  adaptation:

scope:
  decode_only: true
  prefill_modeled: false
  prompt_transmission_modeled: false

models:
  target_model:
  draft_model:
  tokenizer:
  decoding_mode: greedy

resources:
  num_target_resources:
  num_draft_resources:
  draft_target_shared_gpu:
  target_concurrency:
  draft_concurrency:

network:
  enabled:
  uplink:
  downlink:
  rtt:

batching:
  enabled:
  type:
  max_batch_size:
  timeout:

candidate:
  type:
  gamma:
  max_n_beams:
  max_beam_len:
  max_branch_width:
  max_budget:
  proactive_depth:
  proactive_budget:

dip_sd:
  adaptation:
  optimizer:
  acceptance_estimator:
  optimization_epoch:
  max_active_requests:
  min_draft_length:
  max_draft_length:

workload:
  dataset:
  request_count:
  arrival_trace:
  output_length:
  seed:

git:
  commit:
  dirty:
```

Unused fields should be written as `null` rather than omitted where practical.

---

# 10. Correctness test matrix

## 10.1 Shared tests

```text
test_same_workload_across_methods
test_decode_only_no_prefill_events
test_committed_tokens_equal_target_greedy
test_no_unverified_token_commit
test_token_accounting_conservation
test_event_time_monotonicity
test_resource_intervals_do_not_overlap
test_deterministic_replay
```

## 10.2 Target-only

```text
test_target_only_has_no_draft_or_network_events
test_target_only_target_resource_serialization
```

## 10.3 Server-only

```text
test_server_only_uses_separate_draft_and_target_resources
test_server_only_round_order_is_draft_then_verify
test_server_only_has_no_network_events
test_server_only_has_no_proactive_state
test_server_only_one_candidate_tree_per_request
```

## 10.4 SpecEdge

```text
test_specedge_proactive_runs_while_waiting
test_specedge_alignment_success_reuses_subtree
test_specedge_alignment_failure_discards_subtree
test_specedge_dynamic_batch_takes_ready_requests
test_specedge_static_batch_waits_for_full_batch
test_specedge_never_commits_proactive_tokens_early
```

## 10.5 DiP-SD

```text
test_dip_sd_partition_is_complete_and_disjoint
test_dip_sd_batch_order_is_cyclic
test_dip_sd_request_waits_for_sync_before_redraft
test_dip_sd_new_arrivals_wait_until_epoch_barrier
test_dip_sd_optimizer_uses_only_estimated_acceptance
test_dip_sd_solver_is_deterministic
```

---

# 11. Implementation order

The recommended implementation order is:

```text
1. freeze this contract
2. validate target_only
3. extract shared target-verification and commit semantics
4. implement shared SpecExec tree builder
5. implement server_only with separate server draft and target GPUs
6. implement SpecEdge initial tree, server batcher, and proactive reuse
7. implement DiP-SD per-request cycle and ordered pipeline
8. implement and test the DiP-SD optimizer
9. integrate the revised proposed method
10. add equal-resource experiments
```

Do not use old SpecEdge performance numbers until the new implementation passes the contract tests.

---

# 12. Deferred decisions

The following issues are intentionally deferred and must not be changed implicitly:

1. equalizing server GPU counts across baselines;
2. cost-normalized comparison;
3. adding continuous batching to `target_only`;
4. evaluating a shared-GPU `server_only` variant as a separately named resource-sensitivity experiment;
5. online adaptive SpecEdge tree depth;
6. skip-ready scheduling for DiP-SD;
7. explicit target-GPU memory modeling;
8. linearized SpecEdge variants.

Each deferred item should be introduced as a new experiment or a separately named method rather than silently replacing the canonical baseline.

---

# 13. Source attribution

Implementation should be checked against:

- **SpecEdge: Scalable Edge-Assisted Serving for Interactive LLMs**, NeurIPS 2025.
- Official repository: `kaist-ina/specedge`.
- **DiP-SD: Distributed Pipelined Speculative Decoding for Efficient LLM Inference at the Edge**, arXiv:2604.20919.

Record exact repository commits and paper versions in the experiment manifest before producing final paper results.
