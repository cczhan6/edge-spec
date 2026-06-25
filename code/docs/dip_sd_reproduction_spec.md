# DiP-SD Reproduction Specification

Date: 2026-06-25

Source paper: Yaodan Xu, Sheng Zhou, Zhisheng Niu, "DiP-SD:
Distributed Pipelined Speculative Decoding for Efficient LLM Inference at the
Edge", arXiv:2604.20919v1.

This document is the semantic contract for the canonical `dip_sd` method. The
public method name `dip_sd` must mean the original paper method. Static,
round-robin, greedy, or heuristic synchronized pipelines are not acceptable
substitutes.

## 1. Paper-to-code Matrix

| Paper item | Meaning | Current code | Status | Planned implementation |
| --- | --- | --- | --- | --- |
| `M`, `mathcal{M}={1,...,M}` | Fixed active user/request cohort in one optimization horizon | `_run_dip_sd_greedy` uses online `active` list and epoch admission | PARTIAL | `DipSDProblem.users` built from a fixed cohort; online wrapper, if added later, must be `dip_sd_online` |
| `N`, `mathcal{N}={1,...,N}`, `N in {2,...,M}` | Number of ordered verification batches/stages | `optimize_epoch_plan(max_batch_count=...)` scans bounded counts; fixed path accepts configured count | PARTIAL | `scan_batch_counts(problem)` enumerates all feasible `N` in the paper range, with deterministic tie-breaking |
| `x_mn in {0,1}` | User-to-batch assignment | `_assign_for_lengths` greedy load assignment | FAIL | `solve_assignment_subproblem(problem, fixed_lengths, N)` solves paper x-subproblem |
| `sum_n x_mn = 1` | Every user belongs to exactly one batch | Current plans usually partition active ids; not enforced by a paper model | PARTIAL | `validate_assignment(plan)` and optimizer constraints enforce complete/disjoint assignment |
| `b_n=sum_m x_mn`, `b_n >= 1` | Batch sizes and non-empty declared batches | Current plans filter empty batches; fallback can reduce count | PARTIAL | Feasibility constraints require every selected `N` batch to be non-empty |
| `l_m in N_+` and `l_m in {1,...,L_ub}` | Per-user integer draft length | Current `draft_lengths` dict exists but is selected by heuristic search/fixed config | PARTIAL | `solve_draft_length_subproblem(problem, fixed_assignment)` implements paper l-subproblem |
| `L_n >= l_m x_mn` | Batch max draft length | Current span uses `max(draft_lengths)` but not as an explicit auxiliary variable | PARTIAL | `derive_batch_auxiliaries(plan)` computes and validates `L_n` |
| `I_n >= i_m x_mn` | Batch max prefix length | Current optimizer ignores prefix length in objective/span | MISSING | `derive_batch_auxiliaries(plan)` computes `I_n`; simulator passes current context length |
| `alpha_m` | User acceptance parameter | Uses configured device acceptance prior | PARTIAL | `DipSDUser.acceptance_estimate` comes from calibration/profile/past-only input, never future outcomes |
| `u_m(l_m)=(1-alpha_m^(l_m+1))/(1-alpha_m)` | Expected accepted/useful tokens with bonus | `_expected_useful_tokens` matches this form | PASS | Keep as `expected_useful_tokens(alpha, draft_length)` with tests for edge cases |
| `U(l)=sum_m u_m(l_m)` | Total expected useful tokens | Current `useful=sum(...)` | PASS | Keep as `total_expected_useful_tokens(problem, lengths)` |
| `F^d(i)` | Draft model compute primitive | Not modeled; current optimizer uses only draft length | MISSING | `draft_flops(model_profile, prefix_length)` |
| `F^v(L,I)` | Verification compute primitive | Not modeled; current optimizer uses simplified length span | MISSING | `verify_flops(model_profile, max_draft_length, max_prefix_length)` |
| `G_m(b,F)=c_m bF+beta_m` | Device-side latency affine model | Current simulator has analytical `draft_latency_ms`; optimizer does not receive paper parameters | MISSING | `draft_latency(user, draft_length)` with profiled `(c_m,beta_m)` |
| `G^v(b,F)=c^v bF+beta^v` | Server verify latency affine model | Simulator has `verify_latency_ms`; optimizer uses `_batch_span` | MISSING | `verify_latency(problem, batch_aux)` |
| `tau_m^d=l_m G_m(1,F^d(i_m))` | Per-user serial local drafting latency | Simulator computes draft event latency; optimizer ignores device/profile timing | PARTIAL | `user_draft_ready_time(user, l_m)=tau_m^d+tau_m^c` |
| `tau_m^c` | Per-round communication latency | Simulator computes uplink/downlink delays; optimizer ignores them | PARTIAL | `DipSDUser.communication_latency_ms`; optimizer includes ready-time constraints |
| `t_n^d >= x_mn(tau_m^d+tau_m^c)` | Batch ready time governed by slowest member | Simulator uses max edge arrival; optimizer does not model it | PARTIAL | `batch_ready_time(batch)=max(user_draft_ready_time)` |
| `t_n^v=G^v(b_n,F^v(L_n,I_n))` | Batch verification latency | Simulator can batch verify; optimizer uses simplified `_batch_span` | FAIL | `batch_verify_time(problem, batch_aux)` |
| `T_n >= t_n^v` | Stage duration auxiliary | Not explicit | MISSING | `stage_duration_constraints(plan)` |
| `S=sum_n T_n` | Total pipeline span/cycle time | Current `_pipeline_span` is sum of simplified batch spans | FAIL | `pipeline_span(problem, plan)` |
| `S >= t_n^d+t_n^v` | Pipeline precedence/draft-readiness feasibility | Current optimizer does not implement; simulator partially enforces readiness | PARTIAL | `pipeline_feasibility_constraints(plan)` |
| Memory constants `Gamma_p^v`, `Gamma_kv,n^v`, `Gamma_max^v` | Target model parameter/KV memory cap | Not modeled | MISSING | `memory_usage(problem, batch_aux)` and `memory_feasible` |
| `R=U(l)/S` | Throughput objective | Current objective is useful/simplified span | PARTIAL | `objective(problem, plan)=total_expected_useful_tokens/pipeline_span` |
| `P: max_{x,l,N,aux} R` | Joint optimization problem | Current `optimize_epoch_plan` is heuristic | FAIL | `optimize_dip_sd(problem)` |
| `P(N)` | Fixed-N joint problem | No paper-equivalent | MISSING | `solve_fixed_batch_count(problem, N)` |
| x-subproblem `argmin S(N,l^(r))` | Assignment update with fixed lengths | Greedy assignment | FAIL | Exact small-M solver plus MILP-compatible interface; no greedy substitute |
| l-subproblem `argmax U(l)/S(l|x)` | Draft-length update with fixed assignment | Product enumeration over lengths with heuristic assignment | FAIL | Binary reformulation plus Dinkelbach/equivalent exact bounded solver |
| Binary `y_mk`, `z_nk` | Draft length and max-length reformulation | Not present | MISSING | `solve_draft_length_subproblem` uses these logical variables or exact enumeration equivalent for bounded small cohorts |
| Dinkelbach updates `q^(t+1)=U/S`, stop on `|U-qS|<=epsilon` | Fractional-programming solver | Not present | MISSING | `dinkelbach_solve_lengths(...)` |
| Batch count selection `N*=argmax_N R*(N)` | Final deployment decision | Current scan chooses best heuristic plan | PARTIAL | `optimize_dip_sd` returns `N_star`, `assignment`, `draft_lengths`, `R_star` |
| Algorithm 1 | Scan `N`, alternate x/l subproblems, return best | Not implemented faithfully | FAIL | M16 implementation target |

## 2. Optimization Variables

The implementation must use the paper's variables directly:

- `N`: number of ordered server verification batches/stages. Candidate range is
  `N in {2,...,M}` for `M >= 2`. If `M == 1`, the implementation must explicitly
  use a single degenerate batch and record that the paper scan range is not
  applicable.
- `x_mn`: binary user-to-batch assignment. `x_mn=1` means request/user `m` is
  verified in batch/stage `n`.
- `l_m`: integer draft length for user `m`.
- `b_n`: batch size, derived by `b_n=sum_m x_mn`.
- `L_n`: max draft length in batch `n`.
- `I_n`: max prefix/context length in batch `n`.
- `tau_m^d`: local serial draft latency for user `m`.
- `tau_m^c`: per-round communication latency for user `m`.
- `t_n^d`: ready time for batch `n`, governed by the slowest member.
- `t_n^v`: verification latency for batch `n`.
- `T_n`: stage duration.
- `S`: total pipeline span.
- Auxiliary variables from the paper's l-subproblem: `y_mk`, `z_nk`, and
  Dinkelbach scalar `q`.

Project-local names must map one-to-one to these variables. They may not replace
the paper variables with a round-robin grouping, a uniform `gamma`, or a
load-balancing heuristic.

## 3. Objective Function and Constraints

### Expected useful tokens

Paper formula:

```text
u_m(l_m) = (1 - alpha_m^(l_m + 1)) / (1 - alpha_m)
U(l) = sum_m u_m(l_m)
```

Planned functions:

- `expected_useful_tokens(alpha: float, draft_length: int) -> float`
- `total_expected_useful_tokens(problem: DipSDProblem, lengths: Mapping[int, int]) -> float`

Current code status: `_expected_useful_tokens` is reusable after renaming/export
and adding tests.

### Compute and latency primitives

Paper formulas:

```text
F^d(i) = 4 J_d h1_d (2 h1_d + i + 1 + h2_d)
F^v(L,I) = 4 J_v h1_v L (2 h1_v + I + L + h2_v)
G_m(b,F) = c_m b F + beta_m
G^v(b,F) = c^v b F + beta^v
tau_m^d = l_m G_m(1, F^d(i_m))
t_n^v = G^v(b_n, F^v(L_n, I_n))
```

Planned functions:

- `draft_flops(model: DipSDModelProfile, prefix_length: int) -> float`
- `verify_flops(model: DipSDModelProfile, max_draft_length: int, max_prefix_length: int) -> float`
- `draft_token_latency(user: DipSDUser) -> float`
- `user_draft_latency(user: DipSDUser, draft_length: int) -> float`
- `batch_verify_latency(problem: DipSDProblem, batch: DipSDBatchPlan) -> float`

Current code status: missing from optimizer. Simulator has separate analytical
latency helpers, but the optimizer does not receive or model paper parameters.

### Assignment and batch aggregation

Paper constraints:

```text
x_mn in {0,1}
sum_n x_mn = 1
l_m in N_+
b_n = sum_m x_mn, b_n >= 1
L_n >= l_m x_mn
I_n >= i_m x_mn
```

Planned functions:

- `validate_assignment(problem, assignment, batch_count)`
- `derive_batch_auxiliaries(problem, assignment, lengths)`
- `assignment_complete_and_disjoint(problem, assignment)`

Current code status: fixed partition and heuristic assignment are not sufficient.

### Latency coupling and pipeline span

Paper constraints:

```text
t_n^d >= x_mn (tau_m^d + tau_m^c) for all m,n
T_n >= t_n^v
S = sum_n T_n
S >= t_n^d + t_n^v for all n
R = U(l) / S
```

Planned functions:

- `batch_ready_time(problem, batch, lengths)`
- `stage_duration(problem, batch, lengths)`
- `pipeline_span(problem, plan)`
- `objective(problem, plan)`
- `pipeline_feasible(problem, plan)`

Current code status: the event simulator enforces some readiness timing, but the
optimizer's `_pipeline_span` is not paper-equivalent.

### Memory constraints

Paper formulas:

```text
Gamma_p^v = J_v(8(h1_v)^2 + 4 h1_v h2_v)
Gamma_kv,n^v = 4 J_v h1_v b_n I_n
Gamma_p^v + Gamma_kv,n^v <= Gamma_max^v
```

Planned functions:

- `target_parameter_memory(model: DipSDModelProfile) -> float`
- `target_kv_memory(model: DipSDModelProfile, batch_size: int, max_prefix_length: int) -> float`
- `batch_memory_feasible(problem, batch_aux)`

Current code status: missing.

### Bounds

Paper implementation uses practical upper bounds including `L_ub`, `I_ub`,
`T_ub`, and `S_ub`.

Planned configuration fields:

- `dip_sd.max_draft_length` maps to `L_ub`.
- `dip_sd.max_prefix_length` maps to `I_ub`.
- `dip_sd.max_stage_ms` maps to `T_ub`.
- `dip_sd.max_pipeline_span_ms` maps to `S_ub`.
- `dip_sd.target_memory_cap` maps to `Gamma_max^v`.

Existing `min_draft_length` can remain as an additional project guard, but the
paper lower bound is `1`.

## 4. Paper Algorithm

The canonical implementation must follow Algorithm 1:

1. Build a fixed active cohort with `M` users and their parameters.
2. For each candidate `N` in `2..M`:
   - initialize `l^(0)` deterministically, preferably from config
     `initial_draft_length` clipped to `1..L_ub`;
   - set outer iteration `r=0`;
   - solve x-subproblem with fixed `l^(r)`:
     `x^(r+1) <- argmin S(N, l^(r))`;
   - solve l-subproblem with fixed `x^(r+1)`:
     `l^(r+1) <- argmax R(N, x^(r+1))`;
   - update `r`;
   - stop when `l` is unchanged and throughput change is under tolerance, or
     when the configured iteration limit is reached with a recorded diagnostic.
3. Compute converged `R*(N)`.
4. Select `N* = argmax_N R*(N)`.
5. Return `(N*, x*, l*, R*)` plus feasibility diagnostics.

Subproblem requirements:

- x-subproblem: exact MILP-compatible solve of the assignment problem for fixed
  lengths. For this repository, exact exhaustive enumeration is acceptable for
  small bounded cohorts if it is mathematically equivalent and tested against a
  separate brute-force oracle. Greedy load balancing is not acceptable.
- l-subproblem: use binary reformulation with `y_mk` and `z_nk`, and solve the
  fractional objective using Dinkelbach or an exact bounded enumeration that is
  equivalent for the configured finite domain. The implementation must expose
  Dinkelbach diagnostics if the Dinkelbach route is used.
- Batch count selection: scan the paper range and select the highest converged
  throughput. Tie-breaking must be deterministic: higher `R`, then smaller `S`,
  then smaller `N`, then lexicographic assignment, then lexicographic lengths.
- Complexity: exhaustive exact enumeration is allowed only while the configured
  `M` and `L_ub` are small enough for tests and small experiments. If a scalable
  MILP backend is later added, it must preserve the same objective and tests.

## 5. Acceptance Parameters

Paper parameter:

- `alpha_m in (0,1)` is the expected token acceptance rate for user `m`.

Allowed sources:

1. Calibration/profile input computed before test execution.
2. A configured profile value that is independent of current test outcomes.
3. A causal estimate using only previous verification rounds, if a future online
   variant is implemented.

Forbidden:

- reading realized acceptance from future rounds;
- using target verification outcomes from the same optimization horizon;
- changing `alpha_m` after seeing current cohort target outputs.

Planned implementation:

- `DipSDUser.acceptance_estimate` is passed explicitly.
- `optimize_dip_sd(problem)` has no access to `ModelRunner` or generated target
  tokens.
- Tests use sentinel/failing model runners to prove optimizer decisions depend
  only on configured estimates.

## 6. System Assumptions

The paper assumes an approximately stationary active cohort over an optimization
horizon. Therefore canonical `dip_sd` must first implement fixed-cohort behavior:

- form a cohort of active requests;
- optimize `(N, x, l)` for that cohort;
- execute synchronized draft/verify rounds using that plan;
- do not mix newly arrived requests into the cohort mid-horizon;
- if online arrival support is needed later, name it `dip_sd_online`.

Decode-only scope remains unchanged:

- prompt prefill is outside simulated time;
- prefix length `i_m` is the current context length at the beginning of the
  optimization horizon;
- actual verification still uses the real/fake target model and must remain
  lossless against target-only greedy decoding.

## 7. Gap List

### Reusable current pieces

- `src.dip_sd._expected_useful_tokens` matches paper formula (1).
- `DipSDEpochPlan` concept can evolve into a richer `DipSDPlan`.
- Simulator draft/upload/batch-verify/download/state-update event skeleton can
  be reused after optimizer integration.
- Existing tests for sync-before-redraft and greedy equality remain useful as
  regression tests.

### Must rewrite or replace

- `optimize_epoch_plan`: replace heuristic search/greedy assignment with
  paper-equivalent optimization.
- `_assign_for_lengths`: remove from canonical path; it is greedy and fails the
  paper assignment subproblem.
- `_pipeline_span` and `_batch_span`: replace with paper equations for
  readiness, verification, stage duration, memory feasibility, and span.
- `Simulator._run_dip_sd_greedy`: rename/refactor so canonical `dip_sd` runs the
  paper plan. The public `dip_sd_greedy` method must be removed from formal
  registry in M18.
- Config `dip_sd.optimizer: deterministic_search`: replace with explicit paper
  solver settings and profile parameters.

### Static/heuristic substitute pieces

- `build_fixed_epoch_plan` is not a valid public `dip_sd` implementation.
- `dip_sd_greedy` is not a valid public method under the current project goal.
- Existing deterministic search is a heuristic and must not be reported as
  DiP-SD.

### Optimizer not connected to simulation

- Current optimizer does not receive prefix lengths, communication latency,
  draft model cost parameters, target verification parameters, or memory cap.
- Current event trace records optimizer objective but does not guarantee trace
  span matches the optimizer's paper model.

### Test gaps

- No optimizer feasibility test over paper variables.
- No complete/disjoint assignment test over `x_mn`.
- No non-empty batch test for each selected `N`.
- No draft-length bound test for all users.
- No independent brute-force tiny-case oracle.
- No paper objective/manual-case test.
- No no-future-acceptance test at optimizer boundary.
- No event trace test proving simulator uses optimizer assignment and per-user
  draft lengths.
- No trace-span-vs-optimizer-span test.
