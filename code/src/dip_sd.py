from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from math import ceil, inf
from typing import Mapping, Sequence


@dataclass(frozen=True)
class DipSDModelProfile:
    decoder_blocks: int = 1
    hidden_size: int = 1
    ffn_hidden_size: int = 1


@dataclass(frozen=True)
class DipSDUser:
    request_id: int
    prefix_length: int
    acceptance_estimate: float
    communication_latency_ms: float
    draft_latency_scale: float
    draft_latency_overhead_ms: float
    draft_model: DipSDModelProfile = field(default_factory=DipSDModelProfile)


@dataclass(frozen=True)
class DipSDProblem:
    users: tuple[DipSDUser, ...]
    target_model: DipSDModelProfile = field(default_factory=DipSDModelProfile)
    target_latency_scale: float = 0.0
    target_latency_overhead_ms: float = 1.0
    target_memory_cap: float = inf
    min_draft_length: int = 1
    max_draft_length: int = 4
    initial_draft_length: int = 1
    max_batch_size: int | None = None
    max_batch_count: int | None = None
    outer_tolerance: float = 1e-9
    dinkelbach_tolerance: float = 1e-9
    max_outer_iterations: int = 20
    max_dinkelbach_iterations: int = 50
    max_optimizer_states: int = 5_000_000


@dataclass(frozen=True)
class DipSDEpochPlan:
    batches: tuple[tuple[int, ...], ...]
    draft_lengths: dict[int, int]
    objective: float = 0.0
    expected_useful_tokens: float = 0.0
    pipeline_span: float = 0.0
    optimizer: str = "fixed_greedy"
    assignment: dict[int, int] = field(default_factory=dict)
    batch_ready_times: tuple[float, ...] = ()
    verify_times: tuple[float, ...] = ()
    stage_durations: tuple[float, ...] = ()
    max_draft_lengths: tuple[int, ...] = ()
    max_prefix_lengths: tuple[int, ...] = ()
    memory_usage: tuple[float, ...] = ()
    feasible: bool = True
    diagnostics: tuple[str, ...] = ()


def optimize_dip_sd(problem: DipSDProblem) -> DipSDEpochPlan:
    _validate_problem(problem)
    best: DipSDEpochPlan | None = None
    for batch_count in _candidate_batch_counts(problem):
        candidate = _solve_fixed_batch_count(problem, batch_count)
        if _is_better(candidate, best):
            best = candidate
    if best is None:
        raise ValueError("dip_sd problem has no feasible plan")
    return best


def expected_useful_tokens(alpha: float, draft_length: int) -> float:
    return _expected_useful_tokens(alpha, draft_length)


def draft_flops(model: DipSDModelProfile, prefix_length: int) -> float:
    return float(
        4
        * model.decoder_blocks
        * model.hidden_size
        * (2 * model.hidden_size + int(prefix_length) + 1 + model.ffn_hidden_size)
    )


def verify_flops(
    model: DipSDModelProfile,
    max_draft_length: int,
    max_prefix_length: int,
) -> float:
    return float(
        4
        * model.decoder_blocks
        * model.hidden_size
        * int(max_draft_length)
        * (
            2 * model.hidden_size
            + int(max_prefix_length)
            + int(max_draft_length)
            + model.ffn_hidden_size
        )
    )


def evaluate_plan(
    problem: DipSDProblem,
    batches: Sequence[Sequence[int]],
    draft_lengths: Mapping[int, int],
    *,
    optimizer: str = "paper_exact",
    diagnostics: Sequence[str] = (),
) -> DipSDEpochPlan:
    normalized_batches = tuple(tuple(int(request_id) for request_id in batch) for batch in batches)
    user_by_id = _user_by_id(problem)
    assignment = _assignment_from_batches(normalized_batches)
    _validate_batches(problem, normalized_batches)
    _validate_draft_lengths(problem, draft_lengths)

    ready_times: list[float] = []
    verify_times: list[float] = []
    max_lengths: list[int] = []
    max_prefixes: list[int] = []
    memory_usage: list[float] = []
    for batch in normalized_batches:
        users = [user_by_id[request_id] for request_id in batch]
        max_length = max(int(draft_lengths[user.request_id]) for user in users)
        max_prefix = max(user.prefix_length for user in users)
        batch_ready = max(
            user_draft_latency(user, int(draft_lengths[user.request_id]))
            + user.communication_latency_ms
            for user in users
        )
        verify_ms = batch_verify_latency(problem, len(users), max_length, max_prefix)
        memory = target_parameter_memory(problem.target_model) + target_kv_memory(
            problem.target_model,
            len(users),
            max_prefix,
        )
        if memory > problem.target_memory_cap:
            raise ValueError("dip_sd plan exceeds target memory cap")
        ready_times.append(batch_ready)
        verify_times.append(verify_ms)
        max_lengths.append(max_length)
        max_prefixes.append(max_prefix)
        memory_usage.append(memory)

    verify_sum = sum(verify_times)
    readiness_span = max(
        (ready + verify for ready, verify in zip(ready_times, verify_times)),
        default=0.0,
    )
    pipeline_span = max(verify_sum, readiness_span)
    if pipeline_span <= 0:
        raise ValueError("dip_sd pipeline span must be positive")
    if problem.max_optimizer_states <= 0:
        raise ValueError("dip_sd.max_optimizer_states must be positive")
    stage_durations = _stage_durations_with_slack(verify_times, pipeline_span)
    useful = total_expected_useful_tokens(problem, draft_lengths)
    objective = useful / pipeline_span
    return DipSDEpochPlan(
        batches=normalized_batches,
        draft_lengths={int(key): int(value) for key, value in draft_lengths.items()},
        objective=objective,
        expected_useful_tokens=useful,
        pipeline_span=pipeline_span,
        optimizer=optimizer,
        assignment=assignment,
        batch_ready_times=tuple(ready_times),
        verify_times=tuple(verify_times),
        stage_durations=tuple(stage_durations),
        max_draft_lengths=tuple(max_lengths),
        max_prefix_lengths=tuple(max_prefixes),
        memory_usage=tuple(memory_usage),
        feasible=True,
        diagnostics=tuple(diagnostics),
    )


def total_expected_useful_tokens(
    problem: DipSDProblem,
    draft_lengths: Mapping[int, int],
) -> float:
    user_by_id = _user_by_id(problem)
    return sum(
        expected_useful_tokens(user.acceptance_estimate, int(draft_lengths[user.request_id]))
        for user in user_by_id.values()
    )


def user_draft_latency(user: DipSDUser, draft_length: int) -> float:
    return int(draft_length) * (
        user.draft_latency_scale * draft_flops(user.draft_model, user.prefix_length)
        + user.draft_latency_overhead_ms
    )


def batch_verify_latency(
    problem: DipSDProblem,
    batch_size: int,
    max_draft_length: int,
    max_prefix_length: int,
) -> float:
    return (
        problem.target_latency_scale
        * int(batch_size)
        * verify_flops(problem.target_model, max_draft_length, max_prefix_length)
        + problem.target_latency_overhead_ms
    )


def target_parameter_memory(model: DipSDModelProfile) -> float:
    return float(
        model.decoder_blocks
        * (8 * model.hidden_size**2 + 4 * model.hidden_size * model.ffn_hidden_size)
    )


def target_kv_memory(
    model: DipSDModelProfile,
    batch_size: int,
    max_prefix_length: int,
) -> float:
    return float(4 * model.decoder_blocks * model.hidden_size * int(batch_size) * int(max_prefix_length))


def _solve_fixed_batch_count(problem: DipSDProblem, batch_count: int) -> DipSDEpochPlan:
    lengths = {
        user.request_id: _bounded_initial_draft_length(problem)
        for user in _ordered_users(problem)
    }
    previous_objective: float | None = None
    diagnostics: list[str] = [f"batch_count={batch_count}"]
    incumbent: DipSDEpochPlan | None = None
    for outer_iteration in range(problem.max_outer_iterations):
        batches = solve_assignment_subproblem(problem, lengths, batch_count)
        length_plan = solve_draft_length_subproblem(problem, batches)
        candidate = evaluate_plan(
            problem,
            batches,
            length_plan.draft_lengths,
            optimizer="paper_exact",
            diagnostics=(
                *diagnostics,
                f"outer_iterations={outer_iteration + 1}",
                *length_plan.diagnostics,
            ),
        )
        if _is_better(candidate, incumbent):
            incumbent = candidate
        unchanged = candidate.draft_lengths == lengths
        objective_delta = (
            inf
            if previous_objective is None
            else abs(candidate.objective - previous_objective)
        )
        lengths = dict(candidate.draft_lengths)
        previous_objective = candidate.objective
        if unchanged and objective_delta <= problem.outer_tolerance:
            return candidate
    if incumbent is None:
        raise ValueError(f"dip_sd found no feasible plan for batch_count={batch_count}")
    return DipSDEpochPlan(
        **{
            **incumbent.__dict__,
            "diagnostics": (
                *incumbent.diagnostics,
                "outer_iteration_limit_reached",
            ),
        }
    )


def solve_assignment_subproblem(
    problem: DipSDProblem,
    draft_lengths: Mapping[int, int],
    batch_count: int,
) -> tuple[tuple[int, ...], ...]:
    request_ids = tuple(user.request_id for user in _ordered_users(problem))
    state_count = int(batch_count) ** len(request_ids)
    if state_count > problem.max_optimizer_states:
        raise ValueError("dip_sd assignment state space exceeds max_optimizer_states")
    best_plan: DipSDEpochPlan | None = None
    best_batches: tuple[tuple[int, ...], ...] | None = None
    for labels in product(range(batch_count), repeat=len(request_ids)):
        if set(labels) != set(range(batch_count)):
            continue
        batches = tuple(
            tuple(request_id for request_id, label in zip(request_ids, labels) if label == batch_index)
            for batch_index in range(batch_count)
        )
        try:
            candidate = evaluate_plan(
                problem,
                batches,
                draft_lengths,
                optimizer="x_subproblem",
            )
        except ValueError:
            continue
        if _assignment_is_better(candidate, best_plan):
            best_plan = candidate
            best_batches = batches
    if best_batches is None:
        raise ValueError(f"dip_sd x-subproblem infeasible for batch_count={batch_count}")
    return best_batches


def solve_draft_length_subproblem(
    problem: DipSDProblem,
    batches: Sequence[Sequence[int]],
) -> DipSDEpochPlan:
    request_ids = tuple(user.request_id for user in _ordered_users(problem))
    length_values = tuple(range(problem.min_draft_length, problem.max_draft_length + 1))
    state_count = len(length_values) ** len(request_ids)
    if state_count > problem.max_optimizer_states:
        raise ValueError("dip_sd length state space exceeds max_optimizer_states")
    q = 0.0
    best: DipSDEpochPlan | None = None
    diagnostics: list[str] = []
    for iteration in range(problem.max_dinkelbach_iterations):
        iteration_best: DipSDEpochPlan | None = None
        iteration_score: float | None = None
        for values in product(length_values, repeat=len(request_ids)):
            lengths = dict(zip(request_ids, values))
            try:
                candidate = evaluate_plan(
                    problem,
                    batches,
                    lengths,
                    optimizer="l_subproblem_dinkelbach",
                )
            except ValueError:
                continue
            score = candidate.expected_useful_tokens - q * candidate.pipeline_span
            if _dinkelbach_candidate_is_better(score, candidate, iteration_score, iteration_best):
                iteration_score = score
                iteration_best = candidate
        if iteration_best is None or iteration_score is None:
            raise ValueError("dip_sd l-subproblem infeasible")
        best = iteration_best
        diagnostics.append(f"dinkelbach_iterations={iteration + 1}")
        if abs(iteration_score) <= problem.dinkelbach_tolerance:
            break
        q = iteration_best.expected_useful_tokens / iteration_best.pipeline_span
    if best is None:
        raise ValueError("dip_sd l-subproblem produced no solution")
    return DipSDEpochPlan(
        **{
            **best.__dict__,
            "diagnostics": tuple(diagnostics),
        }
    )


def _expected_useful_tokens(alpha: float, draft_length: int) -> float:
    bounded_alpha = max(0.0, min(0.999999, float(alpha)))
    if bounded_alpha == 0.0:
        return 1.0
    return (1.0 - bounded_alpha ** (int(draft_length) + 1)) / (1.0 - bounded_alpha)


def _candidate_batch_counts(problem: DipSDProblem) -> tuple[int, ...]:
    user_count = len(problem.users)
    if user_count == 1:
        return (1,)
    upper = user_count
    if problem.max_batch_count is not None:
        upper = min(upper, int(problem.max_batch_count))
    lower = 2
    if problem.max_batch_size is not None:
        lower = max(lower, ceil(user_count / int(problem.max_batch_size)))
    if lower > upper:
        raise ValueError("dip_sd batch-count range is infeasible")
    return tuple(range(lower, upper + 1))


def _ordered_users(problem: DipSDProblem) -> tuple[DipSDUser, ...]:
    return tuple(sorted(problem.users, key=lambda user: user.request_id))


def _user_by_id(problem: DipSDProblem) -> dict[int, DipSDUser]:
    return {user.request_id: user for user in _ordered_users(problem)}


def _validate_problem(problem: DipSDProblem) -> None:
    if not problem.users:
        raise ValueError("dip_sd requires at least one active user")
    request_ids = [user.request_id for user in problem.users]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("dip_sd request ids must be unique")
    if problem.min_draft_length < 1:
        raise ValueError("dip_sd min_draft_length must be at least 1")
    if problem.max_draft_length < problem.min_draft_length:
        raise ValueError("dip_sd max_draft_length must be >= min_draft_length")
    if problem.max_batch_size is not None and problem.max_batch_size <= 0:
        raise ValueError("dip_sd max_batch_size must be positive")
    if problem.max_batch_count is not None and problem.max_batch_count <= 0:
        raise ValueError("dip_sd max_batch_count must be positive")
    if problem.max_outer_iterations <= 0:
        raise ValueError("dip_sd max_outer_iterations must be positive")
    if problem.max_dinkelbach_iterations <= 0:
        raise ValueError("dip_sd max_dinkelbach_iterations must be positive")
    for user in problem.users:
        if user.prefix_length <= 0:
            raise ValueError("dip_sd prefix_length must be positive")
        if not 0.0 < user.acceptance_estimate < 1.0:
            raise ValueError("dip_sd acceptance_estimate must be in (0, 1)")
        if user.communication_latency_ms < 0.0:
            raise ValueError("dip_sd communication latency must be non-negative")
        if user.draft_latency_scale < 0.0 or user.draft_latency_overhead_ms < 0.0:
            raise ValueError("dip_sd draft latency parameters must be non-negative")
    if problem.target_latency_scale < 0.0 or problem.target_latency_overhead_ms < 0.0:
        raise ValueError("dip_sd target latency parameters must be non-negative")
    if problem.target_memory_cap <= 0.0:
        raise ValueError("dip_sd target_memory_cap must be positive")


def _validate_batches(problem: DipSDProblem, batches: tuple[tuple[int, ...], ...]) -> None:
    if not batches:
        raise ValueError("dip_sd requires non-empty batches")
    request_ids = tuple(user.request_id for user in _ordered_users(problem))
    flattened = tuple(request_id for batch in batches for request_id in batch)
    if tuple(sorted(flattened)) != request_ids:
        raise ValueError("dip_sd assignment must be complete and disjoint")
    if any(not batch for batch in batches):
        raise ValueError("dip_sd batches must be non-empty")
    if problem.max_batch_size is not None and any(
        len(batch) > int(problem.max_batch_size) for batch in batches
    ):
        raise ValueError("dip_sd batch exceeds max_batch_size")


def _validate_draft_lengths(problem: DipSDProblem, draft_lengths: Mapping[int, int]) -> None:
    expected_ids = {user.request_id for user in problem.users}
    if set(draft_lengths) != expected_ids:
        raise ValueError("dip_sd draft lengths must cover every user exactly once")
    for length in draft_lengths.values():
        if not problem.min_draft_length <= int(length) <= problem.max_draft_length:
            raise ValueError("dip_sd draft length outside configured bounds")


def _assignment_from_batches(batches: Sequence[Sequence[int]]) -> dict[int, int]:
    return {
        int(request_id): batch_index
        for batch_index, batch in enumerate(batches)
        for request_id in batch
    }


def _bounded_initial_draft_length(problem: DipSDProblem) -> int:
    return max(
        problem.min_draft_length,
        min(problem.max_draft_length, int(problem.initial_draft_length)),
    )


def _stage_durations_with_slack(
    verify_times: Sequence[float],
    pipeline_span: float,
) -> tuple[float, ...]:
    if not verify_times:
        return ()
    durations = [float(value) for value in verify_times]
    slack = float(pipeline_span) - sum(durations)
    if slack > 0.0:
        durations[-1] += slack
    return tuple(durations)


def _is_better(candidate: DipSDEpochPlan, incumbent: DipSDEpochPlan | None) -> bool:
    if incumbent is None:
        return True
    epsilon = 1e-12
    if candidate.objective > incumbent.objective + epsilon:
        return True
    if abs(candidate.objective - incumbent.objective) > epsilon:
        return False
    if candidate.pipeline_span < incumbent.pipeline_span - epsilon:
        return True
    if abs(candidate.pipeline_span - incumbent.pipeline_span) > epsilon:
        return False
    if len(candidate.batches) != len(incumbent.batches):
        return len(candidate.batches) < len(incumbent.batches)
    if candidate.batches != incumbent.batches:
        return candidate.batches < incumbent.batches
    return tuple(sorted(candidate.draft_lengths.items())) < tuple(
        sorted(incumbent.draft_lengths.items())
    )


def _assignment_is_better(
    candidate: DipSDEpochPlan,
    incumbent: DipSDEpochPlan | None,
) -> bool:
    if incumbent is None:
        return True
    epsilon = 1e-12
    if candidate.pipeline_span < incumbent.pipeline_span - epsilon:
        return True
    if abs(candidate.pipeline_span - incumbent.pipeline_span) > epsilon:
        return False
    return candidate.batches < incumbent.batches


def _dinkelbach_candidate_is_better(
    score: float,
    candidate: DipSDEpochPlan,
    incumbent_score: float | None,
    incumbent: DipSDEpochPlan | None,
) -> bool:
    if incumbent_score is None or incumbent is None:
        return True
    epsilon = 1e-12
    if score > incumbent_score + epsilon:
        return True
    if abs(score - incumbent_score) > epsilon:
        return False
    return _is_better(candidate, incumbent)
