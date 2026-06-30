from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Sequence
from typing import Any

from src.config import resolve_target_latency_profile_path
from src.entities import Device
from src.verification_latency_profile import VerificationLatencyProfile


def draft_latency_ms(device: Device, gamma: int) -> float:
    if gamma < 0:
        raise ValueError("gamma must be >= 0")
    return device.draft_startup_ms + 1000.0 * gamma / device.draft_token_rate_tok_s


def verify_latency_ms(edge: dict[str, Any], target_verify_nodes: Sequence[int]) -> float:
    if not target_verify_nodes:
        raise ValueError("verify batch must not be empty")
    work_units = sum(max(1, int(value)) for value in target_verify_nodes)
    return float(edge["verify_startup_ms"]) + (
        1000.0
        * work_units
        / float(edge["target_only_token_rate_tok_s"])
    )


def target_only_latency_ms(edge: dict[str, Any], output_tokens: int) -> float:
    if output_tokens < 0:
        raise ValueError("output_tokens must be >= 0")
    return float(edge["target_only_startup_ms"]) + (
        1000.0 * output_tokens / float(edge["target_only_token_rate_tok_s"])
    )


class TargetLatencyModel:
    """Shared fixed-capacity target latency facade for one simulator."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.edge = config["edge"]
        target = config.get("target_latency", {"mode": "analytical"})
        self.mode = str(target.get("mode", "analytical"))
        self._profile: VerificationLatencyProfile | None = None
        if self.mode == "profile":
            path = resolve_target_latency_profile_path(str(target["profile_path"]))
            self._profile = VerificationLatencyProfile(
                path,
                metric=str(target.get("metric", "p50_ms")),
            )

    def target_decode_latency_ms(
        self,
        *,
        context_lengths: Sequence[int],
        output_tokens: int = 1,
    ) -> float:
        if self._profile is None:
            return target_only_latency_ms(self.edge, output_tokens)
        if output_tokens != 1:
            raise ValueError(
                "profile target decode represents exactly one output token"
            )
        return self._profile.query(
            "target_decode",
            batch_size=len(context_lengths),
            context_lengths=context_lengths,
        ).total_latency_ms

    def linear_verification_latency_ms(
        self,
        *,
        context_lengths: Sequence[int],
        gamma: int,
        analytical_work_units: Sequence[int],
    ) -> float:
        if self._profile is None:
            return verify_latency_ms(self.edge, analytical_work_units)
        return self._profile.query(
            "linear_verification",
            batch_size=len(context_lengths),
            context_lengths=context_lengths,
            gamma=gamma,
        ).total_latency_ms

    def tree_verification_latency_ms(
        self,
        *,
        context_lengths: Sequence[int],
        tree_nodes: int,
        analytical_work_units: Sequence[int],
    ) -> float:
        if self._profile is None:
            return verify_latency_ms(self.edge, analytical_work_units)
        result = self._profile.query(
            "tree_verification",
            batch_size=len(context_lengths),
            context_lengths=context_lengths,
            tree_nodes=tree_nodes,
        )
        if result.tree_mode != "fixed_forward_approx":
            raise ValueError("tree latency profile must use fixed_forward_approx")
        return result.total_latency_ms


class AcceptanceWindowEstimator:
    """Per-request sliding acceptance history with drafter priors for cold start."""

    def __init__(self, window_rounds: int) -> None:
        if window_rounds <= 0:
            raise ValueError("acceptance window must be positive")
        self.window_rounds = window_rounds
        self._history: dict[int, deque[tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=window_rounds)
        )

    def observe(self, request_id: int, accepted_count: int, proposed_count: int) -> None:
        if accepted_count < 0 or proposed_count < 0 or accepted_count > proposed_count:
            raise ValueError("invalid acceptance observation")
        if proposed_count:
            self._history[request_id].append((accepted_count, proposed_count))

    def estimate(self, request_id: int, prior: float) -> float:
        history = self._history.get(request_id)
        if not history:
            return prior
        accepted = sum(item[0] for item in history)
        proposed = sum(item[1] for item in history)
        return accepted / proposed if proposed else prior


def expected_emitted_tokens(alpha: float, gamma: int) -> float:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if alpha == 1.0:
        return float(gamma + 1)
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)
