from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Sequence
from typing import Any

from src.entities import Device


def draft_latency_ms(device: Device, gamma: int) -> float:
    if gamma < 0:
        raise ValueError("gamma must be >= 0")
    return device.draft_startup_ms + 1000.0 * gamma / device.draft_token_rate_tok_s


def verify_latency_ms(edge: dict[str, Any], gammas: Sequence[int]) -> float:
    if not gammas:
        raise ValueError("verify batch must not be empty")
    return float(edge["verify_startup_ms"]) + (
        1000.0
        * len(gammas)
        / float(edge["target_only_token_rate_tok_s"])
    )


def target_only_latency_ms(edge: dict[str, Any], output_tokens: int) -> float:
    if output_tokens < 0:
        raise ValueError("output_tokens must be >= 0")
    return float(edge["target_only_startup_ms"]) + (
        1000.0 * output_tokens / float(edge["target_only_token_rate_tok_s"])
    )


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
