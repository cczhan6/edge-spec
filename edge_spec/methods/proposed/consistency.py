from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from edge_spec.protocol import DraftSegment
from edge_spec.types import VerificationResult


def stable_prefix_hash(token_ids: list[int]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()


@dataclass
class PrefixCheck:
    status: str
    reason: str = ""

    @property
    def ready(self) -> bool:
        return self.status == "ready"


@dataclass
class RequestState:
    request_id: str
    prompt_ids: list[int]
    generated_ids: list[int] = field(default_factory=list)
    prefix_version: int = 0
    accepted_length_history: list[int] = field(default_factory=list)
    proposed_length_history: list[int] = field(default_factory=list)
    rejected_position_history: list[int] = field(default_factory=list)
    branch_hashes: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.branch_hashes.setdefault(0, stable_prefix_hash(self.prompt_ids))

    @property
    def committed_position(self) -> int:
        return len(self.generated_ids)

    @property
    def committed_prefix_ids(self) -> list[int]:
        return self.prompt_ids + self.generated_ids

    @property
    def prefix_hash(self) -> str:
        return stable_prefix_hash(self.committed_prefix_ids)

    @property
    def acceptance_rate(self) -> float:
        proposed = sum(self.proposed_length_history)
        if proposed == 0:
            return 1.0
        return sum(self.accepted_length_history) / proposed


class PrefixStateManager:
    def __init__(self) -> None:
        self._states: dict[str, RequestState] = {}

    def register_request(self, request_id: str, prompt_ids: list[int]) -> RequestState:
        state = RequestState(request_id=request_id, prompt_ids=list(prompt_ids))
        self._states[request_id] = state
        return state

    def state(self, request_id: str) -> RequestState:
        return self._states[request_id]

    def get(self, request_id: str) -> RequestState | None:
        return self._states.get(request_id)

    def acceptance_rate(self, request_id: str) -> float:
        state = self.get(request_id)
        return state.acceptance_rate if state is not None else 1.0

    def check_segment(self, segment: DraftSegment) -> PrefixCheck:
        state = self.get(segment.request_id)
        if state is None:
            return PrefixCheck("stale", "unknown-request")
        if segment.prefix_version < state.prefix_version:
            return PrefixCheck("stale", "old-prefix-version")
        if segment.prefix_version > state.prefix_version:
            return PrefixCheck("pending", "future-prefix-version")
        if segment.base_position < state.committed_position:
            return PrefixCheck("stale", "already-committed-position")

        expected_hash = state.branch_hashes.get(segment.base_position)
        if expected_hash is None:
            return PrefixCheck("pending", "prefix-position-not-committed")
        if expected_hash != segment.prefix_hash:
            return PrefixCheck("hash-mismatch", "prefix-hash-mismatch")
        return PrefixCheck("ready")

    def apply_verification(
        self,
        segment: DraftSegment,
        verification: VerificationResult,
        emitted_ids: list[int],
    ) -> RequestState:
        state = self.state(segment.request_id)
        check = self.check_segment(segment)
        if not check.ready:
            raise ValueError(f"cannot commit segment: {check.status}:{check.reason}")
        if segment.base_position != state.committed_position:
            raise ValueError("segment is not at the committed frontier")

        if verification.rejected:
            state.generated_ids.extend(emitted_ids)
            rejected_position = segment.base_position + verification.accepted_count
            state.rejected_position_history.append(rejected_position)
            state.accepted_length_history.append(verification.accepted_count)
            state.proposed_length_history.append(verification.proposed_count)
            state.prefix_version += 1
            state.branch_hashes = {
                state.committed_position: stable_prefix_hash(state.committed_prefix_ids)
            }
        else:
            accepted = min(len(segment.draft_ids), len(emitted_ids))
            state.generated_ids.extend(emitted_ids[:accepted])
            state.accepted_length_history.append(accepted)
            state.proposed_length_history.append(verification.proposed_count)
            state.branch_hashes[state.committed_position] = stable_prefix_hash(
                state.committed_prefix_ids
            )
        return state

