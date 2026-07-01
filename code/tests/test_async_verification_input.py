from __future__ import annotations

import pytest

from src.async_verification import AsyncRequestState, AsyncSegmentState


def request_state(
    server_confirmed: list[int],
    segments: list[list[int]],
    current_segment_index: int = 0,
) -> AsyncRequestState:
    return AsyncRequestState(
        request_id=7,
        server_confirmed_ids=list(server_confirmed),
        current_segment_index=current_segment_index,
        segments={
            index: AsyncSegmentState(
                segment_id=index,
                request_id=7,
                segment_index=index,
                path_generation=0,
                draft_ids=tuple(draft_ids),
            )
            for index, draft_ids in enumerate(segments)
        },
    )


def test_successor_verifies_contiguous_path_from_server_prefix() -> None:
    state = request_state(
        server_confirmed=[10],
        segments=[[11, 12], [13, 14]],
    )

    verify = state.build_verify_input(segment_index=1, l_max_ver=4)

    assert verify is not None
    assert verify.prefix_ids == (10,)
    assert verify.draft_ids == (11, 12, 13, 14)
    assert verify.local_slice == slice(2, 4)
    assert verify.dependency_fingerprint == (11, 12)


def test_current_segment_has_stable_local_offsets_and_no_dependencies() -> None:
    state = request_state(
        server_confirmed=[10, 11],
        segments=[[12, 13], [14]],
    )

    verify = state.build_verify_input(segment_index=0, l_max_ver=2)

    assert verify is not None
    assert verify.prefix_ids == (10, 11)
    assert verify.draft_ids == (12, 13)
    assert verify.local_slice == slice(0, 2)
    assert verify.dependency_fingerprint == ()


def test_over_limit_successor_is_not_partially_verified() -> None:
    state = request_state(
        server_confirmed=[10],
        segments=[[11, 12], [13, 14]],
    )

    assert state.build_verify_input(segment_index=1, l_max_ver=3) is None


def test_contiguous_input_requires_every_dependency_segment() -> None:
    state = request_state(
        server_confirmed=[10],
        segments=[[11], [12], [13]],
        current_segment_index=1,
    )
    state.segments.pop(1)

    with pytest.raises(ValueError, match="missing dependency segment: 1"):
        state.build_verify_input(segment_index=2, l_max_ver=3)
