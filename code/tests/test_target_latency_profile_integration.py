from __future__ import annotations

import pytest

from src.simulator import Simulator
from tests.common import accepting_model_runner, small_config


@pytest.mark.parametrize(
    "method",
    (
        "server_only_linear",
        "server_only_tree",
        "specedge_linear",
        "specedge_tree",
        "dip_sd",
    ),
)
def test_segment_prefix_is_verifier_kv_prefix_before_current_draft(
    method: str,
) -> None:
    config, _, workload = small_config(num_requests=2, output_len=6)
    config["speculation"]["gamma_fixed"] = 2
    config["speculation"]["gamma_candidates"] = [2]
    config["specedge"]["server_batch_size"] = 2
    simulator = Simulator(
        config,
        accepting_model_runner(),
        workload,
        "combined_strong_heterogeneous",
        method,
    )
    simulator.run()

    contexts = simulator._verification_context_lengths(simulator.segments)

    assert contexts == tuple(
        len(segment.prefix_ids) for segment in simulator.segments
    )
    for segment, context in zip(simulator.segments, contexts):
        request = simulator.requests[segment.request_id]
        assert context == len(request.prompt_ids) + segment.base_pos
        assert context + segment.verify_gamma == (
            len(segment.prefix_ids) + len(segment.draft_ids)
        )
        assert segment.verify_gamma == len(segment.draft_ids)


def test_verification_context_invariant_rejects_draft_tokens_in_prefix() -> None:
    config, model_runner, workload = small_config(num_requests=1, output_len=4)
    simulator = Simulator(
        config,
        model_runner,
        workload,
        "combined_strong_heterogeneous",
        "server_only_linear",
    )
    simulator.run()
    segment = simulator.segments[0]
    segment.prefix_ids = segment.prefix_ids + segment.draft_ids

    with pytest.raises(RuntimeError, match="verification prefix length"):
        simulator._verification_context_lengths([segment])
