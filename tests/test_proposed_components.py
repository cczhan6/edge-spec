import unittest

from edge_spec.methods.proposed.consistency import (
    PrefixStateManager,
    stable_prefix_hash,
)
from edge_spec.methods.proposed.scheduling import (
    AdaptiveLookaheadPolicy,
    PrefixAwareScheduler,
)
from edge_spec.protocol import DraftSegment, VerifierLaneState
from edge_spec.types import DeviceProfile, SparseProb, VerificationResult


def segment(
    request_id="r",
    prefix_ids=None,
    draft_ids=None,
    version=0,
    base_position=0,
):
    prefix_ids = [1] if prefix_ids is None else prefix_ids
    draft_ids = [2, 3] if draft_ids is None else draft_ids
    dist = SparseProb([2, 3], [0.5, 0.5])
    return DraftSegment(
        microbatch_id=0,
        round_index=0,
        segment_id=0,
        device_id="device-0",
        request_id=request_id,
        draft_model="draft",
        prefix_ids=prefix_ids,
        draft_ids=draft_ids,
        draft_dists=[dist for _ in draft_ids],
        draft_start_s=0.0,
        draft_end_s=0.0,
        draft_elapsed_s=0.0,
        uplink_s=0.0,
        uplink_effective_mbps=1.0,
        uplink_effective_rtt_ms=1.0,
        uplink_jitter_s=0.0,
        uplink_congested=False,
        arrival_s=0.0,
        uplink_payload_bytes=1,
        prefix_version=version,
        base_position=base_position,
        prefix_hash=stable_prefix_hash(prefix_ids),
        lookahead=len(draft_ids),
    )


class ProposedComponentTests(unittest.TestCase):
    def test_stale_prefix_segment_is_rejected(self):
        manager = PrefixStateManager()
        manager.register_request("r", [1])
        manager.state("r").prefix_version = 1
        self.assertEqual(manager.check_segment(segment()).status, "stale")

    def test_rejection_advances_version_and_invalidates_descendant(self):
        manager = PrefixStateManager()
        manager.register_request("r", [1])
        first = segment()
        result = VerificationResult(
            emitted_ids=[9],
            accepted_count=0,
            proposed_count=2,
            rejected=True,
        )
        manager.apply_verification(first, result, [9])
        descendant = segment(prefix_ids=[1, 2, 3], version=0, base_position=2)
        self.assertEqual(manager.state("r").prefix_version, 1)
        self.assertEqual(manager.check_segment(descendant).status, "stale")

    def test_prefix_aware_scheduler_prefers_kv_locality(self):
        manager = PrefixStateManager()
        manager.register_request("r", [1])
        sched = PrefixAwareScheduler("prefix-aware", manager, max_gamma=4)
        seg = segment()
        lanes = [
            VerifierLaneState(lane_id=0, cached_request_id="r"),
            VerifierLaneState(lane_id=1),
        ]
        assignment = sched.assign(seg, lanes, now_s=0.0)
        self.assertEqual(assignment.lane.lane_id, 0)
        self.assertLess(assignment.terms["kv_cache_miss"], 0.001)

    def test_adaptive_lookahead_uses_initial_and_upper_bound(self):
        policy = AdaptiveLookaheadPolicy("adaptive", max_gamma=8, initial_lookahead=4)
        neutral = DeviceProfile("device-0", 60, 100, 40, 0)
        slow = DeviceProfile("device-1", 10, 100, 80, 0)
        fast = DeviceProfile("device-2", 100, 100, 10, 0)

        self.assertEqual(
            policy.select(
                profile=neutral,
                acceptance_rate=0.70,
                edge_queue_depth=2,
                remaining_tokens=10,
            ),
            4,
        )
        self.assertEqual(
            policy.select(
                profile=slow,
                acceptance_rate=0.95,
                edge_queue_depth=0,
                remaining_tokens=10,
            ),
            8,
        )
        self.assertEqual(
            policy.select(
                profile=fast,
                acceptance_rate=0.20,
                edge_queue_depth=8,
                remaining_tokens=10,
            ),
            1,
        )
        self.assertEqual(
            policy.select(
                profile=slow,
                acceptance_rate=0.95,
                edge_queue_depth=0,
                remaining_tokens=3,
            ),
            3,
        )


if __name__ == "__main__":
    unittest.main()
