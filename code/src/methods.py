from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SUPPORTED_METHODS = (
    "target_only",
    "server_only_linear",
    "sync_batch_sd",
    "SpecEdge",
    "server_only",
    "wo_async",
    "wo_scheduling",
    "conservative_rollback",
    "full",
)

DEFAULT_METHODS = (
    "full",
    "target_only",
    "sync_batch_sd",
    "SpecEdge",
    "server_only",
)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    runtime: str
    device_pool: str
    window_size: int
    num_lanes: int
    adaptive_gamma: bool
    lane_assignment: str
    prefix_control: str
    global_batch: bool = False
    batch_timeout: bool = False
    bonus_retarget: bool = False
    candidate_strategy: str | None = None


def get_method_spec(name: str, config: dict[str, Any]) -> MethodSpec:
    if name not in SUPPORTED_METHODS:
        raise ValueError(f"unsupported method: {name}")
    num_lanes = int(config["edge"]["num_lanes"])
    if name == "target_only":
        return MethodSpec(name, "target_only", "heterogeneous", 0, 0, False, "none", "none")
    if name == "server_only_linear":
        return MethodSpec(
            name,
            "server_only_specedge",
            "heterogeneous",
            1,
            0,
            False,
            "none",
            "fine_grained",
            candidate_strategy="linear",
        )
    if name == "sync_batch_sd":
        return MethodSpec(name, "sync", "heterogeneous", 1, 0, True, "global_batch", "fine_grained", global_batch=True, batch_timeout=True)
    if name == "SpecEdge":
        return MethodSpec(name, "specedge", "heterogeneous", 1, 0, False, "global_batch", "fine_grained", global_batch=True, batch_timeout=True)
    if name == "server_only":
        return MethodSpec(name, "server_only_specedge", "heterogeneous", 1, 0, True, "none", "fine_grained")
    if name == "wo_async":
        return MethodSpec(name, "async", "heterogeneous", 1, num_lanes, True, "least_finish", "fine_grained")
    if name == "wo_scheduling":
        return MethodSpec(name, "async", "heterogeneous", 0, num_lanes, True, "round_robin", "fine_grained", bonus_retarget=True)
    if name == "conservative_rollback":
        return MethodSpec(name, "async", "heterogeneous", 0, num_lanes, True, "least_finish", "conservative")
    return MethodSpec(name, "async", "heterogeneous", 0, num_lanes, True, "least_finish", "fine_grained", bonus_retarget=True)
