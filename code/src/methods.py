from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any


SUPPORTED_METHODS = (
    "target_only",
    "server_only_linear",
    "server_only_tree",
    "specedge_linear",
    "specedge_tree",
    "dip_sd",
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
    "server_only_linear",
    "specedge_linear",
    "dip_sd",
    "server_only_tree",
    "specedge_tree",
)

LEGACY_METHOD_ALIASES = {
    "sync_batch_sd": "dip_sd",
    "SpecEdge": "specedge_tree",
    "server_only": "server_only_tree",
}


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
    if name in LEGACY_METHOD_ALIASES:
        canonical = LEGACY_METHOD_ALIASES[name]
        warnings.warn(
            f"method {name!r} is deprecated; use canonical method {canonical!r}",
            FutureWarning,
            stacklevel=2,
        )
        name = canonical
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
    if name == "server_only_tree":
        return MethodSpec(
            name,
            "server_only_specedge",
            "heterogeneous",
            1,
            0,
            False,
            "none",
            "fine_grained",
            candidate_strategy="tree",
        )
    if name == "specedge_linear":
        return MethodSpec(
            name,
            "specedge",
            "heterogeneous",
            1,
            0,
            False,
            "global_batch",
            "fine_grained",
            global_batch=True,
            batch_timeout=True,
            candidate_strategy="linear",
        )
    if name == "specedge_tree":
        return MethodSpec(
            name,
            "specedge",
            "heterogeneous",
            1,
            0,
            False,
            "global_batch",
            "fine_grained",
            global_batch=True,
            batch_timeout=True,
            candidate_strategy="tree",
        )
    if name == "dip_sd":
        return MethodSpec(
            name,
            "dip_sd",
            "heterogeneous",
            1,
            0,
            False,
            "ordered_batches",
            "fine_grained",
            candidate_strategy="linear",
        )
    if name == "wo_async":
        return MethodSpec(name, "async", "heterogeneous", 1, num_lanes, True, "least_finish", "fine_grained")
    if name == "wo_scheduling":
        return MethodSpec(name, "async", "heterogeneous", 0, num_lanes, True, "round_robin", "fine_grained", bonus_retarget=True)
    if name == "conservative_rollback":
        return MethodSpec(name, "async", "heterogeneous", 0, num_lanes, True, "least_finish", "conservative")
    return MethodSpec(name, "async", "heterogeneous", 0, num_lanes, True, "least_finish", "fine_grained", bonus_retarget=True)
