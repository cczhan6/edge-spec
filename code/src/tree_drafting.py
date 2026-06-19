from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


SUPPORTED_TREE_DRAFT_STRATEGIES = {"linear", "specexec", "specexec_approx"}
DEFAULT_TREE_DRAFT_LIMITS = {
    "max_n_beams": 1,
    "max_beam_len": 1,
    "max_branch_width": 1,
    "max_budget": 1,
}


@dataclass(frozen=True)
class DraftTreePlan:
    strategy: str
    path_token_count: int
    tree_budget_nodes: int
    draft_compute_nodes: int
    target_verify_nodes: int
    max_n_beams: int
    max_beam_len: int
    max_branch_width: int
    max_budget: int


class DraftTreeStrategy(Protocol):
    name: str
    max_n_beams: int
    max_beam_len: int
    max_branch_width: int
    max_budget: int

    def plan(self, path_token_count: int) -> DraftTreePlan: ...


@dataclass(frozen=True)
class LinearDraftTreeStrategy:
    max_beam_len: int
    max_budget: int
    name: str = "linear"
    max_n_beams: int = 1
    max_branch_width: int = 1

    def plan(self, path_token_count: int) -> DraftTreePlan:
        count = _clamp_nonnegative(path_token_count, self.max_beam_len)
        return DraftTreePlan(
            strategy=self.name,
            path_token_count=count,
            tree_budget_nodes=count,
            draft_compute_nodes=count,
            target_verify_nodes=1 if count else 0,
            max_n_beams=self.max_n_beams,
            max_beam_len=self.max_beam_len,
            max_branch_width=self.max_branch_width,
            max_budget=self.max_budget,
        )


@dataclass(frozen=True)
class SpecExecDraftTreeStrategy:
    """SpecExec-inspired analytical tree budget, not a strict upstream replay."""

    max_n_beams: int
    max_beam_len: int
    max_branch_width: int
    max_budget: int
    name: str = "specexec_approx"

    def plan(self, path_token_count: int) -> DraftTreePlan:
        depth = _clamp_nonnegative(path_token_count, self.max_beam_len)
        if depth == 0:
            return DraftTreePlan(
                strategy=self.name,
                path_token_count=0,
                tree_budget_nodes=0,
                draft_compute_nodes=0,
                target_verify_nodes=0,
                max_n_beams=self.max_n_beams,
                max_beam_len=self.max_beam_len,
                max_branch_width=self.max_branch_width,
                max_budget=self.max_budget,
            )

        candidate_nodes = 1
        tree_nodes = 0
        draft_compute_nodes = 0
        for _ in range(depth):
            forwarded_nodes = min(candidate_nodes, self.max_n_beams)
            draft_compute_nodes += forwarded_nodes
            child_nodes = forwarded_nodes * self.max_branch_width
            tree_nodes = min(self.max_budget, tree_nodes + child_nodes)
            candidate_nodes = min(
                self.max_budget,
                max(0, candidate_nodes - forwarded_nodes) + child_nodes,
            )

        return DraftTreePlan(
            strategy=self.name,
            path_token_count=depth,
            tree_budget_nodes=max(depth, tree_nodes),
            draft_compute_nodes=draft_compute_nodes,
            target_verify_nodes=max(1, min(max(depth, tree_nodes), self.max_budget)),
            max_n_beams=self.max_n_beams,
            max_beam_len=self.max_beam_len,
            max_branch_width=self.max_branch_width,
            max_budget=self.max_budget,
        )


def build_tree_draft_strategy(
    config: dict[str, Any],
    section_name: str,
    *,
    proactive: bool = False,
) -> DraftTreeStrategy:
    base = config.get("specedge", {})
    section = base if section_name == "specedge" else config.get(section_name, {})
    strategy_name = _strategy_name(base, section, proactive)
    max_n_beams = _tree_int(base, section, "max_n_beams", proactive)
    max_beam_len = _tree_int(base, section, "max_beam_len", proactive)
    max_branch_width = _tree_int(base, section, "max_branch_width", proactive)
    max_budget = _tree_int(base, section, "max_budget", proactive)

    if strategy_name == "linear":
        return LinearDraftTreeStrategy(
            max_beam_len=max_beam_len,
            max_budget=max_budget,
            max_n_beams=1,
            max_branch_width=1,
        )
    if strategy_name in {"specexec", "specexec_approx"}:
        return SpecExecDraftTreeStrategy(
            max_n_beams=max_n_beams,
            max_beam_len=max_beam_len,
            max_branch_width=max_branch_width,
            max_budget=max_budget,
        )
    raise ValueError(f"unknown tree draft strategy: {strategy_name}")


def _strategy_name(
    base: dict[str, Any],
    section: dict[str, Any],
    proactive: bool,
) -> str:
    if proactive:
        return str(
            section.get(
                "proactive_tree_draft_strategy",
                section.get(
                    "tree_draft_strategy",
                    base.get(
                        "proactive_tree_draft_strategy",
                        base.get("tree_draft_strategy", "specexec_approx"),
                    ),
                ),
            )
        )
    return str(
        section.get(
            "tree_draft_strategy",
            base.get("tree_draft_strategy", "specexec_approx"),
        )
    )


def _tree_int(
    base: dict[str, Any],
    section: dict[str, Any],
    key: str,
    proactive: bool,
) -> int:
    if proactive:
        proactive_key = f"proactive_{key}"
        if proactive_key in section:
            return int(section[proactive_key])
        if proactive_key in base:
            return int(base[proactive_key])
    if key in section:
        return int(section[key])
    return int(base.get(key, DEFAULT_TREE_DRAFT_LIMITS[key]))


def _clamp_nonnegative(value: int, upper: int) -> int:
    return max(0, min(int(value), int(upper)))
