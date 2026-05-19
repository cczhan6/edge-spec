from __future__ import annotations

import bisect
import math
import random
from typing import Mapping, Sequence

from .types import SamplingConfig, SparseProb, VerificationResult


def normalize(weights: Mapping[int, float]) -> dict[int, float]:
    cleaned = {int(k): max(0.0, float(v)) for k, v in weights.items()}
    total = sum(cleaned.values())
    if total <= 0:
        raise ValueError("cannot normalize an empty or zero-mass distribution")
    return {k: v / total for k, v in cleaned.items() if v > 0}


def sample_from_probs(probs: Mapping[int, float], rng: random.Random) -> int:
    normalized = normalize(probs)
    items = sorted(normalized.items())
    cdf: list[float] = []
    acc = 0.0
    for _, p in items:
        acc += p
        cdf.append(acc)
    draw = rng.random()
    index = min(bisect.bisect_left(cdf, draw), len(items) - 1)
    return items[index][0]


def apply_top_k_top_p_py(
    probs: Sequence[float], config: SamplingConfig
) -> list[float]:
    config.validate()
    if not probs:
        raise ValueError("probs must not be empty")
    scaled: list[float] = []
    for p in probs:
        if p < 0:
            raise ValueError("probabilities must be non-negative")
        scaled.append(float(p) ** (1.0 / config.temperature))
    total = sum(scaled)
    if total <= 0:
        raise ValueError("probabilities must have positive mass")
    scaled = [p / total for p in scaled]

    indexed = sorted(enumerate(scaled), key=lambda item: item[1], reverse=True)
    if config.top_k > 0:
        indexed = indexed[: config.top_k]
    if config.top_p < 1.0:
        kept = []
        cumulative = 0.0
        for item in indexed:
            kept.append(item)
            cumulative += item[1]
            if cumulative >= config.top_p:
                break
        indexed = kept
    sparse = normalize(dict(indexed))
    return [sparse.get(i, 0.0) for i in range(len(probs))]


def sparse_from_dense(probs: Sequence[float], min_prob: float = 0.0) -> SparseProb:
    ids: list[int] = []
    values: list[float] = []
    for i, p in enumerate(probs):
        fp = float(p)
        if fp > min_prob:
            ids.append(i)
            values.append(fp)
    normalized = normalize(dict(zip(ids, values)))
    return SparseProb(list(normalized.keys()), list(normalized.values()))


def positive_difference(target: SparseProb, draft: SparseProb) -> dict[int, float]:
    draft_map = draft.as_dict()
    diff: dict[int, float] = {}
    for token_id, p in target.as_dict().items():
        value = p - draft_map.get(token_id, 0.0)
        if value > 0:
            diff[token_id] = value
    return diff


def verify_draft_exact(
    draft_ids: Sequence[int],
    draft_dists: Sequence[SparseProb],
    target_dists: Sequence[SparseProb],
    rng: random.Random,
) -> VerificationResult:
    """Leviathan-style exact speculative sampling for one request.

    target_dists must contain len(draft_ids) + 1 distributions. The final
    distribution is used for the bonus token when every draft token is accepted.
    """
    if len(draft_dists) != len(draft_ids):
        raise ValueError("draft_dists must align with draft_ids")
    if len(target_dists) != len(draft_ids) + 1:
        raise ValueError("target_dists must contain one bonus distribution")

    emitted: list[int] = []
    for index, token_id in enumerate(draft_ids):
        target_dist = target_dists[index]
        draft_dist = draft_dists[index]
        p = target_dist.prob(int(token_id))
        q = draft_dist.prob(int(token_id))
        if q <= 0:
            accept_prob = 1.0 if p > 0 else 0.0
        else:
            accept_prob = min(1.0, p / q)
        if rng.random() <= accept_prob:
            emitted.append(int(token_id))
            continue

        rejection_dist = positive_difference(target_dist, draft_dist)
        if not rejection_dist:
            rejection_dist = target_dist.as_dict()
        emitted.append(sample_from_probs(rejection_dist, rng))
        return VerificationResult(
            emitted_ids=emitted,
            accepted_count=index,
            proposed_count=len(draft_ids),
            rejected=True,
        )

    bonus_token = sample_from_probs(target_dists[len(draft_ids)].as_dict(), rng)
    emitted.append(bonus_token)
    return VerificationResult(
        emitted_ids=emitted,
        accepted_count=len(draft_ids),
        proposed_count=len(draft_ids),
        rejected=False,
    )


def warp_logits_torch(logits, config: SamplingConfig):
    """Return a torch probability tensor after temperature, top-k and top-p."""
    config.validate()
    import torch

    filtered = logits.float() / config.temperature
    if config.top_k > 0 and config.top_k < filtered.shape[-1]:
        threshold = torch.topk(filtered, config.top_k).values[..., -1, None]
        filtered = torch.where(
            filtered < threshold,
            torch.full_like(filtered, -math.inf),
            filtered,
        )
    probs = torch.softmax(filtered, dim=-1)
    if config.top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > config.top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        probs = torch.zeros_like(probs).scatter(-1, sorted_indices, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return probs


def sparse_from_torch_probs(probs, min_prob: float = 0.0) -> SparseProb:
    nonzero = probs > min_prob
    ids = nonzero.nonzero(as_tuple=False).flatten()
    values = probs[ids]
    if ids.numel() == 0:
        raise ValueError("probability tensor has no positive entries")
    ids_list = [int(x) for x in ids.detach().cpu().tolist()]
    probs_list = [float(x) for x in values.detach().cpu().tolist()]
    normalized = normalize(dict(zip(ids_list, probs_list)))
    return SparseProb(list(normalized.keys()), list(normalized.values()))
