from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

from src.tree_drafting import DraftTreePlan


@dataclass(frozen=True)
class SemanticVerifyInput:
    prefix_ids: list[int]
    draft_ids: list[int]


@dataclass(frozen=True)
class DraftTokenCandidate:
    token_id: int
    logprob: float


@dataclass(frozen=True)
class DraftTreeNode:
    node_id: int
    parent_id: int | None
    token_id: int
    depth: int
    logprob: float = 0.0
    processed: bool = True


@dataclass(frozen=True)
class DraftCandidateTree:
    prefix_ids: list[int]
    primary_ids: list[int]
    primary_node_ids: list[int]
    nodes: list[DraftTreeNode]
    processed_candidate_count: int = 0
    retained_tree_nodes: int = 0
    target_verify_tree_nodes: int = 0

    @property
    def node_count(self) -> int:
        return len(self.nodes)


@dataclass(frozen=True)
class SemanticTreeVerifyInput:
    prefix_ids: list[int]
    draft_tree: DraftCandidateTree


@dataclass(frozen=True)
class VerificationResult:
    accepted_count: int
    emitted_ids: list[int]
    rejected: bool
    bonus_token: int | None = None


class ModelRunner(Protocol):
    eos_token_id: int | None

    def encode_prompt(self, prompt: str) -> list[int]: ...

    def prompt_token_count(self, prompt: str) -> int: ...

    def draft(self, drafter_profile: str, prefix_ids: Sequence[int], gamma: int) -> list[int]: ...

    def draft_tree(
        self,
        drafter_profile: str,
        prefix_ids: Sequence[int],
        plan: DraftTreePlan,
    ) -> DraftCandidateTree: ...

    def draft_bonus_tree(
        self,
        drafter_profile: str,
        draft_tree: DraftCandidateTree,
        plan: DraftTreePlan,
    ) -> DraftCandidateTree: ...

    def verify(self, prefix_ids: Sequence[int], draft_ids: Sequence[int]) -> VerificationResult: ...

    def verify_tree(
        self,
        prefix_ids: Sequence[int],
        draft_tree: DraftCandidateTree,
    ) -> VerificationResult: ...

    def verify_batch(self, requests: Sequence[SemanticVerifyInput]) -> list[VerificationResult]: ...

    def verify_tree_batch(
        self,
        requests: Sequence[SemanticTreeVerifyInput],
    ) -> list[VerificationResult]: ...

    def target_only(self, prefix_ids: Sequence[int], max_new_tokens: int) -> list[int]: ...


class FakeModelRunner:
    """Deterministic model runner for tests and smoke runs."""

    def __init__(
        self,
        vocab_size: int = 97,
        eos_token_id: int | None = None,
        target_token_fn: Callable[[Sequence[int]], int] | None = None,
        draft_token_fn: Callable[[str, Sequence[int]], int] | None = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self._target_token_fn = target_token_fn
        self._draft_token_fn = draft_token_fn

    def encode_prompt(self, prompt: str) -> list[int]:
        values = [int(value) % self.vocab_size for value in prompt.encode("utf-8")]
        return values or [1]

    def prompt_token_count(self, prompt: str) -> int:
        return len(self.encode_prompt(prompt))

    def draft(self, drafter_profile: str, prefix_ids: Sequence[int], gamma: int) -> list[int]:
        context = list(prefix_ids)
        draft_ids: list[int] = []
        for _ in range(gamma):
            token_id = self._draft_token(drafter_profile, context)
            draft_ids.append(token_id)
            context.append(token_id)
            if token_id == self.eos_token_id:
                break
        return draft_ids

    def draft_tree(
        self,
        drafter_profile: str,
        prefix_ids: Sequence[int],
        plan: DraftTreePlan,
    ) -> DraftCandidateTree:
        return build_draft_candidate_tree(
            prefix_ids,
            plan,
            lambda context, width: self._draft_top_candidates(drafter_profile, context, width),
            self.eos_token_id,
        )

    def draft_bonus_tree(
        self,
        drafter_profile: str,
        draft_tree: DraftCandidateTree,
        plan: DraftTreePlan,
    ) -> DraftCandidateTree:
        return build_bonus_candidate_tree(
            draft_tree,
            plan,
            lambda context, width: self._draft_top_candidates(drafter_profile, context, width),
            self.eos_token_id,
        )

    def verify(self, prefix_ids: Sequence[int], draft_ids: Sequence[int]) -> VerificationResult:
        context = list(prefix_ids)
        emitted: list[int] = []
        for index, draft_token in enumerate(draft_ids):
            target_token = self._target_token(context)
            if int(draft_token) != target_token:
                emitted.append(target_token)
                return VerificationResult(index, emitted, True)
            emitted.append(target_token)
            context.append(target_token)
            if target_token == self.eos_token_id:
                return VerificationResult(index + 1, emitted, False)
        bonus = self._target_token(context)
        emitted.append(bonus)
        return VerificationResult(len(draft_ids), emitted, False, bonus)

    def verify_batch(self, requests: Sequence[SemanticVerifyInput]) -> list[VerificationResult]:
        if not requests:
            raise ValueError("verify batch must not be empty")
        return [self.verify(item.prefix_ids, item.draft_ids) for item in requests]

    def verify_tree(
        self,
        prefix_ids: Sequence[int],
        draft_tree: DraftCandidateTree,
    ) -> VerificationResult:
        return verify_candidate_tree(
            prefix_ids,
            draft_tree,
            self._target_token,
            self.eos_token_id,
        )

    def verify_tree_batch(
        self,
        requests: Sequence[SemanticTreeVerifyInput],
    ) -> list[VerificationResult]:
        if not requests:
            raise ValueError("verify tree batch must not be empty")
        return [self.verify_tree(item.prefix_ids, item.draft_tree) for item in requests]

    def target_only(self, prefix_ids: Sequence[int], max_new_tokens: int) -> list[int]:
        context = list(prefix_ids)
        generated: list[int] = []
        for _ in range(max_new_tokens):
            token_id = self._target_token(context)
            generated.append(token_id)
            context.append(token_id)
            if token_id == self.eos_token_id:
                break
        return generated

    def _draft_top_candidates(
        self,
        drafter_profile: str,
        prefix_ids: Sequence[int],
        width: int,
    ) -> list[DraftTokenCandidate]:
        primary = self._draft_token(drafter_profile, prefix_ids)
        target = self._target_token(prefix_ids)
        candidates = [primary, target]
        offset = 1
        while len(candidates) < width:
            candidates.append((primary + offset) % self.vocab_size)
            offset += 1
        token_ids = _unique_token_ids(candidates, width)
        return [
            DraftTokenCandidate(token_id=token_id, logprob=-0.1 * index)
            for index, token_id in enumerate(token_ids)
        ]

    def _target_token(self, prefix_ids: Sequence[int]) -> int:
        if self._target_token_fn is not None:
            return int(self._target_token_fn(prefix_ids))
        return int((sum(prefix_ids[-8:]) + len(prefix_ids) + 1) % self.vocab_size)

    def _draft_token(self, drafter_profile: str, prefix_ids: Sequence[int]) -> int:
        if self._draft_token_fn is not None:
            return int(self._draft_token_fn(drafter_profile, prefix_ids))
        target = self._target_token(prefix_ids)
        cadence = {"small": 3, "medium": 5, "large": 11}.get(drafter_profile, 4)
        if (sum(prefix_ids[-4:]) + len(prefix_ids)) % cadence == 0:
            return (target + 1) % self.vocab_size
        return target


class HuggingFaceModelRunner:
    """Real-model greedy semantics. Host wall time is intentionally ignored."""

    def __init__(self, config: dict[str, Any]) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "model runner requires torch and transformers; install requirements.txt"
            ) from exc
        self.torch = torch
        model_runner = config.get("model_runner", config.get("oracle"))
        if model_runner is None:
            raise KeyError("config must define model_runner")
        self._hf_load_kwargs = _huggingface_load_kwargs(model_runner)
        target_name = str(model_runner["target_model"])
        self.tokenizer = self._from_pretrained(
            AutoTokenizer,
            target_name,
            "target tokenizer",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            raise ValueError("target tokenizer must define eos_token_id or pad_token_id")
        self.target_device = str(model_runner["target_device"])
        self.target_model = self._load_model(AutoModelForCausalLM, target_name, self.target_device)
        self.drafters: dict[str, Any] = {}
        self.drafter_devices: dict[str, str] = {}
        self.vocab_size = self._tokenizer_vocab_size(self.tokenizer)
        target_model_vocab = self._model_vocab_size(self.target_model)
        if target_model_vocab < self.vocab_size:
            raise ValueError(
                "target model vocabulary "
                f"({target_model_vocab}) cannot cover tokenizer vocabulary ({self.vocab_size})"
            )
        target_tokenizer_mapping = (
            self.tokenizer.get_vocab() if hasattr(self.tokenizer, "get_vocab") else None
        )
        for profile, values in model_runner["drafter_models"].items():
            model_name = str(values["model"])
            device = str(values["device"])
            tokenizer = self._from_pretrained(
                AutoTokenizer,
                model_name,
                f"drafter {profile} tokenizer",
            )
            model = self._load_model(AutoModelForCausalLM, model_name, device)
            if self._tokenizer_vocab_size(tokenizer) != self.vocab_size:
                raise ValueError(f"drafter {profile} tokenizer vocabulary is incompatible with target")
            if (
                target_tokenizer_mapping is not None
                and hasattr(tokenizer, "get_vocab")
                and tokenizer.get_vocab() != target_tokenizer_mapping
            ):
                raise ValueError(f"drafter {profile} tokenizer mapping is incompatible with target")
            model_vocab = self._model_vocab_size(model)
            if model_vocab < self.vocab_size:
                raise ValueError(
                    f"drafter {profile} model vocabulary ({model_vocab}) "
                    f"cannot cover tokenizer vocabulary ({self.vocab_size})"
                )
            self.drafters[str(profile)] = model
            self.drafter_devices[str(profile)] = device

    def encode_prompt(self, prompt: str) -> list[int]:
        return [int(value) for value in self.tokenizer.encode(prompt)]

    def prompt_token_count(self, prompt: str) -> int:
        return len(self.encode_prompt(prompt))

    def draft(self, drafter_profile: str, prefix_ids: Sequence[int], gamma: int) -> list[int]:
        return self._incremental_greedy(
            self.drafters[drafter_profile],
            self.drafter_devices[drafter_profile],
            prefix_ids,
            gamma,
        )

    def draft_tree(
        self,
        drafter_profile: str,
        prefix_ids: Sequence[int],
        plan: DraftTreePlan,
    ) -> DraftCandidateTree:
        return build_draft_candidate_tree_batched(
            prefix_ids,
            plan,
            lambda contexts, width: self._draft_top_candidates_batch(
                drafter_profile,
                contexts,
                width,
            ),
            self.eos_token_id,
        )

    def draft_bonus_tree(
        self,
        drafter_profile: str,
        draft_tree: DraftCandidateTree,
        plan: DraftTreePlan,
    ) -> DraftCandidateTree:
        return build_bonus_candidate_tree(
            draft_tree,
            plan,
            lambda context, width: self._draft_top_candidates(drafter_profile, context, width),
            self.eos_token_id,
        )

    def verify(self, prefix_ids: Sequence[int], draft_ids: Sequence[int]) -> VerificationResult:
        return self.verify_batch(
            [SemanticVerifyInput(list(prefix_ids), list(draft_ids))]
        )[0]

    def verify_tree(
        self,
        prefix_ids: Sequence[int],
        draft_tree: DraftCandidateTree,
    ) -> VerificationResult:
        return self.verify_tree_batch(
            [SemanticTreeVerifyInput(list(prefix_ids), draft_tree)]
        )[0]

    def verify_batch(self, requests: Sequence[SemanticVerifyInput]) -> list[VerificationResult]:
        if not requests:
            raise ValueError("verify batch must not be empty")
        torch = self.torch
        sequences = [item.prefix_ids + item.draft_ids for item in requests]
        max_len = max(len(sequence) for sequence in sequences)
        input_ids = torch.full(
            (len(sequences), max_len),
            int(self.pad_token_id),
            dtype=torch.long,
            device=self.target_device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, sequence in enumerate(sequences):
            input_ids[row, : len(sequence)] = torch.tensor(
                sequence,
                dtype=torch.long,
                device=self.target_device,
            )
            attention_mask[row, : len(sequence)] = 1
        with torch.inference_mode():
            logits = self.target_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
        results: list[VerificationResult] = []
        for row, item in enumerate(requests):
            emitted: list[int] = []
            accepted = 0
            rejected = False
            bonus: int | None = None
            for offset, draft_token in enumerate(item.draft_ids):
                position = len(item.prefix_ids) - 1 + offset
                target_token = int(
                    logits[row, position, : self.vocab_size].argmax(dim=-1).item()
                )
                if int(draft_token) != target_token:
                    emitted.append(target_token)
                    rejected = True
                    break
                emitted.append(target_token)
                accepted += 1
                if target_token == self.eos_token_id:
                    break
            if not rejected and accepted == len(item.draft_ids):
                if not emitted or emitted[-1] != self.eos_token_id:
                    position = len(item.prefix_ids) - 1 + len(item.draft_ids)
                    bonus = int(
                        logits[row, position, : self.vocab_size].argmax(dim=-1).item()
                    )
                    emitted.append(bonus)
            results.append(VerificationResult(accepted, emitted, rejected, bonus))
        return results

    def verify_tree_batch(
        self,
        requests: Sequence[SemanticTreeVerifyInput],
    ) -> list[VerificationResult]:
        if not requests:
            raise ValueError("verify tree batch must not be empty")
        torch = self.torch
        packed_trees = [
            _tree_with_prefix(item.draft_tree, item.prefix_ids)
            for item in requests
        ]
        sequences = [
            list(tree.prefix_ids) + [node.token_id for node in tree.nodes]
            for tree in packed_trees
        ]
        max_len = max(len(sequence) for sequence in sequences)
        input_ids = torch.full(
            (len(sequences), max_len),
            int(self.pad_token_id),
            dtype=torch.long,
            device=self.target_device,
        )
        position_ids = torch.full(
            (len(sequences), max_len),
            0,
            dtype=torch.long,
            device=self.target_device,
        )
        mask_dtype = getattr(self.target_model, "dtype", None) or torch.float32
        attention_mask = torch.tensor(
            [
                _tree_attention_mask_4d_data(tree, max_len)
                for tree in packed_trees
            ],
            dtype=mask_dtype,
            device=self.target_device,
        )
        for row, (tree, sequence) in enumerate(zip(packed_trees, sequences)):
            input_ids[row, : len(sequence)] = torch.tensor(
                sequence,
                dtype=torch.long,
                device=self.target_device,
            )
            position_ids[row, : len(sequence)] = torch.tensor(
                _tree_position_ids(tree),
                dtype=torch.long,
                device=self.target_device,
            )
        with torch.inference_mode():
            logits = self.target_model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
        return [
            _verify_candidate_tree_from_logits(
                logits,
                row,
                tree,
                self.vocab_size,
                self.eos_token_id,
            )
            for row, tree in enumerate(packed_trees)
        ]

    def target_only(self, prefix_ids: Sequence[int], max_new_tokens: int) -> list[int]:
        return self._incremental_greedy(
            self.target_model,
            self.target_device,
            prefix_ids,
            max_new_tokens,
        )

    def _draft_top_candidates(
        self,
        drafter_profile: str,
        prefix_ids: Sequence[int],
        width: int,
    ) -> list[DraftTokenCandidate]:
        if width <= 0:
            return []
        torch = self.torch
        model = self.drafters[drafter_profile]
        device = self.drafter_devices[drafter_profile]
        input_ids = torch.tensor([list(prefix_ids)], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, use_cache=False).logits
            logprobs = torch.log_softmax(logits[:, -1, : self.vocab_size], dim=-1)
            top_values, top_ids = torch.topk(
                logprobs,
                k=min(width, self.vocab_size),
                dim=-1,
            )
        return [
            DraftTokenCandidate(token_id=int(token_id), logprob=float(logprob))
            for token_id, logprob in zip(top_ids[0].tolist(), top_values[0].tolist())
        ]

    def _draft_top_candidates_batch(
        self,
        drafter_profile: str,
        prefix_id_batches: Sequence[Sequence[int]],
        width: int,
    ) -> list[list[DraftTokenCandidate]]:
        contexts = [list(prefix_ids) for prefix_ids in prefix_id_batches]
        if not contexts:
            return []
        if width <= 0:
            return [[] for _ in contexts]
        if any(not context for context in contexts):
            raise ValueError("draft prefix must not be empty")

        torch = self.torch
        model = self.drafters[drafter_profile]
        device = self.drafter_devices[drafter_profile]
        max_len = max(len(context) for context in contexts)
        input_ids = torch.full(
            (len(contexts), max_len),
            int(self.pad_token_id),
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, context in enumerate(contexts):
            input_ids[row, : len(context)] = torch.tensor(
                context,
                dtype=torch.long,
                device=device,
            )
            attention_mask[row, : len(context)] = 1

        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
            results: list[list[DraftTokenCandidate]] = []
            for row, context in enumerate(contexts):
                logprobs = torch.log_softmax(
                    logits[row, len(context) - 1, : self.vocab_size],
                    dim=-1,
                )
                top_values, top_ids = torch.topk(
                    logprobs,
                    k=min(width, self.vocab_size),
                    dim=-1,
                )
                results.append(
                    [
                        DraftTokenCandidate(
                            token_id=int(token_id),
                            logprob=float(logprob),
                        )
                        for token_id, logprob in zip(
                            top_ids.tolist(),
                            top_values.tolist(),
                        )
                    ]
                )
        return results

    def _target_token(self, prefix_ids: Sequence[int]) -> int:
        generated = self._incremental_greedy(
            self.target_model,
            self.target_device,
            prefix_ids,
            1,
        )
        if not generated:
            raise RuntimeError("target model returned no token")
        return generated[0]

    def _incremental_greedy(
        self,
        model: Any,
        device: str,
        prefix_ids: Sequence[int],
        max_new_tokens: int,
    ) -> list[int]:
        if max_new_tokens <= 0:
            return []
        torch = self.torch
        input_ids = torch.tensor([list(prefix_ids)], dtype=torch.long, device=device)
        generated: list[int] = []
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, use_cache=True)
            past = outputs.past_key_values
            for index in range(max_new_tokens):
                token_id = int(
                    outputs.logits[:, -1, : self.vocab_size].argmax(dim=-1).item()
                )
                generated.append(token_id)
                if token_id == self.eos_token_id or index + 1 == max_new_tokens:
                    break
                next_ids = torch.tensor([[token_id]], dtype=torch.long, device=device)
                outputs = model(
                    input_ids=next_ids,
                    past_key_values=past,
                    use_cache=True,
                )
                past = outputs.past_key_values
        return generated

    def _load_model(self, model_class: Any, name: str, device: str) -> Any:
        return self._from_pretrained(
            model_class,
            name,
            f"model {name}",
            torch_dtype="auto",
        ).to(device).eval()

    def _from_pretrained(
        self,
        factory: Any,
        name: str,
        description: str,
        **kwargs: Any,
    ) -> Any:
        load_kwargs = {**self._hf_load_kwargs, **kwargs}
        try:
            return factory.from_pretrained(name, **load_kwargs)
        except Exception as exc:
            if load_kwargs.get("local_files_only") or not _looks_like_network_error(exc):
                raise
            retry_kwargs = {**load_kwargs, "local_files_only": True}
            try:
                return factory.from_pretrained(name, **retry_kwargs)
            except Exception as local_exc:
                online_error = f"{type(exc).__name__}: {exc}"
                cache_error = f"{type(local_exc).__name__}: {local_exc}"
                raise RuntimeError(
                    f"failed to load {description} '{name}' from Hugging Face, "
                    "then failed to load it from the local cache. "
                    f"online error: {online_error}; local cache error: {cache_error}. "
                    "If the model is already cached, set model_runner.local_files_only: true "
                    "or HF_HUB_OFFLINE=1; otherwise fix the proxy/network or pre-download "
                    "the model files."
                ) from local_exc

    @staticmethod
    def _model_vocab_size(model: Any) -> int:
        size = getattr(model.config, "vocab_size", None)
        if size is None:
            raise ValueError("language model config must expose vocab_size")
        return int(size)

    @staticmethod
    def _tokenizer_vocab_size(tokenizer: Any) -> int:
        size = len(tokenizer)
        if hasattr(tokenizer, "get_vocab"):
            vocab = tokenizer.get_vocab()
            if vocab:
                size = max(size, max(int(token_id) for token_id in vocab.values()) + 1)
        return int(size)


def build_model_runner(
    config: dict[str, Any],
    use_fake_model_runner: bool = False,
) -> ModelRunner:
    if use_fake_model_runner:
        return FakeModelRunner()
    return HuggingFaceModelRunner(config)


def build_draft_candidate_tree(
    prefix_ids: Sequence[int],
    plan: DraftTreePlan,
    top_tokens_fn: Callable[[Sequence[int], int], Sequence[DraftTokenCandidate]],
    eos_token_id: int | None = None,
) -> DraftCandidateTree:
    max_depth = max(0, int(plan.path_token_count))
    if max_depth == 0:
        return DraftCandidateTree(list(prefix_ids), [], [], [])

    retained_limit = max(0, int(plan.max_budget))
    if retained_limit == 0:
        return DraftCandidateTree(list(prefix_ids), [], [], [])

    return _build_logprob_pruned_tree(
        list(prefix_ids),
        max_depth,
        retained_limit,
        max(1, int(plan.max_n_beams)),
        max(1, int(plan.max_branch_width)),
        top_tokens_fn,
        eos_token_id,
        initial_candidates=[_BuildCandidate(None, [], 0, 0.0)],
        processed_offset=0,
    )


def build_draft_candidate_tree_batched(
    prefix_ids: Sequence[int],
    plan: DraftTreePlan,
    top_tokens_batch_fn: Callable[
        [Sequence[Sequence[int]], int],
        Sequence[Sequence[DraftTokenCandidate]],
    ],
    eos_token_id: int | None = None,
) -> DraftCandidateTree:
    max_depth = max(0, int(plan.path_token_count))
    if max_depth == 0:
        return DraftCandidateTree(list(prefix_ids), [], [], [])

    retained_limit = max(0, int(plan.max_budget))
    if retained_limit == 0:
        return DraftCandidateTree(list(prefix_ids), [], [], [])

    return _build_logprob_pruned_tree_batched(
        list(prefix_ids),
        max_depth,
        retained_limit,
        max(1, int(plan.max_n_beams)),
        max(1, int(plan.max_branch_width)),
        top_tokens_batch_fn,
        eos_token_id,
        initial_candidates=[_BuildCandidate(None, [], 0, 0.0)],
        processed_offset=0,
    )


@dataclass
class _BuildNode:
    source_id: int
    parent_source_id: int | None
    token_id: int
    depth: int
    logprob: float
    path: list[int]
    processed: bool = False


@dataclass(frozen=True)
class _BuildCandidate:
    source_id: int | None
    path: list[int]
    depth: int
    logprob: float


def _build_logprob_pruned_tree(
    prefix_ids: list[int],
    max_depth: int,
    retained_limit: int,
    max_n_beams: int,
    max_branch_width: int,
    top_tokens_fn: Callable[[Sequence[int], int], Sequence[DraftTokenCandidate]],
    eos_token_id: int | None,
    initial_candidates: list[_BuildCandidate],
    processed_offset: int,
) -> DraftCandidateTree:
    return _build_logprob_pruned_tree_batched(
        prefix_ids,
        max_depth,
        retained_limit,
        max_n_beams,
        max_branch_width,
        lambda contexts, width: [
            top_tokens_fn(context, width)
            for context in contexts
        ],
        eos_token_id,
        initial_candidates,
        processed_offset,
    )


def _build_logprob_pruned_tree_batched(
    prefix_ids: list[int],
    max_depth: int,
    retained_limit: int,
    max_n_beams: int,
    max_branch_width: int,
    top_tokens_batch_fn: Callable[
        [Sequence[Sequence[int]], int],
        Sequence[Sequence[DraftTokenCandidate]],
    ],
    eos_token_id: int | None,
    initial_candidates: list[_BuildCandidate],
    processed_offset: int,
) -> DraftCandidateTree:
    decay = math.log(0.9)
    next_source_id = 1
    nodes: list[_BuildNode] = []
    root_candidates = list(initial_candidates)
    processed_count = int(processed_offset)

    for _ in range(max_depth):
        candidates: list[_BuildCandidate] = []
        candidates.extend(root_candidates)
        root_candidates = []
        candidates.extend(
            _BuildCandidate(node.source_id, node.path, node.depth, node.logprob)
            for node in nodes
            if not node.processed and node.token_id != eos_token_id
        )
        candidates = [
            candidate
            for candidate in candidates
            if candidate.depth < max_depth
        ]
        if not candidates:
            break

        candidates.sort(key=lambda item: (-item.logprob, item.depth, item.source_id or 0))
        selected = candidates[:max_n_beams]
        incoming: list[_BuildNode] = []
        selected_for_expansion: list[_BuildCandidate] = []
        for candidate in selected:
            if candidate.source_id is not None:
                node = next(
                    item for item in nodes if item.source_id == candidate.source_id
                )
                if node.processed:
                    continue
                node.processed = True
            processed_count += 1
            selected_for_expansion.append(candidate)

        if not selected_for_expansion:
            break

        child_batches = top_tokens_batch_fn(
            [prefix_ids + candidate.path for candidate in selected_for_expansion],
            max_branch_width,
        )
        for candidate, child_batch in zip(selected_for_expansion, child_batches):
            child_candidates = _unique_token_candidates(child_batch, max_branch_width)
            for child in child_candidates:
                path = candidate.path + [child.token_id]
                incoming.append(
                    _BuildNode(
                        source_id=next_source_id,
                        parent_source_id=candidate.source_id,
                        token_id=child.token_id,
                        depth=candidate.depth + 1,
                        logprob=candidate.logprob + decay + child.logprob,
                        path=path,
                    )
                )
                next_source_id += 1

        if not incoming:
            break

        if len(nodes) >= retained_limit:
            lowest_retained = min(node.logprob for node in nodes)
            if max(node.logprob for node in incoming) < lowest_retained:
                break

        nodes.extend(incoming)
        nodes = _prune_build_nodes(nodes, retained_limit)

    nodes = _prune_build_nodes(nodes, retained_limit)
    return _finalize_build_tree(prefix_ids, nodes, processed_count)


def build_bonus_candidate_tree(
    draft_tree: DraftCandidateTree,
    plan: DraftTreePlan,
    top_tokens_fn: Callable[[Sequence[int], int], Sequence[DraftTokenCandidate]],
    eos_token_id: int | None = None,
) -> DraftCandidateTree:
    path_count = max(0, int(plan.path_token_count))
    if path_count == 0 or not draft_tree.nodes:
        return DraftCandidateTree(list(draft_tree.prefix_ids), [], [], [])

    leaves = _leaf_nodes(draft_tree)
    if not leaves:
        return DraftCandidateTree(list(draft_tree.prefix_ids), [], [], [])
    leaves.sort(key=lambda node: (-node.logprob, node.depth, node.node_id))
    selected_leaves = leaves[: max(1, int(plan.max_n_beams))]

    best_leaf: DraftTreeNode | None = None
    best_token: DraftTokenCandidate | None = None
    best_score = float("-inf")
    bonus_width = 1024
    for leaf in selected_leaves:
        leaf_path = _path_to_node(draft_tree, leaf.node_id)
        for candidate in _unique_token_candidates(
            top_tokens_fn(draft_tree.prefix_ids + leaf_path, bonus_width),
            bonus_width,
        ):
            score = leaf.logprob + math.log(0.95) + candidate.logprob
            if score > best_score:
                best_score = score
                best_leaf = leaf
                best_token = candidate

    if best_leaf is None or best_token is None:
        return DraftCandidateTree(list(draft_tree.prefix_ids), [], [], [])

    leaf_path = _path_to_node(draft_tree, best_leaf.node_id)
    prefix_ids = list(draft_tree.prefix_ids) + leaf_path
    if path_count == 1 or best_token.token_id == eos_token_id:
        return _with_tree_stats(
            make_linear_draft_tree(prefix_ids, [best_token.token_id]),
            processed_candidate_count=len(selected_leaves),
        )

    suffix_plan = DraftTreePlan(
        strategy=plan.strategy,
        path_token_count=path_count - 1,
        tree_budget_nodes=max(0, int(plan.tree_budget_nodes) - 1),
        draft_compute_nodes=0,
        target_verify_nodes=0,
        max_n_beams=plan.max_n_beams,
        max_beam_len=max(0, plan.max_beam_len - 1),
        max_branch_width=plan.max_branch_width,
        max_budget=max(1, int(plan.max_budget) - 1),
    )
    suffix_tree = build_draft_candidate_tree(
        prefix_ids + [best_token.token_id],
        suffix_plan,
        top_tokens_fn,
        eos_token_id,
    )
    tree = concat_linear_prefix_tree(prefix_ids, [best_token.token_id], suffix_tree)
    return _with_tree_stats(
        tree,
        processed_candidate_count=len(selected_leaves)
        + suffix_tree.processed_candidate_count,
    )


def _prune_build_nodes(nodes: list[_BuildNode], retained_limit: int) -> list[_BuildNode]:
    if len(nodes) <= retained_limit:
        return nodes
    ranked = sorted(nodes, key=lambda item: (-item.logprob, item.depth, item.source_id))
    keep_ids = {node.source_id for node in ranked[:retained_limit]}
    by_id = {node.source_id: node for node in nodes}
    for node_id in list(keep_ids):
        parent_id = by_id[node_id].parent_source_id
        while parent_id is not None:
            keep_ids.add(parent_id)
            parent_id = by_id[parent_id].parent_source_id
    if len(keep_ids) > retained_limit:
        ranked_keep = [node for node in ranked if node.source_id in keep_ids]
        keep_ids = {node.source_id for node in ranked_keep[:retained_limit]}
    return [
        node
        for node in nodes
        if node.source_id in keep_ids
        and (node.parent_source_id is None or node.parent_source_id in keep_ids)
    ]


def _finalize_build_tree(
    prefix_ids: list[int],
    build_nodes: list[_BuildNode],
    processed_count: int,
) -> DraftCandidateTree:
    ordered = sorted(build_nodes, key=lambda item: (item.depth, item.source_id))
    id_map: dict[int, int] = {}
    nodes: list[DraftTreeNode] = []
    for build_node in ordered:
        parent_id = (
            None
            if build_node.parent_source_id is None
            else id_map[build_node.parent_source_id]
        )
        node = _new_tree_node(
            nodes,
            parent_id,
            build_node.token_id,
            build_node.depth,
            logprob=build_node.logprob,
            processed=build_node.processed,
        )
        id_map[build_node.source_id] = node.node_id

    primary_node_ids = _best_primary_node_ids(nodes)
    primary_ids = [
        next(node.token_id for node in nodes if node.node_id == node_id)
        for node_id in primary_node_ids
    ]
    retained = len(nodes)
    target_verify_nodes = max(1, retained) if retained else 0
    return DraftCandidateTree(
        prefix_ids,
        primary_ids,
        primary_node_ids,
        nodes,
        processed_candidate_count=processed_count,
        retained_tree_nodes=retained,
        target_verify_tree_nodes=target_verify_nodes,
    )


def make_linear_draft_tree(
    prefix_ids: Sequence[int],
    draft_ids: Sequence[int],
) -> DraftCandidateTree:
    nodes: list[DraftTreeNode] = []
    primary_node_ids: list[int] = []
    parent_id: int | None = None
    for depth, token_id in enumerate(draft_ids, start=1):
        node = _new_tree_node(nodes, parent_id, int(token_id), depth)
        primary_node_ids.append(node.node_id)
        parent_id = node.node_id
    return DraftCandidateTree(
        list(prefix_ids),
        [int(token_id) for token_id in draft_ids],
        primary_node_ids,
        nodes,
        processed_candidate_count=len(nodes),
        retained_tree_nodes=len(nodes),
        target_verify_tree_nodes=1 if nodes else 0,
    )


def build_tree_attention_mask(draft_tree: DraftCandidateTree) -> list[list[bool]]:
    prefix_len = len(draft_tree.prefix_ids)
    tree_len = len(draft_tree.nodes)
    total_len = prefix_len + tree_len
    mask = [[False for _ in range(total_len)] for _ in range(total_len)]

    for row in range(prefix_len):
        for col in range(row + 1):
            mask[row][col] = True

    node_index_by_id = {
        node.node_id: prefix_len + index
        for index, node in enumerate(draft_tree.nodes)
    }
    node_by_id = {node.node_id: node for node in draft_tree.nodes}
    if len(node_by_id) != tree_len:
        raise ValueError("draft tree contains duplicate node ids")

    ancestor_cache: dict[int, list[int]] = {}

    def ancestor_ids(node_id: int, visiting: set[int] | None = None) -> list[int]:
        if node_id in ancestor_cache:
            return ancestor_cache[node_id]
        if visiting is None:
            visiting = set()
        if node_id in visiting:
            raise ValueError("draft tree contains a parent cycle")
        visiting.add(node_id)
        node = node_by_id[node_id]
        if node.parent_id is None:
            ancestors: list[int] = []
        else:
            if node.parent_id not in node_by_id:
                raise ValueError(f"draft tree references unknown parent id {node.parent_id}")
            ancestors = ancestor_ids(node.parent_id, visiting) + [node.parent_id]
        visiting.remove(node_id)
        ancestor_cache[node_id] = ancestors
        return ancestors

    for node in draft_tree.nodes:
        row = node_index_by_id[node.node_id]
        for col in range(prefix_len):
            mask[row][col] = True
        for ancestor_id in ancestor_ids(node.node_id):
            mask[row][node_index_by_id[ancestor_id]] = True
        mask[row][row] = True

    return mask


def build_tree_attention_mask_tensor(
    draft_tree: DraftCandidateTree,
    *,
    device: Any = None,
    dtype: Any = None,
) -> Any:
    import torch

    tensor = torch.tensor(build_tree_attention_mask(draft_tree), device=device)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _tree_attention_mask_4d_data(
    draft_tree: DraftCandidateTree,
    max_len: int,
) -> list[list[list[float]]]:
    allowed = build_tree_attention_mask(draft_tree)
    valid_len = len(allowed)
    matrix: list[list[float]] = []
    for row in range(max_len):
        values: list[float] = []
        for col in range(max_len):
            visible = row < valid_len and col < valid_len and allowed[row][col]
            values.append(0.0 if visible else -10000.0)
        matrix.append(values)
    return [matrix]


def _tree_position_ids(draft_tree: DraftCandidateTree) -> list[int]:
    prefix_len = len(draft_tree.prefix_ids)
    return list(range(prefix_len)) + [
        prefix_len + max(0, int(node.depth) - 1)
        for node in draft_tree.nodes
    ]


def _tree_with_prefix(
    draft_tree: DraftCandidateTree,
    prefix_ids: Sequence[int],
) -> DraftCandidateTree:
    prefix = list(prefix_ids)
    if draft_tree.prefix_ids == prefix:
        return draft_tree
    return DraftCandidateTree(
        prefix,
        list(draft_tree.primary_ids),
        list(draft_tree.primary_node_ids),
        list(draft_tree.nodes),
        processed_candidate_count=draft_tree.processed_candidate_count,
        retained_tree_nodes=draft_tree.retained_tree_nodes,
        target_verify_tree_nodes=draft_tree.target_verify_tree_nodes,
    )


def concat_linear_prefix_tree(
    prefix_ids: Sequence[int],
    prefix_draft_ids: Sequence[int],
    suffix_tree: DraftCandidateTree,
) -> DraftCandidateTree:
    if not prefix_draft_ids:
        return suffix_tree
    nodes: list[DraftTreeNode] = []
    primary_node_ids: list[int] = []
    parent_id: int | None = None
    for depth, token_id in enumerate(prefix_draft_ids, start=1):
        node = _new_tree_node(nodes, parent_id, int(token_id), depth)
        primary_node_ids.append(node.node_id)
        parent_id = node.node_id
    id_map: dict[int, int] = {}
    depth_offset = len(prefix_draft_ids)
    for suffix_node in suffix_tree.nodes:
        mapped_parent = (
            parent_id
            if suffix_node.parent_id is None
            else id_map[suffix_node.parent_id]
        )
        node = _new_tree_node(
            nodes,
            mapped_parent,
            suffix_node.token_id,
            depth_offset + suffix_node.depth,
        )
        id_map[suffix_node.node_id] = node.node_id
    primary_node_ids.extend(id_map[node_id] for node_id in suffix_tree.primary_node_ids)
    return DraftCandidateTree(
        list(prefix_ids),
        [int(token_id) for token_id in prefix_draft_ids] + list(suffix_tree.primary_ids),
        primary_node_ids,
        nodes,
        processed_candidate_count=len(prefix_draft_ids) + suffix_tree.processed_candidate_count,
        retained_tree_nodes=len(nodes),
        target_verify_tree_nodes=(
            1
            if suffix_tree.target_verify_tree_nodes <= 1
            else len(prefix_draft_ids) + suffix_tree.target_verify_tree_nodes
        ),
    )


def rebase_draft_tree(
    draft_tree: DraftCandidateTree,
    accepted_count: int,
) -> DraftCandidateTree:
    consumed = max(0, min(int(accepted_count), len(draft_tree.primary_node_ids)))
    if consumed == 0:
        return draft_tree
    prefix_ids = list(draft_tree.prefix_ids) + list(draft_tree.primary_ids[:consumed])
    remaining_primary = list(draft_tree.primary_ids[consumed:])
    if not remaining_primary:
        return DraftCandidateTree(prefix_ids, [], [], [])

    new_root = draft_tree.primary_node_ids[consumed - 1]
    descendants = _descendant_nodes(draft_tree, new_root)
    id_map: dict[int, int] = {}
    nodes: list[DraftTreeNode] = []
    for node in descendants:
        mapped_parent = None if node.parent_id == new_root else id_map[node.parent_id]
        new_node = _new_tree_node(
            nodes,
            mapped_parent,
            node.token_id,
            node.depth - consumed,
        )
        id_map[node.node_id] = new_node.node_id
    primary_node_ids = [
        id_map[node_id]
        for node_id in draft_tree.primary_node_ids[consumed:]
        if node_id in id_map
    ]
    return DraftCandidateTree(
        prefix_ids,
        remaining_primary,
        primary_node_ids,
        nodes,
        processed_candidate_count=sum(1 for node in nodes if node.processed),
        retained_tree_nodes=len(nodes),
        target_verify_tree_nodes=1 if nodes else 0,
    )


def verify_candidate_tree(
    prefix_ids: Sequence[int],
    draft_tree: DraftCandidateTree,
    target_token_fn: Callable[[Sequence[int]], int],
    eos_token_id: int | None = None,
) -> VerificationResult:
    children = _children_by_parent(draft_tree)
    context = list(prefix_ids)
    emitted: list[int] = []
    accepted = 0
    parent_id: int | None = None

    while True:
        target_token = int(target_token_fn(context))
        matching_child = next(
            (
                child
                for child in children.get(parent_id, [])
                if child.token_id == target_token
            ),
            None,
        )
        if matching_child is None:
            emitted.append(target_token)
            rejected = bool(children.get(parent_id))
            return VerificationResult(accepted, emitted, rejected, None if rejected else target_token)
        emitted.append(target_token)
        accepted += 1
        context.append(target_token)
        parent_id = matching_child.node_id
        if target_token == eos_token_id:
            return VerificationResult(accepted, emitted, False)
        if not children.get(parent_id):
            bonus = int(target_token_fn(context))
            emitted.append(bonus)
            return VerificationResult(accepted, emitted, False, bonus)


def _verify_candidate_tree_from_logits(
    logits: Any,
    row: int,
    draft_tree: DraftCandidateTree,
    vocab_size: int,
    eos_token_id: int | None,
) -> VerificationResult:
    children = _children_by_parent(draft_tree)
    prefix_len = len(draft_tree.prefix_ids)
    if prefix_len <= 0:
        raise ValueError("tree verification requires a non-empty prefix")
    node_position_by_id = {
        node.node_id: prefix_len + index
        for index, node in enumerate(draft_tree.nodes)
    }

    def target_token(parent_id: int | None) -> int:
        position = prefix_len - 1 if parent_id is None else node_position_by_id[parent_id]
        return int(logits[row, position, :vocab_size].argmax(dim=-1).item())

    emitted: list[int] = []
    accepted = 0
    parent_id: int | None = None
    while True:
        token = target_token(parent_id)
        matching_child = next(
            (
                child
                for child in children.get(parent_id, [])
                if child.token_id == token
            ),
            None,
        )
        if matching_child is None:
            emitted.append(token)
            rejected = bool(children.get(parent_id))
            return VerificationResult(accepted, emitted, rejected, None if rejected else token)
        emitted.append(token)
        accepted += 1
        parent_id = matching_child.node_id
        if token == eos_token_id:
            return VerificationResult(accepted, emitted, False)
        if not children.get(parent_id):
            bonus = target_token(parent_id)
            emitted.append(bonus)
            return VerificationResult(accepted, emitted, False, bonus)


def _new_tree_node(
    nodes: list[DraftTreeNode],
    parent_id: int | None,
    token_id: int,
    depth: int,
    logprob: float = 0.0,
    processed: bool = True,
) -> DraftTreeNode:
    node = DraftTreeNode(
        node_id=len(nodes) + 1,
        parent_id=parent_id,
        token_id=int(token_id),
        depth=int(depth),
        logprob=float(logprob),
        processed=bool(processed),
    )
    nodes.append(node)
    return node


def _ordered_tree_parents(
    parent_ids: Sequence[int | None],
    primary_node_ids: Sequence[int],
) -> list[int | None]:
    primary_rank = {node_id: index for index, node_id in enumerate(primary_node_ids)}

    def key(parent_id: int | None) -> tuple[int, int]:
        if parent_id is None:
            return (0, -1)
        if parent_id in primary_rank:
            return (0, primary_rank[parent_id])
        return (1, int(parent_id))

    return sorted(dict.fromkeys(parent_ids), key=key)


def _unique_token_ids(token_ids: Sequence[int], limit: int) -> list[int]:
    unique: list[int] = []
    seen: set[int] = set()
    for token_id in token_ids:
        value = int(token_id)
        if value in seen:
            continue
        unique.append(value)
        seen.add(value)
        if len(unique) >= limit:
            break
    return unique


def _unique_token_candidates(
    candidates: Sequence[DraftTokenCandidate],
    limit: int,
) -> list[DraftTokenCandidate]:
    unique: list[DraftTokenCandidate] = []
    seen: set[int] = set()
    for candidate in candidates:
        token_id = int(candidate.token_id)
        if token_id in seen:
            continue
        unique.append(DraftTokenCandidate(token_id, float(candidate.logprob)))
        seen.add(token_id)
        if len(unique) >= limit:
            break
    return unique


def _with_tree_stats(
    draft_tree: DraftCandidateTree,
    processed_candidate_count: int,
) -> DraftCandidateTree:
    retained = len(draft_tree.nodes)
    return DraftCandidateTree(
        list(draft_tree.prefix_ids),
        list(draft_tree.primary_ids),
        list(draft_tree.primary_node_ids),
        list(draft_tree.nodes),
        processed_candidate_count=processed_candidate_count,
        retained_tree_nodes=retained,
        target_verify_tree_nodes=1 if draft_tree.primary_ids else 0,
    )


def _best_primary_node_ids(nodes: Sequence[DraftTreeNode]) -> list[int]:
    children = _children_by_parent_nodes(nodes)
    primary: list[int] = []
    parent_id: int | None = None
    while True:
        child_options = children.get(parent_id, [])
        if not child_options:
            return primary
        child = max(child_options, key=lambda node: (node.logprob, -node.node_id))
        primary.append(child.node_id)
        parent_id = child.node_id


def _leaf_nodes(draft_tree: DraftCandidateTree) -> list[DraftTreeNode]:
    parent_ids = {node.parent_id for node in draft_tree.nodes if node.parent_id is not None}
    return [node for node in draft_tree.nodes if node.node_id not in parent_ids]


def _path_to_node(draft_tree: DraftCandidateTree, node_id: int) -> list[int]:
    by_id = {node.node_id: node for node in draft_tree.nodes}
    path: list[int] = []
    current_id: int | None = node_id
    while current_id is not None:
        node = by_id[current_id]
        path.append(node.token_id)
        current_id = node.parent_id
    return list(reversed(path))


def _children_by_parent_nodes(
    nodes: Sequence[DraftTreeNode],
) -> dict[int | None, list[DraftTreeNode]]:
    children: dict[int | None, list[DraftTreeNode]] = {}
    for node in nodes:
        children.setdefault(node.parent_id, []).append(node)
    return children


def _children_by_parent(
    draft_tree: DraftCandidateTree,
) -> dict[int | None, list[DraftTreeNode]]:
    children: dict[int | None, list[DraftTreeNode]] = {}
    for node in draft_tree.nodes:
        children.setdefault(node.parent_id, []).append(node)
    return children


def _descendant_nodes(
    draft_tree: DraftCandidateTree,
    root_node_id: int,
) -> list[DraftTreeNode]:
    children = _children_by_parent(draft_tree)
    descendants: list[DraftTreeNode] = []
    stack = list(children.get(root_node_id, []))
    while stack:
        node = stack.pop(0)
        descendants.append(node)
        stack.extend(children.get(node.node_id, []))
    return descendants


def _huggingface_load_kwargs(model_runner: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    local_files_only = _optional_bool(
        model_runner.get("local_files_only"),
        "model_runner.local_files_only",
    )
    if local_files_only is None:
        local_files_only = _offline_env_enabled()
    if local_files_only is not None:
        kwargs["local_files_only"] = local_files_only
    for key in ("cache_dir", "revision"):
        value = model_runner.get(key)
        if value is not None:
            kwargs[key] = str(value)
    return kwargs


def _offline_env_enabled() -> bool | None:
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        value = os.environ.get(name)
        if value is not None:
            return _optional_bool(value, name)
    return None


def _optional_bool(value: object, name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _looks_like_network_error(exc: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        module = type(current).__module__
        name = type(current).__name__
        if module.startswith(("httpx", "httpcore", "requests", "urllib3")):
            return True
        if name in {
            "ConnectionError",
            "ConnectError",
            "ConnectTimeout",
            "ProxyError",
            "ProtocolError",
            "ReadTimeout",
            "RemoteProtocolError",
            "Timeout",
            "TimeoutException",
        }:
            return True
        current = current.__cause__ or current.__context__
    return False
