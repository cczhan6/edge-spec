from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence


@dataclass(frozen=True)
class SemanticVerifyInput:
    prefix_ids: list[int]
    draft_ids: list[int]


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

    def verify(self, prefix_ids: Sequence[int], draft_ids: Sequence[int]) -> VerificationResult: ...

    def verify_batch(self, requests: Sequence[SemanticVerifyInput]) -> list[VerificationResult]: ...

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

    def verify(self, prefix_ids: Sequence[int], draft_ids: Sequence[int]) -> VerificationResult:
        return self.verify_batch(
            [SemanticVerifyInput(list(prefix_ids), list(draft_ids))]
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

    def target_only(self, prefix_ids: Sequence[int], max_new_tokens: int) -> list[int]:
        return self._incremental_greedy(
            self.target_model,
            self.target_device,
            prefix_ids,
            max_new_tokens,
        )

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
