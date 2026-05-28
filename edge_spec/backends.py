from __future__ import annotations

import random
import time
from collections.abc import Mapping
from typing import Protocol, Sequence

from .sampling import (
    apply_top_k_top_p_py,
    sample_from_probs,
    sparse_from_dense,
    sparse_from_torch_probs,
    warp_logits_torch,
)
from .types import DraftOutput, SamplingConfig, SparseProb


def _token_ids_to_list(value) -> list[int]:
    if isinstance(value, Mapping) or (
        hasattr(value, "keys") and "input_ids" in value.keys()
    ):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise ValueError("expected a single prompt, got batched input_ids")
        value = value[0]
    return [int(token_id) for token_id in value]


class ModelBackend(Protocol):
    model_name: str
    eos_token_id: int | None

    def encode_prompt(self, prompt: str) -> list[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...

    def draft(
        self,
        prefix_ids: Sequence[int],
        gamma: int,
        sampling: SamplingConfig,
        rng: random.Random,
    ) -> DraftOutput: ...

    def target_distributions(
        self,
        prefixes: Sequence[Sequence[int]],
        draft_ids_batch: Sequence[Sequence[int]],
        sampling: SamplingConfig,
    ) -> tuple[list[list[SparseProb]], float]: ...

    def generate_target_only(
        self,
        prefix_ids: Sequence[int],
        max_new_tokens: int,
        sampling: SamplingConfig,
        rng: random.Random,
    ) -> tuple[list[int], float]: ...


class FakeBackend:
    def __init__(
        self,
        model_name: str,
        vocab_size: int = 16,
        seed: int = 0,
        eos_token_id: int | None = 0,
        delay_s: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.seed = seed
        self.eos_token_id = eos_token_id
        self.delay_s = delay_s
        self.layer_count = 2
        self.hidden_size = 32
        self.intermediate_size = 128

    def encode_prompt(self, prompt: str) -> list[int]:
        ids = [1]
        ids.extend((ord(ch) % (self.vocab_size - 2)) + 2 for ch in prompt[:32])
        return ids

    def decode(self, token_ids: Sequence[int]) -> str:
        return " ".join(str(int(token_id)) for token_id in token_ids)

    def _base_probs(self, context: Sequence[int]) -> list[float]:
        center = (sum(context) + len(context) * 3 + self.seed) % self.vocab_size
        weights: list[float] = []
        for token_id in range(self.vocab_size):
            distance = abs(token_id - center)
            wrapped = min(distance, self.vocab_size - distance)
            weights.append(1.0 / (1.0 + wrapped))
        if len(context) > 6:
            weights[self.eos_token_id or 0] += 0.2
        total = sum(weights)
        return [weight / total for weight in weights]

    def _dist(self, context: Sequence[int], sampling: SamplingConfig) -> SparseProb:
        probs = apply_top_k_top_p_py(self._base_probs(context), sampling)
        return sparse_from_dense(probs)

    def draft(
        self,
        prefix_ids: Sequence[int],
        gamma: int,
        sampling: SamplingConfig,
        rng: random.Random,
    ) -> DraftOutput:
        start = time.perf_counter()
        if self.delay_s:
            time.sleep(self.delay_s)
        context = list(prefix_ids)
        draft_ids: list[int] = []
        dists: list[SparseProb] = []
        for _ in range(gamma):
            dist = self._dist(context, sampling)
            token_id = sample_from_probs(dist.as_dict(), rng)
            draft_ids.append(token_id)
            dists.append(dist)
            context.append(token_id)
            if token_id == self.eos_token_id:
                break
        return DraftOutput(draft_ids, dists, time.perf_counter() - start)

    def target_distributions(
        self,
        prefixes: Sequence[Sequence[int]],
        draft_ids_batch: Sequence[Sequence[int]],
        sampling: SamplingConfig,
    ) -> tuple[list[list[SparseProb]], float]:
        start = time.perf_counter()
        if self.delay_s:
            time.sleep(self.delay_s)
        batch_dists: list[list[SparseProb]] = []
        for prefix, draft_ids in zip(prefixes, draft_ids_batch):
            row: list[SparseProb] = []
            context = list(prefix)
            for token_id in list(draft_ids) + [None]:
                row.append(self._dist(context, sampling))
                if token_id is not None:
                    context.append(int(token_id))
            batch_dists.append(row)
        return batch_dists, time.perf_counter() - start

    def generate_target_only(
        self,
        prefix_ids: Sequence[int],
        max_new_tokens: int,
        sampling: SamplingConfig,
        rng: random.Random,
    ) -> tuple[list[int], float]:
        start = time.perf_counter()
        context = list(prefix_ids)
        generated: list[int] = []
        for _ in range(max_new_tokens):
            dist = self._dist(context, sampling)
            token_id = sample_from_probs(dist.as_dict(), rng)
            generated.append(token_id)
            context.append(token_id)
            if token_id == self.eos_token_id:
                break
        return generated, time.perf_counter() - start


class HuggingFaceBackend:
    def __init__(
        self,
        model_name: str,
        device: str,
        torch_dtype: str = "auto",
    ) -> None:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device was requested, but this PyTorch build has no CUDA support. "
                "Install a CUDA PyTorch build, for example: "
                "mamba install -n edge-spec -c pytorch -c nvidia --override-channels "
                "pytorch=2.5.* pytorch-cuda=12.4"
            )
        self.torch = torch
        self.model_name = model_name
        self.device = device
        dtype = torch_dtype
        if torch_dtype == "auto" and device.startswith("cuda"):
            dtype = torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"[edge-spec] loading {model_name} on {device}", flush=True)
        major_version = int(transformers.__version__.split(".", 1)[0])
        dtype_kwargs = {"dtype": dtype} if major_version >= 5 else {"torch_dtype": dtype}
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **dtype_kwargs,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model.to(device)
        self.model.eval()
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.layer_count = int(
            getattr(self.model.config, "num_hidden_layers", 0)
            or getattr(self.model.config, "n_layer", 0)
        )
        self.hidden_size = int(
            getattr(self.model.config, "hidden_size", 0)
            or getattr(self.model.config, "n_embd", 0)
        )
        self.intermediate_size = int(
            getattr(self.model.config, "intermediate_size", 0)
            or (4 * self.hidden_size if self.hidden_size else 0)
        )
        self.vocab_size = int(getattr(self.model.config, "vocab_size", 0) or 0)

    def _sync(self) -> None:
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)

    def encode_prompt(self, prompt: str) -> list[int]:
        messages = [{"role": "user", "content": prompt}]
        ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return _token_ids_to_list(ids)

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=True)

    def _sample_incremental(
        self,
        prefix_ids: Sequence[int],
        max_new_tokens: int,
        sampling: SamplingConfig,
        rng: random.Random,
    ) -> tuple[list[int], list[SparseProb]]:
        torch = self.torch
        generated: list[int] = []
        dists: list[SparseProb] = []
        if max_new_tokens <= 0:
            return generated, dists

        input_ids = torch.tensor([list(prefix_ids)], device=self.device, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            for _ in range(max_new_tokens):
                logits = outputs.logits[0, -1, :]
                probs = warp_logits_torch(logits, sampling)
                dist = sparse_from_torch_probs(probs)
                token_id = sample_from_probs(dist.as_dict(), rng)
                generated.append(token_id)
                dists.append(dist)
                if token_id == self.eos_token_id:
                    break
                next_input_ids = torch.tensor(
                    [[token_id]], device=self.device, dtype=torch.long
                )
                attention_mask = torch.ones(
                    (1, len(prefix_ids) + len(generated)),
                    device=self.device,
                    dtype=torch.long,
                )
                outputs = self.model(
                    input_ids=next_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
        return generated, dists

    def draft(
        self,
        prefix_ids: Sequence[int],
        gamma: int,
        sampling: SamplingConfig,
        rng: random.Random,
    ) -> DraftOutput:
        self._sync()
        start = time.perf_counter()
        draft_ids, dists = self._sample_incremental(prefix_ids, gamma, sampling, rng)
        self._sync()
        return DraftOutput(draft_ids, dists, time.perf_counter() - start)

    def target_distributions(
        self,
        prefixes: Sequence[Sequence[int]],
        draft_ids_batch: Sequence[Sequence[int]],
        sampling: SamplingConfig,
    ) -> tuple[list[list[SparseProb]], float]:
        torch = self.torch
        sequences = [list(prefix) + list(draft_ids) for prefix, draft_ids in zip(prefixes, draft_ids_batch)]
        max_len = max(len(sequence) for sequence in sequences)
        input_ids = torch.full(
            (len(sequences), max_len),
            int(self.pad_token_id),
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, sequence in enumerate(sequences):
            input_ids[row, : len(sequence)] = torch.tensor(
                sequence, dtype=torch.long, device=self.device
            )
            attention_mask[row, : len(sequence)] = 1

        self._sync()
        start = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
            batch_dists: list[list[SparseProb]] = []
            for row, (prefix, draft_ids) in enumerate(zip(prefixes, draft_ids_batch)):
                row_dists: list[SparseProb] = []
                for offset in range(len(draft_ids) + 1):
                    position = len(prefix) - 1 + offset
                    probs = warp_logits_torch(logits[row, position, :], sampling)
                    row_dists.append(sparse_from_torch_probs(probs))
                batch_dists.append(row_dists)
        self._sync()
        return batch_dists, time.perf_counter() - start

    def generate_target_only(
        self,
        prefix_ids: Sequence[int],
        max_new_tokens: int,
        sampling: SamplingConfig,
        rng: random.Random,
    ) -> tuple[list[int], float]:
        self._sync()
        start = time.perf_counter()
        generated, _ = self._sample_incremental(
            prefix_ids,
            max_new_tokens,
            sampling,
            rng,
        )
        self._sync()
        return generated, time.perf_counter() - start
