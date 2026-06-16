from __future__ import annotations

import sys
import types
import unittest
from contextlib import nullcontext
from unittest.mock import patch

from src.config import load_config
from src.model_runner import HuggingFaceModelRunner, SemanticVerifyInput


class _FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _FakeSelectedLogits:
    def argmax(self, *, dim):
        return _FakeScalar(7)


class _FakeLogits:
    keys = []

    def __getitem__(self, key):
        self.keys.append(key)
        return _FakeSelectedLogits()


class _FakeTensor:
    def __init__(self, data):
        self.data = [list(row) for row in data]

    @property
    def shape(self):
        return len(self.data), len(self.data[0])

    def __setitem__(self, key, value):
        row, column = key
        if isinstance(column, slice):
            if hasattr(value, "data"):
                self.data[row][column] = value.data[0]
            else:
                start, stop, step = column.indices(len(self.data[row]))
                for index in range(start, stop, step):
                    self.data[row][index] = value
        else:
            self.data[row][column] = value


class _FakeModel:
    def __init__(self, name):
        self.name = name
        self.calls = []
        self.config = types.SimpleNamespace(
            vocab_size=_FakeModelFactory.vocab_sizes.get(name, 32)
        )

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(logits=_FakeLogits(), past_key_values=object())


class _FakeModelFactory:
    models = {}
    calls = []
    fail_first_online_for = set()
    vocab_sizes = {}

    @classmethod
    def from_pretrained(cls, name, **kwargs):
        cls.calls.append((name, dict(kwargs)))
        if (
            name in cls.fail_first_online_for
            and not kwargs.get("local_files_only", False)
        ):
            cls.fail_first_online_for.remove(name)
            raise _FakeRemoteProtocolError("server disconnected")
        model = _FakeModel(name)
        cls.models[name] = model
        return model


class _FakeRemoteProtocolError(Exception):
    pass


_FakeRemoteProtocolError.__module__ = "httpx"


class _FakeTokenizer:
    calls = []
    fail_first_online_for = set()
    eos_token_id = 9
    pad_token_id = 0
    eos_token = "<eos>"
    pad_token = "<pad>"

    @classmethod
    def from_pretrained(cls, name, **kwargs):
        cls.calls.append((name, dict(kwargs)))
        if (
            name in cls.fail_first_online_for
            and not kwargs.get("local_files_only", False)
        ):
            cls.fail_first_online_for.remove(name)
            raise _FakeRemoteProtocolError("server disconnected")
        return cls()

    def __len__(self):
        return 32

    def get_vocab(self):
        return {str(index): index for index in range(32)}

    def encode(self, prompt):
        return [index + 1 for index, _ in enumerate(prompt.split())]


def _fake_modules():
    torch = types.ModuleType("torch")
    torch.long = object()
    torch.inference_mode = nullcontext
    torch.tensor = lambda values, **kwargs: _FakeTensor(
        values if values and isinstance(values[0], list) else [values]
    )
    torch.full = lambda shape, fill, **kwargs: _FakeTensor(
        [[fill for _ in range(shape[1])] for _ in range(shape[0])]
    )
    torch.zeros_like = lambda tensor: _FakeTensor(
        [[0 for _ in row] for row in tensor.data]
    )
    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = _FakeModelFactory
    transformers.AutoTokenizer = _FakeTokenizer
    return {"torch": torch, "transformers": transformers}


class HuggingFaceModelRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        _FakeModelFactory.models = {}
        _FakeModelFactory.calls = []
        _FakeModelFactory.fail_first_online_for = set()
        _FakeModelFactory.vocab_sizes = {}
        _FakeTokenizer.calls = []
        _FakeTokenizer.fail_first_online_for = set()
        _FakeLogits.keys = []

    def test_draft_and_target_only_use_kv_cache_while_verify_batch_does_not(self) -> None:
        with patch.dict(sys.modules, _fake_modules()):
            model_runner = HuggingFaceModelRunner(load_config("configs/default.yaml"))
            draft = model_runner.draft("small", [1, 2], 3)
            small = _FakeModelFactory.models["Qwen/Qwen2.5-0.5B-Instruct"]
            self.assertEqual(draft, [7, 7, 7])
            self.assertEqual(len(small.calls), 3)
            self.assertTrue(all(call["use_cache"] is True for call in small.calls))
            self.assertNotIn("past_key_values", small.calls[0])
            self.assertIn("past_key_values", small.calls[1])

            target = _FakeModelFactory.models["Qwen/Qwen2.5-7B-Instruct"]
            results = model_runner.verify_batch(
                [
                    SemanticVerifyInput([1], [7, 7]),
                    SemanticVerifyInput([1, 2, 3], [7]),
                ]
            )
            self.assertEqual([result.accepted_count for result in results], [2, 1])
            self.assertEqual(len(target.calls), 1)
            self.assertFalse(target.calls[0]["use_cache"])
            self.assertEqual(target.calls[0]["input_ids"].shape, (2, 4))

            self.assertEqual(model_runner.target_only([1, 2], 2), [7, 7])
            self.assertTrue(all(call["use_cache"] is True for call in target.calls[1:]))

    def test_allows_padded_model_vocabulary_sizes(self) -> None:
        _FakeModelFactory.vocab_sizes = {
            "Qwen/Qwen2.5-7B-Instruct": 64,
            "Qwen/Qwen2.5-0.5B-Instruct": 40,
            "Qwen/Qwen2.5-1.5B-Instruct": 48,
            "Qwen/Qwen2.5-3B-Instruct": 56,
        }
        with patch.dict(sys.modules, _fake_modules()):
            model_runner = HuggingFaceModelRunner(load_config("configs/default.yaml"))
            self.assertEqual(model_runner.vocab_size, 32)
            self.assertEqual(model_runner.draft("small", [1, 2], 1), [7])
            self.assertEqual(model_runner.verify([1, 2], [7]).accepted_count, 1)

        vocab_slices = [
            key[-1]
            for key in _FakeLogits.keys
            if isinstance(key, tuple) and isinstance(key[-1], slice)
        ]
        self.assertTrue(any(item.stop == 32 for item in vocab_slices))

    def test_rejects_model_vocabulary_that_cannot_cover_tokenizer(self) -> None:
        _FakeModelFactory.vocab_sizes = {
            "Qwen/Qwen2.5-0.5B-Instruct": 31,
        }
        with patch.dict(sys.modules, _fake_modules()):
            with self.assertRaisesRegex(
                ValueError,
                r"drafter small model vocabulary \(31\) cannot cover tokenizer vocabulary \(32\)",
            ):
                HuggingFaceModelRunner(load_config("configs/default.yaml"))

    def test_passes_huggingface_load_options_to_tokenizers_and_models(self) -> None:
        config = load_config("configs/default.yaml")
        config["model_runner"]["local_files_only"] = True
        config["model_runner"]["cache_dir"] = "/tmp/hf-cache"
        config["model_runner"]["revision"] = "main"
        with patch.dict(sys.modules, _fake_modules()):
            HuggingFaceModelRunner(config)

        for _, kwargs in [*_FakeTokenizer.calls, *_FakeModelFactory.calls]:
            self.assertEqual(kwargs["local_files_only"], True)
            self.assertEqual(kwargs["cache_dir"], "/tmp/hf-cache")
            self.assertEqual(kwargs["revision"], "main")

    def test_retries_huggingface_network_failures_from_local_cache(self) -> None:
        _FakeTokenizer.fail_first_online_for = {"Qwen/Qwen2.5-7B-Instruct"}
        with patch.dict(sys.modules, _fake_modules()):
            HuggingFaceModelRunner(load_config("configs/default.yaml"))

        target_tokenizer_calls = [
            kwargs
            for name, kwargs in _FakeTokenizer.calls
            if name == "Qwen/Qwen2.5-7B-Instruct"
        ]
        self.assertEqual(len(target_tokenizer_calls), 2)
        self.assertNotIn("local_files_only", target_tokenizer_calls[0])
        self.assertEqual(target_tokenizer_calls[1]["local_files_only"], True)


if __name__ == "__main__":
    unittest.main()
