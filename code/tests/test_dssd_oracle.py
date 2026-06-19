from __future__ import annotations

import sys
import types
import unittest
from contextlib import nullcontext
from unittest.mock import patch

from src.config import load_config
from src.model_runner import (
    DraftCandidateTree,
    DraftTreeNode,
    DraftTokenCandidate,
    HuggingFaceModelRunner,
    SemanticTreeVerifyInput,
    SemanticVerifyInput,
    make_linear_draft_tree,
)
from src.tree_drafting import SpecExecDraftTreeStrategy


class _FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _FakeVector:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


class _FakeSelectedLogits:
    def __init__(self, key=None):
        self.key = key

    def argmax(self, *, dim):
        return _FakeScalar(7)

    def topk(self, k):
        return (
            _FakeVector([-0.1 * index for index in range(k)]),
            _FakeVector([7 + index for index in range(k)]),
        )


class _FakeLogits:
    keys = []

    def __getitem__(self, key):
        self.keys.append(key)
        return _FakeSelectedLogits(key)


class _FakeTensor:
    def __init__(self, data, dtype=None):
        self.data = data
        self.dtype = dtype

    @property
    def shape(self):
        shape = []
        value = self.data
        while isinstance(value, list):
            shape.append(len(value))
            value = value[0] if value else None
        return tuple(shape)

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
        self.dtype = _FakeModelFactory.dtypes.get(name, None)
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
    dtypes = {}

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
    torch.float32 = object()
    torch.float16 = object()
    torch.inference_mode = nullcontext
    torch.tensor = lambda values, **kwargs: _FakeTensor(
        values if values and isinstance(values[0], list) else [values],
        dtype=kwargs.get("dtype"),
    )
    torch.full = lambda shape, fill, **kwargs: _FakeTensor(
        [[fill for _ in range(shape[1])] for _ in range(shape[0])],
        dtype=kwargs.get("dtype"),
    )
    torch.zeros_like = lambda tensor: _FakeTensor(
        [[0 for _ in row] for row in tensor.data],
        dtype=tensor.dtype,
    )
    torch.log_softmax = lambda values, dim: values
    torch.topk = lambda values, k, dim: values.topk(k)
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
        _FakeModelFactory.dtypes = {}
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

    def test_huggingface_runner_supports_proactive_bonus_tree(self) -> None:
        with patch.dict(sys.modules, _fake_modules()):
            model_runner = HuggingFaceModelRunner(load_config("configs/default.yaml"))
            model_runner._draft_top_candidates = lambda profile, prefix, width: [
                DraftTokenCandidate(token_id=11 + index, logprob=-0.1 * index)
                for index in range(width)
            ]
            draft_tree = make_linear_draft_tree([1, 2], [3, 4])
            strategy = SpecExecDraftTreeStrategy(
                max_n_beams=4,
                max_beam_len=3,
                max_branch_width=2,
                max_budget=6,
            )

            proactive = model_runner.draft_bonus_tree("small", draft_tree, strategy.plan(2))

            self.assertTrue(proactive.primary_ids)
            self.assertGreaterEqual(proactive.processed_candidate_count, 1)

    def test_huggingface_runner_batches_tree_draft_candidates_by_depth(self) -> None:
        with patch.dict(sys.modules, _fake_modules()):
            model_runner = HuggingFaceModelRunner(load_config("configs/default.yaml"))
            calls = []

            def batch_top_candidates(profile, contexts, width):
                calls.append((profile, [list(context) for context in contexts], width))
                batches = []
                for row, _ in enumerate(contexts):
                    batches.append(
                        [
                            DraftTokenCandidate(
                                token_id=20 + len(calls) * 10 + row * 2 + index,
                                logprob=-0.1 * index,
                            )
                            for index in range(width)
                        ]
                    )
                return batches

            model_runner._draft_top_candidates = lambda profile, prefix, width: (
                _ for _ in ()
            ).throw(AssertionError("tree draft should use batched top-k"))
            model_runner._draft_top_candidates_batch = batch_top_candidates
            strategy = SpecExecDraftTreeStrategy(
                max_n_beams=2,
                max_beam_len=3,
                max_branch_width=2,
                max_budget=8,
            )

            tree = model_runner.draft_tree("small", [1, 2], strategy.plan(3))

            self.assertEqual(len(calls), 3)
            self.assertEqual([len(contexts) for _, contexts, _ in calls], [1, 2, 2])
            self.assertEqual([width for _, _, width in calls], [2, 2, 2])
            self.assertEqual(calls[0][1], [[1, 2]])
            self.assertTrue(tree.primary_ids)
            self.assertEqual(tree.processed_candidate_count, 5)

    def test_draft_top_candidates_batch_uses_one_padded_forward(self) -> None:
        with patch.dict(sys.modules, _fake_modules()):
            model_runner = HuggingFaceModelRunner(load_config("configs/default.yaml"))

            candidates = model_runner._draft_top_candidates_batch(
                "small",
                [[1, 2], [3]],
                2,
            )

            small = _FakeModelFactory.models["Qwen/Qwen2.5-0.5B-Instruct"]
            self.assertEqual(len(small.calls), 1)
            call = small.calls[0]
            self.assertFalse(call["use_cache"])
            self.assertEqual(call["input_ids"].shape, (2, 2))
            self.assertEqual(call["input_ids"].data, [[1, 2], [3, 0]])
            self.assertEqual(call["attention_mask"].data, [[1, 1], [1, 0]])
            self.assertEqual(
                [[candidate.token_id for candidate in batch] for batch in candidates],
                [[7, 8], [7, 8]],
            )

    def test_tree_verify_batch_uses_single_target_forward_with_tree_mask(self) -> None:
        modules = _fake_modules()
        _FakeModelFactory.dtypes = {
            "Qwen/Qwen2.5-7B-Instruct": modules["torch"].float16,
        }
        with patch.dict(sys.modules, modules):
            model_runner = HuggingFaceModelRunner(load_config("configs/default.yaml"))
            model_runner._target_token = lambda prefix: (_ for _ in ()).throw(
                AssertionError("tree verification should use batched logits")
            )
            tree = DraftCandidateTree(
                prefix_ids=[1, 2],
                primary_ids=[7, 7],
                primary_node_ids=[1, 3],
                nodes=[
                    DraftTreeNode(1, None, 7, 1),
                    DraftTreeNode(2, None, 8, 1),
                    DraftTreeNode(3, 1, 7, 2),
                ],
            )

            results = model_runner.verify_tree_batch(
                [SemanticTreeVerifyInput([1, 2], tree)]
            )

            target = _FakeModelFactory.models["Qwen/Qwen2.5-7B-Instruct"]
            self.assertEqual(len(target.calls), 1)
            call = target.calls[0]
            self.assertFalse(call["use_cache"])
            self.assertEqual(call["input_ids"].shape, (1, 5))
            self.assertEqual(call["position_ids"].data[0], [0, 1, 2, 2, 3])
            self.assertEqual(call["attention_mask"].shape, (1, 1, 5, 5))
            self.assertIs(call["attention_mask"].dtype, modules["torch"].float16)
            self.assertEqual(results[0].accepted_count, 2)
            self.assertEqual(results[0].emitted_ids, [7, 7, 7])
            self.assertFalse(results[0].rejected)

    def test_single_tree_verify_uses_tree_forward_path(self) -> None:
        with patch.dict(sys.modules, _fake_modules()):
            model_runner = HuggingFaceModelRunner(load_config("configs/default.yaml"))
            model_runner._target_token = lambda prefix: (_ for _ in ()).throw(
                AssertionError("tree verification should use batched logits")
            )
            tree = DraftCandidateTree(
                prefix_ids=[1, 2],
                primary_ids=[7],
                primary_node_ids=[1],
                nodes=[DraftTreeNode(1, None, 7, 1)],
            )

            result = model_runner.verify_tree([1, 2], tree)

            target = _FakeModelFactory.models["Qwen/Qwen2.5-7B-Instruct"]
            self.assertEqual(len(target.calls), 1)
            self.assertEqual(result.accepted_count, 1)
            self.assertEqual(result.emitted_ids, [7, 7])

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
