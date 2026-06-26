from __future__ import annotations

import sys
import types
import unittest
from contextlib import contextmanager, nullcontext
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


TARGET_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
SMALL_DRAFTER_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MEDIUM_DRAFTER_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LARGE_DRAFTER_NAME = "Qwen/Qwen2.5-3B-Instruct"


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
    def __init__(self, key=None, token_id=7):
        self.key = key
        self.token_id = token_id

    def argmax(self, *, dim):
        return _FakeScalar(self.token_id)

    def topk(self, k):
        return (
            _FakeVector([-0.1 * index for index in range(k)]),
            _FakeVector([self.token_id + index for index in range(k)]),
        )


class _FakeLogits:
    keys = []

    def __getitem__(self, key):
        self.keys.append(key)
        return _FakeSelectedLogits(key)


class _ContextualLogits:
    def __init__(
        self,
        input_ids,
        attention_mask,
        target_token_fn,
        corrupt_token_fn=None,
    ):
        self.input_ids = [list(row) for row in input_ids]
        self.attention_mask = (
            [list(row) for row in attention_mask]
            if attention_mask is not None
            else [[1 for _ in row] for row in input_ids]
        )
        self.target_token_fn = target_token_fn
        self.corrupt_token_fn = corrupt_token_fn
        self.real_lengths = [
            sum(1 for value in row if value)
            for row in self.attention_mask
        ]
        self.max_len = max((len(row) for row in self.input_ids), default=0)
        self.has_mixed_lengths = len(set(self.real_lengths)) > 1

    def __getitem__(self, key):
        _FakeLogits.keys.append(key)
        row, position, _ = key
        if isinstance(row, slice):
            row = 0
        if isinstance(position, slice):
            position = self.real_lengths[row] - 1
        context = self.input_ids[row][: position + 1]
        token_id = self.target_token_fn(context)
        if self.has_mixed_lengths and self.real_lengths[row] < self.max_len:
            if self.corrupt_token_fn is None:
                corrupt_token = 31 if position == self.real_lengths[row] - 1 else None
            else:
                corrupt_token = self.corrupt_token_fn(
                    context,
                    row,
                    position,
                    self.real_lengths[row],
                    self.max_len,
                )
            if corrupt_token is not None:
                token_id = corrupt_token
        return _FakeSelectedLogits(key, token_id)


class _TreeRootCorruptLogits:
    def __init__(self, input_ids, prefix_len, target_token_fn, corrupt_token):
        self.input_ids = [list(row) for row in input_ids]
        self.prefix_len = int(prefix_len)
        self.target_token_fn = target_token_fn
        self.corrupt_token = int(corrupt_token)

    def __getitem__(self, key):
        _FakeLogits.keys.append(key)
        row, position, _ = key
        if isinstance(row, slice):
            row = 0
        if isinstance(position, slice):
            position = self.prefix_len - 1
        if position == self.prefix_len - 1:
            token_id = self.corrupt_token
        else:
            token_id = self.target_token_fn(self.input_ids[row][: position + 1])
        return _FakeSelectedLogits(key, token_id)


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
        logits_factory = _FakeModelFactory.logits_factories.get(self.name)
        logits = logits_factory(kwargs) if logits_factory is not None else _FakeLogits()
        return types.SimpleNamespace(logits=logits, past_key_values=object())


class _FakeModelFactory:
    models = {}
    calls = []
    fail_first_online_for = set()
    vocab_sizes = {}
    dtypes = {}
    logits_factories = {}

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
    vocab_size = 32
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
        return self.vocab_size

    def get_vocab(self):
        return {str(index): index for index in range(self.vocab_size)}

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


def _default_target_token(context):
    return 10 + ((sum(context) + len(context) * 7) % 20)


@contextmanager
def _contextual_huggingface_runner(
    token_by_context=None,
    *,
    vocab_size=64,
    corrupt_token_fn=None,
):
    mapping = {tuple(key): value for key, value in (token_by_context or {}).items()}

    def target_token(context):
        return int(mapping.get(tuple(context), _default_target_token(context)))

    modules = _fake_modules()
    _FakeTokenizer.vocab_size = vocab_size
    _FakeModelFactory.vocab_sizes = {
        TARGET_MODEL_NAME: vocab_size,
        SMALL_DRAFTER_NAME: vocab_size,
        MEDIUM_DRAFTER_NAME: vocab_size,
        LARGE_DRAFTER_NAME: vocab_size,
    }
    _FakeModelFactory.logits_factories = {
        TARGET_MODEL_NAME: lambda kwargs: _ContextualLogits(
            kwargs["input_ids"].data,
            kwargs.get("attention_mask").data if kwargs.get("attention_mask") else None,
            target_token,
            corrupt_token_fn,
        )
    }
    try:
        with patch.dict(sys.modules, modules):
            yield HuggingFaceModelRunner(load_config("configs/default.yaml"))
    finally:
        _FakeModelFactory.logits_factories = {}
        _FakeTokenizer.vocab_size = 32


class HuggingFaceModelRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        _FakeModelFactory.models = {}
        _FakeModelFactory.calls = []
        _FakeModelFactory.fail_first_online_for = set()
        _FakeModelFactory.vocab_sizes = {}
        _FakeModelFactory.dtypes = {}
        _FakeModelFactory.logits_factories = {}
        _FakeTokenizer.calls = []
        _FakeTokenizer.fail_first_online_for = set()
        _FakeTokenizer.vocab_size = 32
        _FakeLogits.keys = []

    def assertVerificationEqual(
        self,
        actual,
        expected,
        *,
        eos_token_id=None,
    ) -> None:
        self.assertEqual(actual.accepted_count, expected.accepted_count)
        self.assertEqual(
            actual.committed_tokens[: actual.accepted_count],
            expected.committed_tokens[: expected.accepted_count],
        )
        self.assertEqual(actual.correction_token, expected.correction_token)
        self.assertEqual(actual.bonus_token, expected.bonus_token)
        self.assertEqual(actual.committed_tokens, expected.committed_tokens)
        self.assertEqual(actual.emitted_ids, expected.emitted_ids)
        self.assertEqual(actual.rejected, expected.rejected)
        if eos_token_id is not None:
            self.assertEqual(
                bool(
                    actual.committed_tokens
                    and actual.committed_tokens[-1] == eos_token_id
                ),
                bool(
                    expected.committed_tokens
                    and expected.committed_tokens[-1] == eos_token_id
                ),
            )

    def assertBatchMatchesIndividualVerify(self, model_runner, requests) -> None:
        individual_results = [
            model_runner.verify(request.prefix_ids, request.draft_ids)
            for request in requests
        ]
        batch_results = model_runner.verify_batch(requests)

        self.assertEqual(len(batch_results), len(individual_results))
        for actual, expected in zip(batch_results, individual_results):
            self.assertVerificationEqual(
                actual,
                expected,
                eos_token_id=model_runner.eos_token_id,
            )

    def test_draft_and_target_only_use_kv_cache_while_verify_batch_does_not(self) -> None:
        with patch.dict(sys.modules, _fake_modules()):
            model_runner = HuggingFaceModelRunner(load_config("configs/default.yaml"))
            draft = model_runner.draft("small", [1, 2], 3)
            small = _FakeModelFactory.models[SMALL_DRAFTER_NAME]
            self.assertEqual(draft, [7, 7, 7])
            self.assertEqual(len(small.calls), 3)
            self.assertTrue(all(call["use_cache"] is True for call in small.calls))
            self.assertNotIn("past_key_values", small.calls[0])
            self.assertIn("past_key_values", small.calls[1])

            target = _FakeModelFactory.models[TARGET_MODEL_NAME]
            results = model_runner.verify_batch(
                [
                    SemanticVerifyInput([1], [7, 7]),
                    SemanticVerifyInput([1, 2, 3], [7]),
                ]
            )
            self.assertEqual([result.accepted_count for result in results], [2, 1])
            self.assertEqual(len(target.calls), 2)
            self.assertTrue(all(call["use_cache"] is False for call in target.calls))
            self.assertEqual(
                [call["input_ids"].shape for call in target.calls],
                [(1, 3), (1, 4)],
            )

            verify_call_count = len(target.calls)
            self.assertEqual(model_runner.target_only([1, 2], 2), [7, 7])
            self.assertTrue(
                all(
                    call["use_cache"] is True
                    for call in target.calls[verify_call_count:]
                )
            )

    def test_verify_batch_mixed_lengths_matches_individual_verify(self) -> None:
        with _contextual_huggingface_runner() as model_runner:
            prefix = [1, 2]
            first = _default_target_token(prefix)
            second = _default_target_token(prefix + [first])
            third = _default_target_token(prefix + [first, second])
            requests = [
                SemanticVerifyInput(prefix, [first]),
                SemanticVerifyInput(prefix, [first, second]),
                SemanticVerifyInput(prefix, [first, second, third]),
            ]

            self.assertBatchMatchesIndividualVerify(model_runner, requests)

    def test_verify_batch_mixed_prefix_lengths_matches_individual_verify(self) -> None:
        with _contextual_huggingface_runner() as model_runner:
            prefixes = [[3], [3, 4], [3, 4, 5]]
            requests = [
                SemanticVerifyInput(prefix, [_default_target_token(prefix)])
                for prefix in prefixes
            ]

            self.assertBatchMatchesIndividualVerify(model_runner, requests)

    def test_verify_batch_preserves_original_request_order(self) -> None:
        with _contextual_huggingface_runner() as model_runner:
            prefix = [1, 4]
            first = _default_target_token(prefix)
            requests = [
                SemanticVerifyInput([8, 1], [_default_target_token([8, 1])]),
                SemanticVerifyInput([2], [_default_target_token([2]), 29]),
                SemanticVerifyInput([5, 5, 5], [_default_target_token([5, 5, 5])]),
                SemanticVerifyInput(
                    prefix,
                    [first, _default_target_token(prefix + [first])],
                ),
            ]

            self.assertBatchMatchesIndividualVerify(model_runner, requests)

    def test_verify_batch_mixed_lengths_preserves_bonus_token(self) -> None:
        prefix = [101, 102, 103]
        token_by_context = {
            tuple(prefix): 330,
            tuple(prefix + [330]): 3838,
            tuple(prefix + [330, 3838]): 444,
        }
        with _contextual_huggingface_runner(
            token_by_context,
            vocab_size=5000,
            corrupt_token_fn=lambda context, row, position, real_length, max_len: (
                2610 if position == real_length - 1 else None
            ),
        ) as model_runner:
            requests = [
                SemanticVerifyInput(prefix, [330]),
                SemanticVerifyInput(prefix, [330, 3838]),
            ]

            self.assertBatchMatchesIndividualVerify(model_runner, requests)
            batch_results = model_runner.verify_batch(requests)
            self.assertEqual(batch_results[0].committed_tokens, [330, 3838])
            self.assertEqual(batch_results[0].bonus_token, 3838)

    def test_verify_batch_mixed_lengths_preserves_correction_token(self) -> None:
        prefix = [7]
        token_by_context = {
            tuple(prefix): 21,
            (8,): 22,
            (8, 22): 23,
        }

        def corrupt_correction_position(context, row, position, real_length, max_len):
            return 29 if context == prefix else None

        with _contextual_huggingface_runner(
            token_by_context,
            corrupt_token_fn=corrupt_correction_position,
        ) as model_runner:
            requests = [
                SemanticVerifyInput(prefix, [20]),
                SemanticVerifyInput([8], [22, 23]),
            ]

            self.assertBatchMatchesIndividualVerify(model_runner, requests)
            batch_results = model_runner.verify_batch(requests)
            self.assertEqual(batch_results[0].accepted_count, 0)
            self.assertEqual(batch_results[0].correction_token, 21)
            self.assertEqual(batch_results[0].committed_tokens, [21])

    def test_verify_batch_mixed_lengths_handles_eos(self) -> None:
        eos = _FakeTokenizer.eos_token_id
        token_by_context = {
            (2,): eos,
            (3,): eos,
            (4,): 11,
            (4, 11): eos,
            (5,): 12,
            (5, 12): 13,
        }
        with _contextual_huggingface_runner(token_by_context) as model_runner:
            requests = [
                SemanticVerifyInput([2], [eos, 14]),
                SemanticVerifyInput([3], [14]),
                SemanticVerifyInput([4], [11]),
                SemanticVerifyInput([5], [12, 13]),
            ]

            self.assertBatchMatchesIndividualVerify(model_runner, requests)
            batch_results = model_runner.verify_batch(requests)
            self.assertEqual(batch_results[0].committed_tokens, [eos])
            self.assertFalse(batch_results[0].rejected)
            self.assertEqual(batch_results[1].correction_token, eos)
            self.assertTrue(batch_results[1].rejected)
            self.assertEqual(batch_results[2].bonus_token, eos)
            self.assertEqual(batch_results[2].committed_tokens, [11, eos])

    def test_verify_batch_groups_by_prefix_and_draft_length(self) -> None:
        with _contextual_huggingface_runner() as model_runner:
            model_runner.verify_batch(
                [
                    SemanticVerifyInput([1], [18, 19, 20]),
                    SemanticVerifyInput([1, 2], [20, 21]),
                ]
            )

            target = _FakeModelFactory.models[TARGET_MODEL_NAME]
            self.assertEqual(len(target.calls), 2)
            self.assertEqual(
                [call["input_ids"].shape for call in target.calls],
                [(1, 4), (1, 4)],
            )

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

    def test_tree_verify_batch_uses_safe_prefix_forward_and_tree_mask(self) -> None:
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
            self.assertEqual(len(target.calls), 2)
            prefix_call = target.calls[0]
            self.assertFalse(prefix_call["use_cache"])
            self.assertEqual(prefix_call["input_ids"].shape, (1, 2))
            self.assertNotIn("position_ids", prefix_call)
            call = target.calls[1]
            self.assertFalse(call["use_cache"])
            self.assertEqual(call["input_ids"].shape, (1, 5))
            self.assertEqual(call["position_ids"].data[0], [0, 1, 2, 2, 3])
            self.assertEqual(call["attention_mask"].shape, (1, 1, 5, 5))
            self.assertIs(call["attention_mask"].dtype, modules["torch"].float16)
            self.assertEqual(results[0].accepted_count, 2)
            self.assertEqual(results[0].emitted_ids, [7, 7, 7])
            self.assertFalse(results[0].rejected)

    def test_tree_verify_batch_preserves_root_token_from_safe_prefix_forward(self) -> None:
        prefix = [101, 102, 103]
        token_by_context = {
            tuple(prefix): 3838,
            tuple(prefix + [3838]): 444,
        }

        def target_token(context):
            return int(token_by_context.get(tuple(context), _default_target_token(context)))

        modules = _fake_modules()
        _FakeTokenizer.vocab_size = 5000
        _FakeModelFactory.vocab_sizes = {
            TARGET_MODEL_NAME: 5000,
            SMALL_DRAFTER_NAME: 5000,
            MEDIUM_DRAFTER_NAME: 5000,
            LARGE_DRAFTER_NAME: 5000,
        }

        def logits_factory(kwargs):
            if "position_ids" in kwargs:
                return _TreeRootCorruptLogits(
                    kwargs["input_ids"].data,
                    len(prefix),
                    target_token,
                    corrupt_token=2610,
                )
            return _ContextualLogits(
                kwargs["input_ids"].data,
                kwargs.get("attention_mask").data if kwargs.get("attention_mask") else None,
                target_token,
            )

        _FakeModelFactory.logits_factories = {
            TARGET_MODEL_NAME: logits_factory,
        }
        try:
            with patch.dict(sys.modules, modules):
                model_runner = HuggingFaceModelRunner(load_config("configs/default.yaml"))
                tree = DraftCandidateTree(
                    prefix_ids=prefix,
                    primary_ids=[3838],
                    primary_node_ids=[1],
                    nodes=[DraftTreeNode(1, None, 3838, 1)],
                )

                result = model_runner.verify_tree_batch(
                    [SemanticTreeVerifyInput(prefix, tree)]
                )[0]

                self.assertEqual(result.accepted_count, 1)
                self.assertEqual(result.committed_tokens, [3838, 444])
                self.assertEqual(result.bonus_token, 444)
        finally:
            _FakeTokenizer.vocab_size = 32

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
            self.assertEqual(len(target.calls), 2)
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
