from __future__ import annotations

import csv
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from scripts import profile_target_verification_latency as profiler


def test_default_matrix_matches_requested_formal_grid() -> None:
    assert profiler.DEFAULT_BATCH_SIZES == (1, 2, 4, 8, 16)
    assert profiler.DEFAULT_CONTEXT_LENGTHS == (128, 512, 1024, 2048)
    assert profiler.DEFAULT_GAMMAS == (1, 2, 4, 8)
    assert profiler.DEFAULT_TREE_NODES == (8, 16, 32, 64)
    assert profiler.DEFAULT_WARMUP == 10
    assert profiler.DEFAULT_REPEAT == 30


def test_smoke_matrix_expands_to_expected_rows() -> None:
    specs = profiler.expand_profile_specs(
        batch_sizes=(1, 2),
        context_lengths=(128,),
        gammas=(1, 4),
        tree_nodes=(8,),
    )

    assert len(specs) == 8
    assert sum(spec.method == "target_decode" for spec in specs) == 2
    assert sum(spec.method == "linear_verification" for spec in specs) == 4
    assert sum(spec.method == "tree_verification" for spec in specs) == 2

    decode = next(spec for spec in specs if spec.method == "target_decode")
    assert decode.gamma is None
    assert decode.tree_nodes is None
    assert decode.timed_input_length == 1
    assert decode.tree_mode == ""

    linear = next(
        spec
        for spec in specs
        if spec.method == "linear_verification" and spec.gamma == 4
    )
    assert linear.tree_nodes is None
    assert linear.timed_input_length == 4

    tree = next(spec for spec in specs if spec.method == "tree_verification")
    assert tree.gamma is None
    assert tree.tree_nodes == 8
    assert tree.timed_input_length == 1
    assert tree.tree_mode == "fixed_forward_approx"


def test_statistics_use_population_std_and_linear_percentiles() -> None:
    stats = profiler.compute_statistics([1.0, 2.0, 3.0, 4.0])

    assert stats == {
        "mean_ms": 2.5,
        "p50_ms": 2.5,
        "p95_ms": 3.8499999999999996,
        "std_ms": 1.118033988749895,
    }


def test_csv_contract_contains_required_provenance_and_scope_fields() -> None:
    required = {
        "method",
        "batch_size",
        "context_length",
        "gamma",
        "tree_nodes",
        "mean_ms",
        "p50_ms",
        "p95_ms",
        "std_ms",
        "gpu_name",
        "model_name",
        "tree_mode",
        "status",
        "error",
        "dtype",
        "attention_implementation",
        "use_cache",
        "torch_version",
        "transformers_version",
        "cuda_version",
        "model_revision",
        "past_length",
        "timed_input_length",
        "warmup",
        "repeat",
        "latency_scope",
        "device",
        "peak_memory_mb",
    }

    assert required <= set(profiler.CSV_FIELDS)


def test_positive_integer_list_parser_rejects_nonpositive_values() -> None:
    assert profiler.parse_positive_ints("1, 4,8") == (1, 4, 8)

    try:
        profiler.parse_positive_ints("1,0")
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("zero must be rejected")


def test_cached_forward_inputs_cover_past_and_timed_tokens() -> None:
    inputs = profiler.build_forward_inputs(
        torch_module=torch,
        batch_size=2,
        context_length=3,
        timed_input_length=4,
        vocab_size=32,
        device="cpu",
    )

    assert inputs.input_ids.shape == (2, 4)
    assert inputs.attention_mask.shape == (2, 7)
    assert inputs.attention_mask.tolist() == [[1] * 7, [1] * 7]
    assert inputs.position_ids.tolist() == [[3, 4, 5, 6], [3, 4, 5, 6]]
    assert inputs.cache_position.tolist() == [3, 4, 5, 6]
    assert inputs.input_ids.min().item() >= 0
    assert inputs.input_ids.max().item() < 32


def test_forward_kwargs_include_cache_position_only_when_supported() -> None:
    class WithCachePosition:
        def forward(
            self,
            input_ids,
            past_key_values,
            attention_mask,
            position_ids,
            use_cache,
            cache_position=None,
        ):
            raise AssertionError("not called")

    class WithoutCachePosition:
        def forward(
            self,
            input_ids,
            past_key_values,
            attention_mask,
            position_ids,
            use_cache,
        ):
            raise AssertionError("not called")

    inputs = profiler.build_forward_inputs(
        torch_module=torch,
        batch_size=1,
        context_length=3,
        timed_input_length=1,
        vocab_size=32,
        device="cpu",
    )
    cache = object()

    with_cache_position = profiler.build_forward_kwargs(
        WithCachePosition(), inputs, cache
    )
    without_cache_position = profiler.build_forward_kwargs(
        WithoutCachePosition(), inputs, cache
    )

    assert with_cache_position["cache_position"].tolist() == [3]
    assert "cache_position" not in without_cache_position
    assert with_cache_position["use_cache"] is True


def test_legacy_prefix_cache_is_cloned_for_each_sample() -> None:
    key = torch.arange(12).reshape(1, 1, 3, 4)
    value = key + 100
    canonical = ((key, value),)

    first = profiler.fresh_prefix_cache(canonical, lambda: None)
    second = profiler.fresh_prefix_cache(canonical, lambda: None)

    assert profiler.cache_sequence_length(first) == 3
    assert profiler.cache_sequence_length(second) == 3
    assert first[0][0].data_ptr() != key.data_ptr()
    assert second[0][0].data_ptr() != first[0][0].data_ptr()
    first[0][0].add_(1)
    assert torch.equal(key, torch.arange(12).reshape(1, 1, 3, 4))


def test_nonconvertible_cache_is_rebuilt_instead_of_reused() -> None:
    class NonConvertibleCache:
        def __init__(self, generation: int) -> None:
            self.generation = generation

        def get_seq_length(self) -> int:
            return 3

    canonical = NonConvertibleCache(0)
    built: list[NonConvertibleCache] = []

    def rebuild() -> NonConvertibleCache:
        cache = NonConvertibleCache(len(built) + 1)
        built.append(cache)
        return cache

    first = profiler.fresh_prefix_cache(canonical, rebuild)
    second = profiler.fresh_prefix_cache(canonical, rebuild)

    assert first is built[0]
    assert second is built[1]
    assert first is not second
    assert canonical.generation == 0


def test_cache_fingerprint_detects_in_place_tensor_changes() -> None:
    cache = ((torch.zeros(1, 1, 3, 2), torch.ones(1, 1, 3, 2)),)
    before = profiler.cache_fingerprint(cache)

    cache[0][0].add_(1)

    assert profiler.cache_fingerprint(cache) != before


class FakeBackend:
    def __init__(self) -> None:
        self.metadata = {
            "gpu_name": "Fake GPU",
            "model_name": "fake/model",
            "dtype": "float16",
            "attention_implementation": "sdpa",
            "torch_version": "2.test",
            "transformers_version": "4.test",
            "cuda_version": "12.test",
            "model_revision": "revision-test",
            "device": "cuda:0",
        }
        self.prefix_oom: set[tuple[int, int]] = set()
        self.measure_oom: set[tuple[str, int, int, int | None]] = set()
        self.probes: list[tuple[int, int]] = []
        self.measure_calls: list[profiler.ProfileSpec] = []
        self.cleanup_calls = 0

    def prepare_prefix(self, batch_size: int, context_length: int):
        self.probes.append((batch_size, context_length))
        if (batch_size, context_length) in self.prefix_oom:
            raise RuntimeError("CUDA out of memory in prefix")
        return {"past_length": context_length}

    def measure(self, spec, prefix_state, warmup: int, repeat: int):
        self.measure_calls.append(spec)
        key = (spec.method, spec.batch_size, spec.context_length, spec.gamma)
        if key in self.measure_oom:
            raise RuntimeError("CUDA out of memory in timed forward")
        base = float(spec.batch_size + spec.context_length + spec.timed_input_length)
        return profiler.Measurement([base + index for index in range(repeat)], 123.5)

    @staticmethod
    def is_oom(exc: BaseException) -> bool:
        return "out of memory" in str(exc).lower()

    def cleanup_after_oom(self) -> None:
        self.cleanup_calls += 1


def test_tree_nodes_share_one_physical_measurement_and_identical_stats(
    tmp_path: Path,
) -> None:
    backend = FakeBackend()
    specs = profiler.expand_profile_specs(
        batch_sizes=(1,),
        context_lengths=(128,),
        gammas=(1,),
        tree_nodes=(8, 16, 32),
    )

    rows = profiler.profile_matrix(
        backend=backend,
        specs=specs,
        warmup=2,
        repeat=3,
        output_path=tmp_path / "profile.csv",
    )

    tree_calls = [
        spec for spec in backend.measure_calls if spec.method == "tree_verification"
    ]
    tree_rows = [row for row in rows if row["method"] == "tree_verification"]
    assert len(tree_calls) == 1
    assert [row["tree_nodes"] for row in tree_rows] == [8, 16, 32]
    shared_fields = ("mean_ms", "p50_ms", "p95_ms", "std_ms", "peak_memory_mb")
    assert len({tuple(row[field] for field in shared_fields) for row in tree_rows}) == 1
    assert all(row["tree_mode"] == "fixed_forward_approx" for row in tree_rows)


def test_prefix_oom_fans_out_to_group_and_later_groups_continue(tmp_path: Path) -> None:
    backend = FakeBackend()
    backend.prefix_oom.add((1, 128))
    specs = profiler.expand_profile_specs(
        batch_sizes=(1, 2),
        context_lengths=(128,),
        gammas=(1, 4),
        tree_nodes=(8, 16),
    )

    rows = profiler.profile_matrix(
        backend=backend,
        specs=specs,
        warmup=1,
        repeat=2,
        output_path=tmp_path / "profile.csv",
    )

    failed = [row for row in rows if row["batch_size"] == 1]
    continued = [row for row in rows if row["batch_size"] == 2]
    assert len(failed) == 5
    assert all(row["status"] == "oom" for row in failed)
    assert all(row["mean_ms"] == "" for row in failed)
    assert all(row["status"] == "success" for row in continued)
    assert backend.cleanup_calls == 1
    assert not any(spec.batch_size == 1 for spec in backend.measure_calls)


def test_timed_oom_records_one_combination_and_continues(tmp_path: Path) -> None:
    backend = FakeBackend()
    backend.measure_oom.add(("linear_verification", 1, 128, 4))
    specs = profiler.expand_profile_specs(
        batch_sizes=(1,),
        context_lengths=(128,),
        gammas=(1, 4),
        tree_nodes=(8,),
    )

    rows = profiler.profile_matrix(
        backend=backend,
        specs=specs,
        warmup=1,
        repeat=2,
        output_path=tmp_path / "profile.csv",
    )

    failed = [row for row in rows if row["status"] == "oom"]
    assert len(failed) == 1
    assert failed[0]["method"] == "linear_verification"
    assert failed[0]["gamma"] == 4
    assert rows[-1]["method"] == "tree_verification"
    assert rows[-1]["status"] == "success"
    assert backend.cleanup_calls == 1


def test_csv_is_rewritten_after_every_emitted_row(tmp_path: Path) -> None:
    backend = FakeBackend()
    write_sizes: list[int] = []

    def recording_writer(path, rows):
        write_sizes.append(len(rows))
        profiler.write_csv(path, rows)

    specs = profiler.expand_profile_specs(
        batch_sizes=(1,),
        context_lengths=(128,),
        gammas=(1,),
        tree_nodes=(8,),
    )
    output = tmp_path / "profile.csv"

    rows = profiler.profile_matrix(
        backend=backend,
        specs=specs,
        warmup=10,
        repeat=30,
        output_path=output,
        csv_writer=recording_writer,
    )

    assert write_sizes == [1, 2, 3]
    with output.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == len(rows) == 3
    assert written[0]["latency_scope"] == "cuda_device_elapsed"
    assert written[0]["past_length"] == "128"
    assert written[0]["timed_input_length"] == "1"
    assert written[0]["warmup"] == "10"
    assert written[0]["repeat"] == "30"
    assert written[0]["peak_memory_mb"] == "123.5"


def test_incremental_csv_preserves_every_batch_and_context_group(
    tmp_path: Path,
) -> None:
    backend = FakeBackend()
    specs = profiler.expand_profile_specs(
        batch_sizes=(1, 2),
        context_lengths=(128, 512),
        gammas=(1,),
        tree_nodes=(8,),
    )
    output = tmp_path / "profile.csv"

    accumulated_rows = profiler.profile_matrix(
        backend=backend,
        specs=specs,
        warmup=1,
        repeat=2,
        output_path=output,
    )

    with output.open(newline="", encoding="utf-8") as handle:
        persisted_rows = list(csv.DictReader(handle))
    expected_groups = {(1, 128), (1, 512), (2, 128), (2, 512)}
    persisted_groups = {
        (int(row["batch_size"]), int(row["context_length"]))
        for row in persisted_rows
    }
    assert len(specs) == 12
    assert len(accumulated_rows) == len(persisted_rows) == len(specs)
    assert persisted_groups == expected_groups


def test_write_csv_uses_os_replace_for_atomic_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replacements: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def recording_replace(source, destination) -> None:
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(profiler.os, "replace", recording_replace)
    output = tmp_path / "profile.csv"

    profiler.write_csv(output, [{"method": "target_decode"}])

    assert replacements == [(output.with_suffix(".csv.tmp"), output)]


def test_final_validation_rejects_truncated_persisted_csv(tmp_path: Path) -> None:
    backend = FakeBackend()
    specs = profiler.expand_profile_specs(
        batch_sizes=(1, 2),
        context_lengths=(128, 512),
        gammas=(1,),
        tree_nodes=(8,),
    )

    def truncating_writer(path, accumulated_rows) -> None:
        profiler.write_csv(path, accumulated_rows[-3:])

    with pytest.raises(RuntimeError, match="persisted CSV row count"):
        profiler.profile_matrix(
            backend=backend,
            specs=specs,
            warmup=1,
            repeat=2,
            output_path=tmp_path / "profile.csv",
            csv_writer=truncating_writer,
        )


def test_non_oom_backend_failure_is_not_silenced(tmp_path: Path) -> None:
    class BrokenBackend(FakeBackend):
        def prepare_prefix(self, batch_size: int, context_length: int):
            raise ValueError("invalid model configuration")

    with pytest.raises(ValueError, match="invalid model configuration"):
        profiler.profile_matrix(
            backend=BrokenBackend(),
            specs=profiler.expand_profile_specs(
                batch_sizes=(1,),
                context_lengths=(128,),
                gammas=(1,),
                tree_nodes=(8,),
            ),
            warmup=1,
            repeat=1,
            output_path=tmp_path / "profile.csv",
        )


def test_target_only_runner_config_does_not_mutate_or_load_drafters() -> None:
    config = {
        "model_runner": {
            "target_model": "original/model",
            "target_device": "cuda:1",
            "revision": "old-revision",
            "drafter_models": {"small": {"model": "draft/model", "device": "cuda:0"}},
        }
    }

    result = profiler.target_only_runner_config(
        config,
        model_name="override/model",
        device="cuda:0",
        revision="new-revision",
        cache_dir="/cache",
        local_files_only=True,
    )

    assert result["model_runner"] == {
        "target_model": "override/model",
        "target_device": "cuda:0",
        "revision": "new-revision",
        "cache_dir": "/cache",
        "local_files_only": True,
        "drafter_models": {},
    }
    assert config["model_runner"]["target_model"] == "original/model"
    assert "small" in config["model_runner"]["drafter_models"]


def test_cuda_event_timing_synchronizes_before_start_and_after_end() -> None:
    events: list[str] = []

    class InferenceMode:
        def __enter__(self):
            events.append("inference_enter")

        def __exit__(self, exc_type, exc, traceback):
            events.append("inference_exit")

    class Event:
        def __init__(self, name: str) -> None:
            self.name = name

        def record(self) -> None:
            events.append(f"{self.name}_record")

        def elapsed_time(self, other) -> float:
            assert other.name == "end"
            events.append("elapsed_time")
            return 1.25

    class Cuda:
        event_count = 0

        @classmethod
        def Event(cls, enable_timing: bool):
            assert enable_timing is True
            name = "start" if cls.event_count % 2 == 0 else "end"
            cls.event_count += 1
            return Event(name)

        @staticmethod
        def synchronize(device: str) -> None:
            events.append(f"synchronize:{device}")

    class Torch:
        cuda = Cuda

        @staticmethod
        def inference_mode():
            return InferenceMode()

    class Model:
        def __call__(self, **kwargs):
            assert kwargs["use_cache"] is True
            events.append("model_forward")
            return "outputs"

    outputs, elapsed_ms = profiler.measure_cuda_forward(
        torch_module=Torch,
        model=Model(),
        device="cuda:0",
        kwargs={"use_cache": True},
    )

    assert outputs == "outputs"
    assert elapsed_ms == 1.25
    assert events == [
        "inference_enter",
        "synchronize:cuda:0",
        "start_record",
        "model_forward",
        "end_record",
        "synchronize:cuda:0",
        "elapsed_time",
        "inference_exit",
    ]


def test_script_entrypoint_can_import_repository_modules() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/profile_target_verification_latency.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--batch-sizes" in completed.stdout
