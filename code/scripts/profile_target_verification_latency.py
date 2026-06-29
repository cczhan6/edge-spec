from __future__ import annotations

import argparse
import copy
import csv
import gc
import math
import inspect
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_BATCH_SIZES = (1, 2, 4, 8, 16)
DEFAULT_CONTEXT_LENGTHS = (128, 512, 1024, 2048)
DEFAULT_GAMMAS = (1, 2, 4, 8)
DEFAULT_TREE_NODES = (8, 16, 32, 64)
DEFAULT_WARMUP = 10
DEFAULT_REPEAT = 30

CSV_FIELDS = (
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
)


@dataclass(frozen=True)
class ProfileSpec:
    method: str
    batch_size: int
    context_length: int
    gamma: int | None = None
    tree_nodes: int | None = None
    timed_input_length: int = 1
    tree_mode: str = ""


@dataclass(frozen=True)
class ForwardInputs:
    input_ids: Any
    attention_mask: Any
    position_ids: Any
    cache_position: Any


@dataclass(frozen=True)
class Measurement:
    samples_ms: list[float]
    peak_memory_mb: float


def parse_positive_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise ValueError("values must be a comma-separated list of positive integers")
    return values


def expand_profile_specs(
    *,
    batch_sizes: Sequence[int],
    context_lengths: Sequence[int],
    gammas: Sequence[int],
    tree_nodes: Sequence[int],
) -> list[ProfileSpec]:
    specs: list[ProfileSpec] = []
    for batch_size in batch_sizes:
        for context_length in context_lengths:
            specs.append(ProfileSpec("target_decode", batch_size, context_length))
            specs.extend(
                ProfileSpec(
                    "linear_verification",
                    batch_size,
                    context_length,
                    gamma=gamma,
                    timed_input_length=gamma,
                )
                for gamma in gammas
            )
            specs.extend(
                ProfileSpec(
                    "tree_verification",
                    batch_size,
                    context_length,
                    tree_nodes=node_count,
                    tree_mode="fixed_forward_approx",
                )
                for node_count in tree_nodes
            )
    return specs


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def compute_statistics(samples_ms: Sequence[float]) -> dict[str, float]:
    if not samples_ms:
        raise ValueError("latency samples must not be empty")
    values = [float(value) for value in samples_ms]
    return {
        "mean_ms": statistics.fmean(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "std_ms": statistics.pstdev(values),
    }


def build_forward_inputs(
    *,
    torch_module: Any,
    batch_size: int,
    context_length: int,
    timed_input_length: int,
    vocab_size: int,
    device: str,
) -> ForwardInputs:
    token_row = torch_module.arange(
        context_length,
        context_length + timed_input_length,
        dtype=torch_module.long,
        device=device,
    ).remainder(vocab_size)
    input_ids = token_row.unsqueeze(0).expand(batch_size, -1).clone()
    attention_mask = torch_module.ones(
        (batch_size, context_length + timed_input_length),
        dtype=torch_module.long,
        device=device,
    )
    position_ids = torch_module.arange(
        context_length,
        context_length + timed_input_length,
        dtype=torch_module.long,
        device=device,
    ).unsqueeze(0).expand(batch_size, -1)
    cache_position = torch_module.arange(
        context_length,
        context_length + timed_input_length,
        dtype=torch_module.long,
        device=device,
    )
    return ForwardInputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache_position=cache_position,
    )


def build_forward_kwargs(model: Any, inputs: ForwardInputs, past_key_values: Any) -> dict[str, Any]:
    kwargs = {
        "input_ids": inputs.input_ids,
        "past_key_values": past_key_values,
        "attention_mask": inputs.attention_mask,
        "position_ids": inputs.position_ids,
        "use_cache": True,
    }
    parameters = inspect.signature(model.forward).parameters
    if "cache_position" in parameters:
        kwargs["cache_position"] = inputs.cache_position
    return kwargs


def _is_tensor(value: Any) -> bool:
    return all(hasattr(value, name) for name in ("clone", "shape", "data_ptr"))


def _clone_legacy_cache(value: Any) -> Any:
    if _is_tensor(value):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone_legacy_cache(item) for item in value)
    if isinstance(value, list):
        return [_clone_legacy_cache(item) for item in value]
    return value


def fresh_prefix_cache(canonical_cache: Any, rebuild: Callable[[], Any]) -> Any:
    if isinstance(canonical_cache, (tuple, list)):
        return _clone_legacy_cache(canonical_cache)
    to_legacy_cache = getattr(canonical_cache, "to_legacy_cache", None)
    if callable(to_legacy_cache):
        legacy_cache = to_legacy_cache()
        if isinstance(legacy_cache, (tuple, list)):
            return _clone_legacy_cache(legacy_cache)
    if hasattr(canonical_cache, "key_cache") or hasattr(canonical_cache, "value_cache"):
        try:
            cloned = copy.deepcopy(canonical_cache)
        except (RuntimeError, TypeError):
            pass
        else:
            if cloned is not canonical_cache:
                return cloned
    return rebuild()


def cache_sequence_length(cache: Any) -> int:
    get_seq_length = getattr(cache, "get_seq_length", None)
    if callable(get_seq_length):
        return int(get_seq_length())
    if isinstance(cache, (tuple, list)) and cache:
        layer = cache[0]
        if isinstance(layer, (tuple, list)) and layer and _is_tensor(layer[0]):
            return int(layer[0].shape[-2])
    raise TypeError(f"cannot determine sequence length for cache type {type(cache).__name__}")


def cache_fingerprint(cache: Any) -> tuple[Any, ...]:
    entries: list[Any] = []

    def visit(value: Any) -> None:
        if _is_tensor(value):
            entries.append(
                (
                    tuple(int(item) for item in value.shape),
                    int(value.data_ptr()),
                    int(getattr(value, "_version", 0)),
                )
            )
            return
        if isinstance(value, (tuple, list)):
            entries.append((type(value).__name__, len(value)))
            for item in value:
                visit(item)
            return
        key_cache = getattr(value, "key_cache", None)
        value_cache = getattr(value, "value_cache", None)
        if key_cache is not None or value_cache is not None:
            entries.append(type(value).__name__)
            visit([] if key_cache is None else key_cache)
            visit([] if value_cache is None else value_cache)
            return
        entries.append((type(value).__name__, cache_sequence_length(value)))

    visit(cache)
    return tuple(entries)


def target_only_runner_config(
    config: dict[str, Any],
    *,
    model_name: str | None = None,
    device: str | None = None,
    revision: str | None = None,
    cache_dir: str | None = None,
    local_files_only: bool | None = None,
) -> dict[str, Any]:
    result = copy.deepcopy(config)
    model_runner = result["model_runner"]
    model_runner["drafter_models"] = {}
    overrides = {
        "target_model": model_name,
        "target_device": device,
        "revision": revision,
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }
    for key, value in overrides.items():
        if value is not None:
            model_runner[key] = value
    return result


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output_path)


def _persisted_csv_row_count(path: str | Path) -> int:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _base_row(
    spec: ProfileSpec,
    metadata: dict[str, Any],
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {field: "" for field in CSV_FIELDS}
    row.update(metadata)
    row.update(
        {
            "method": spec.method,
            "batch_size": spec.batch_size,
            "context_length": spec.context_length,
            "gamma": "" if spec.gamma is None else spec.gamma,
            "tree_nodes": "" if spec.tree_nodes is None else spec.tree_nodes,
            "tree_mode": spec.tree_mode,
            "use_cache": "true",
            "past_length": spec.context_length,
            "timed_input_length": spec.timed_input_length,
            "warmup": warmup,
            "repeat": repeat,
            "latency_scope": "cuda_device_elapsed",
        }
    )
    return row


def _success_row(
    spec: ProfileSpec,
    measurement: Measurement,
    metadata: dict[str, Any],
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    row = _base_row(spec, metadata, warmup, repeat)
    row.update(compute_statistics(measurement.samples_ms))
    row["peak_memory_mb"] = measurement.peak_memory_mb
    row["status"] = "success"
    return row


def _oom_row(
    spec: ProfileSpec,
    exc: BaseException,
    metadata: dict[str, Any],
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    row = _base_row(spec, metadata, warmup, repeat)
    row["status"] = "oom"
    row["error"] = " ".join(str(exc).split())[:500]
    return row


def profile_matrix(
    *,
    backend: Any,
    specs: Sequence[ProfileSpec],
    warmup: int,
    repeat: int,
    output_path: str | Path,
    csv_writer: Callable[[str | Path, Sequence[dict[str, Any]]], None] = write_csv,
) -> list[dict[str, Any]]:
    if warmup < 0:
        raise ValueError("warmup must be nonnegative")
    if repeat <= 0:
        raise ValueError("repeat must be positive")

    groups: dict[tuple[int, int], list[ProfileSpec]] = {}
    for spec in specs:
        groups.setdefault((spec.batch_size, spec.context_length), []).append(spec)

    expected_row_count = len(specs)
    accumulated_rows: list[dict[str, Any]] = []

    def emit(row: dict[str, Any]) -> None:
        accumulated_rows.append(row)
        csv_writer(output_path, tuple(accumulated_rows))

    for (batch_size, context_length), group_specs in groups.items():
        try:
            prefix_state = backend.prepare_prefix(batch_size, context_length)
        except Exception as exc:
            if not backend.is_oom(exc):
                raise
            for spec in group_specs:
                emit(_oom_row(spec, exc, backend.metadata, warmup, repeat))
            backend.cleanup_after_oom()
            continue

        non_tree_specs = [spec for spec in group_specs if spec.method != "tree_verification"]
        tree_specs = [spec for spec in group_specs if spec.method == "tree_verification"]
        for spec in non_tree_specs:
            try:
                measurement = backend.measure(spec, prefix_state, warmup, repeat)
            except Exception as exc:
                if not backend.is_oom(exc):
                    raise
                emit(_oom_row(spec, exc, backend.metadata, warmup, repeat))
                backend.cleanup_after_oom()
                continue
            emit(_success_row(spec, measurement, backend.metadata, warmup, repeat))

        if tree_specs:
            measured_spec = tree_specs[0]
            try:
                measurement = backend.measure(measured_spec, prefix_state, warmup, repeat)
            except Exception as exc:
                if not backend.is_oom(exc):
                    raise
                for spec in tree_specs:
                    emit(_oom_row(spec, exc, backend.metadata, warmup, repeat))
                backend.cleanup_after_oom()
            else:
                for spec in tree_specs:
                    emit(_success_row(spec, measurement, backend.metadata, warmup, repeat))

    if len(accumulated_rows) != expected_row_count:
        raise RuntimeError(
            "accumulated profiling row count mismatch: "
            f"expected {expected_row_count}, got {len(accumulated_rows)}"
        )
    persisted_row_count = _persisted_csv_row_count(output_path)
    if persisted_row_count != expected_row_count:
        raise RuntimeError(
            "persisted CSV row count mismatch: "
            f"expected {expected_row_count}, got {persisted_row_count}"
        )
    return accumulated_rows


def measure_cuda_forward(
    *,
    torch_module: Any,
    model: Any,
    device: str,
    kwargs: dict[str, Any],
) -> tuple[Any, float]:
    start = torch_module.cuda.Event(enable_timing=True)
    end = torch_module.cuda.Event(enable_timing=True)
    with torch_module.inference_mode():
        torch_module.cuda.synchronize(device)
        start.record()
        outputs = model(**kwargs)
        end.record()
        torch_module.cuda.synchronize(device)
        elapsed_ms = float(start.elapsed_time(end))
    return outputs, elapsed_ms


@dataclass(frozen=True)
class PrefixState:
    batch_size: int
    context_length: int
    canonical_cache: Any
    canonical_fingerprint: tuple[Any, ...]


class CudaTargetProfiler:
    def __init__(
        self,
        *,
        runner: Any,
        runner_config: dict[str, Any],
        transformers_version: str,
    ) -> None:
        self.runner = runner
        self.torch = runner.torch
        self.model = runner.target_model.eval()
        self.device = str(runner.target_device)
        if not self.device.startswith("cuda"):
            raise ValueError("target verification profiling requires a CUDA target device")
        self.vocab_size = int(runner.vocab_size)
        model_runner = runner_config["model_runner"]
        model_config = self.model.config
        revision = getattr(model_config, "_commit_hash", None)
        if revision is None:
            revision = model_runner.get("revision")
        if revision is None:
            revision = (
                runner_config.get("model_bindings", {})
                .get("target", {})
                .get("revision", "")
            )
        attention_implementation = getattr(
            model_config,
            "_attn_implementation",
            getattr(model_config, "_attn_implementation_internal", "unknown"),
        )
        try:
            dtype = str(next(self.model.parameters()).dtype).removeprefix("torch.")
        except StopIteration:
            dtype = "unknown"
        self.metadata = {
            "gpu_name": self.torch.cuda.get_device_name(self.device),
            "model_name": str(model_runner["target_model"]),
            "dtype": dtype,
            "attention_implementation": str(attention_implementation or "unknown"),
            "torch_version": str(self.torch.__version__),
            "transformers_version": str(transformers_version),
            "cuda_version": str(self.torch.version.cuda or "unknown"),
            "model_revision": str(revision or "unknown"),
            "device": self.device,
        }
        self.torch.cuda.synchronize(self.device)

    @classmethod
    def from_config(cls, runner_config: dict[str, Any]) -> "CudaTargetProfiler":
        import transformers

        from src.model_runner import HuggingFaceModelRunner

        runner = HuggingFaceModelRunner(runner_config)
        return cls(
            runner=runner,
            runner_config=runner_config,
            transformers_version=transformers.__version__,
        )

    def _build_prefix(self, batch_size: int, context_length: int) -> PrefixState:
        inputs = build_forward_inputs(
            torch_module=self.torch,
            batch_size=batch_size,
            context_length=0,
            timed_input_length=context_length,
            vocab_size=self.vocab_size,
            device=self.device,
        )
        kwargs = build_forward_kwargs(self.model, inputs, None)
        with self.torch.inference_mode():
            outputs = self.model(**kwargs)
            self.torch.cuda.synchronize(self.device)
        cache = outputs.past_key_values
        del outputs
        actual_length = cache_sequence_length(cache)
        if actual_length != context_length:
            raise RuntimeError(
                f"prefix cache length mismatch: expected {context_length}, got {actual_length}"
            )
        return PrefixState(
            batch_size=batch_size,
            context_length=context_length,
            canonical_cache=cache,
            canonical_fingerprint=cache_fingerprint(cache),
        )

    def prepare_prefix(self, batch_size: int, context_length: int) -> PrefixState:
        return self._build_prefix(batch_size, context_length)

    def _fresh_cache(self, prefix_state: PrefixState) -> Any:
        cache = fresh_prefix_cache(
            prefix_state.canonical_cache,
            lambda: self._build_prefix(
                prefix_state.batch_size,
                prefix_state.context_length,
            ).canonical_cache,
        )
        actual_length = cache_sequence_length(cache)
        if actual_length != prefix_state.context_length:
            raise RuntimeError(
                "isolated prefix cache length mismatch: "
                f"expected {prefix_state.context_length}, got {actual_length}"
            )
        return cache

    def measure(
        self,
        spec: ProfileSpec,
        prefix_state: PrefixState,
        warmup: int,
        repeat: int,
    ) -> Measurement:
        if (
            prefix_state.batch_size != spec.batch_size
            or prefix_state.context_length != spec.context_length
        ):
            raise ValueError("profile spec does not match prepared prefix state")
        inputs = build_forward_inputs(
            torch_module=self.torch,
            batch_size=spec.batch_size,
            context_length=spec.context_length,
            timed_input_length=spec.timed_input_length,
            vocab_size=self.vocab_size,
            device=self.device,
        )
        self.torch.cuda.reset_peak_memory_stats(self.device)
        samples: list[float] = []
        for index in range(warmup + repeat):
            sample_cache = self._fresh_cache(prefix_state)
            kwargs = build_forward_kwargs(self.model, inputs, sample_cache)
            outputs, elapsed_ms = measure_cuda_forward(
                torch_module=self.torch,
                model=self.model,
                device=self.device,
                kwargs=kwargs,
            )
            if index >= warmup:
                samples.append(elapsed_ms)
            del outputs
            del sample_cache
        if cache_fingerprint(prefix_state.canonical_cache) != prefix_state.canonical_fingerprint:
            raise RuntimeError("canonical prefix cache was modified in place")
        peak_memory_mb = float(self.torch.cuda.max_memory_allocated(self.device)) / (1024.0**2)
        return Measurement(samples_ms=samples, peak_memory_mb=peak_memory_mb)

    def is_oom(self, exc: BaseException) -> bool:
        out_of_memory = getattr(self.torch.cuda, "OutOfMemoryError", ())
        return isinstance(exc, out_of_memory) or "out of memory" in str(exc).lower()

    def cleanup_after_oom(self) -> None:
        gc.collect()
        self.torch.cuda.empty_cache()
        self.torch.cuda.synchronize(self.device)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile KV-cached target decode and verification latency with CUDA events."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--device")
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir")
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--batch-sizes",
        type=parse_positive_ints,
        default=DEFAULT_BATCH_SIZES,
    )
    parser.add_argument(
        "--context-lengths",
        type=parse_positive_ints,
        default=DEFAULT_CONTEXT_LENGTHS,
    )
    parser.add_argument("--gammas", type=parse_positive_ints, default=DEFAULT_GAMMAS)
    parser.add_argument(
        "--tree-nodes",
        type=parse_positive_ints,
        default=DEFAULT_TREE_NODES,
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from src.config import load_config

    args = _build_argument_parser().parse_args(argv)
    config = load_config(args.config)
    runner_config = target_only_runner_config(
        config,
        model_name=args.model_name,
        device=args.device,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    backend = CudaTargetProfiler.from_config(runner_config)
    specs = expand_profile_specs(
        batch_sizes=args.batch_sizes,
        context_lengths=args.context_lengths,
        gammas=args.gammas,
        tree_nodes=args.tree_nodes,
    )
    rows = profile_matrix(
        backend=backend,
        specs=specs,
        warmup=args.warmup,
        repeat=args.repeat,
        output_path=args.output,
    )
    success_count = sum(row["status"] == "success" for row in rows)
    oom_count = sum(row["status"] == "oom" for row in rows)
    print(f"wrote {len(rows)} rows to {args.output}: success={success_count}, oom={oom_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
