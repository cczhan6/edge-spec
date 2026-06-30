from __future__ import annotations

import csv
import math
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_PROFILE_PATH = Path(
    "outputs/profiling/target_verification_latency_full_merged.csv"
)
SUPPORTED_METHODS = (
    "target_decode",
    "linear_verification",
    "tree_verification",
)
SUPPORTED_METRICS = ("p50_ms", "mean_ms", "p95_ms")
STATISTIC_FIELDS = ("mean_ms", "p50_ms", "p95_ms", "std_ms")
REQUIRED_FIELDS = (
    "method",
    "batch_size",
    "context_length",
    "gamma",
    "tree_nodes",
    *STATISTIC_FIELDS,
    "tree_mode",
    "status",
)


class ProfileValidationError(ValueError):
    """The profile CSV is malformed or internally inconsistent."""


class ProfileQueryError(ValueError):
    """A latency query is invalid or has no feasible measured row."""


@dataclass(frozen=True)
class ProfileRow:
    method: str
    batch_size: int
    context_length: int
    gamma: int | None
    tree_nodes: int | None
    mean_ms: float | None
    p50_ms: float | None
    p95_ms: float | None
    std_ms: float | None
    tree_mode: str | None
    status: str
    raw_row: tuple[tuple[str, str], ...]

    @property
    def statistics(self) -> tuple[float | None, ...]:
        return (self.mean_ms, self.p50_ms, self.p95_ms, self.std_ms)

    def metric_value(self, metric: str) -> float:
        value = getattr(self, metric)
        if value is None:
            raise ProfileQueryError(
                f"profile row {self.method}/{self.batch_size}/{self.context_length} "
                f"has no {metric} value"
            )
        return float(value)


@dataclass(frozen=True)
class ProfileSourceRow:
    method: str
    batch_size: int
    context_length: int
    gamma: int | None
    tree_nodes: int | None
    tree_mode: str | None
    status: str
    actual_subbatch_size: int
    metric: str
    metric_value: float
    requested_tree_nodes: int | None
    raw_row: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class VerificationLatencyQueryResult:
    actual_batch_size: int
    profile_batch_size: int
    profile_batch_sizes: tuple[int, ...]
    subbatch_count: int
    subbatch_sizes: tuple[int, ...]
    profile_context_length: int
    profile_gamma: int | None
    total_latency_ms: float
    tree_mode: str | None
    source_rows: tuple[ProfileSourceRow, ...]


class VerificationLatencyProfile:
    """One-time CSV loader and conservative in-memory latency query index."""

    def __init__(
        self,
        path: str | Path = DEFAULT_PROFILE_PATH,
        *,
        metric: str = "p50_ms",
    ) -> None:
        if metric not in SUPPORTED_METRICS:
            raise ProfileValidationError(
                f"unsupported metric {metric!r}; expected one of {SUPPORTED_METRICS}"
            )
        self.metric = metric
        self.path = Path(path)
        rows = self._load_rows(self.path)

        self.legal_batch_sizes = tuple(sorted({row.batch_size for row in rows}))
        self.legal_context_lengths = tuple(
            sorted({row.context_length for row in rows})
        )
        self.legal_gammas = tuple(
            sorted({row.gamma for row in rows if row.gamma is not None})
        )
        self.legal_tree_nodes = tuple(
            sorted({row.tree_nodes for row in rows if row.tree_nodes is not None})
        )
        self.oom_rows = tuple(row for row in rows if row.status == "oom")

        self._target_index: dict[tuple[str, int, int], ProfileRow] = {}
        self._linear_index: dict[tuple[str, int, int, int], ProfileRow] = {}
        self._tree_index: dict[tuple[str, int, int], ProfileRow] = {}
        self._feasible_batches: dict[tuple[str, int, int | None], tuple[int, ...]] = {}
        self._build_indexes(rows)

    @staticmethod
    def _load_rows(path: Path) -> tuple[ProfileRow, ...]:
        try:
            handle = path.open(newline="", encoding="utf-8")
        except OSError as exc:
            raise ProfileValidationError(f"cannot read profile CSV {path}: {exc}") from exc
        with handle:
            reader = csv.DictReader(handle)
            missing = [field for field in REQUIRED_FIELDS if field not in (reader.fieldnames or ())]
            if missing:
                raise ProfileValidationError(
                    f"profile CSV is missing required fields: {', '.join(missing)}"
                )
            rows: list[ProfileRow] = []
            raw_keys: set[tuple[str, int, int, int | None, int | None]] = set()
            for line_number, raw in enumerate(reader, start=2):
                row = _parse_row(raw, line_number)
                raw_key = (
                    row.method,
                    row.batch_size,
                    row.context_length,
                    row.gamma,
                    row.tree_nodes,
                )
                if raw_key in raw_keys:
                    raise ProfileValidationError(
                        f"duplicate profile row key {raw_key} at line {line_number}"
                    )
                raw_keys.add(raw_key)
                rows.append(row)
        if not rows:
            raise ProfileValidationError("profile CSV must contain at least one row")
        return tuple(rows)

    def _build_indexes(self, rows: Sequence[ProfileRow]) -> None:
        tree_groups: dict[tuple[int, int], list[ProfileRow]] = {}
        feasible: dict[tuple[str, int, int | None], set[int]] = {}

        for row in rows:
            if row.status != "success":
                continue
            if row.method == "target_decode":
                key = (row.method, row.batch_size, row.context_length)
                self._target_index[key] = row
                condition = (row.method, row.context_length, None)
                feasible.setdefault(condition, set()).add(row.batch_size)
            elif row.method == "linear_verification":
                assert row.gamma is not None
                key = (row.method, row.batch_size, row.context_length, row.gamma)
                self._linear_index[key] = row
                condition = (row.method, row.context_length, row.gamma)
                feasible.setdefault(condition, set()).add(row.batch_size)
            else:
                tree_groups.setdefault((row.batch_size, row.context_length), []).append(row)

        for (batch_size, context_length), group in tree_groups.items():
            expected = group[0].statistics
            if any(row.statistics != expected for row in group[1:]):
                raise ProfileValidationError(
                    "tree statistics must be exactly identical for all tree_nodes "
                    f"at batch_size={batch_size}, context_length={context_length}"
                )
            canonical = min(group, key=lambda row: int(row.tree_nodes or 0))
            key = ("tree_verification", batch_size, context_length)
            self._tree_index[key] = canonical
            condition = ("tree_verification", context_length, None)
            feasible.setdefault(condition, set()).add(batch_size)

        self._feasible_batches = {
            condition: tuple(sorted(batch_sizes))
            for condition, batch_sizes in feasible.items()
        }

    def query(
        self,
        method: str,
        *,
        batch_size: int,
        context_length: int | None = None,
        context_lengths: Sequence[int] | None = None,
        gamma: int | None = None,
        tree_nodes: int | None = None,
    ) -> VerificationLatencyQueryResult:
        if method not in SUPPORTED_METHODS:
            raise ProfileQueryError(
                f"unsupported method {method!r}; expected one of {SUPPORTED_METHODS}"
            )
        _require_positive_int(batch_size, "batch_size", ProfileQueryError)
        self._validate_method_arguments(method, gamma, tree_nodes)
        actual_context = self._resolve_context(
            batch_size, context_length, context_lengths
        )
        profile_context = _ceil_tier(
            actual_context,
            self.legal_context_lengths,
            "context",
        )
        profile_gamma = None
        if method == "linear_verification":
            assert gamma is not None
            profile_gamma = _ceil_tier(gamma, self.legal_gammas, "gamma")

        condition = (method, profile_context, profile_gamma)
        feasible_batches = self._feasible_batches.get(condition, ())
        if not feasible_batches:
            raise ProfileQueryError(
                "no feasible success batch for "
                f"method={method}, context={profile_context}, gamma={profile_gamma}"
            )
        subbatch_sizes, profile_batch_sizes = self._plan_batches(
            batch_size, feasible_batches
        )

        sources: list[ProfileSourceRow] = []
        for actual_subbatch_size, profile_batch in zip(
            subbatch_sizes, profile_batch_sizes
        ):
            row = self._lookup_success_row(
                method,
                profile_batch,
                profile_context,
                profile_gamma,
            )
            value = row.metric_value(self.metric)
            sources.append(
                ProfileSourceRow(
                    method=row.method,
                    batch_size=row.batch_size,
                    context_length=row.context_length,
                    gamma=row.gamma,
                    tree_nodes=row.tree_nodes,
                    tree_mode=row.tree_mode,
                    status=row.status,
                    actual_subbatch_size=actual_subbatch_size,
                    metric=self.metric,
                    metric_value=value,
                    requested_tree_nodes=(
                        tree_nodes if method == "tree_verification" else None
                    ),
                    raw_row=row.raw_row,
                )
            )

        source_rows = tuple(sources)
        return VerificationLatencyQueryResult(
            actual_batch_size=batch_size,
            profile_batch_size=max(profile_batch_sizes),
            profile_batch_sizes=profile_batch_sizes,
            subbatch_count=len(subbatch_sizes),
            subbatch_sizes=subbatch_sizes,
            profile_context_length=profile_context,
            profile_gamma=profile_gamma,
            total_latency_ms=sum(row.metric_value for row in source_rows),
            tree_mode=(
                "fixed_forward_approx" if method == "tree_verification" else None
            ),
            source_rows=source_rows,
        )

    @staticmethod
    def _validate_method_arguments(
        method: str,
        gamma: int | None,
        tree_nodes: int | None,
    ) -> None:
        if method == "target_decode":
            if gamma is not None:
                raise ProfileQueryError("target_decode does not accept gamma")
            if tree_nodes is not None:
                raise ProfileQueryError("target_decode does not accept tree_nodes")
        elif method == "linear_verification":
            if gamma is None:
                raise ProfileQueryError("linear_verification requires gamma")
            _require_positive_int(gamma, "gamma", ProfileQueryError)
            if tree_nodes is not None:
                raise ProfileQueryError(
                    "linear_verification does not accept tree_nodes"
                )
        else:
            if tree_nodes is None:
                raise ProfileQueryError("tree_verification requires tree_nodes")
            _require_positive_int(tree_nodes, "tree_nodes", ProfileQueryError)
            if gamma is not None:
                raise ProfileQueryError("tree_verification does not accept gamma")

    @staticmethod
    def _resolve_context(
        batch_size: int,
        context_length: int | None,
        context_lengths: Sequence[int] | None,
    ) -> int:
        if (context_length is None) == (context_lengths is None):
            raise ProfileQueryError(
                "exactly one of context_length or context_lengths is required"
            )
        if context_length is not None:
            _require_positive_int(context_length, "context", ProfileQueryError)
            return context_length
        assert context_lengths is not None
        if isinstance(context_lengths, (str, bytes)):
            raise ProfileQueryError("context_lengths must be a sequence of integers")
        values = tuple(context_lengths)
        if len(values) != batch_size:
            raise ProfileQueryError(
                "context_lengths length must equal batch_size: "
                f"expected {batch_size}, got {len(values)}"
            )
        for value in values:
            _require_positive_int(value, "context", ProfileQueryError)
        return max(values)

    def _plan_batches(
        self,
        actual_batch_size: int,
        feasible_batches: tuple[int, ...],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if actual_batch_size <= self.legal_batch_sizes[-1]:
            rounded = _ceil_tier(
                actual_batch_size,
                self.legal_batch_sizes,
                "batch_size",
            )
            if rounded in feasible_batches:
                return (actual_batch_size,), (rounded,)

        largest_feasible = feasible_batches[-1]
        remaining = actual_batch_size
        actual_chunks: list[int] = []
        profile_chunks: list[int] = []
        while remaining:
            actual_chunk = min(remaining, largest_feasible)
            profile_chunk = _ceil_tier(
                actual_chunk,
                feasible_batches,
                "feasible batch_size",
            )
            actual_chunks.append(actual_chunk)
            profile_chunks.append(profile_chunk)
            remaining -= actual_chunk
        return tuple(actual_chunks), tuple(profile_chunks)

    def _lookup_success_row(
        self,
        method: str,
        batch_size: int,
        context_length: int,
        gamma: int | None,
    ) -> ProfileRow:
        if method == "target_decode":
            return self._target_index[(method, batch_size, context_length)]
        if method == "linear_verification":
            assert gamma is not None
            return self._linear_index[(method, batch_size, context_length, gamma)]
        return self._tree_index[(method, batch_size, context_length)]


def _parse_row(raw: dict[str, str], line_number: int) -> ProfileRow:
    method = str(raw.get("method", "")).strip()
    if method not in SUPPORTED_METHODS:
        raise ProfileValidationError(
            f"unsupported method {method!r} at line {line_number}"
        )
    status = str(raw.get("status", "")).strip().lower()
    if status not in {"success", "oom"}:
        raise ProfileValidationError(
            f"unsupported status {status!r} at line {line_number}"
        )
    batch_size = _parse_integral(raw.get("batch_size", ""), "batch_size", line_number)
    context_length = _parse_integral(
        raw.get("context_length", ""), "context_length", line_number
    )
    gamma = _parse_optional_integral(raw.get("gamma", ""), "gamma", line_number)
    tree_nodes = _parse_optional_integral(
        raw.get("tree_nodes", ""), "tree_nodes", line_number
    )
    tree_mode = str(raw.get("tree_mode", "")).strip() or None

    if method == "target_decode" and (gamma is not None or tree_nodes is not None):
        raise ProfileValidationError(
            f"target_decode row must not define gamma or tree_nodes at line {line_number}"
        )
    if method == "linear_verification" and (gamma is None or tree_nodes is not None):
        raise ProfileValidationError(
            f"linear_verification row requires gamma and no tree_nodes at line {line_number}"
        )
    if method == "tree_verification":
        if gamma is not None or tree_nodes is None:
            raise ProfileValidationError(
                f"tree_verification row requires tree_nodes and no gamma at line {line_number}"
            )
        if tree_mode != "fixed_forward_approx":
            raise ProfileValidationError(
                "tree rows must use tree_mode=fixed_forward_approx "
                f"at line {line_number}"
            )

    statistics: dict[str, float | None] = {}
    for field in STATISTIC_FIELDS:
        text = str(raw.get(field, "")).strip()
        if status == "success":
            statistics[field] = _parse_nonnegative_float(text, field, line_number)
        else:
            statistics[field] = (
                None if not text else _parse_nonnegative_float(text, field, line_number)
            )

    return ProfileRow(
        method=method,
        batch_size=batch_size,
        context_length=context_length,
        gamma=gamma,
        tree_nodes=tree_nodes,
        mean_ms=statistics["mean_ms"],
        p50_ms=statistics["p50_ms"],
        p95_ms=statistics["p95_ms"],
        std_ms=statistics["std_ms"],
        tree_mode=tree_mode,
        status=status,
        raw_row=tuple((key, str(value)) for key, value in raw.items()),
    )


def _parse_integral(value: object, name: str, line_number: int) -> int:
    try:
        number = float(str(value).strip())
    except ValueError as exc:
        raise ProfileValidationError(
            f"{name} must be a positive integer at line {line_number}"
        ) from exc
    if not math.isfinite(number) or not number.is_integer() or number <= 0:
        raise ProfileValidationError(
            f"{name} must be a positive integer at line {line_number}"
        )
    return int(number)


def _parse_optional_integral(
    value: object,
    name: str,
    line_number: int,
) -> int | None:
    if not str(value).strip():
        return None
    return _parse_integral(value, name, line_number)


def _parse_nonnegative_float(value: str, name: str, line_number: int) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise ProfileValidationError(
            f"{name} must be a finite nonnegative number at line {line_number}"
        ) from exc
    if not math.isfinite(number) or number < 0:
        raise ProfileValidationError(
            f"{name} must be a finite nonnegative number at line {line_number}"
        )
    return number


def _require_positive_int(
    value: object,
    name: str,
    error_type: type[ValueError],
) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise error_type(f"must be a positive integer {name}")


def _ceil_tier(value: int, tiers: Sequence[int], name: str) -> int:
    if not tiers:
        raise ProfileQueryError(f"no legal {name} tiers are available")
    index = bisect_left(tiers, value)
    if index == len(tiers):
        raise ProfileQueryError(
            f"{name} {value} exceeds maximum measured tier {tiers[-1]}"
        )
    return int(tiers[index])
