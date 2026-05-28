from __future__ import annotations

import argparse
import os
from pathlib import Path

from .backends import FakeBackend, HuggingFaceBackend
from .dataset import (
    SPECBENCH_SIX_CATEGORIES,
    fallback_items,
    iter_microbatches,
    load_specbench,
    normalize_specbench_category,
    select_one_per_category,
    select_one_per_category_per_device,
    sort_specbench_categories,
)
from .io import write_json, write_jsonl
from .methods.base import RunConfig
from .methods.baselines.sync_batch import SyncBatchRunner
from .methods.baselines.target_only import TargetOnlyRunner
from .methods.proposed.async_runtime import ProposedAsyncRunner
from .simulation import load_device_profiles
from .types import SamplingConfig, SpecBenchItem


DEFAULT_DRAFT_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]
DEFAULT_TARGET_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def make_progress(enabled: bool, total: int, desc: str):
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    unit = "req" if desc in {"proposed", "target_only"} else "batch"
    return tqdm(total=total, desc=desc, unit=unit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Three-device edge speculative decoding experiments."
    )
    parser.add_argument("--dataset-path", default="data/spec_bench/question.jsonl")
    parser.add_argument("--profile-config", default="configs/edge_hetero.yaml")
    parser.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", "results"))
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument(
        "--category",
        default=os.environ.get("CATEGORY"),
        help=(
            "Run only one SpecBench six-task group. Supported values: "
            f"{', '.join(SPECBENCH_SIX_CATEGORIES)}. Can also be set via CATEGORY."
        ),
    )
    parser.add_argument(
        "--dataset-mode",
        choices=(
            "limit",
            "one-per-category",
            "one-per-category-per-device",
            "all",
        ),
        default=os.environ.get("DATASET_MODE", "limit"),
        help=(
            "Dataset selection mode. 'one-per-category' samples one request from "
            "each SpecBench six-task group; 'all' runs every selected request "
            "without dropping the final partial microbatch."
        ),
    )
    parser.add_argument(
        "--one-per-category",
        action="store_true",
        help=(
            "Evaluate one SpecBench sample from each six-task group. "
            "Ignores the default --limit=3."
        ),
    )
    parser.add_argument(
        "--one-per-category-per-device",
        action="store_true",
        help=(
            "Evaluate one sample per device from each SpecBench six-task group. "
            "Each task group becomes one 3-request microbatch."
        ),
    )
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--draft-models", nargs=3, default=DEFAULT_DRAFT_MODELS)
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--client-device", default="cuda:0")
    parser.add_argument("--server-device", default="cuda:1")
    parser.add_argument("--torch-dtype", default="auto")
    gamma_env = os.environ.get("GAMMA")
    parser.add_argument(
        "--gamma",
        type=int,
        default=int(gamma_env) if gamma_env is not None else None,
        help=(
            "Maximum speculative draft length. Defaults to 8 for proposed "
            "adaptive lookahead and 4 otherwise."
        ),
    )
    parser.add_argument(
        "--initial-lookahead",
        type=int,
        default=int(os.environ.get("INITIAL_LOOKAHEAD", "4")),
        help="Initial adaptive lookahead before device/request adjustments.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    network_seed_env = os.environ.get("NETWORK_SEED")
    parser.add_argument(
        "--network-seed",
        type=int,
        default=int(network_seed_env) if network_seed_env is not None else None,
        help="Seed for the dynamic network trace. Defaults to --seed.",
    )
    parser.add_argument(
        "--network-trace-slot-s",
        type=float,
        default=float(os.environ.get("NETWORK_TRACE_SLOT_S", "0.05")),
        help="Seconds per deterministic network trace slot.",
    )
    parser.add_argument(
        "--method",
        choices=("proposed", "sync_batch", "target_only"),
        default="proposed",
        help="Experiment method to run.",
    )
    parser.add_argument(
        "--lane-count",
        type=int,
        default=int(os.environ.get("LANE_COUNT", "3")),
        help="Number of independent verifier lanes used by --method proposed.",
    )
    parser.add_argument(
        "--max-inflight-segments",
        type=int,
        default=2,
        help="Maximum unverified draft segments per active request.",
    )
    parser.add_argument(
        "--lookahead-policy",
        choices=("adaptive", "fixed"),
        default="adaptive",
    )
    parser.add_argument(
        "--scheduler",
        choices=("prefix-aware", "queue-only"),
        default="prefix-aware",
    )
    parser.add_argument("--lane-batch-size", type=int, default=2)
    parser.add_argument("--lane-batch-timeout-s", type=float, default=0.001)
    parser.add_argument(
        "--use-fake-models",
        action="store_true",
        help="Run without torch/transformers using deterministic fake models.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser.parse_args()


def build_backends(args: argparse.Namespace):
    if args.use_fake_models:
        target = FakeBackend(args.target_model, seed=9, delay_s=0.001)
        if args.method == "target_only":
            return [target for _ in args.draft_models], target
        drafts = [
            FakeBackend(args.draft_models[0], seed=1, delay_s=0.001),
            FakeBackend(args.draft_models[1], seed=3, delay_s=0.001),
            FakeBackend(args.draft_models[2], seed=5, delay_s=0.001),
        ]
        return drafts, target
    target = HuggingFaceBackend(args.target_model, args.server_device, args.torch_dtype)
    if args.method == "target_only":
        return [target for _ in args.draft_models], target
    drafts = [
        HuggingFaceBackend(model_name, args.client_device, args.torch_dtype)
        for model_name in args.draft_models
    ]
    return drafts, target


def build_runner(args: argparse.Namespace, draft_backends, target_backend, profiles, sampling):
    gamma = args.gamma
    if gamma is None:
        gamma = 8 if args.method == "proposed" and args.lookahead_policy == "adaptive" else 4
    config = RunConfig(
        method=args.method,
        sampling=sampling,
        gamma=gamma,
        initial_lookahead=args.initial_lookahead,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        network_seed=args.network_seed,
        network_trace_slot_s=args.network_trace_slot_s,
        lane_count=args.lane_count,
        max_inflight_segments=args.max_inflight_segments,
        lookahead_policy=args.lookahead_policy,
        scheduler=args.scheduler,
        lane_batch_size=args.lane_batch_size,
        lane_batch_timeout_s=args.lane_batch_timeout_s,
    )
    kwargs = {
        "draft_backends": draft_backends,
        "target_backend": target_backend,
        "profiles": profiles,
        "config": config,
    }
    if args.method == "proposed":
        return ProposedAsyncRunner(**kwargs)
    if args.method == "sync_batch":
        return SyncBatchRunner(**kwargs)
    if args.method == "target_only":
        return TargetOnlyRunner(**kwargs)
    raise ValueError(f"unsupported method: {args.method}")


def print_task_metrics(summary: dict) -> None:
    task_metrics = summary.get("task_metrics") or {}
    if not task_metrics:
        return
    print("task_metrics:")
    print(
        "task                 requests  eff_recv_tok/s  e2e_ttft_s  e2e_mean_latency_s"
    )
    for task in sort_specbench_categories(task_metrics):
        metrics = task_metrics[task]
        ttft = metrics.get("e2e_first_token_latency_s")
        ttft_text = f"{ttft:.4f}" if ttft is not None else "n/a"
        print(
            f"{task:<20} "
            f"{metrics['request_count']:>8} "
            f"{metrics['effective_received_throughput_tokens_per_s']:>15.4f} "
            f"{ttft_text:>11} "
            f"{metrics['e2e_mean_latency_s']:>19.4f}"
        )


def select_dataset(args: argparse.Namespace) -> tuple[list[SpecBenchItem], str, str | None]:
    legacy_modes = [
        args.one_per_category,
        args.one_per_category_per_device,
    ]
    if sum(bool(mode) for mode in legacy_modes) > 1:
        raise ValueError(
            "--one-per-category and --one-per-category-per-device are mutually exclusive"
        )
    dataset_mode = args.dataset_mode
    if any(legacy_modes):
        if args.dataset_mode != "limit":
            raise ValueError("Use either --dataset-mode or legacy dataset flags, not both")
        if args.one_per_category:
            dataset_mode = "one-per-category"
        elif args.one_per_category_per_device:
            dataset_mode = "one-per-category-per-device"

    category_filter = normalize_specbench_category(args.category) if args.category else None
    try:
        full_dataset_mode = dataset_mode in {
            "one-per-category",
            "one-per-category-per-device",
            "all",
        }
        items = load_specbench(
            args.dataset_path,
            limit=None if full_dataset_mode else args.limit,
            category=category_filter,
            shuffle=args.shuffle,
            seed=args.seed,
        )
    except FileNotFoundError:
        if not args.use_fake_models:
            raise
        items = fallback_items()
    if category_filter and not items:
        raise ValueError(
            f"no requests found for category {args.category!r} "
            f"(normalized to {category_filter!r}); supported categories: "
            f"{', '.join(SPECBENCH_SIX_CATEGORIES)}"
        )
    if dataset_mode == "one-per-category":
        items = select_one_per_category(items)
    if dataset_mode == "one-per-category-per-device":
        items = select_one_per_category_per_device(
            items, device_count=len(args.draft_models)
        )
    return items, dataset_mode, category_filter


def main() -> None:
    args = parse_args()
    sampling = SamplingConfig(args.temperature, args.top_p, args.top_k)
    sampling.validate()
    profiles = load_device_profiles(args.profile_config)
    items, dataset_mode, category_filter = select_dataset(args)

    microbatches = list(
        iter_microbatches(
            items,
            batch_size=len(args.draft_models),
            drop_last=args.drop_last and dataset_mode != "all",
        )
    )
    if not microbatches:
        raise ValueError("no requests to run")
    run_request_count = sum(len(batch) for batch in microbatches)
    dropped_request_count = len(items) - run_request_count

    draft_backends, target_backend = build_backends(args)
    runner = build_runner(args, draft_backends, target_backend, profiles, sampling)
    progress_total = (
        run_request_count
        if args.method in {"proposed", "target_only"}
        else len(microbatches)
    )
    progress = make_progress(
        enabled=not args.no_progress,
        total=progress_total,
        desc=args.method,
    )
    try:
        result = runner.run_dataset(microbatches, progress=progress)
    finally:
        if progress is not None:
            progress.close()

    records = result.records
    traces = result.traces
    summary = result.summary
    summary["dataset_selection"] = {
        "dataset_path": args.dataset_path,
        "category": category_filter,
        "dataset_mode": dataset_mode,
        "one_per_category": dataset_mode == "one-per-category",
        "one_per_category_per_device": dataset_mode == "one-per-category-per-device",
        "all": dataset_mode == "all",
        "selected_request_count": len(items),
        "request_count": run_request_count,
        "dropped_request_count": dropped_request_count,
        "microbatch_count": len(microbatches),
        "categories": sort_specbench_categories({item.category for item in items}),
    }
    summary["run_config"] = runner.run_config_summary()

    output_dir = Path(args.results_dir)
    write_jsonl(output_dir / "request_records.jsonl", records)
    write_jsonl(output_dir / "event_trace.jsonl", traces)
    write_json(output_dir / "summary.json", summary)
    print(f"wrote {len(records)} request records to {output_dir}")
    print(f"method={summary.get('method', args.method)}")
    print(f"throughput_tokens_per_s={summary['throughput_tokens_per_s']:.4f}")
    print_task_metrics(summary)


if __name__ == "__main__":
    main()
