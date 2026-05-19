from __future__ import annotations

import argparse
import os
from pathlib import Path

from .backends import FakeBackend, HuggingFaceBackend
from .dataset import (
    fallback_items,
    iter_microbatches,
    load_specbench,
    select_one_per_category,
    select_one_per_category_per_device,
)
from .io import write_json, write_jsonl
from .runner import HeteroAsyncPipelineRunner, HeteroSyncRunner
from .simulation import load_device_profiles
from .types import SamplingConfig


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
    return tqdm(total=total, desc=desc, unit="req" if desc == "async" else "batch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Three-edge-device edge speculative decoding experiments."
    )
    parser.add_argument("--dataset-path", default="data/spec_bench/question.jsonl")
    parser.add_argument("--profile-config", default="configs/edge_hetero.yaml")
    parser.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", "results"))
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--category", default=None)
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
            "each category; 'all' runs every selected request without dropping "
            "the final partial microbatch."
        ),
    )
    parser.add_argument(
        "--one-per-category",
        action="store_true",
        help="Evaluate one SpecBench sample from each category. Ignores the default --limit=3.",
    )
    parser.add_argument(
        "--one-per-category-per-device",
        action="store_true",
        help=(
            "Evaluate one sample per device from each SpecBench category. "
            "Each category becomes one 3-request microbatch."
        ),
    )
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--draft-models", nargs=3, default=DEFAULT_DRAFT_MODELS)
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--client-device", default="cuda:0")
    parser.add_argument("--server-device", default="cuda:1")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-target-baseline", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("sync", "async"),
        default="sync",
        help="Run synchronous barrier baseline or asynchronous multi-pipeline scheme.",
    )
    parser.add_argument(
        "--pipeline-count",
        type=int,
        default=3,
        help="Number of independent verification pipelines used by --mode async.",
    )
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
        drafts = [
            FakeBackend(args.draft_models[0], seed=1, delay_s=0.001),
            FakeBackend(args.draft_models[1], seed=3, delay_s=0.001),
            FakeBackend(args.draft_models[2], seed=5, delay_s=0.001),
        ]
        target = FakeBackend(args.target_model, seed=9, delay_s=0.001)
        return drafts, target
    drafts = [
        HuggingFaceBackend(model_name, args.client_device, args.torch_dtype)
        for model_name in args.draft_models
    ]
    target = HuggingFaceBackend(args.target_model, args.server_device, args.torch_dtype)
    return drafts, target


def print_task_metrics(summary: dict) -> None:
    task_metrics = summary.get("task_metrics") or {}
    if not task_metrics:
        return
    print("task_metrics:")
    print(
        "task                 requests  eff_recv_tok/s  e2e_ttft_s  e2e_mean_latency_s"
    )
    for task, metrics in sorted(task_metrics.items()):
        ttft = metrics.get("e2e_first_token_latency_s")
        ttft_text = f"{ttft:.4f}" if ttft is not None else "n/a"
        print(
            f"{task:<20} "
            f"{metrics['request_count']:>8} "
            f"{metrics['effective_received_throughput_tokens_per_s']:>15.4f} "
            f"{ttft_text:>11} "
            f"{metrics['e2e_mean_latency_s']:>19.4f}"
        )


def main() -> None:
    args = parse_args()
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
            raise ValueError(
                "Use either --dataset-mode or legacy dataset flags, not both"
            )
        if args.one_per_category:
            dataset_mode = "one-per-category"
        elif args.one_per_category_per_device:
            dataset_mode = "one-per-category-per-device"

    sampling = SamplingConfig(args.temperature, args.top_p, args.top_k)
    sampling.validate()
    profiles = load_device_profiles(args.profile_config)

    try:
        full_dataset_mode = dataset_mode in {
            "one-per-category",
            "one-per-category-per-device",
            "all",
        }
        items = load_specbench(
            args.dataset_path,
            limit=None if full_dataset_mode else args.limit,
            category=args.category,
            shuffle=args.shuffle,
            seed=args.seed,
        )
    except FileNotFoundError:
        if not args.use_fake_models:
            raise
        items = fallback_items()
    if dataset_mode == "one-per-category":
        items = select_one_per_category(items)
    if dataset_mode == "one-per-category-per-device":
        items = select_one_per_category_per_device(
            items, device_count=len(args.draft_models)
        )

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
    runner_kwargs = {
        "draft_backends": draft_backends,
        "target_backend": target_backend,
        "profiles": profiles,
        "sampling": sampling,
        "gamma": args.gamma,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "run_target_baseline": not args.skip_target_baseline,
    }
    if args.mode == "async":
        runner = HeteroAsyncPipelineRunner(
            **runner_kwargs,
            pipeline_count=args.pipeline_count,
        )
    else:
        runner = HeteroSyncRunner(**runner_kwargs)
    progress_total = len(items) if args.mode == "async" else len(microbatches)
    progress = make_progress(
        enabled=not args.no_progress,
        total=progress_total,
        desc=args.mode,
    )
    try:
        records, traces, summary = runner.run_dataset(microbatches, progress=progress)
    finally:
        if progress is not None:
            progress.close()
    summary["dataset_selection"] = {
        "dataset_path": args.dataset_path,
        "dataset_mode": dataset_mode,
        "one_per_category": dataset_mode == "one-per-category",
        "one_per_category_per_device": dataset_mode == "one-per-category-per-device",
        "all": dataset_mode == "all",
        "selected_request_count": len(items),
        "request_count": run_request_count,
        "dropped_request_count": dropped_request_count,
        "microbatch_count": len(microbatches),
        "categories": sorted({item.category for item in items}),
    }
    summary["run_config"] = {
        "mode": args.mode,
        "pipeline_count": args.pipeline_count if args.mode == "async" else None,
        "gamma": args.gamma,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "skip_target_baseline": args.skip_target_baseline,
    }

    output_dir = Path(args.results_dir)
    write_jsonl(output_dir / "specbench_sync_hetero.jsonl", records)
    write_jsonl(output_dir / "round_trace.jsonl", traces)
    write_json(output_dir / "summary.json", summary)
    print(f"wrote {len(records)} request records to {output_dir}")
    print(f"mode={summary.get('mode', args.mode)}")
    print(f"throughput_tokens_per_s={summary['throughput_tokens_per_s']:.4f}")
    print_task_metrics(summary)


if __name__ == "__main__":
    main()
