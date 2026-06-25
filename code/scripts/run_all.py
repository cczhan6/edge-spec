from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import apply_tree_draft_strategy, load_config
from src.methods import DEFAULT_METHODS, SUPPORTED_METHODS
from src.metrics import (
    CATEGORY_MAIN_FIELDS,
    DEVICE_FIELDS,
    EVENT_FIELDS,
    MAIN_FIELDS,
    REQUEST_FIELDS,
    SEGMENT_FIELDS,
    SYSTEM_FIELDS,
    category_rows,
    device_rows,
    enrich_comparisons,
    event_rows,
    request_rows,
    segment_rows,
    summarize,
    write_csv,
)
from src.simulator import Simulator
from src.model_runner import build_model_runner
from src.workload import load_workload
from scripts.progress import ProgressReporter


DEFAULT_SCENARIOS = (
    "homogeneous",
    "combined_strong_heterogeneous",
)


def run_experiments(
    config_path: str,
    dataset_path: str,
    scenarios: list[str],
    methods: list[str],
    out_dir: str,
    summary_out: str | None = None,
    single_out: str | None = None,
    use_fake_model_runner: bool = False,
    samples_per_category: int | None = None,
    write_details: bool = True,
    tree_draft_strategy: str | None = None,
) -> list[dict]:
    if single_out and (len(scenarios) != 1 or len(methods) != 1):
        raise ValueError("--out requires exactly one scenario and one method")
    model_runner = build_model_runner(
        apply_tree_draft_strategy(load_config(config_path), tree_draft_strategy),
        use_fake_model_runner=use_fake_model_runner,
    )
    raw_dir = Path(out_dir)
    all_main_rows: list[dict] = []
    all_category_rows: list[dict] = []
    progress_stream = sys.stderr
    method_label_width = max(len(method) for method in methods) + 2
    for scenario in scenarios:
        print(f"\nscenario: {scenario}", file=progress_stream)
        print(f"method order: {' -> '.join(methods)}", file=progress_stream)
        progress_stream.flush()
        config = apply_tree_draft_strategy(
            load_config(config_path, scenario),
            tree_draft_strategy,
        )
        workload = load_workload(
            dataset_path,
            int(config["simulation"]["num_requests"]),
            int(config["simulation"]["seed"]),
            model_runner.prompt_token_count,
            samples_per_category=samples_per_category,
        )
        config["simulation"]["num_requests"] = len(workload)
        main_rows = []
        category_main_rows = []
        system_rows = []
        for method in methods:
            request_progress = ProgressReporter(
                len(workload),
                f"  {method}",
                stream=progress_stream,
                unit="req",
                label_width=method_label_width,
            )
            request_progress.start()

            def update_request_progress(completed: int, total: int) -> None:
                request_progress.update(completed)

            simulator = Simulator(
                config,
                model_runner,
                workload,
                scenario,
                method,
                progress_callback=update_request_progress,
            )
            result = simulator.run()
            request_progress.finish_line()
            main, system = summarize(result, int(config["simulation"]["num_devices"]))
            main_rows.append(main)
            category_main_rows.extend(
                category_rows(result, int(config["simulation"]["num_devices"]))
            )
            system_rows.append(system)
            _print_method_metrics(main, progress_stream)
            if write_details:
                detail_method = result.method
                segment_detail_rows = segment_rows(result)
                write_csv(
                    raw_dir / f"request_details_{scenario}_{detail_method}.csv",
                    request_rows(result),
                    REQUEST_FIELDS,
                )
                write_csv(
                    raw_dir / f"segment_details_{scenario}_{detail_method}.csv",
                    segment_detail_rows,
                    SEGMENT_FIELDS,
                )
                write_csv(
                    raw_dir / f"event_details_{scenario}_{detail_method}.csv",
                    event_rows(result),
                    EVENT_FIELDS,
                )
                write_csv(
                    raw_dir / f"device_metrics_{scenario}_{detail_method}.csv",
                    device_rows(result),
                    DEVICE_FIELDS,
                )
                write_csv(
                    raw_dir / f"round_trace_{scenario}_{detail_method}.csv",
                    segment_detail_rows,
                    SEGMENT_FIELDS,
                )
        enrich_comparisons(main_rows)
        _print_scenario_comparison(main_rows, progress_stream)
        for category in sorted({row["category"] for row in category_main_rows}):
            enrich_comparisons(
                [row for row in category_main_rows if row["category"] == category]
            )
        all_main_rows.extend(main_rows)
        all_category_rows.extend(category_main_rows)
        write_csv(raw_dir / f"main_results_{scenario}.csv", main_rows, MAIN_FIELDS)
        write_csv(
            raw_dir / f"category_results_{scenario}.csv",
            category_main_rows,
            CATEGORY_MAIN_FIELDS,
        )
        write_csv(raw_dir / f"system_metrics_{scenario}.csv", system_rows, SYSTEM_FIELDS)

    if single_out:
        write_csv(single_out, all_main_rows, MAIN_FIELDS)
    summary_path = summary_out or str(raw_dir.parent / "summary" / "all_results.csv")
    write_csv(summary_path, all_main_rows, MAIN_FIELDS)
    category_summary_path = str(Path(summary_path).with_name("category_results.csv"))
    write_csv(category_summary_path, all_category_rows, CATEGORY_MAIN_FIELDS)
    return all_main_rows


def _print_method_metrics(row: dict, stream) -> None:
    print(
        "  metrics: "
        f"{row['method']} "
        f"avg={_format_ms(row['avg_latency_ms'])} "
        f"p95={_format_ms(row['p95_latency_ms'])} "
        f"tpot={_format_ms(row['avg_tpot_ms'])} "
        f"tbt={_format_ms(row['avg_tbt_ms'])} "
        f"makespan={_format_ms(row['makespan_ms'])} "
        f"goodput={_format_float(row['goodput_tok_s'])} tok/s "
        f"acceptance={_format_percent(row['avg_acceptance_rate'])} "
        f"gamma={_format_float(row['avg_selected_gamma'])}",
        file=stream,
    )
    stream.flush()


def _print_scenario_comparison(rows: list[dict], stream) -> None:
    if not rows:
        return
    print("comparison:", file=stream)
    for row in rows:
        print(
            "  "
            f"{row['method']}: "
            f"avg={_format_ms(row['avg_latency_ms'])} "
            f"goodput={_format_float(row['goodput_tok_s'])} tok/s "
            f"vs_target={_format_x(row.get('latency_speedup_vs_autoregressive'))} "
            f"vs_sync={_format_x(row.get('latency_ratio_vs_sync_batch_sd'))} "
            f"vs_specedge={_format_x(row.get('latency_ratio_vs_specedge'))}",
            file=stream,
        )
    stream.flush()


def _format_ms(value: object) -> str:
    try:
        milliseconds = float(value)
    except (TypeError, ValueError):
        return "n/a"
    seconds = milliseconds / 1000.0
    if seconds >= 100:
        return f"{seconds:.1f}s"
    if seconds >= 10:
        return f"{seconds:.2f}s"
    return f"{seconds:.3f}s"


def _format_float(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.2f}"


def _format_percent(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{100.0 * number:.1f}%"


def _format_x(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.2f}x"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run edge speculative decoding simulations.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", default="data/spec_bench/question.jsonl")
    scenario = parser.add_mutually_exclusive_group()
    scenario.add_argument("--scenario")
    scenario.add_argument("--scenarios", nargs="+")
    method = parser.add_mutually_exclusive_group()
    method.add_argument("--method", choices=SUPPORTED_METHODS)
    method.add_argument("--methods", nargs="+", choices=SUPPORTED_METHODS)
    parser.add_argument("--out")
    parser.add_argument("--out_dir", default="outputs/raw")
    parser.add_argument("--summary_out")
    parser.add_argument("--use-fake-model-runner", action="store_true")
    parser.add_argument("--use-fake-oracle", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--samples-per-category", type=int)
    parser.add_argument(
        "--tree-draft-strategy",
        choices=("linear", "specexec", "specexec_approx"),
        help="Override SpecEdge/server_only tree drafting strategy for this run.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Write only scenario/category/system summaries and skip per-method detail CSVs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scenarios = [args.scenario] if args.scenario else args.scenarios or list(DEFAULT_SCENARIOS)
    methods = [args.method] if args.method else args.methods or list(DEFAULT_METHODS)
    rows = run_experiments(
        config_path=args.config,
        dataset_path=args.dataset,
        scenarios=scenarios,
        methods=methods,
        out_dir=args.out_dir,
        summary_out=args.summary_out,
        single_out=args.out,
        use_fake_model_runner=args.use_fake_model_runner or args.use_fake_oracle,
        samples_per_category=args.samples_per_category,
        write_details=not args.summary_only,
        tree_draft_strategy=args.tree_draft_strategy,
    )
    print(f"wrote {len(rows)} method rows")


if __name__ == "__main__":
    main()
