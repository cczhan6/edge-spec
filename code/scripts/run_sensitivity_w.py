from __future__ import annotations

import argparse
import sys

from src.config import load_config
from src.metrics import MAIN_FIELDS, summarize, write_csv
from src.model_runner import build_model_runner
from src.simulator import Simulator
from src.workload import load_workload
from scripts.progress import ProgressReporter


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Full sensitivity analysis for speculative window W.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", default="data/spec_bench/question.jsonl")
    parser.add_argument("--scenario", default="combined_strong_heterogeneous")
    parser.add_argument("--values", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--out", default="outputs/raw/sensitivity_w.csv")
    parser.add_argument("--use-fake-model-runner", action="store_true")
    parser.add_argument("--use-fake-oracle", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--samples-per-category", type=int)
    args = parser.parse_args()
    print(
        "warning: current full uses DSI-style unbounded speculation; "
        "W_default is retained only for legacy fixed-window experiments.",
        file=sys.stderr,
    )
    model_runner = build_model_runner(
        load_config(args.config),
        use_fake_model_runner=args.use_fake_model_runner or args.use_fake_oracle,
    )
    rows = []
    for value in args.values:
        config = load_config(args.config, args.scenario)
        config["speculation"]["W_default"] = value
        workload = load_workload(
            args.dataset,
            int(config["simulation"]["num_requests"]),
            int(config["simulation"]["seed"]),
            model_runner.prompt_token_count,
            samples_per_category=args.samples_per_category,
        )
        config["simulation"]["num_requests"] = len(workload)
        progress_item = f"{args.scenario}/full W={value}"
        request_progress = ProgressReporter(len(workload), progress_item, unit="req")
        request_progress.start()

        def update_request_progress(completed: int, total: int) -> None:
            request_progress.update(completed)

        main, _ = summarize(
            Simulator(
                config,
                model_runner,
                workload,
                args.scenario,
                "full",
                progress_callback=update_request_progress,
            ).run(),
            config["simulation"]["num_devices"],
        )
        request_progress.finish_line()
        main["W"] = value
        rows.append(main)
    write_csv(args.out, rows, ["W", *MAIN_FIELDS])


if __name__ == "__main__":
    main()
