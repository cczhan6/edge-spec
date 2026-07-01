from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from scripts.audit_experiment_config import (
    audit_experiment_config,
    write_resolved_config,
)
from scripts.baseline_trace import _write_yaml, write_trace_bundle
from scripts.summarize_baseline_performance_eval import (
    METHODS,
    SCENARIO,
    SEEDS,
    initialize_runs_csv,
)
from src.config import build_devices, load_config, validate_config
from src.edge_compute import EdgeComputeModel, deterministic_draft_rate
from src.entities import Request
from src.events import EventType
from src.metrics import summarize
from src.model_runner import ModelRunner, build_model_runner
from src.simulator import Simulator
from src.workload import WorkloadItem, load_workload


@dataclass(frozen=True)
class SharedRequest:
    request_id: int
    device_id: int
    prompt_id: str
    prompt: str
    prompt_token_count: int
    category: str
    category_group: str
    output_len: int
    arrival_time_ms: float
    decode_ready_time_ms: float

    def workload_item(self) -> WorkloadItem:
        return WorkloadItem(
            prompt_id=self.prompt_id,
            prompt=self.prompt,
            prompt_token_count=self.prompt_token_count,
            category=self.category,
            category_group=self.category_group,
        )


class SharedTraceSimulator(Simulator):
    def __init__(
        self,
        config: dict[str, Any],
        model_runner: ModelRunner,
        shared_requests: Sequence[SharedRequest],
        scenario: str,
        method: str,
        **kwargs: Any,
    ) -> None:
        self._shared_requests = list(shared_requests)
        super().__init__(
            config,
            model_runner,
            [row.workload_item() for row in self._shared_requests],
            scenario,
            method,
            **kwargs,
        )

    def _schedule_request_arrivals(self) -> None:
        if self.requests:
            raise RuntimeError("shared requests were already scheduled")
        for row in self._shared_requests:
            prompt_ids = self.model_runner.encode_prompt(row.prompt)
            if len(prompt_ids) != row.prompt_token_count:
                raise ValueError(
                    f"shared prompt token count differs for request {row.request_id}"
                )
            request = Request(
                request_id=row.request_id,
                device_id=row.device_id,
                output_len=row.output_len,
                arrival_time_ms=row.arrival_time_ms,
                decode_ready_time_ms=row.decode_ready_time_ms,
                prompt_id=row.prompt_id,
                category=row.category,
                category_group=row.category_group,
                prompt=row.prompt,
                prompt_token_count=len(prompt_ids),
                prompt_ids=prompt_ids,
            )
            self.requests.append(request)
            self.device_runtimes[row.device_id].assigned_requests += 1
            self._schedule(
                row.arrival_time_ms,
                EventType.REQUEST_ARRIVE,
                row.request_id,
            )


def materialize_shared_trace(
    config: dict[str, Any],
    workload: Sequence[WorkloadItem],
    path: str | Path,
) -> str:
    simulation = config["simulation"]
    if len(workload) != int(simulation["num_requests"]):
        raise ValueError("workload size does not match simulation.num_requests")
    rng = random.Random(int(simulation["seed"]))
    current_ms = 0.0
    rows = []
    for request_id, item in enumerate(workload):
        if request_id and simulation["request_arrival"] == "poisson":
            current_ms += (
                rng.expovariate(float(simulation["poisson_rate_per_s"])) * 1000.0
            )
        rows.append(
            SharedRequest(
                request_id=request_id,
                device_id=request_id % int(simulation["num_devices"]),
                prompt_id=item.prompt_id,
                prompt=item.prompt,
                prompt_token_count=item.prompt_token_count,
                category=item.category,
                category_group=item.category_group,
                output_len=int(rng.choice(simulation["output_len_choices"])),
                arrival_time_ms=current_ms,
                decode_ready_time_ms=current_ms,
            )
        )
    payload = b"".join(
        (
            json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        for row in rows
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    try:
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() != payload:
                raise ValueError(f"existing shared trace differs: {destination}")
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def load_shared_trace(
    path: str | Path,
    config: dict[str, Any],
) -> list[SharedRequest]:
    with Path(path).open(encoding="utf-8") as handle:
        rows = [SharedRequest(**json.loads(line)) for line in handle if line.strip()]
    expected = int(config["simulation"]["num_requests"])
    if len(rows) != expected or [row.request_id for row in rows] != list(
        range(expected)
    ):
        raise ValueError("shared trace request IDs are incomplete or unordered")
    devices = int(config["simulation"]["num_devices"])
    if any(row.device_id != row.request_id % devices for row in rows):
        raise ValueError("shared trace device mapping is invalid")
    return rows


def build_resource_fingerprint(config: dict[str, Any]) -> dict[str, Any]:
    pool_name = "heterogeneous"
    devices = build_devices(config, pool_name)
    compute = EdgeComputeModel(config, devices, pool_name)
    templates = config["device_pools"][pool_name]["templates"]
    epoch_rates = []
    for device in devices:
        bounds = tuple(
            float(value)
            for value in templates[device.device_type][
                "dynamic_draft_token_rate_range_tok_s"
            ]
        )
        epoch_rates.append(
            {
                "device_id": device.device_id,
                "device_type": device.device_type,
                "rates": [
                    compute.current_rate(device.device_id),
                    deterministic_draft_rate(
                        int(config["simulation"]["seed"]),
                        device.device_id,
                        device.device_type,
                        1,
                        bounds,
                    ),
                    deterministic_draft_rate(
                        int(config["simulation"]["seed"]),
                        device.device_id,
                        device.device_type,
                        2,
                        bounds,
                    ),
                ],
            }
        )
    mapping_payload = json.dumps(
        epoch_rates,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "seed": int(config["simulation"]["seed"]),
        "devices": [asdict(device) for device in devices],
        "epoch0_rates": [
            compute.current_rate(device.device_id) for device in devices
        ],
        "epoch_rate_mapping_sha256": hashlib.sha256(mapping_payload).hexdigest(),
        "dynamic_edge_compute": dict(config["dynamic_edge_compute"]),
        "dynamic_ranges": {
            name: list(values["dynamic_draft_token_rate_range_tok_s"])
            for name, values in templates.items()
            if int(values["count"]) > 0
        },
        "block_probabilities": {
            name: float(values["block_probability"])
            for name, values in templates.items()
            if int(values["count"]) > 0
        },
        "target_latency": dict(config["target_latency"]),
    }


def run_cell(
    config_path: str | Path,
    trace_path: str | Path,
    method: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"unsupported canonical method: {method}")
    config = load_config(config_path)
    shared = load_shared_trace(trace_path, config)
    model_runner = build_model_runner(config, use_fake_model_runner=False)
    simulator = SharedTraceSimulator(
        config,
        model_runner,
        shared,
        SCENARIO,
        method,
    )
    result = simulator.run()
    main, system = summarize(result, int(config["simulation"]["num_devices"]))
    destination = Path(output_dir)
    write_trace_bundle(destination, config, result, main, system)
    status = {
        "scenario": SCENARIO,
        "seed": int(config["simulation"]["seed"]),
        "method": method,
        "success": True,
        "return_code": 0,
        "failure_reason": "",
        "shared_trace_path": str(Path(trace_path)),
        "shared_trace_sha256": _sha256_file(trace_path),
        "resource_fingerprint": build_resource_fingerprint(config),
    }
    _write_json_atomic(destination / "run_status.json", status)
    return status


def run_matrix_cells(
    *,
    root: str | Path,
    seeds: Sequence[int],
    methods: Sequence[str],
    command_for: Callable[[int, str], Sequence[str]],
    run_process: Callable[..., Any] = subprocess.run,
) -> list[dict[str, Any]]:
    output_root = Path(root)
    statuses = []
    for seed in seeds:
        for method in methods:
            directory = output_root / SCENARIO / str(seed) / method
            directory.mkdir(parents=True, exist_ok=True)
            log_path = directory / "stdout.log"
            with log_path.open("w", encoding="utf-8") as log:
                completed = run_process(
                    list(command_for(int(seed), str(method))),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            status_path = directory / "run_status.json"
            if int(completed.returncode) == 0 and status_path.is_file():
                status = _read_json(status_path)
            elif int(completed.returncode) == 0:
                status = {
                    "scenario": SCENARIO,
                    "seed": int(seed),
                    "method": str(method),
                    "success": False,
                    "return_code": 0,
                    "failure_reason": "worker returned zero but run_status.json is missing",
                }
                _write_json_atomic(status_path, status)
            else:
                status = {
                    "scenario": SCENARIO,
                    "seed": int(seed),
                    "method": str(method),
                    "success": False,
                    "return_code": int(completed.returncode),
                    "failure_reason": f"worker return code {completed.returncode}",
                }
                _write_json_atomic(status_path, status)
            statuses.append(status)
    return statuses


def prepare_matrix_inputs(
    *,
    root: str | Path,
    config_path: str | Path,
    dataset_path: str | Path,
) -> dict[int, tuple[Path, Path]]:
    output_root = Path(root)
    output_root.mkdir(parents=True, exist_ok=True)
    first_config = load_config(config_path, SCENARIO)
    token_counter = build_model_runner(
        first_config,
        use_fake_model_runner=False,
    ).prompt_token_count
    prepared = {}
    for seed in SEEDS:
        config = load_config(config_path, SCENARIO)
        config["simulation"]["seed"] = seed
        validate_config(config)
        config_output = output_root / "_configs" / f"{SCENARIO}_seed_{seed}.yaml"
        _write_yaml(config_output, config)
        audit = audit_experiment_config(
            config,
            scenario=SCENARIO,
            methods=METHODS,
            use_fake_model_runner=False,
            repo_root=Path(__file__).resolve().parents[1],
        )
        write_resolved_config(config_output.with_suffix(".audit.json"), audit)
        workload = load_workload(
            dataset_path,
            int(config["simulation"]["num_requests"]),
            seed,
            token_counter,
        )
        trace_output = output_root / "_workloads" / f"{SCENARIO}_seed_{seed}.jsonl"
        materialize_shared_trace(config, workload, trace_output)
        prepared[seed] = (config_output, trace_output)
    return prepared


def run_formal_matrix(
    *,
    root: str | Path,
    config_path: str | Path,
    dataset_path: str | Path,
) -> int:
    initialize_runs_csv(root)
    prepared = prepare_matrix_inputs(
        root=root,
        config_path=config_path,
        dataset_path=dataset_path,
    )

    def command_for(seed: int, method: str) -> list[str]:
        config_output, trace_output = prepared[seed]
        directory = Path(root) / SCENARIO / str(seed) / method
        return [
            sys.executable,
            "-m",
            "scripts.run_baseline_performance_eval",
            "cell",
            "--config",
            str(config_output),
            "--trace",
            str(trace_output),
            "--method",
            method,
            "--output-dir",
            str(directory),
        ]

    statuses = run_matrix_cells(
        root=root,
        seeds=SEEDS,
        methods=METHODS,
        command_for=command_for,
    )
    from scripts.summarize_baseline_performance_eval import summarize_results

    failures = summarize_results(root)
    return int(any(not bool(status.get("success")) for status in statuses) or failures)


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_atomic(path: str | Path, value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    temporary.replace(destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run canonical baseline performance evaluation cells."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    matrix = subparsers.add_parser("matrix", help="Run the formal 30-cell matrix.")
    matrix.add_argument("--root", default="outputs/baseline_performance_eval")
    matrix.add_argument("--config", default="configs/default.yaml")
    matrix.add_argument("--dataset", default="data/spec_bench/question.jsonl")
    matrix.add_argument("--execute-formal-matrix", action="store_true")
    cell = subparsers.add_parser("cell", help="Run one pre-materialized cell.")
    cell.add_argument("--config", required=True)
    cell.add_argument("--trace", required=True)
    cell.add_argument("--method", required=True, choices=METHODS)
    cell.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "matrix":
        if not args.execute_formal_matrix:
            raise SystemExit("formal matrix requires --execute-formal-matrix")
        raise SystemExit(
            run_formal_matrix(
                root=args.root,
                config_path=args.config,
                dataset_path=args.dataset,
            )
        )
    try:
        run_cell(args.config, args.trace, args.method, args.output_dir)
    except Exception as exc:
        _write_json_atomic(
            Path(args.output_dir) / "run_status.json",
            {
                "scenario": SCENARIO,
                "method": args.method,
                "success": False,
                "return_code": 1,
                "failure_reason": f"{type(exc).__name__}: {exc}",
            },
        )
        raise


if __name__ == "__main__":
    main()
