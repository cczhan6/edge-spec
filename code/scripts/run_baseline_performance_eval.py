from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from src.entities import Request
from src.events import EventType
from src.model_runner import ModelRunner
from src.simulator import Simulator
from src.workload import WorkloadItem


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
