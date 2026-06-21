from __future__ import annotations

from src.config import load_config
from src.model_runner import FakeModelRunner
from src.workload import WorkloadItem


def small_config(
    num_requests: int = 3,
    output_len: int = 12,
) -> tuple[dict, FakeModelRunner, list[WorkloadItem]]:
    config = load_config("configs/default.yaml")
    num_devices = max(1, min(3, num_requests))
    config["simulation"]["num_requests"] = num_requests
    config["simulation"]["num_devices"] = num_devices
    config["simulation"]["output_len_choices"] = [output_len]
    config["edge"]["num_lanes"] = 2
    templates = config["device_pools"]["heterogeneous"]["templates"]
    templates["low_end"]["count"] = num_devices
    templates["mid_end"]["count"] = 0
    templates["high_end"]["count"] = 0
    config["device_pools"]["medium_only"]["templates"]["medium"]["count"] = num_devices
    for pool in config["device_pools"].values():
        for template in pool["templates"].values():
            template["draft_token_rate_tok_s"] = 500
            template["uplink_mbps"] = 100
            template["downlink_mbps"] = 300
            template["rtt_ms"] = 100
            template["jitter_ms"] = 0
    model_runner = FakeModelRunner()
    workload = [
        WorkloadItem(str(index), f"prompt {index}", 2)
        for index in range(num_requests)
    ]
    return config, model_runner, workload


def rejecting_model_runner() -> FakeModelRunner:
    return FakeModelRunner(
        target_token_fn=lambda prefix: 1,
        draft_token_fn=lambda profile, prefix: 2,
    )


def accepting_model_runner() -> FakeModelRunner:
    def next_token(prefix) -> int:
        return (int(prefix[-1]) + 1) % 97

    return FakeModelRunner(
        target_token_fn=next_token,
        draft_token_fn=lambda profile, prefix: next_token(prefix),
    )
