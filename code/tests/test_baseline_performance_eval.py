from __future__ import annotations

from src.config import load_config


def test_dynamic_heterogeneous_configuration_contract() -> None:
    config = load_config("configs/default.yaml", "dynamic_heterogeneous")

    assert config["simulation"]["num_requests"] == 80
    assert config["simulation"]["request_arrival"] == "poisson"
    assert config["dynamic_edge_compute"] == {
        "enabled": True,
        "resample_every_completed_requests": 5,
    }
    assert config["target_latency"] == {
        "mode": "profile",
        "profile_path": "outputs/profiling/target_verification_latency_full_merged.csv",
        "metric": "p50_ms",
    }
    templates = config["device_pools"]["heterogeneous"]["templates"]
    assert {name: values["count"] for name, values in templates.items()} == {
        "low_end": 3,
        "mid_end": 3,
        "high_end": 2,
    }
    assert all(values["block_probability"] == 0.2 for values in templates.values())
    assert all(
        "dynamic_draft_token_rate_range_tok_s" in values
        for values in templates.values()
    )
