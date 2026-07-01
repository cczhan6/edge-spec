from __future__ import annotations

import copy

import pytest

from src.config import load_config, validate_config
from src.methods import DEFAULT_METHODS, get_method_spec


CANONICAL_METHODS = (
    "target_only",
    "server_only_linear",
    "server_only_tree",
    "specedge_linear",
    "specedge_tree",
    "dip_sd",
)


def test_async_speculative_is_independent_and_fixed_config_is_valid() -> None:
    config = load_config("configs/default.yaml")

    spec = get_method_spec("async_speculative", config)

    assert spec.runtime == "async_speculative"
    assert spec.name == "async_speculative"
    assert "async_speculative" not in DEFAULT_METHODS
    validate_config(config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("mode", "dynamic", "async_speculative.mode must be fixed"),
        ("num_channels", 0, "async_speculative.num_channels must be positive"),
        ("gamma_fixed", 0, "async_speculative.gamma_fixed must be positive"),
        (
            "lookahead_depth_fixed",
            0,
            "async_speculative.lookahead_depth_fixed must be positive",
        ),
        ("l_max_ver", 0, "async_speculative.l_max_ver must be positive"),
    ),
)
def test_fixed_config_rejects_invalid_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    config = load_config("configs/default.yaml")
    config["async_speculative"][field] = value

    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_fixed_config_requires_verification_limit_to_cover_gamma() -> None:
    config = load_config("configs/default.yaml")
    config["async_speculative"]["gamma_fixed"] = 4
    config["async_speculative"]["l_max_ver"] = 3

    with pytest.raises(
        ValueError,
        match="async_speculative.l_max_ver must be >= gamma_fixed",
    ):
        validate_config(config)


def test_canonical_method_specs_are_unchanged_by_registration() -> None:
    config = load_config("configs/default.yaml")
    without_proposed = copy.deepcopy(config)
    without_proposed.pop("async_speculative")

    assert {
        name: get_method_spec(name, config)
        for name in CANONICAL_METHODS
    } == {
        name: get_method_spec(name, without_proposed)
        for name in CANONICAL_METHODS
    }
