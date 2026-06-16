from __future__ import annotations

import unittest

from src.config import load_config, validate_config


class ConfigTest(unittest.TestCase):
    def test_requires_positive_analytical_target_rate(self) -> None:
        config = load_config("configs/default.yaml")
        config["edge"]["target_only_token_rate_tok_s"] = 0
        with self.assertRaisesRegex(ValueError, "target_only_token_rate"):
            validate_config(config)

    def test_requires_model_runner_model_for_each_drafter_profile(self) -> None:
        config = load_config("configs/default.yaml")
        del config["model_runner"]["drafter_models"]["small"]
        with self.assertRaisesRegex(ValueError, "small"):
            validate_config(config)

    def test_device_pool_count_must_match_num_devices(self) -> None:
        config = load_config("configs/default.yaml")
        config["device_pools"]["heterogeneous"]["templates"]["low_end"]["count"] = 4
        with self.assertRaisesRegex(ValueError, "defines 9 devices"):
            validate_config(config)

    def test_server_only_drafter_profile_must_be_known(self) -> None:
        config = load_config("configs/default.yaml")
        config["server_only"]["drafter_profile"] = "unknown"
        with self.assertRaisesRegex(ValueError, "server_only"):
            validate_config(config)

    def test_server_only_draft_rate_must_be_positive(self) -> None:
        config = load_config("configs/default.yaml")
        config["server_only"]["draft_token_rate_tok_s"] = 0
        with self.assertRaisesRegex(ValueError, "server_only.draft_token_rate"):
            validate_config(config)

    def test_specedge_proactive_budget_must_not_exceed_main_budget(self) -> None:
        config = load_config("configs/default.yaml")
        config["specedge"]["proactive_max_budget"] = config["specedge"]["max_budget"] + 1
        with self.assertRaisesRegex(ValueError, "proactive_max_budget"):
            validate_config(config)

    def test_specedge_scheduler_types_must_be_known(self) -> None:
        config = load_config("configs/default.yaml")
        config["specedge"]["server_batch_type"] = "round_robin"
        with self.assertRaisesRegex(ValueError, "server_batch_type"):
            validate_config(config)
        config = load_config("configs/default.yaml")
        config["specedge"]["proactive_type"] = "eager"
        with self.assertRaisesRegex(ValueError, "proactive_type"):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
