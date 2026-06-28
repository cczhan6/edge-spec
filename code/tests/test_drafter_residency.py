from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.check_drafter_residency import (
    build_residency_manifest,
    tokenizer_compatibility,
    write_manifest,
)


class _Tokenizer:
    def __init__(self, vocab: dict[str, int], *, eos: int = 2) -> None:
        self._vocab = vocab
        self.eos_token_id = eos
        self.bos_token_id = 1
        self.pad_token_id = None
        self.unk_token_id = 0

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)


class DrafterResidencyManifestTest(unittest.TestCase):
    def test_script_is_directly_executable_from_repository_root(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/check_drafter_residency.py", "--help"],
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("cuda:0", completed.stdout)

    def test_simultaneous_success_keeps_all_models_resident_policy(self) -> None:
        manifest = build_residency_manifest(
            individual=[{"profile": "small"}, {"profile": "medium"}, {"profile": "large"}],
            simultaneous={"success": True, "oom": False},
            tokenizer_result={"compatible": True},
            device={"device": "cuda:0", "name": "GPU"},
        )

        self.assertEqual(manifest["residency_policy"], "all_configured_models_simultaneous")
        self.assertFalse(manifest["model_loading_in_decode_latency"])
        self.assertFalse(manifest["simultaneous"]["oom"])

    def test_simultaneous_oom_records_sequential_lazy_policy_without_rebinding(self) -> None:
        manifest = build_residency_manifest(
            individual=[{"profile": "small"}, {"profile": "medium"}, {"profile": "large"}],
            simultaneous={"success": False, "oom": True, "error": "CUDA out of memory"},
            tokenizer_result={"compatible": True},
            device={"device": "cuda:0", "name": "GPU"},
        )

        self.assertEqual(manifest["residency_policy"], "sequential_lazy_model_loading")
        self.assertEqual(manifest["virtual_device_binding_changed"], False)
        self.assertEqual(manifest["drafter_profiles_merged"], False)
        self.assertFalse(manifest["model_loading_in_decode_latency"])

    def test_tokenizer_compatibility_requires_exact_vocab_and_special_ids(self) -> None:
        reference = _Tokenizer({"a": 0, "b": 1})
        same = _Tokenizer({"a": 0, "b": 1})
        remapped = _Tokenizer({"a": 1, "b": 0})

        compatible = tokenizer_compatibility(
            {"target": reference, "small": same, "medium": same, "large": same}
        )
        incompatible = tokenizer_compatibility(
            {"target": reference, "small": same, "medium": remapped, "large": same}
        )

        self.assertTrue(compatible["compatible"])
        self.assertFalse(incompatible["compatible"])
        self.assertFalse(incompatible["comparisons"]["target:medium"]["exact_token_id_mapping"])

    def test_manifest_writer_rejects_non_finite_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "manifest.json"
            write_manifest(output, {"value": 1.0})
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"value": 1.0})

            with self.assertRaises(ValueError):
                write_manifest(output, {"value": float("nan")})


if __name__ == "__main__":
    unittest.main()
