from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config


PROFILE_ENV = {
    "small": "DRAFTER_SMALL_MODEL_PATH",
    "medium": "DRAFTER_MEDIUM_MODEL_PATH",
    "large": "DRAFTER_LARGE_MODEL_PATH",
}


def tokenizer_compatibility(tokenizers: dict[str, Any]) -> dict[str, Any]:
    if "target" not in tokenizers:
        raise ValueError("target tokenizer is required for vocabulary compatibility")
    target = tokenizers["target"]
    target_vocab = target.get_vocab()
    special_names = (
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
    )
    comparisons: dict[str, dict[str, Any]] = {}
    compatible = True
    for name, tokenizer in tokenizers.items():
        if name == "target":
            continue
        exact_mapping = tokenizer.get_vocab() == target_vocab
        special_ids_match = all(
            getattr(tokenizer, field, None) == getattr(target, field, None)
            for field in special_names
        )
        pair_compatible = exact_mapping and special_ids_match
        comparisons[f"target:{name}"] = {
            "exact_token_id_mapping": exact_mapping,
            "special_token_ids_match": special_ids_match,
            "compatible": pair_compatible,
            "target_vocab_entries": len(target_vocab),
            "drafter_vocab_entries": len(tokenizer.get_vocab()),
        }
        compatible = compatible and pair_compatible
    return {
        "policy": "exact_token_id_mapping",
        "compatible": compatible,
        "comparisons": comparisons,
    }


def build_residency_manifest(
    *,
    individual: list[dict[str, Any]],
    simultaneous: dict[str, Any],
    tokenizer_result: dict[str, Any],
    device: dict[str, Any],
) -> dict[str, Any]:
    simultaneous_success = bool(simultaneous.get("success"))
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "individual": individual,
        "simultaneous": simultaneous,
        "tokenizer_vocabulary_compatibility": tokenizer_result,
        "residency_policy": (
            "all_configured_models_simultaneous"
            if simultaneous_success
            else "sequential_lazy_model_loading"
        ),
        "virtual_device_binding_changed": False,
        "drafter_profiles_merged": False,
        "model_loading_in_decode_latency": False,
    }


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        manifest,
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


def check_drafter_residency(
    config_path: str | Path,
    *,
    device: str = "cuda:0",
) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except (ImportError, OSError) as exc:
        raise RuntimeError("drafter residency check requires torch and transformers") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; cannot check drafter residency on cuda:0")
    if device != "cuda:0":
        raise ValueError("formal drafter residency must be checked on cuda:0")
    torch.cuda.set_device(device)

    config = load_config(config_path)
    runner = config["model_runner"]
    local_files_only = _optional_bool(
        os.environ.get("LOCAL_FILES_ONLY", runner.get("local_files_only"))
    )
    load_kwargs: dict[str, Any] = {"torch_dtype": "auto"}
    tokenizer_kwargs: dict[str, Any] = {}
    if local_files_only is not None:
        load_kwargs["local_files_only"] = local_files_only
        tokenizer_kwargs["local_files_only"] = local_files_only
    if runner.get("cache_dir"):
        load_kwargs["cache_dir"] = runner["cache_dir"]
        tokenizer_kwargs["cache_dir"] = runner["cache_dir"]

    references = {
        profile: os.environ.get(PROFILE_ENV[profile])
        or str(runner["drafter_models"][profile]["model"])
        for profile in PROFILE_ENV
    }
    target_reference = os.environ.get("TARGET_MODEL_PATH") or str(runner["target_model"])
    target_tokenizer = AutoTokenizer.from_pretrained(target_reference, **tokenizer_kwargs)
    tokenizers: dict[str, Any] = {"target": target_tokenizer}
    individual: list[dict[str, Any]] = []

    for profile, reference in references.items():
        _release_cuda(torch)
        torch.cuda.reset_peak_memory_stats(device)
        before_free, total = torch.cuda.mem_get_info(device)
        tokenizer = AutoTokenizer.from_pretrained(reference, **tokenizer_kwargs)
        tokenizers[profile] = tokenizer
        model = None
        try:
            model = AutoModelForCausalLM.from_pretrained(reference, **load_kwargs).to(device).eval()
            torch.cuda.synchronize(device)
            after_free, _ = torch.cuda.mem_get_info(device)
            individual.append(
                {
                    "profile": profile,
                    "model": reference,
                    "device": device,
                    "success": True,
                    "oom": False,
                    "dtype": _model_dtype(model),
                    "peak_allocated_mib": _mib(torch.cuda.max_memory_allocated(device)),
                    "peak_reserved_mib": _mib(torch.cuda.max_memory_reserved(device)),
                    "free_before_load_mib": _mib(before_free),
                    "free_after_load_mib": _mib(after_free),
                    "total_memory_mib": _mib(total),
                }
            )
        except Exception as exc:
            if not _is_cuda_oom(torch, exc):
                raise
            individual.append(
                {
                    "profile": profile,
                    "model": reference,
                    "device": device,
                    "success": False,
                    "oom": True,
                    "dtype": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "peak_allocated_mib": _mib(torch.cuda.max_memory_allocated(device)),
                    "peak_reserved_mib": _mib(torch.cuda.max_memory_reserved(device)),
                    "free_before_load_mib": _mib(before_free),
                    "free_after_load_mib": _mib(torch.cuda.mem_get_info(device)[0]),
                    "total_memory_mib": _mib(total),
                }
            )
        finally:
            del model
            _release_cuda(torch)

    _release_cuda(torch)
    torch.cuda.reset_peak_memory_stats(device)
    simultaneous_before_free, total = torch.cuda.mem_get_info(device)
    loaded_models: dict[str, Any] = {}
    simultaneous: dict[str, Any]
    try:
        for profile, reference in references.items():
            loaded_models[profile] = (
                AutoModelForCausalLM.from_pretrained(reference, **load_kwargs).to(device).eval()
            )
        torch.cuda.synchronize(device)
        simultaneous_after_free, _ = torch.cuda.mem_get_info(device)
        simultaneous = {
            "success": True,
            "oom": False,
            "models": list(references.values()),
            "dtypes": {
                profile: _model_dtype(model) for profile, model in loaded_models.items()
            },
            "peak_allocated_mib": _mib(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_mib": _mib(torch.cuda.max_memory_reserved(device)),
            "free_before_load_mib": _mib(simultaneous_before_free),
            "free_after_load_mib": _mib(simultaneous_after_free),
            "total_memory_mib": _mib(total),
        }
    except Exception as exc:
        if not _is_cuda_oom(torch, exc):
            raise
        simultaneous = {
            "success": False,
            "oom": True,
            "models": list(references.values()),
            "error": f"{type(exc).__name__}: {exc}",
            "peak_allocated_mib": _mib(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_mib": _mib(torch.cuda.max_memory_reserved(device)),
            "free_before_load_mib": _mib(simultaneous_before_free),
            "free_after_attempt_mib": _mib(torch.cuda.mem_get_info(device)[0]),
            "total_memory_mib": _mib(total),
        }
    finally:
        loaded_models.clear()
        _release_cuda(torch)

    properties = torch.cuda.get_device_properties(device)
    return build_residency_manifest(
        individual=individual,
        simultaneous=simultaneous,
        tokenizer_result=tokenizer_compatibility(tokenizers),
        device={
            "device": device,
            "index": 0,
            "name": properties.name,
            "total_memory_mib": _mib(properties.total_memory),
            "free_after_check_mib": _mib(torch.cuda.mem_get_info(device)[0]),
        },
    )


def _release_cuda(torch: Any) -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _model_dtype(model: Any) -> str:
    try:
        return str(next(model.parameters()).dtype)
    except StopIteration:
        return str(getattr(model, "dtype", "unknown"))


def _is_cuda_oom(torch: Any, exc: Exception) -> bool:
    oom_type = getattr(torch, "OutOfMemoryError", None)
    return (oom_type is not None and isinstance(exc, oom_type)) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )


def _mib(value: int | float) -> float:
    return float(value) / (1024.0 * 1024.0)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError("LOCAL_FILES_ONLY must be a boolean")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check formal drafter residency on cuda:0.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="outputs/drafter_residency_manifest.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = check_drafter_residency(args.config)
    write_manifest(args.output, manifest)
    print(f"wrote drafter residency manifest: {args.output}")
    print(
        "simultaneous residency: "
        + ("PASS" if manifest["simultaneous"]["success"] else "OOM; sequential/lazy required")
    )


if __name__ == "__main__":
    main()
