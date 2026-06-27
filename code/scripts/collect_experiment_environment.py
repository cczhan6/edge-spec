from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def collect_environment(
    *,
    repo_root: str | Path,
    nvidia_smi_command: Sequence[str] = ("nvidia-smi",),
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    errors: list[str] = []
    torch_version, cuda_runtime_version, cuda_available = _torch_metadata(errors)
    transformers_version = _package_version("transformers", errors)
    gpu_result = _gpu_metadata(nvidia_smi_command, errors)
    git_result = _git_metadata(root, errors)

    return {
        "schema_version": 1,
        "collected_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "software": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "pytorch_version": torch_version,
            "transformers_version": transformers_version,
            "cuda_runtime_version": cuda_runtime_version,
            "cuda_available_to_pytorch": cuda_available,
        },
        "hardware": gpu_result,
        "git": git_result,
        "collection_errors": errors,
    }


def write_environment_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
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


def _package_version(distribution: str, errors: list[str]) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        errors.append(f"Python package is unavailable: {distribution}")
        return None


def _torch_metadata(errors: list[str]) -> tuple[str | None, str | None, bool | None]:
    try:
        import torch
    except (ImportError, OSError) as exc:
        errors.append(f"PyTorch metadata is unavailable: {type(exc).__name__}: {exc}")
        return _package_version("torch", errors), None, None
    return str(torch.__version__), getattr(torch.version, "cuda", None), bool(torch.cuda.is_available())


def _gpu_metadata(command: Sequence[str], errors: list[str]) -> dict[str, Any]:
    query_command = [
        *command,
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            query_command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        errors.append(f"nvidia-smi GPU query failed: {type(exc).__name__}: {exc}")
        return {
            "gpu_query_status": "unavailable",
            "cuda_driver_version": None,
            "gpus": [],
        }
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit code {completed.returncode}"
        errors.append(f"nvidia-smi GPU query failed: {detail}")
        return {
            "gpu_query_status": "unavailable",
            "cuda_driver_version": None,
            "gpus": [],
        }

    gpus: list[dict[str, Any]] = []
    driver_versions: set[str] = set()
    for line_number, raw_line in enumerate(completed.stdout.splitlines(), start=1):
        if not raw_line.strip():
            continue
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 4:
            errors.append(f"nvidia-smi row {line_number} has {len(parts)} fields, expected 4")
            continue
        index, model, memory_total_mib, driver_version = parts
        try:
            parsed_index = int(index)
            parsed_memory = int(memory_total_mib)
        except ValueError:
            errors.append(f"nvidia-smi row {line_number} contains non-integer index or memory")
            continue
        driver_versions.add(driver_version)
        gpus.append(
            {
                "index": parsed_index,
                "model": model,
                "memory_total_mib": parsed_memory,
                "driver_version": driver_version,
            }
        )

    return {
        "gpu_query_status": "ok" if gpus else "empty",
        "cuda_driver_version": ",".join(sorted(driver_versions)) or None,
        "gpus": sorted(gpus, key=lambda gpu: gpu["index"]),
    }


def _git_metadata(repo_root: Path, errors: list[str]) -> dict[str, Any]:
    commit = _run_git(repo_root, ("rev-parse", "HEAD"), errors)
    status = _run_git(repo_root, ("status", "--porcelain"), errors)
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
    }


def _run_git(
    repo_root: Path,
    args: Sequence[str],
    errors: list[str],
) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        errors.append(f"git {' '.join(args)} failed: {type(exc).__name__}: {exc}")
        return None
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit code {completed.returncode}"
        errors.append(f"git {' '.join(args)} failed: {detail}")
        return None
    return completed.stdout.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect formal experiment environment metadata.")
    parser.add_argument("--output", default="outputs/environment_manifest.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    manifest = collect_environment(repo_root=repo_root)
    write_environment_manifest(args.output, manifest)
    print(f"wrote environment manifest: {args.output}")
    if manifest["collection_errors"]:
        print("environment collection completed with explicit unavailable fields:", file=sys.stderr)
        for error in manifest["collection_errors"]:
            print(f"- {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
