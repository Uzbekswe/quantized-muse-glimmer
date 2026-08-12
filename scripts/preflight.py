"""Validate remote GPU, llama.cpp, memory, disk, and artifact prerequisites."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

from .common import binary_path, hardware_metadata, load_config, project_path, write_json


def command_output(executable: str, args: Sequence[str], cwd: Path | None = None) -> str | None:
    """Run only the fixed system commands used by preflight.

    Keeping the executable separate prevents a caller-controlled executable
    path from being passed directly to subprocess.run.
    """

    if executable not in {"git", "nvidia-smi", "llama-cli"}:
        raise ValueError(f"Unsupported preflight executable: {executable}")
    if executable == "git":
        command = ["git", *args]
    elif executable == "nvidia-smi":
        command = ["nvidia-smi", *args]
    else:
        command = ["./llama-cli", *args]
    try:
        result = subprocess.run(command, cwd=cwd, check=False, capture_output=True, text=True)
    except OSError:
        return None
    return (result.stdout or result.stderr).strip() if result.returncode == 0 else None


def llama_build(version: str) -> int | None:
    match = re.search(r"b(\d+)", version)
    return int(match.group(1)) if match else None


def llama_revision(llama_cpp_dir: Path) -> tuple[str | None, int | None]:
    """Return the checkout revision and optional bNNNN branch number."""
    if not llama_cpp_dir.is_dir():
        return None, None
    git_args = ["-C", str(llama_cpp_dir)]
    branch = command_output("git", [*git_args, "symbolic-ref", "--short", "HEAD"])
    commit = command_output("git", [*git_args, "rev-parse", "--short", "HEAD"])
    tags = command_output("git", [*git_args, "tag", "--points-at", "HEAD"])
    build = llama_build(branch or "") or llama_build(tags or "")
    return commit or branch or tags, build


def parse_gpu_info(raw: str | None) -> list[dict[str, str | float]]:
    """Parse nvidia-smi's `name,memory.total` CSV output."""

    if not raw:
        return []
    rows = []
    for row in csv.reader(line for line in raw.splitlines() if line.strip()):
        if len(row) < 2:
            continue
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*MiB", row[1], re.IGNORECASE)
        if match:
            rows.append({"name": row[0].strip(), "memory_mib": float(match.group(1))})
    return rows


def check(
    config: dict,
    artifact_dir: Path,
    llama_cpp_dir: Path,
    required_artifacts: Sequence[str] | None = None,
) -> dict:
    requirements = config["hardware_requirements"]
    memory = hardware_metadata().get("memory_bytes", 0)
    disk = shutil.disk_usage(artifact_dir if artifact_dir.exists() else artifact_dir.parent)
    llama_cli = binary_path(str(llama_cpp_dir), "llama-cli")
    version = command_output("llama-cli", ["--version"], llama_cli.parent) if llama_cli.is_file() else None
    revision, checkout_build = llama_revision(llama_cpp_dir)
    build = llama_build(version or "") or checkout_build
    gpu_info = command_output(
        "nvidia-smi", ["--query-gpu=name,memory.total", "--format=csv,noheader"]
    )
    gpus = parse_gpu_info(gpu_info)
    required = {
        "bf16": artifact_dir / "converted" / "Muse-Glimmer-30B-BF16.gguf",
        "q4": artifact_dir / "quantized" / "Muse-Glimmer-30B-q4_k_m.gguf",
    }
    artifact_status = {name: path.is_file() for name, path in required.items()}
    required_artifacts = list(required_artifacts or [])
    unknown_artifacts = sorted(set(required_artifacts) - set(required))
    artifact_requirements_ok = not unknown_artifacts and all(
        artifact_status[name] for name in required_artifacts
    )
    minimum_gpu_mib = requirements["minimum_gpu_memory_gib"] * 1024
    gpu_name_pattern = requirements.get("required_gpu_name_pattern", "")
    gpu_ok = (
        len(gpus) == requirements["minimum_gpu_count"]
        and all(float(gpu["memory_mib"]) >= minimum_gpu_mib for gpu in gpus)
        and all(re.search(gpu_name_pattern, str(gpu["name"]), re.IGNORECASE) for gpu in gpus)
    )
    result = {
        "memory_bytes": memory,
        "memory_ok": memory >= requirements["minimum_system_ram_gib"] * 1024**3,
        "free_disk_bytes": disk.free,
        "free_disk_ok": disk.free >= requirements["recommended_free_disk_gib"] * 1024**3,
        "gpu": gpu_info,
        "gpu_devices": gpus,
        "cuda_ok": bool(gpus),
        "gpu_ok": gpu_ok,
        "llama_cli": str(llama_cli),
        "llama_version": version,
        "llama_revision": revision,
        "llama_build": build,
        "llama_build_ok": build is not None and build >= config["model"]["llama_cpp_min_build"],
        "artifact_status": artifact_status,
        "required_artifacts": required_artifacts,
        "unknown_required_artifacts": unknown_artifacts,
        "artifacts_ok": artifact_requirements_ok,
    }
    result["ok"] = all(
        value for key, value in result.items() if key.endswith("_ok")
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--llama-cpp-dir", default="llama.cpp")
    parser.add_argument(
        "--required-artifacts",
        default="",
        help="Comma-separated artifact names to require: bf16,q4",
    )
    parser.add_argument(
        "--require-artifacts",
        action="store_true",
        help="Deprecated alias requiring both bf16 and q4",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    artifact_dir = project_path(args.artifact_dir or config["paths"]["artifacts_dir"])
    required = [item.strip() for item in args.required_artifacts.split(",") if item.strip()]
    if args.require_artifacts:
        required = ["bf16", "q4"]
    result = check(config, artifact_dir, project_path(args.llama_cpp_dir), required)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        output = write_json(args.output, result)
        print(f"Wrote preflight record to {output}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
