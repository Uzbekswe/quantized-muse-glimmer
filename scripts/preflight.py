"""Validate remote GPU, llama.cpp, memory, disk, and artifact prerequisites."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

from .common import binary_path, hardware_metadata, load_config, project_path, sha256_file, write_json


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError:
        return None
    return (result.stdout or result.stderr).strip() if result.returncode == 0 else None


def llama_build(version: str) -> int | None:
    match = re.search(r"b(\d+)", version)
    return int(match.group(1)) if match else None


def llama_revision(llama_cpp_dir: Path) -> tuple[str | None, int | None]:
    """Return the checkout revision and optional bNNNN branch number."""
    branch = command_output(["git", "-C", str(llama_cpp_dir), "symbolic-ref", "--short", "HEAD"])
    commit = command_output(["git", "-C", str(llama_cpp_dir), "rev-parse", "--short", "HEAD"])
    build = llama_build(branch or "")
    return commit or branch, build


def check(config: dict, artifact_dir: Path, llama_cpp_dir: Path, require_artifacts: bool) -> dict:
    requirements = config["hardware_requirements"]
    memory = hardware_metadata().get("memory_bytes", 0)
    disk = shutil.disk_usage(artifact_dir if artifact_dir.exists() else artifact_dir.parent)
    llama_cli = binary_path(str(llama_cpp_dir), "llama-cli")
    version = command_output([str(llama_cli), "--version"]) if llama_cli.is_file() else None
    revision, checkout_build = llama_revision(llama_cpp_dir)
    build = llama_build(version or "") or checkout_build
    gpu_info = command_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    required = {
        "bf16_gguf": project_path(config["paths"]["bf16_gguf"]),
        "q4_gguf": project_path(config["paths"]["quantized_dir"]) / "Muse-Glimmer-30B-q4_k_m.gguf",
    }
    artifact_status = {name: path.is_file() for name, path in required.items()}
    result = {
        "memory_bytes": memory,
        "memory_ok": memory >= requirements["minimum_system_ram_gib"] * 1024**3,
        "free_disk_bytes": disk.free,
        "free_disk_ok": disk.free >= requirements["recommended_free_disk_gib"] * 1024**3,
        "gpu": gpu_info,
        "cuda_ok": bool(gpu_info),
        "llama_cli": str(llama_cli),
        "llama_version": version,
        "llama_revision": revision,
        "llama_build": build,
        "llama_build_ok": build is not None and build >= config["model"]["llama_cpp_min_build"],
        "artifact_status": artifact_status,
        "artifacts_ok": all(artifact_status.values()) if require_artifacts else True,
    }
    result["ok"] = all(value for key, value in result.items() if key.endswith("_ok"))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--llama-cpp-dir", default="llama.cpp")
    parser.add_argument("--require-artifacts", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    artifact_dir = project_path(args.artifact_dir or config["paths"]["artifacts_dir"])
    result = check(config, artifact_dir, project_path(args.llama_cpp_dir), args.require_artifacts)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        output = write_json(args.output, result)
        print(f"Wrote preflight record to {output}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
