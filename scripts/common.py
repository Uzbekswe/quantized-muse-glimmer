"""Shared helpers for the quantization lab CLI tools."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    """Resolve a path relative to the repository unless it is absolute."""

    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path = "configs/experiment.yaml") -> dict[str, Any]:
    with project_path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return config


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def command_string(command: Sequence[str]) -> str:
    return shlex.join(str(part) for part in command)


def run_command(
    command: Sequence[str],
    *,
    execute: bool,
    cwd: str | Path | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str] | None:
    """Print a command, and run it only when execute=True."""

    print(f"$ {command_string(command)}")
    if not execute:
        return None
    return subprocess.run(
        list(map(str, command)),
        cwd=project_path(cwd) if cwd else PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def require_file(path: str | Path, description: str = "file") -> Path:
    resolved = project_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def require_directory(path: str | Path, description: str = "directory") -> Path:
    resolved = project_path(path)
    if not resolved.is_dir():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with project_path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> Path:
    resolved = project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return resolved


def append_jsonl(path: str | Path, value: dict[str, Any]) -> Path:
    resolved = project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("a", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True)
        handle.write("\n")
    return resolved


def hardware_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
    }
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                check=True,
                capture_output=True,
                text=True,
            )
            metadata["memory_bytes"] = int(result.stdout.strip())
        except (OSError, ValueError, subprocess.CalledProcessError):
            pass
    elif Path("/proc/meminfo").is_file():
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                metadata["memory_bytes"] = int(line.split()[1]) * 1024
                break
    return metadata


def binary_path(llama_cpp_dir: str | Path, name: str) -> Path:
    root = project_path(llama_cpp_dir)
    candidates = [root / "build" / "bin" / name, root / name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    # Return the conventional path so dry-run output remains useful.
    return candidates[0]


def llama_version(llama_cli: Path) -> str:
    if not llama_cli.is_file():
        return "unavailable"
    try:
        result = subprocess.run(
            [str(llama_cli), "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unavailable"
    return (result.stdout or result.stderr).strip().splitlines()[0] if (result.stdout or result.stderr) else "unknown"


def parse_tokens_per_second(text: str, label: str) -> float | None:
    """Parse llama.cpp's human-readable benchmark statistics."""

    pattern = rf"{label}.*?([0-9]+(?:\.[0-9]+)?)\s+tokens per second"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return float(match.group(1))

    # llama-bench emits a compact table, for example:
    # ``| pp512 | 3395.93 ± 318.07 |`` and ``| tg128 | 28.62 ± 0.02 |``.
    table_label = rf"{re.escape(label)}\d+" if label in {"pp", "tg"} else re.escape(label)
    table_match = re.search(
        rf"\|\s*{table_label}\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*(?:±|\+/-)",
        text,
        flags=re.IGNORECASE,
    )
    return float(table_match.group(1)) if table_match else None


def parse_token_count(text: str, label: str) -> int | None:
    pattern = rf"{label}.*?/\s*([0-9]+)\s+(?:runs|tokens)"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return int(match.group(1)) if match else None


def parse_perplexity(text: str) -> float | None:
    """Parse llama-perplexity's human-readable PPL output."""

    match = re.search(r"(?:PPL|perplexity)\s*=\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    return float(match.group(1)) if match else None


def parse_peak_memory(text: str) -> int | None:
    """Parse GNU time's maximum resident set size in bytes."""

    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    return int(match.group(1)) * 1024 if match else None


def extract_completion(output: str, prompt: str) -> str:
    """Remove at most one leading echoed prompt from captured model output."""

    text = output.replace("\r", "").strip()
    if text.startswith(prompt):
        return text[len(prompt) :].lstrip()
    return text
