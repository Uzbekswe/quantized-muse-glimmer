"""Run reproducible text, speed, and perplexity benchmarks with llama.cpp."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from statistics import median

from .common import (
    append_jsonl,
    binary_path,
    command_string,
    extract_completion,
    hardware_metadata,
    llama_version,
    load_config,
    parse_token_count,
    parse_perplexity,
    parse_peak_memory,
    parse_tokens_per_second,
    project_path,
    require_file,
    sha256_file,
    utc_now,
)


_SHA256_CACHE: dict[Path, str] = {}


def cached_sha256(path: Path) -> str:
    """Hash each large artifact at most once per benchmark process."""
    resolved = path.resolve()
    if resolved not in _SHA256_CACHE:
        _SHA256_CACHE[resolved] = sha256_file(resolved)
    return _SHA256_CACHE[resolved]


def discover_models(config: dict, artifact_dir: Path) -> list[tuple[str, Path, str]]:
    models: list[tuple[str, Path, str]] = []
    converted = artifact_dir / "converted" / "Muse-Glimmer-30B-BF16.gguf"
    if converted.is_file():
        models.append(("bf16", converted, "bf16"))
    quantized = artifact_dir / "quantized"
    if quantized.is_dir():
        for path in sorted(quantized.glob("*.gguf")):
            models.append((path.stem, path, path.stem))
    official = artifact_dir / "official"
    for baseline in config["official_baselines"]:
        path = official / baseline["filename"]
        if path.is_file():
            models.append((baseline["id"], path, baseline["id"]))
    return models


def parse_expected(prompt: dict, output: str) -> float | None:
    expected = prompt.get("expected")
    if expected is None:
        return None
    return 1.0 if str(expected).lower() in output.lower() else 0.0


def run_process(command: list[str], execute: bool) -> tuple[str, str, int, float, int | None]:
    print(f"$ {command_string(command)}")
    if not execute:
        return "", "", 0, 0.0, None
    timed_command = command
    time_binary = "/usr/bin/time"
    has_time = shutil.which(time_binary) is not None
    started = time.perf_counter()
    transcript_path = None
    if "--single-turn" in command and shutil.which("script"):
        # llama-cli's conversation output is TTY-aware. Capture a pseudo-TTY
        # transcript so task outputs are recorded instead of silently empty.
        handle = tempfile.NamedTemporaryFile(prefix="muse-tty-", suffix=".log", delete=False)
        transcript_path = Path(handle.name)
        handle.close()
        script_command = ["script", "-q", "-c", shlex.join(command), str(transcript_path)]
        timed_command = [time_binary, "-v", *script_command] if has_time else script_command
    elif has_time:
        timed_command = [time_binary, "-v", *command]
    result = subprocess.run(timed_command, check=False, capture_output=True, text=True)
    elapsed = (time.perf_counter() - started) * 1000
    peak_memory = parse_peak_memory(result.stderr)
    stdout = result.stdout
    if transcript_path is not None:
        try:
            stdout = transcript_path.read_text(encoding="utf-8", errors="replace")
        finally:
            transcript_path.unlink(missing_ok=True)
    return stdout, result.stderr, result.returncode, elapsed, peak_memory


def base_record(config, variant, model, llama_cli, prompt_id=None, context_length=None):
    calibration = project_path(config["paths"]["calibration_dir"] + "/calibration.txt")
    return {
        "timestamp": utc_now(),
        "kind": "prompt",
        "variant": variant,
        "model_path": str(model),
        "model_size_bytes": model.stat().st_size if model.is_file() else None,
        "model_sha256": cached_sha256(model) if model.is_file() else None,
        "source_model_revision": config["model"]["source_revision"],
        "llama_cpp_revision": llama_version(llama_cli),
        "quantization_recipe": variant,
        "calibration_dataset_hash": cached_sha256(calibration) if calibration.is_file() else "unavailable",
        "hardware": hardware_metadata(),
        "prompt_id": prompt_id,
        "seed": config["benchmark"]["seed"],
        "context_length": context_length,
        "prompt_tokens": None,
        "output_tokens": None,
        "prefill_tokens_per_second": None,
        "decode_tokens_per_second": None,
        "latency_ms": None,
        "peak_memory_bytes": None,
        "quality_metric": None,
        "error": None,
        "command": None,
    }


def run_prompt_benchmark(config, models, llama_cpp_dir, output_path, execute):
    prompts_path = project_path(config["benchmark"]["prompts_file"])
    prompts = [json.loads(line) for line in prompts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    llama_cli = binary_path(llama_cpp_dir, "llama-cli")
    records = 0
    prompt_repetitions = config["benchmark"].get("prompt_repetitions", config["benchmark"]["repetitions"])
    for variant, model, recipe in models:
        for repetition in range(prompt_repetitions):
            for prompt in prompts:
                context_length = config["benchmark"].get("context_length", 512)
                command = [
                    str(llama_cli),
                    "-m",
                    str(model),
                    "-p",
                    prompt["prompt"],
                    "-n",
                    str(config["benchmark"]["max_tokens"]),
                    "--seed",
                    str(config["benchmark"]["seed"]),
                    "--temp",
                    str(config["benchmark"]["temperature"]),
                    "--top-p",
                    str(config["benchmark"]["top_p"]),
                    "--top-k",
                    str(config["benchmark"]["top_k"]),
                    "-ngl",
                    str(config["benchmark"]["gpu_layers"]),
                    "-c",
                    str(context_length),
                    "--single-turn",
                    "--no-display-prompt",
                ]
                if config["benchmark"].get("jinja", True):
                    command.insert(command.index("--single-turn"), "--jinja")
                stdout, stderr, returncode, latency, peak_memory = run_process(command, execute)
                combined = f"{stdout}\n{stderr}"
                visible_output = stdout if stdout.strip() else stderr
                completion = extract_completion(stdout, prompt["prompt"])
                record = base_record(config, variant, model, llama_cli, prompt["id"], context_length)
                record.update(
                    {
                        "repeat": repetition,
                        "output": visible_output,
                        "output_tokens": parse_token_count(combined, "eval time"),
                        "prefill_tokens_per_second": parse_tokens_per_second(combined, "prompt eval time"),
                        "decode_tokens_per_second": parse_tokens_per_second(combined, "eval time"),
                        "latency_ms": latency,
                        "peak_memory_bytes": peak_memory,
                        "command": command_string(command),
                        "quality_metric": parse_expected(prompt, completion),
                        "error": None if returncode == 0 else (stderr[-2000:] or f"exit {returncode}"),
                    }
                )
                append_jsonl(output_path, record)
                records += 1
    return records


def run_speed_benchmark(config, models, llama_cpp_dir, output_path, execute):
    bench = binary_path(llama_cpp_dir, "llama-bench")
    records = 0
    for variant, model, recipe in models:
        command = [
            str(bench),
            "-m",
            str(model),
            "-p",
            "512",
            "-n",
            str(config["benchmark"]["max_tokens"]),
            "-c",
            str(config["benchmark"].get("context_length", 512)),
            "-r",
            str(config["benchmark"]["repetitions"]),
            "-ngl",
            str(config["benchmark"]["gpu_layers"]),
        ]
        stdout, stderr, returncode, latency, peak_memory = run_process(command, execute)
        combined = f"{stdout}\n{stderr}"
        context_length = config["benchmark"].get("context_length", 512)
        record = base_record(config, variant, model, bench, None, context_length)
        record.update(
            {
                "kind": "speed",
                "prefill_tokens_per_second": parse_tokens_per_second(combined, "pp") or parse_tokens_per_second(combined, "prompt eval time"),
                "decode_tokens_per_second": parse_tokens_per_second(combined, "tg") or parse_tokens_per_second(combined, "eval time"),
                "latency_ms": latency,
                "peak_memory_bytes": peak_memory,
                "command": command_string(command),
                "raw_output": combined[-8000:],
                "error": None if returncode == 0 else (stderr[-2000:] or f"exit {returncode}"),
            }
        )
        append_jsonl(output_path, record)
        records += 1
    return records


def run_perplexity_benchmark(config, models, llama_cpp_dir, output_path, execute):
    perplexity = binary_path(llama_cpp_dir, "llama-perplexity")
    text_file = require_file(config["benchmark"]["perplexity_file"], "perplexity evaluation text")
    records = 0
    for variant, model, recipe in models:
        command = [
            str(perplexity),
            "-m",
            str(model),
            "-f",
            str(text_file),
            "-ngl",
            str(config["benchmark"]["gpu_layers"]),
        ]
        stdout, stderr, returncode, latency, peak_memory = run_process(command, execute)
        combined = f"{stdout}\n{stderr}"
        record = base_record(config, variant, model, perplexity, None, None)
        record.update(
            {
                "kind": "perplexity",
                "quality_metric": parse_perplexity(combined),
                "latency_ms": latency,
                "peak_memory_bytes": peak_memory,
                "command": command_string(command),
                "raw_output": combined[-8000:],
                "error": None if returncode == 0 else (stderr[-2000:] or f"exit {returncode}"),
            }
        )
        append_jsonl(output_path, record)
        records += 1
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--models", default=None, help="Comma-separated model paths")
    parser.add_argument("--llama-cpp-dir", default="llama.cpp")
    parser.add_argument("--mode", choices=["prompt", "speed", "perplexity", "all"], default="all")
    parser.add_argument("--output", default="results/benchmarks.jsonl")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    parser.add_argument("--execute", action="store_true", help="Run llama.cpp benchmarks")
    parser.add_argument("--append", action="store_true", help="Append records instead of replacing the output JSONL")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    artifact_dir = project_path(args.artifact_dir or config["paths"]["artifacts_dir"])
    if args.models:
        models = []
        for raw in args.models.split(","):
            path = project_path(raw.strip())
            require_file(path, "benchmark model")
            models.append((path.stem, path, path.stem))
    else:
        models = discover_models(config, artifact_dir)
    if not models:
        print("No model files found. Dry-run still shows the benchmark contract; prepare artifacts first for execution.")
        models = [("<model-variant>", project_path("artifacts/quantized/<model>.gguf"), "unknown")]

    output = project_path(args.output)
    if args.execute and not args.append:
        output.unlink(missing_ok=True)
    print(f"Benchmark mode={args.mode}; execution={'enabled' if args.execute else 'dry-run'}")
    if args.mode in {"prompt", "all"}:
        run_prompt_benchmark(config, models, args.llama_cpp_dir, output, args.execute)
    if args.mode in {"speed", "all"}:
        run_speed_benchmark(config, models, args.llama_cpp_dir, output, args.execute)
    if args.mode in {"perplexity", "all"}:
        run_perplexity_benchmark(config, models, args.llama_cpp_dir, output, args.execute)
    print(f"Benchmark records: {output if args.execute else 'dry-run only'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
