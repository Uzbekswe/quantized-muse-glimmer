from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.benchmark as benchmark_module
from scripts.benchmark import run_process
from scripts.cleanup_source import main as cleanup_main
from scripts.common import (
    extract_completion,
    load_config,
    parse_peak_memory,
    parse_perplexity,
    parse_tokens_per_second,
    project_path,
    sha256_file,
)
from scripts.convert import build_command
from scripts.prepare import download_commands
from scripts.preflight import check, command_output, parse_gpu_info
from scripts.preserve import preserve
from scripts.quantize import recipe_commands
from scripts.report import summarize, write_outputs
from scripts.smoke import assert_smoke, build_command as smoke_command


ROOT = Path(__file__).resolve().parents[1]


def test_config_has_required_experiment_contract():
    config = load_config()
    assert config["project"]["name"] == "quantized-muse-glimmer"
    assert config["model"]["llama_cpp_min_build"] >= 10353
    assert {recipe["quant_type"] for recipe in config["recipes"]} >= {"Q4_K_M", "Q5_K_M", "Q6_K"}
    assert any(recipe["calibrated"] for recipe in config["recipes"])
    assert config["benchmark"]["prompt_repetitions"] == 1
    assert config["benchmark"]["jinja"] is True
    assert config["stages"]["first_gpu"]["recipes"] == ["q4_k_m"]
    assert config["stages"]["first_gpu"]["text_only"] is True
    assert config["model"]["source_revision"] == "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
    assert config["benchmark"]["context_length"] == 512
    assert config["hardware_requirements"]["minimum_gpu_memory_gib"] == 80


def test_conversion_command_preserves_bf16_source():
    command = build_command(
        ROOT / "artifacts/source/Muse-Glimmer-30B",
        ROOT / "artifacts/converted/Muse-Glimmer-30B-BF16.gguf",
        "llama.cpp",
        "bf16",
    )
    assert "convert_hf_to_gguf.py" in command[1]
    assert "--outtype" in command
    assert command[-1] == "bf16"


def test_text_only_download_excludes_multimodal_artifacts(tmp_path):
    config = load_config()
    commands = download_commands(config, tmp_path, text_only=True)
    rendered = " ".join(" ".join(command) for command in commands)
    assert "mmproj-kquant.gguf" not in rendered
    assert "dflash-kquant.gguf" not in rendered
    assert "muse-glimmer-30B-kquant-17gb.gguf" in rendered


def test_quantization_recipes_are_named_and_do_not_target_source(tmp_path):
    config = load_config()
    source = tmp_path / "source.gguf"
    output_dir = tmp_path / "quantized"
    source.write_bytes(b"source")
    imatrix = tmp_path / "imatrix.gguf"
    imatrix.write_bytes(b"matrix")
    commands = recipe_commands(config, source, output_dir, "llama.cpp", imatrix, 8)
    assert len(commands) == len(config["recipes"])
    assert all(output != source for _, output, _ in commands)
    assert {recipe_id for recipe_id, _, _ in commands} >= {"q4_k_m", "q5_k_m", "q6_k"}
    assert any("--imatrix" in command for _, _, command in commands)


def test_quantization_requires_matrix_for_calibrated_recipes(tmp_path):
    config = load_config()
    with pytest.raises(ValueError, match="require --imatrix"):
        recipe_commands(config, tmp_path / "source.gguf", tmp_path, "llama.cpp", None, 8)


def test_report_summary_and_outputs(tmp_path):
    records = [
        {
            "variant": "q4_k_m",
            "kind": "prompt",
            "model_size_bytes": 4 * 1024**3,
            "prefill_tokens_per_second": 100.0,
            "decode_tokens_per_second": 20.0,
            "latency_ms": 50.0,
            "quality_metric": 0.9,
            "error": None,
        },
        {
            "variant": "q4_k_m",
            "kind": "prompt",
            "model_size_bytes": 4 * 1024**3,
            "prefill_tokens_per_second": 110.0,
            "decode_tokens_per_second": 22.0,
            "latency_ms": 60.0,
            "quality_metric": 1.0,
            "error": None,
        },
    ]
    rows = summarize(records)
    assert rows[0]["variant"] == "q4_k_m"
    assert rows[0]["kind"] == "prompt"
    assert rows[0]["size_gib"] == 4.0
    assert rows[0]["decode_tok_s_median"] == 21.0
    assert rows[0]["peak_memory_bytes_median"] is None
    write_outputs(rows, tmp_path / "results.md", tmp_path / "summary.csv")
    assert (tmp_path / "results.md").is_file()
    assert "q4_k_m" in (tmp_path / "results.md").read_text()
    assert "q4_k_m" in (tmp_path / "summary.csv").read_text()


def test_preserve_copies_and_verifies_checksum(tmp_path):
    source = tmp_path / "source.gguf"
    destination = tmp_path / "persistent" / "q4.gguf"
    source.write_bytes(b"verified q4 artifact")
    record = preserve(source, destination)
    assert destination.read_bytes() == source.read_bytes()
    assert record["sha256"] == sha256_file(source)
    assert record["size_bytes"] == source.stat().st_size


def test_calibration_and_evaluation_inputs_are_hashed_and_valid_jsonl():
    calibration = ROOT / "calibration/calibration.txt"
    prompts = ROOT / "evaluation/prompts.jsonl"
    assert len(sha256_file(calibration)) == 64
    rows = [json.loads(line) for line in prompts.read_text().splitlines() if line.strip()]
    assert len(rows) >= 10
    assert all("id" in row and "category" in row and "prompt" in row for row in rows)


def test_shared_parsers_cover_runtime_formats():
    output = "prompt eval time = 12.5 ms / 512 tokens (40.96 tokens per second)\n"
    table = "| pp512 | 3395.93 ± 318.07 |\n| tg128 | 28.62 +/- 0.02 |"
    assert parse_tokens_per_second(output, "prompt eval time") == 40.96
    assert parse_tokens_per_second(table, "pp") == 3395.93
    assert parse_tokens_per_second(table, "tg") == 28.62
    assert parse_perplexity("Final perplexity = 9.5372") == 9.5372
    assert parse_peak_memory("Maximum resident set size (kbytes): 1234") == 1234 * 1024


def test_pty_benchmark_wraps_script_inside_time(monkeypatch):
    seen = {}

    class Result:
        stdout = ""
        stderr = "Maximum resident set size (kbytes): 2048"
        returncode = 0

    monkeypatch.setattr(
        benchmark_module.shutil,
        "which",
        lambda name: "/usr/bin/time" if name.endswith("time") else "/usr/bin/script",
    )

    def fake_run(command, **kwargs):
        seen["command"] = command
        return Result()

    monkeypatch.setattr(benchmark_module.subprocess, "run", fake_run)
    _, _, returncode, _, peak_memory = run_process(["echo", "--single-turn", "hello"], True)
    assert returncode == 0
    assert seen["command"][0:3] == ["/usr/bin/time", "-v", "script"]
    assert peak_memory == 2048 * 1024


def test_completion_removes_only_a_leading_echo():
    prompt = "repeat this"
    assert extract_completion(f"{prompt} {prompt} once", prompt) == f"{prompt} once"
    assert extract_completion("model output", prompt) == "model output"


def test_gpu_parser_and_preflight_contract(tmp_path, monkeypatch):
    assert parse_gpu_info("NVIDIA A100-SXM4-80GB, 81920 MiB\n") == [
        {"name": "NVIDIA A100-SXM4-80GB", "memory_mib": 81920.0}
    ]
    with pytest.raises(ValueError, match="Unsupported"):
        command_output("sh", ["-c", "echo unsafe"])

    artifact_dir = tmp_path / "artifacts"
    (artifact_dir / "converted").mkdir(parents=True)
    (artifact_dir / "converted" / "Muse-Glimmer-30B-BF16.gguf").write_bytes(b"bf16")
    llama_dir = tmp_path / "llama.cpp" / "build" / "bin"
    llama_dir.mkdir(parents=True)
    (llama_dir / "llama-cli").write_bytes(b"placeholder")
    monkeypatch.setattr("scripts.preflight.hardware_metadata", lambda: {"memory_bytes": 64 * 1024**3})
    monkeypatch.setattr(
        "scripts.preflight.shutil.disk_usage",
        lambda _: type("Usage", (), {"free": 200 * 1024**3})(),
    )
    monkeypatch.setattr(
        "scripts.preflight.command_output",
        lambda executable, args, cwd=None: (
            "NVIDIA A100-SXM4-80GB, 81920 MiB"
            if executable == "nvidia-smi"
            else "b10353"
        ),
    )
    result = check(load_config(), artifact_dir, tmp_path / "llama.cpp", ["bf16"])
    assert result["ok"] is True
    assert result["artifacts_ok"] is True
    assert result["gpu_ok"] is True
    assert result["artifact_status"]["q4"] is False

    monkeypatch.setattr(
        "scripts.preflight.command_output",
        lambda executable, args, cwd=None: (
            "NVIDIA A100-SXM4-80GB, 40960 MiB\nNVIDIA A100-SXM4-80GB, 81920 MiB"
            if executable == "nvidia-smi"
            else "b10353"
        ),
    )
    rejected = check(load_config(), artifact_dir, tmp_path / "llama.cpp", ["bf16"])
    assert rejected["gpu_ok"] is False
    assert rejected["ok"] is False


def test_cleanup_requires_confirmation_and_matching_checksum(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    converted = tmp_path / "converted.gguf"
    converted.write_bytes(b"converted")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "project: {name: test}\n"
        f"paths:\n  source_dir: {source_dir}\n  bf16_gguf: {converted}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="confirm-source-cleanup"):
        cleanup_main(["--config", str(config_path), "--execute"])
    cleanup_main(
        [
            "--config",
            str(config_path),
            "--execute",
            "--confirm-source-cleanup",
            "--expected-sha256",
            sha256_file(converted),
        ]
    )
    assert not source_dir.exists()


def test_cleanup_rejects_checksum_mismatch(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    converted = tmp_path / "converted.gguf"
    converted.write_bytes(b"converted")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "project: {name: test}\n"
        f"paths:\n  source_dir: {source_dir}\n  bf16_gguf: {converted}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        cleanup_main(
            [
                "--config",
                str(config_path),
                "--execute",
                "--confirm-source-cleanup",
                "--expected-sha256",
                "0" * 64,
            ]
        )
    assert source_dir.exists()


def test_smoke_command_and_assertion():
    command = smoke_command("model.gguf")
    assert "--jinja" in command
    assert "-c" in command
    assert assert_smoke("Q4 smoke test passed.", "", 0, "Q4 smoke test passed.")
    with pytest.raises(AssertionError, match="not generated"):
        assert_smoke("different output", "", 0, "expected text")
