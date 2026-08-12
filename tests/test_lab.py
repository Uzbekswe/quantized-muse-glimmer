from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.common import load_config, project_path, sha256_file
from scripts.convert import build_command
from scripts.quantize import recipe_commands
from scripts.report import summarize, write_outputs


ROOT = Path(__file__).resolve().parents[1]


def test_config_has_required_experiment_contract():
    config = load_config()
    assert config["project"]["name"] == "quantized-muse-glimmer"
    assert config["model"]["llama_cpp_min_build"] >= 10353
    assert {recipe["quant_type"] for recipe in config["recipes"]} >= {"Q4_K_M", "Q5_K_M", "Q6_K"}
    assert any(recipe["calibrated"] for recipe in config["recipes"])


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
    write_outputs(rows, tmp_path / "results.md", tmp_path / "summary.csv")
    assert (tmp_path / "results.md").is_file()
    assert "q4_k_m" in (tmp_path / "results.md").read_text()
    assert "q4_k_m" in (tmp_path / "summary.csv").read_text()


def test_calibration_and_evaluation_inputs_are_hashed_and_valid_jsonl():
    calibration = ROOT / "calibration/calibration.txt"
    prompts = ROOT / "evaluation/prompts.jsonl"
    assert len(sha256_file(calibration)) == 64
    rows = [json.loads(line) for line in prompts.read_text().splitlines() if line.strip()]
    assert len(rows) >= 10
    assert all("id" in row and "category" in row and "prompt" in row for row in rows)
