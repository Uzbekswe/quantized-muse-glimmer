"""Create named GGUF quantization variants from an immutable BF16 GGUF."""

from __future__ import annotations

import argparse
import re

from .common import binary_path, command_string, load_config, project_path, require_file, run_command


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def recipe_commands(config, input_path, output_dir, llama_cpp_dir, imatrix, threads, recipe_ids=None):
    selected = set(recipe_ids or [])
    commands = []
    quantizer = binary_path(llama_cpp_dir, "llama-quantize")
    for recipe in config["recipes"]:
        if selected and recipe["id"] not in selected:
            continue
        output_name = f"Muse-Glimmer-30B-{safe_name(recipe['id'])}.gguf"
        output = output_dir / output_name
        command = [str(quantizer)]
        if recipe.get("calibrated"):
            if imatrix is None:
                raise ValueError("Calibrated recipes require --imatrix")
            command.extend(["--imatrix", str(imatrix)])
        command.extend([str(input_path), str(output), recipe["quant_type"], str(threads)])
        commands.append((recipe["id"], output, command))
    return commands


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--imatrix", default=None)
    parser.add_argument("--recipes", default=None, help="Comma-separated recipe IDs")
    parser.add_argument("--llama-cpp-dir", default="llama.cpp")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    parser.add_argument("--execute", action="store_true", help="Run llama-quantize")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    input_path = project_path(args.input or config["paths"]["bf16_gguf"])
    output_dir = project_path(args.output_dir or config["paths"]["quantized_dir"])
    imatrix = project_path(args.imatrix) if args.imatrix else None
    recipe_ids = [item.strip() for item in args.recipes.split(",")] if args.recipes else None
    selected_recipes = [recipe for recipe in config["recipes"] if not recipe_ids or recipe["id"] in recipe_ids]
    if args.execute:
        require_file(input_path, "BF16 GGUF input")
        if imatrix is None and any(recipe.get("calibrated") for recipe in selected_recipes):
            raise ValueError("Execution requires --imatrix for calibrated recipes")
        if imatrix is not None:
            require_file(imatrix, "importance matrix")
    elif not input_path.is_file():
        print(f"Note: BF16 GGUF is not present in dry-run mode: {input_path}")
    if imatrix is None:
        imatrix = project_path("artifacts/converted/imatrix.gguf")
    if any(input_path.resolve() == path.resolve() for _, path, _ in recipe_commands(config, input_path, output_dir, args.llama_cpp_dir, imatrix, args.threads)):
        raise ValueError("Quantized output must not overwrite the BF16 source")
    output_dir.mkdir(parents=True, exist_ok=True)
    commands = recipe_commands(config, input_path, output_dir, args.llama_cpp_dir, imatrix, args.threads, recipe_ids)
    print("Quantization is safe by default; pass --execute to create large GGUF files.")
    for recipe_id, output, command in commands:
        print(f"[{recipe_id}] {command_string(command)}")
        if args.execute:
            run_command(command, execute=True)
            print(f"Created {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
