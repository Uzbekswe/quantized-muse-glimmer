"""Convert a local Hugging Face checkpoint to a BF16 GGUF file."""

from __future__ import annotations

import argparse
import sys

from .common import (
    binary_path,
    command_string,
    load_config,
    project_path,
    require_directory,
    run_command,
)


def build_command(model_dir, output, llama_cpp_dir, outtype):
    converter = project_path(llama_cpp_dir) / "convert_hf_to_gguf.py"
    return [sys.executable, str(converter), str(model_dir), "--outfile", str(output), "--outtype", outtype]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--llama-cpp-dir", default="llama.cpp")
    parser.add_argument("--outtype", default="bf16", choices=["bf16", "f16", "f32"])
    parser.add_argument("--dry-run", action="store_true", help="Print the command without executing it")
    parser.add_argument("--execute", action="store_true", help="Run conversion")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    model_dir = project_path(args.model_dir or config["paths"]["source_dir"])
    output = project_path(args.output or config["paths"]["bf16_gguf"])
    if args.execute:
        require_directory(model_dir, "Hugging Face model directory")
    elif not model_dir.is_dir():
        print(f"Note: source directory is not present in dry-run mode: {model_dir}")
    if output.resolve() == model_dir.resolve():
        raise ValueError("Output must not overwrite the source model directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(model_dir, output, args.llama_cpp_dir, args.outtype)
    print("Conversion is safe by default; pass --execute on a GPU-capable machine.")
    print(f"$ {command_string(command)}")
    if args.execute:
        run_command(command, execute=True)
        print(f"Created {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
