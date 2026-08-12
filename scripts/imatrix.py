"""Create a llama.cpp activation importance matrix from calibration text."""

from __future__ import annotations

import argparse

from .common import binary_path, command_string, load_config, project_path, require_file, run_command


def build_command(model, calibration, output, llama_cpp_dir, gpu_layers):
    return [
        str(binary_path(llama_cpp_dir, "llama-imatrix")),
        "-m",
        str(model),
        "-f",
        str(calibration),
        "-o",
        str(output),
        "-ngl",
        str(gpu_layers),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--output", default="artifacts/converted/imatrix.gguf")
    parser.add_argument("--llama-cpp-dir", default="llama.cpp")
    parser.add_argument("--gpu-layers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print the command without executing it")
    parser.add_argument("--execute", action="store_true", help="Run llama-imatrix")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    model = project_path(args.model or config["paths"]["bf16_gguf"])
    calibration = project_path(args.calibration or config["paths"]["calibration_dir"] + "/calibration.txt")
    output = project_path(args.output)
    if args.execute:
        require_file(model, "BF16 GGUF model")
        require_file(calibration, "calibration corpus")
    else:
        if not model.is_file():
            print(f"Note: BF16 GGUF is not present in dry-run mode: {model}")
        if not calibration.is_file():
            print(f"Note: calibration corpus is not present in dry-run mode: {calibration}")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(
        model,
        calibration,
        output,
        args.llama_cpp_dir,
        args.gpu_layers if args.gpu_layers is not None else config["benchmark"]["gpu_layers"],
    )
    print("Importance-matrix generation is safe by default; pass --execute to run it.")
    print(f"$ {command_string(command)}")
    if args.execute:
        run_command(command, execute=True)
        print(f"Created {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
