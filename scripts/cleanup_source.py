"""Safely remove the downloaded HF source after BF16 conversion is verified."""

from __future__ import annotations

import argparse
import shutil

from .common import load_config, project_path, require_directory, require_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--converted", default=None)
    parser.add_argument("--execute", action="store_true", help="Delete the validated source directory")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    source = project_path(args.source_dir or config["paths"]["source_dir"])
    converted = project_path(args.converted or config["paths"]["bf16_gguf"])
    require_file(converted, "converted BF16 GGUF")
    require_directory(source, "HF source directory")
    expected_root = project_path(config["paths"]["source_dir"]).resolve()
    if source.resolve() != expected_root:
        raise ValueError(f"Refusing to delete outside configured source directory: {source}")
    print(f"Source cleanup {'enabled' if args.execute else 'dry-run'}: {source}")
    if args.execute:
        shutil.rmtree(source)
        print(f"Removed validated source directory: {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
