"""Safely remove the downloaded HF source after BF16 conversion is verified."""

from __future__ import annotations

import argparse
import json
import re
import shutil

from .common import load_config, project_path, require_directory, require_file, sha256_file


def expected_checksum(converted, checksum_file=None, expected_sha256=None):
    if expected_sha256:
        return expected_sha256.strip().lower()
    if checksum_file:
        data = json.loads(project_path(checksum_file).read_text(encoding="utf-8"))
        try:
            key = str(converted.resolve().relative_to(project_path(".").resolve()))
        except ValueError:
            key = str(converted.resolve())
        return data.get(key) or data.get(str(converted))
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--converted", default=None)
    parser.add_argument("--expected-sha256", default=None)
    parser.add_argument("--checksum-file", default=None)
    parser.add_argument(
        "--confirm-source-cleanup",
        action="store_true",
        help="Confirm that the verified source checkpoint may be removed",
    )
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
    expected = expected_checksum(converted, args.checksum_file, args.expected_sha256)
    actual = sha256_file(converted)
    if args.execute:
        if not args.confirm_source_cleanup:
            raise ValueError("Pass --confirm-source-cleanup before deleting the source checkpoint")
        if not expected:
            raise ValueError("Pass --expected-sha256 or --checksum-file before deleting the source checkpoint")
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Expected BF16 checksum must be a 64-character SHA-256 hex digest")
        if expected != actual:
            raise ValueError(f"BF16 checksum mismatch: expected {expected}, got {actual}")
    print(f"Source cleanup {'enabled' if args.execute else 'dry-run'}: {source}")
    print(f"Verified BF16 checksum: {actual}")
    if args.execute:
        shutil.rmtree(source)
        print(f"Removed validated source directory: {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
