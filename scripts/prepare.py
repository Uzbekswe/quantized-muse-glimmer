"""Plan or execute model downloads and write artifact checksums."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import (
    append_jsonl,
    command_string,
    load_config,
    project_path,
    run_command,
    sha256_file,
    utc_now,
    write_json,
)


def download_commands(config: dict, artifact_dir: Path, *, text_only: bool = False) -> list[list[str]]:
    model = config["model"]
    source_dir = artifact_dir / "source" / "Muse-Glimmer-30B"
    official_dir = artifact_dir / "official"
    official_files = [item["filename"] for item in config["official_baselines"]]
    if not text_only:
        official_files.extend(["mmproj-kquant.gguf", "dflash-kquant.gguf"])
    return [
        [
            "hf",
            "download",
            model["source_repo"],
            "--revision",
            model["source_revision"],
            "--local-dir",
            str(source_dir),
        ],
        [
            "hf",
            "download",
            model["official_gguf_repo"],
            *official_files,
            "--local-dir",
            str(official_dir),
        ],
    ]


def collect_checksums(root: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    if not root.exists():
        return checksums
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.name.endswith((".json", ".jsonl")):
            checksums[str(path.relative_to(project_path(".")))] = sha256_file(path)
    return checksums


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--text-only", action="store_true", help="Download only the BF16 source and text GGUF baselines")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    parser.add_argument("--execute", action="store_true", help="Actually download model artifacts")
    parser.add_argument("--verify-only", action="store_true", help="Only hash existing artifacts")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    artifact_dir = project_path(args.artifact_dir or config["paths"]["artifacts_dir"])
    if args.verify_only:
        checksums = collect_checksums(artifact_dir)
        output = write_json(config["paths"]["results_dir"] + "/checksums.json", checksums)
        print(f"Wrote {len(checksums)} checksums to {output}")
        return 0

    commands = download_commands(config, artifact_dir, text_only=args.text_only)
    print("Preparation is safe by default; pass --execute to download weights.")
    for command in commands:
        print(f"$ {command_string(command)}")
        if args.execute:
            run_command(command, execute=True)

    if args.execute:
        checksums = collect_checksums(artifact_dir)
        output = write_json(config["paths"]["results_dir"] + "/checksums.json", checksums)
        append_jsonl(
            config["paths"]["results_dir"] + "/events.jsonl",
            {"event": "prepare", "timestamp": utc_now(), "artifact_count": len(checksums)},
        )
        print(f"Wrote {len(checksums)} checksums to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
