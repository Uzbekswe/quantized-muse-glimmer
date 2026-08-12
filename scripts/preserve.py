"""Copy a validated model to persistent storage and verify its checksum."""

from __future__ import annotations

import argparse
import shutil

from .common import project_path, require_file, sha256_file, utc_now, write_json


def preserve(source, destination):
    source = require_file(source, "source model")
    destination = project_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing destination: {destination}")
    partial = destination.with_name(destination.name + ".part")
    if partial.exists():
        partial.unlink()
    shutil.copy2(source, partial)
    source_hash = sha256_file(source)
    destination_hash = sha256_file(partial)
    if source_hash != destination_hash:
        partial.unlink(missing_ok=True)
        raise IOError("Persistent copy checksum does not match source")
    partial.replace(destination)
    return {
        "timestamp": utc_now(),
        "source": str(source),
        "destination": str(destination),
        "size_bytes": destination.stat().st_size,
        "sha256": destination_hash,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--output", default="results/preservation.json")
    parser.add_argument("--execute", action="store_true", help="Perform the large copy")
    args = parser.parse_args(argv)

    source = project_path(args.source)
    destination = project_path(args.destination)
    print(f"Preservation {'enabled' if args.execute else 'dry-run'}: {source} -> {destination}")
    if not args.execute:
        return 0
    record = preserve(source, destination)
    output = write_json(args.output, record)
    print(f"Verified persistent copy: {record['sha256']}")
    print(f"Wrote preservation record to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
