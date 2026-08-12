"""Summarize benchmark JSONL records into Markdown and CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from .common import project_path


def numeric(values):
    return [float(value) for value in values if isinstance(value, (int, float))]


def summarize(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record.get("variant", "unknown"), record.get("kind", "unknown"))].append(record)
    rows = []
    for (variant, kind), items in sorted(groups.items()):
        sizes = numeric(item.get("model_size_bytes") for item in items)
        prefill = numeric(item.get("prefill_tokens_per_second") for item in items)
        decode = numeric(item.get("decode_tokens_per_second") for item in items)
        latency = numeric(item.get("latency_ms") for item in items)
        quality = numeric(item.get("quality_metric") for item in items)
        rows.append(
            {
                "variant": variant,
                "kind": kind,
                "records": len(items),
                "size_gib": round(median(sizes) / (1024**3), 3) if sizes else None,
                "prefill_tok_s_median": round(median(prefill), 3) if prefill else None,
                "decode_tok_s_median": round(median(decode), 3) if decode else None,
                "latency_ms_median": round(median(latency), 3) if latency else None,
                "quality_metric_mean": round(mean(quality), 4) if quality else None,
                "errors": sum(bool(item.get("error")) for item in items),
            }
        )
    return rows


def write_outputs(rows, markdown_path, csv_path):
    markdown = [
        "# Benchmark summary",
        "",
        "Generated from machine-readable benchmark records. Empty cells mean the corresponding runtime metric was not parsed or was not run.",
        "",
        "| Variant | Kind | Records | Size (GiB) | Prefill tok/s | Decode tok/s | Latency (ms) | Quality metric | Errors |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            "| {variant} | {kind} | {records} | {size_gib} | {prefill_tok_s_median} | {decode_tok_s_median} | {latency_ms_median} | {quality_metric_mean} | {errors} |".format(**row)
        )
    markdown_path = project_path(markdown_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")

    csv_path = project_path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["variant"])
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/benchmarks.jsonl")
    parser.add_argument("--markdown", default="report/generated/results.md")
    parser.add_argument("--csv", default="results/summary.csv")
    args = parser.parse_args(argv)

    input_path = project_path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Benchmark input not found: {input_path}")
    records = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = summarize(records)
    write_outputs(rows, args.markdown, args.csv)
    print(f"Summarized {len(records)} records across {len(rows)} variants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
