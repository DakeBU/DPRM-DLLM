#!/usr/bin/env python3
"""Freeze disjoint Omni development and confirmation prompt sets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--seed", default="20260821-partiprompts-v1")
    parser.add_argument("--development-count", type=int, default=128)
    parser.add_argument("--confirmation-count", type=int, default=512)
    args = parser.parse_args()

    observed_sha = sha256(args.source)
    if observed_sha != args.expected_sha256:
        raise SystemExit(
            f"PartiPrompts SHA-256 mismatch: {observed_sha} != {args.expected_sha256}"
        )

    with args.source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"Prompt", "Category", "Challenge"}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit(f"unexpected PartiPrompts columns: {rows[0].keys() if rows else []}")

    records = []
    seen = set()
    for source_index, row in enumerate(rows):
        prompt = row["Prompt"].strip()
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        key = hashlib.sha256(
            f"{args.seed}\0{source_index}\0{prompt}".encode("utf-8")
        ).hexdigest()
        records.append(
            {
                "prompt_id": f"parti_{source_index:04d}",
                "source_index": source_index,
                "prompt": prompt,
                "category": row["Category"].strip(),
                "challenge": row["Challenge"].strip(),
                "split_key": key,
            }
        )
    records.sort(key=lambda row: row["split_key"])
    required_count = args.development_count + args.confirmation_count
    if len(records) < required_count:
        raise SystemExit(f"only {len(records)} unique prompts for {required_count} requested")

    split_rows = {
        "development": records[: args.development_count],
        "confirmation": records[args.development_count : required_count],
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    for split, selected in split_rows.items():
        path = args.output_root / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")

    development_text = {row["prompt"] for row in split_rows["development"]}
    confirmation_text = {row["prompt"] for row in split_rows["confirmation"]}
    manifest = {
        "schema_version": 1,
        "source": "Google Research PartiPrompts",
        "source_url": args.source_url,
        "source_sha256": observed_sha,
        "source_rows": len(rows),
        "unique_prompt_rows": len(records),
        "split_seed": args.seed,
        "selection": "ascending SHA-256 of seed, source index, and prompt text",
        "development_count": len(split_rows["development"]),
        "confirmation_count": len(split_rows["confirmation"]),
        "development_confirmation_text_overlap": len(
            development_text & confirmation_text
        ),
        "files": {
            split: {
                "path": f"{split}.jsonl",
                "sha256": sha256(args.output_root / f"{split}.jsonl"),
            }
            for split in split_rows
        },
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
