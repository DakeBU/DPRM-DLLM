#!/usr/bin/env python3
"""Create a deterministic JourneyDB training source with evaluation prompts removed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def normalized_prompt(row: dict) -> str:
    content = row["messages"][0]["content"].strip()
    lines = content.splitlines()
    return "\n".join(lines[1:]).strip() if len(lines) > 1 else content


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--forbidden-prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument(
        "--unique-offset",
        type=int,
        default=0,
        help="Skip this many deduplicated prompts before collecting training rows.",
    )
    args = parser.parse_args()
    forbidden = {
        line.strip()
        for line in args.forbidden_prompts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    seen: set[str] = set()
    kept = 0
    skipped_forbidden = 0
    skipped_duplicate = 0
    unique_seen = 0
    with args.source.open(encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as target:
        for source_index, line in enumerate(source):
            row = json.loads(line)
            prompt = normalized_prompt(row)
            if prompt in seen:
                skipped_duplicate += 1
                continue
            seen.add(prompt)
            if unique_seen < args.unique_offset:
                unique_seen += 1
                continue
            unique_seen += 1
            if prompt in forbidden:
                skipped_forbidden += 1
                continue
            row["dprm_source_index"] = source_index
            target.write(json.dumps(row) + "\n")
            kept += 1
            if kept >= args.count:
                break
    if kept != args.count:
        raise RuntimeError(f"requested {args.count} eligible rows, found {kept}")
    print(
        json.dumps(
            {
                "source": str(args.source),
                "output": str(args.output),
                "kept": kept,
                "skipped_forbidden": skipped_forbidden,
                "skipped_duplicate": skipped_duplicate,
                "unique_offset": args.unique_offset,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
