#!/usr/bin/env python3
"""Collect matched confidence and forced-action Omni outputs for CLIP scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path, method: str) -> dict:
    row = json.loads(path.read_text(encoding="utf-8"))
    row["method"] = method
    row["prompt_id"] = path.parent.name
    row["json_path"] = str(path)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records: dict[str, list[dict]] = {"confidence": []}
    for path in sorted(args.root.glob("baseline/prompt_*/*.json")):
        records["confidence"].append(load(path, "confidence"))
    random_paths = sorted(args.root.glob("random/prompt_*/*.json"))
    if random_paths:
        records["random"] = [load(path, "random") for path in random_paths]
    for branch_root in sorted((args.root / "branches").glob("step*_q*")):
        method = branch_root.name
        records[method] = [
            load(path, method) for path in sorted(branch_root.glob("prompt_*/*.json"))
        ]
    if not records["confidence"] or len(records) < 2:
        raise SystemExit(f"incomplete action rollout tree under {args.root}")
    expected = len(records["confidence"])
    if any(len(rows) != expected for rows in records.values()):
        raise SystemExit({key: len(rows) for key, rows in records.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: len(rows) for key, rows in records.items()}, indent=2))


if __name__ == "__main__":
    main()
