#!/usr/bin/env python3
"""Build an offline DPRM bucket table from public trace/reward files.

Trace rows are JSONL records containing at least:

  {"sample_id": "...", "phase": 0, "bucket_counts": [{"confidence_bin": 3, "aux_bin": 0, "count": 8}]}

Rewards are CSV rows containing a join key and a scalar `reward` column. The
default join key is `sample_id`; for VQA-style logs use for example
`--key-fields order_policy task doc_id`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dprm import build_bucket_table_from_trace_records


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_rewards(path: Path, key_fields: tuple[str, ...]) -> dict[str, float]:
    rewards: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = "|".join(str(row.get(field, "")) for field in key_fields)
            rewards[key] = float(row["reward"])
    return rewards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--reward-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--key-fields", nargs="+", default=["sample_id"])
    parser.add_argument("--num-phases", type=int, default=8)
    parser.add_argument("--confidence-bins", type=int, default=16)
    parser.add_argument("--aux-bins", type=int, default=1)
    parser.add_argument("--reward-temperature", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--switch-steps", type=int, default=1000)
    parser.add_argument("--ready-count", type=int, default=64)
    args = parser.parse_args()

    key_fields = tuple(args.key_fields)
    rewards = read_rewards(args.reward_csv, key_fields)
    records: list[dict[str, Any]] = []
    for path in args.trace_jsonl:
        records.extend(iter_jsonl(path))

    payload = build_bucket_table_from_trace_records(
        records,
        rewards,
        key_fields=key_fields,
        num_phases=args.num_phases,
        confidence_bins=args.confidence_bins,
        aux_bins=args.aux_bins,
        reward_temperature=args.reward_temperature,
        guidance_scale=args.guidance_scale,
        warmup_steps=args.warmup_steps,
        switch_steps=args.switch_steps,
        ready_count=args.ready_count,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["metadata"], indent=2))


if __name__ == "__main__":
    main()
