#!/usr/bin/env python3
"""Merge disjoint Omni evaluation-record shards with overlap checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--orders",
        nargs="+",
        help="Explicit method subset to retain from every input shard.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    merged: dict[str, list[dict]] = {}
    seen: dict[str, dict[str, str]] = {}
    expected_orders: set[str] | None = None
    for path in args.inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if args.orders:
            missing = set(args.orders) - set(payload)
            if missing:
                raise ValueError(f"missing orders in {path}: {sorted(missing)}")
            payload = {order: payload[order] for order in args.orders}
        orders = set(payload)
        if expected_orders is None:
            expected_orders = orders
        elif orders != expected_orders:
            raise ValueError(
                f"order mismatch in {path}: {sorted(orders)} != "
                f"{sorted(expected_orders)}"
            )
        for order, rows in payload.items():
            merged.setdefault(order, [])
            seen.setdefault(order, {})
            for raw in rows:
                row = dict(raw)
                prompt_id = str(row["prompt_id"])
                prompt = str(row.get("prompt", "")).strip()
                if prompt_id in seen[order]:
                    raise ValueError(f"duplicate {order}/{prompt_id} across shards")
                seen[order][prompt_id] = prompt
                merged[order].append(row)

    for order in merged:
        merged[order].sort(key=lambda row: str(row["prompt_id"]))
    reference_order = sorted(merged)[0]
    reference = seen[reference_order]
    for order, prompts in seen.items():
        if set(prompts) != set(reference):
            raise ValueError(f"prompt-id mismatch between {reference_order} and {order}")
        mismatched = [key for key in reference if prompts[key] != reference[key]]
        if mismatched:
            raise ValueError(
                f"prompt-text mismatch between {reference_order} and {order}: "
                f"{mismatched[:3]}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "inputs": [str(path) for path in args.inputs],
                "orders": sorted(merged),
                "prompts_per_order": len(reference),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
