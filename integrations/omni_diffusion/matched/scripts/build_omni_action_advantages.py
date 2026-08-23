#!/usr/bin/env python3
"""Convert scored forced-action continuations into paired DPRM rewards."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary-weight", type=float, default=0.5)
    parser.add_argument("--secondary-weight", type=float, default=0.5)
    args = parser.parse_args()

    payload = json.loads(args.records.read_text(encoding="utf-8"))
    baseline = {
        (str(row["prompt"]), int(row["seed"])): row
        for row in payload.get("confidence", [])
    }
    branches = [row for method, rows in payload.items() if method != "confidence" for row in rows]
    raw = []
    for row in branches:
        key = str(row["prompt"]), int(row["seed"])
        if key not in baseline:
            raise ValueError(f"missing confidence baseline for seed {row['seed']}")
        base = baseline[key]
        force = dict(row.get("counterfactual_override") or {})
        if not force.get("applied"):
            continue
        primary = float(row["clip_cosine"]) - float(base["clip_cosine"])
        secondary = float(row["clip_b32_cosine"]) - float(base["clip_b32_cosine"])
        raw.append((row, force, primary, secondary))
    if not raw:
        raise SystemExit("no applied forced actions with matched scores")
    primary_scale = max(statistics.pstdev(value[2] for value in raw), 1e-6)
    secondary_scale = max(statistics.pstdev(value[3] for value in raw), 1e-6)
    output = []
    for row, force, primary, secondary in raw:
        reward = (
            args.primary_weight * primary / primary_scale
            + args.secondary_weight * secondary / secondary_scale
        )
        output.append(
            {
                **force,
                "prompt": row["prompt"],
                "seed": row["seed"],
                "clip_advantage": primary,
                "clip_b32_advantage": secondary,
                "advantage": reward,
            }
        )
    result = {
        "design": "paired action-conditioned continuation rewards",
        "primary_weight": args.primary_weight,
        "secondary_weight": args.secondary_weight,
        "primary_advantage_scale": primary_scale,
        "secondary_advantage_scale": secondary_scale,
        "branches": output,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "branches"}, indent=2))
    print(json.dumps({"branches": len(output)}, indent=2))


if __name__ == "__main__":
    main()
