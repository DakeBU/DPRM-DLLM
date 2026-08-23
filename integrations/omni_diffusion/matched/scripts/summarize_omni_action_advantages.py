#!/usr/bin/env python3
"""Summarize paired Omni action advantages by intervention stage and rank."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def interval(values: np.ndarray, *, seed: int, resamples: int) -> tuple[float, float]:
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty action group")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()

    payload = json.loads(args.records.read_text(encoding="utf-8"))
    rows = payload.get("branches", [])
    groups: dict[tuple[int, float], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(int(row["step"]), float(row["requested_quantile"]))].append(row)
    if not groups:
        raise SystemExit(f"no action advantages in {args.records}")

    summaries = []
    for group_index, ((step, quantile), group) in enumerate(sorted(groups.items())):
        primary = np.asarray([float(row["clip_advantage"]) for row in group])
        secondary = np.asarray([float(row["clip_b32_advantage"]) for row in group])
        confidence_gap = np.asarray(
            [float(row["confidence_gap_from_default"]) for row in group]
        )
        primary_ci = interval(
            primary, seed=args.seed + 2 * group_index, resamples=args.resamples
        )
        secondary_ci = interval(
            secondary, seed=args.seed + 2 * group_index + 1, resamples=args.resamples
        )
        summaries.append(
            {
                "step": step,
                "requested_quantile": quantile,
                "n": len(group),
                "clip_cosine": {
                    "mean_delta": float(primary.mean()),
                    "ci95_low": primary_ci[0],
                    "ci95_high": primary_ci[1],
                    "wins": int((primary > 0).sum()),
                    "ties": int((primary == 0).sum()),
                    "losses": int((primary < 0).sum()),
                },
                "clip_b32_cosine": {
                    "mean_delta": float(secondary.mean()),
                    "ci95_low": secondary_ci[0],
                    "ci95_high": secondary_ci[1],
                    "wins": int((secondary > 0).sum()),
                    "ties": int((secondary == 0).sum()),
                    "losses": int((secondary < 0).sum()),
                },
                "both_metrics_improved": int(((primary > 0) & (secondary > 0)).sum()),
                "mean_confidence_gap_from_default": float(confidence_gap.mean()),
                "median_confidence_gap_from_default": float(np.median(confidence_gap)),
            }
        )
    result = {
        "design": payload.get("design"),
        "records_sha256": hashlib.sha256(args.records.read_bytes()).hexdigest(),
        "resamples": args.resamples,
        "seed": args.seed,
        "groups": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
