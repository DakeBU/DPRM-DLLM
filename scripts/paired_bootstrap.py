#!/usr/bin/env python3
"""Recompute a paired mean difference and percentile interval from raw units."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def nested(record: dict[str, Any], field: str) -> Any:
    value: Any = record
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"missing field {field!r}")
        value = value[part]
    return value


def numeric(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if text in {"true", "yes"}:
        return 1.0
    if text in {"false", "no"}:
        return 0.0
    return float(text)


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for field in ("records", "samples", "examples"):
            if isinstance(payload.get(field), list):
                return payload[field]
    raise ValueError(f"cannot find a record list in {path}")


def keyed_values(path: Path, key_field: str, value_field: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in load_rows(path):
        key = str(nested(row, key_field))
        if key in values:
            raise ValueError(f"duplicate key {key!r} in {path}")
        values[key] = numeric(nested(row, value_field))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--value", required=True)
    parser.add_argument("--baseline-value")
    parser.add_argument("--method-value")
    parser.add_argument("--direction", choices=("higher", "lower"), default="higher")
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=956)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.draws <= 0:
        raise ValueError("draws must be positive")
    baseline = keyed_values(
        args.baseline, args.key, args.baseline_value or args.value
    )
    method = keyed_values(args.method, args.key, args.method_value or args.value)
    if set(baseline) != set(method):
        raise ValueError(
            "paired inputs have different keys: "
            f"baseline_only={sorted(set(baseline) - set(method))[:5]}, "
            f"method_only={sorted(set(method) - set(baseline))[:5]}"
        )
    keys = sorted(baseline)
    if not keys:
        raise ValueError("paired inputs are empty")
    base = np.asarray([baseline[key] for key in keys], dtype=np.float64)
    treatment = np.asarray([method[key] for key in keys], dtype=np.float64)
    raw_delta = treatment - base
    benefit = raw_delta if args.direction == "higher" else -raw_delta
    rng = np.random.default_rng(args.seed)
    sampled = benefit[
        rng.integers(0, len(keys), size=(args.draws, len(keys)))
    ].mean(axis=1)
    scale = float(args.scale)
    result = {
        "n": len(keys),
        "key_field": args.key,
        "baseline_value_field": args.baseline_value or args.value,
        "method_value_field": args.method_value or args.value,
        "direction": args.direction,
        "scale": scale,
        "baseline_mean": float(base.mean() * scale),
        "method_mean": float(treatment.mean() * scale),
        "method_minus_baseline": float(raw_delta.mean() * scale),
        "benefit_delta": float(benefit.mean() * scale),
        "benefit_ci95": [
            float(np.quantile(sampled, 0.025) * scale),
            float(np.quantile(sampled, 0.975) * scale),
        ],
        "wins": int((benefit > 0).sum()),
        "ties": int((benefit == 0).sum()),
        "losses": int((benefit < 0).sum()),
        "bootstrap_draws": args.draws,
        "bootstrap_seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
