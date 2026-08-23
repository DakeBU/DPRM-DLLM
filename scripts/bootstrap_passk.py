#!/usr/bin/env python3
"""Paired bootstrap for pass@K success matrices.

Each input contains one row per shared example and one Boolean column per
decoded sample. Supported formats are NPY, NPZ, JSON, and JSONL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def nested_value(payload, field: str):
    value = payload
    for part in field.split("."):
        value = value[part]
    return value


def load_matrix(path: Path, field: str) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        value = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        key = field if field in archive.files else archive.files[0]
        value = archive[key]
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = nested_value(payload, field) if isinstance(payload, dict) else payload
    elif suffix == ".jsonl":
        rows = [
            nested_value(json.loads(line), field)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        value = rows
    else:
        raise ValueError(f"unsupported matrix format: {path}")
    matrix = np.asarray(value, dtype=bool)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"expected a nonempty 2D success matrix in {path}")
    return matrix


def per_example_mean_passk(matrix: np.ndarray, ks: list[int]) -> np.ndarray:
    if max(ks) > matrix.shape[1]:
        raise ValueError(
            f"requested K={max(ks)} but matrix has only {matrix.shape[1]} samples"
        )
    cumulative = np.maximum.accumulate(matrix, axis=1)
    return np.stack([cumulative[:, k - 1] for k in ks], axis=1).mean(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--baseline-field", default="successes")
    parser.add_argument("--method-field", default="successes")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--scale", type=float, default=100.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ks = sorted(set(args.ks))
    if not ks or min(ks) < 1 or args.bootstrap < 1:
        raise SystemExit("K values and bootstrap count must be positive")
    baseline = load_matrix(args.baseline, args.baseline_field)
    method = load_matrix(args.method, args.method_field)
    if baseline.shape != method.shape:
        raise SystemExit(
            f"paired matrices differ in shape: {baseline.shape} != {method.shape}"
        )

    baseline_stat = per_example_mean_passk(baseline, ks)
    method_stat = per_example_mean_passk(method, ks)
    delta = method_stat - baseline_stat
    rng = np.random.default_rng(args.seed)
    draws = np.empty(args.bootstrap, dtype=np.float64)
    n = delta.shape[0]
    for draw in range(args.bootstrap):
        indices = rng.integers(0, n, size=n)
        draws[draw] = delta[indices].mean()

    payload = {
        "n": int(n),
        "samples_per_example": int(baseline.shape[1]),
        "k_values": ks,
        "statistic": "arithmetic mean of per-example pass@K",
        "scale": args.scale,
        "baseline_mean": float(baseline_stat.mean() * args.scale),
        "method_mean": float(method_stat.mean() * args.scale),
        "paired_delta": float(delta.mean() * args.scale),
        "ci95": [
            float(np.quantile(draws, 0.025) * args.scale),
            float(np.quantile(draws, 0.975) * args.scale),
        ],
        "bootstrap_draws": args.bootstrap,
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
