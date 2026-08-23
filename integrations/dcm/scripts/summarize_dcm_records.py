#!/usr/bin/env python3
"""Recompute DCM means and paired bootstrap intervals from per-cell records."""

from __future__ import annotations

import argparse
import csv
import json
import math
import zlib
from pathlib import Path

import numpy as np


METRICS = (
    "token_recovery",
    "mae",
    "nonzero_recovery",
    "nonzero_mae",
    "zero_accuracy",
)
HIGHER_IS_BETTER = {
    "token_recovery": True,
    "mae": False,
    "nonzero_recovery": True,
    "nonzero_mae": False,
    "zero_accuracy": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("records_dir", type=Path)
    parser.add_argument("--baseline", default="Confidence")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--reference-summary",
        type=Path,
        help="Fail if recomputed metrics differ from this evaluator summary.",
    )
    parser.add_argument("--atol", type=float, default=1e-12)
    return parser.parse_args()


def stable_seed(base: int, *parts: str) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return int((base + zlib.crc32(payload)) % (2**32 - 1))


def read_records(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(("cell_index",) + METRICS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} contains no records")
    indices = np.asarray([int(row["cell_index"]) for row in rows], dtype=np.int64)
    values = {
        metric: np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        for metric in METRICS
    }
    return indices, values


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int, higher: bool) -> dict[str, float | bool]:
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, len(values), size=len(values))
        means[i] = values[idx].mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "higher_is_better": higher,
    }


def paired_bootstrap_delta(
    baseline: np.ndarray, candidate: np.ndarray, n_boot: int, seed: int
) -> dict[str, float]:
    diff = candidate - baseline
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, len(diff), size=len(diff))
        means[i] = diff[idx].mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {
        "delta_mean": float(diff.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }


def summarize(
    records_dir: Path,
    baseline: str,
    n_boot: int,
    seed: int,
    series_order: list[str] | None = None,
) -> dict:
    files = sorted(records_dir.glob("*_per_cell.csv"))
    if not files:
        raise FileNotFoundError(f"No *_per_cell.csv files under {records_dir}")
    series: dict[str, dict[str, np.ndarray]] = {}
    reference_indices: np.ndarray | None = None
    for path in files:
        label = path.name[: -len("_per_cell.csv")]
        indices, values = read_records(path)
        if reference_indices is None:
            reference_indices = indices
        elif not np.array_equal(indices, reference_indices):
            raise ValueError(f"Cell indices in {path} do not match the other series")
        series[label] = values
    if baseline not in series:
        raise ValueError(f"Baseline {baseline!r} is absent; found {sorted(series)}")

    if series_order is None:
        labels = [baseline] + sorted(label for label in series if label != baseline)
    else:
        labels = list(series_order)
        if set(labels) != set(series):
            raise ValueError(
                f"Series order {labels} does not match record labels {sorted(series)}"
            )
    summary = {
        "records_dir": str(records_dir),
        "num_cells": int(len(reference_indices)),
        "bootstrap": int(n_boot),
        "seed": int(seed),
        "metrics": {},
        "paired_deltas": {},
    }
    for label in labels:
        summary["metrics"][label] = {
            metric: bootstrap_ci(
                series[label][metric],
                n_boot,
                stable_seed(seed, label, metric),
                HIGHER_IS_BETTER[metric],
            )
            for metric in METRICS
        }
    for base_idx, base_label in enumerate(labels[:-1]):
        for label in labels[base_idx + 1 :]:
            summary["paired_deltas"][f"{label}_minus_{base_label}"] = {
                metric: paired_bootstrap_delta(
                    series[base_label][metric],
                    series[label][metric],
                    n_boot,
                    stable_seed(seed, "delta", label, metric),
                )
                for metric in METRICS
            }
    return summary


def verify_reference(actual: dict, reference_path: Path, atol: float) -> None:
    reference = json.loads(reference_path.read_text())
    mismatches: list[str] = []
    for section in ("metrics", "paired_deltas"):
        expected_section = reference.get(section, {})
        for label, expected_metrics in expected_section.items():
            if label not in actual[section]:
                mismatches.append(f"missing {section}.{label}")
                continue
            for metric, expected_fields in expected_metrics.items():
                if metric not in actual[section][label]:
                    mismatches.append(f"missing {section}.{label}.{metric}")
                    continue
                for field, expected in expected_fields.items():
                    if field not in actual[section][label][metric] or isinstance(expected, bool):
                        continue
                    observed = actual[section][label][metric][field]
                    if not math.isclose(float(observed), float(expected), abs_tol=atol, rel_tol=0.0):
                        mismatches.append(
                            f"{section}.{label}.{metric}.{field}: {observed} != {expected}"
                        )
    if mismatches:
        raise RuntimeError("Reference mismatch:\n" + "\n".join(mismatches[:20]))


def main() -> None:
    args = parse_args()
    reference = None
    if args.reference_summary:
        reference = json.loads(args.reference_summary.read_text())
    series_order = list(reference["metrics"]) if reference else None
    summary = summarize(
        args.records_dir,
        args.baseline,
        args.bootstrap,
        args.seed,
        series_order=series_order,
    )
    if args.reference_summary:
        verify_reference(summary, args.reference_summary, args.atol)
        summary["reference_verified"] = str(args.reference_summary)
    payload = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
