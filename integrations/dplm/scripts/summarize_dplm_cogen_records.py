#!/usr/bin/env python3
"""Recompute the bounded DPLM CoGen-200 gate from per-sample records."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def parse_candidate(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("candidate must be LABEL=CSV")
    return label, Path(path)


def values(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} contains no records")
    tm = np.clip(np.asarray([float(row["bb_tmscore"]) for row in rows]), 0.0, 1.0)
    plddt = np.clip(np.asarray([float(row["mean_plddt"]) for row in rows]) / 100.0, 0.0, 1.0)
    return {"tm": tm, "plddt": plddt, "balanced": np.sqrt(tm * plddt)}


def independent_delta(
    baseline: np.ndarray, candidate: np.ndarray, seed: int, draws: int
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=float)
    for draw in range(draws):
        left = baseline[rng.integers(0, len(baseline), size=len(baseline))]
        right = candidate[rng.integers(0, len(candidate), size=len(candidate))]
        estimates[draw] = right.mean() - left.mean()
    return {
        "delta": float(candidate.mean() - baseline.mean()),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
    }


def verify(actual: dict, reference_path: Path, atol: float) -> None:
    reference = json.loads(reference_path.read_text())
    failures = []
    for metric, observed in actual["baseline"].items():
        expected = reference["baseline"][metric]
        if not math.isclose(observed, expected, abs_tol=atol, rel_tol=0.0):
            failures.append(f"baseline.{metric}: {observed} != {expected}")
    for label, candidate in actual["candidates"].items():
        expected_candidate = reference["candidates"][label]
        for metric, observed in candidate["means"].items():
            expected = expected_candidate["means"][metric]
            if not math.isclose(observed, expected, abs_tol=atol, rel_tol=0.0):
                failures.append(f"{label}.means.{metric}: {observed} != {expected}")
        for metric, delta in candidate["deltas"].items():
            for field, observed in delta.items():
                expected = expected_candidate["deltas"][metric][field]
                if not math.isclose(observed, expected, abs_tol=atol, rel_tol=0.0):
                    failures.append(f"{label}.{metric}.{field}: {observed} != {expected}")
        if candidate["eligible"] != expected_candidate["eligible"]:
            failures.append(f"{label}.eligible differs")
    if actual["selected"] != reference["selected"]:
        failures.append("selected differs")
    if failures:
        raise RuntimeError("Reference mismatch:\n" + "\n".join(failures[:20]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--reference-summary", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args()

    baseline = values(args.baseline)
    payload = {
        "protocol": {
            "length": 200,
            "selection_metric": "sqrt(bb_tmscore * mean_plddt/100)",
            "gate": (
                "balanced utility lower CI > 0; TM lower CI > -0.02; "
                "normalized-pLDDT lower CI > -0.02"
            ),
            "bootstrap": "independent generated-sample bootstrap",
            "bootstrap_draws": args.bootstrap,
        },
        "baseline": {metric: float(array.mean()) for metric, array in baseline.items()},
        "candidates": {},
        "selected": None,
        "confirmation_eligible": False,
    }
    ranked = []
    for index, (label, path) in enumerate(args.candidate):
        candidate = values(path)
        deltas = {
            metric: independent_delta(
                baseline[metric], candidate[metric], args.seed + 10 * index + offset, args.bootstrap
            )
            for offset, metric in enumerate(("tm", "plddt", "balanced"))
        }
        eligible = (
            deltas["balanced"]["ci_low"] > 0.0
            and deltas["tm"]["ci_low"] > -0.02
            and deltas["plddt"]["ci_low"] > -0.02
        )
        payload["candidates"][label] = {
            "record_file": str(path),
            "means": {metric: float(array.mean()) for metric, array in candidate.items()},
            "deltas": deltas,
            "eligible": eligible,
        }
        if eligible:
            ranked.append((candidate["balanced"].mean(), -index, label))
    if ranked:
        payload["selected"] = max(ranked)[2]
        payload["confirmation_eligible"] = True

    if args.reference_summary:
        verify(payload, args.reference_summary, args.atol)
        payload["reference_verified"] = str(args.reference_summary)
    text = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
