#!/usr/bin/env python3
"""Recompute the released SDPO-DNA summary from compressed evaluation records."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


METRICS = ("hepg2_mean", "log_lik_mean", "atac_acc", "kmer_pearson", "total_metric")


def interval(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path)
    parser.add_argument("--reference-summary", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args()

    with np.load(args.records) as data:
        n = len(data["sequences"])
        if not (len(data["hepg2"]) == len(data["atac_success"]) == len(data["log_likelihood"]) == n):
            raise ValueError("Per-sequence SDPO record arrays have inconsistent lengths")
        if data["bootstrap_indices"].shape[1] != n:
            raise ValueError("Bootstrap indices do not match the sample count")
        summary = {metric: interval(data[f"bootstrap_{metric}"]) for metric in METRICS}
        summary["n_samples"] = int(n)
        summary["bootstrap_draws"] = int(data["bootstrap_indices"].shape[0])

    if args.reference_summary:
        reference = json.loads(args.reference_summary.read_text())
        errors = []
        for metric in METRICS:
            for field in ("mean", "ci_low", "ci_high"):
                actual = summary[metric][field]
                expected = reference[metric][field]
                if not math.isclose(actual, expected, abs_tol=args.atol, rel_tol=0.0):
                    errors.append(f"{metric}.{field}: {actual} != {expected}")
        if int(reference["n_samples"]) != summary["n_samples"]:
            errors.append(f"n_samples: {summary['n_samples']} != {reference['n_samples']}")
        if errors:
            raise RuntimeError("Reference mismatch:\n" + "\n".join(errors))
        summary["reference_verified"] = str(args.reference_summary)

    payload = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
