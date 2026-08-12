#!/usr/bin/env python3
"""Replace DCM and GenMol rows from the scientific native-value artifact."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = [
    "host", "task", "method", "variant", "metric", "direction",
    "value", "ci95_low", "ci95_high", "n", "protocol",
]

METHODS = {
    "DCM": {
        "confidence": ("Confidence", "confidence"),
        "recovery": ("DPRM-recovery", "dprm_recovery"),
        "mae": ("DPRM-MAE", "dprm_mae"),
        "balanced": ("DPRM-balanced", "dprm_balanced"),
        "zero": ("DPRM-zero", "dprm_zero"),
    },
    "GenMol": {
        "confidence": ("Confidence", "confidence"),
        "qed": ("DPRM-QED", "dprm_qed"),
        "balanced": ("DPRM-balanced", "dprm_balanced"),
        "sa": ("DPRM-SA", "dprm_sa"),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--native",
        type=Path,
        default=Path("results/artifacts/scientific_preference_native_values.csv"),
    )
    parser.add_argument(
        "--results", type=Path, default=Path("results/paper_results.csv")
    )
    args = parser.parse_args()

    with args.results.open(newline="", encoding="utf-8") as handle:
        existing = list(csv.DictReader(handle))
    with args.native.open(newline="", encoding="utf-8") as handle:
        native = list(csv.DictReader(handle))

    retained = [
        row for row in existing if row["host"] not in {"DCM", "GenMol V2"}
    ]
    generated = []
    for row in native:
        source_host = row["host"]
        if source_host not in METHODS:
            continue
        method, variant = METHODS[source_host][row["preference"]]
        if source_host == "DCM":
            host, task, n, protocol = (
                "DCM", "Dentate Gyrus", "293", "fixed_model_preference"
            )
        else:
            host, task, n, protocol = (
                "GenMol V2", "De novo", "1000", "declared_preference_sweep"
            )
        generated.append(
            {
                "host": host,
                "task": task,
                "method": method,
                "variant": variant,
                "metric": row["metric"],
                "direction": "higher",
                "value": row["mean"],
                "ci95_low": row["ci95_low"],
                "ci95_high": row["ci95_high"],
                "n": n,
                "protocol": protocol,
            }
        )

    if len(generated) != 35:
        raise ValueError(f"expected 35 scientific result rows, found {len(generated)}")
    with args.results.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(retained + generated)
    print(f"wrote {len(retained) + len(generated)} rows to {args.results}")


if __name__ == "__main__":
    main()
