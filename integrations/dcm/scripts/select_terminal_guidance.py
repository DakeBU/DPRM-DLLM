#!/usr/bin/env python3
"""Select DCM decode guidance from disjoint training-cell summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PREFERENCES = {
    "DPRM-recovery": (0.90, 0.075, 0.025),
    "DPRM-MAE": (0.05, 0.90, 0.05),
    "DPRM-balanced": (0.45, 0.35, 0.20),
    "DPRM-zero": (0.025, 0.075, 0.90),
}


def benefits(metrics: dict, label: str) -> tuple[float, float, float]:
    row = metrics[label]
    return (
        float(row["nonzero_recovery"]["mean"]),
        1.0 - float(row["nonzero_mae"]["mean"]) / 7.0,
        float(row["zero_accuracy"]["mean"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_root", type=Path)
    args = parser.parse_args()

    records = []
    for path in sorted(args.eval_root.glob("development_g*/summary.json")):
        payload = json.loads(path.read_text())
        confidence = benefits(payload["metrics"], "Confidence")
        gains = {}
        endpoint_utilities = []
        confidence_utilities = []
        for label, weights in PREFERENCES.items():
            value = benefits(payload["metrics"], label)
            endpoint_utility = sum(w * x for w, x in zip(weights, value))
            confidence_utility = sum(w * x for w, x in zip(weights, confidence))
            endpoint_utilities.append(endpoint_utility)
            confidence_utilities.append(confidence_utility)
            gains[label] = endpoint_utility - confidence_utility
        diagnostics = payload["controller_diagnostics"]
        order_changes = {
            label: float(
                diagnostics[label].get("decode", {}).get("order_changed_row_rate", 0.0)
            )
            for label in PREFERENCES
        }
        records.append(
            {
                "guidance": float(path.parent.name.removeprefix("development_g")),
                "mean_declared_utility_gain": (
                    sum(endpoint_utilities) - sum(confidence_utilities)
                )
                / len(endpoint_utilities),
                "positive_endpoint_count": sum(value > 0.0 for value in gains.values()),
                "endpoint_utility_gains": gains,
                "active_endpoint_count": sum(value >= 0.01 for value in order_changes.values()),
                "order_change_rates": order_changes,
            }
        )
    if not records:
        raise SystemExit(f"no development summaries under {args.eval_root}")

    selected = max(
        records, key=lambda item: (item["mean_declared_utility_gain"], -item["guidance"])
    )
    passed = (
        selected["mean_declared_utility_gain"] > 0.0
        and selected["positive_endpoint_count"] >= 3
        and selected["active_endpoint_count"] >= 3
    )
    output = {
        "criterion": "mean declared endpoint utility gain over confidence",
        "candidates": records,
        "selected": selected,
        "promotion_gate": {
            "passed": passed,
            "requirements": {
                "mean_declared_utility_gain": "> 0",
                "positive_endpoint_count": ">= 3 of 4",
                "active_endpoint_count": ">= 3 of 4 with order-change rate >= 0.01",
            },
        },
    }
    (args.eval_root / "development_selection.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    if not passed:
        raise SystemExit("development promotion gate failed")


if __name__ == "__main__":
    main()
