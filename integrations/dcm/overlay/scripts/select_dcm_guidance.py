#!/usr/bin/env python3
"""Select DCM guidance on development cells with a prespecified utility gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap_delta(diff: np.ndarray, seed: int, n_boot: int) -> dict:
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        sample = rng.integers(0, len(diff), size=len(diff))
        means[index] = diff[sample].mean()
    lo, hi = np.quantile(means, (0.025, 0.975))
    return {"mean": float(diff.mean()), "ci95_low": float(lo), "ci95_high": float(hi)}


def utility(frame: pd.DataFrame, max_bin_distance: float) -> np.ndarray:
    mae_benefit = 1.0 - frame.mae.to_numpy(dtype=float) / max_bin_distance
    return (
        0.45 * frame.token_recovery.to_numpy(dtype=float)
        + 0.35 * mae_benefit
        + 0.20 * frame.zero_accuracy.to_numpy(dtype=float)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--max-bin-distance", type=float, default=7.0)
    args = parser.parse_args()

    root = Path(args.root)
    baseline = pd.read_csv(root / "baseline" / "Progressive-DCM_per_cell.csv")
    baseline_utility = utility(baseline, args.max_bin_distance)
    candidates = []
    for summary_path in sorted(root.glob("*_g*/summary.json")):
        payload = json.loads(summary_path.read_text())
        variant_dir = summary_path.parent
        method = next(iter(payload["metrics"]))
        frame = pd.read_csv(variant_dir / f"{method}_per_cell.csv")
        diag = payload["controller_diagnostics"][method]["decode"]
        method_utility = utility(frame, args.max_bin_distance)
        seed = 20260811 + len(candidates) * 101
        row = {
            "method": method,
            "variant_dir": str(variant_dir),
            "checkpoint": payload["controller_diagnostics"][method].get("checkpoint"),
            "guidance": float(payload["guidance_scale_override"]),
            "order_changed_row_rate": float(diag["order_changed_row_rate"]),
            "selected_ready_rate": float(diag["selected_ready_rate"]),
            "selected_mean_abs_score_delta": float(diag["selected_mean_abs_score_delta"]),
            "utility": bootstrap_delta(method_utility - baseline_utility, seed, args.bootstrap),
            "token_recovery": bootstrap_delta(
                frame.token_recovery.to_numpy() - baseline.token_recovery.to_numpy(), seed + 1, args.bootstrap
            ),
            "mae_benefit": bootstrap_delta(
                (baseline.mae.to_numpy() - frame.mae.to_numpy()) / args.max_bin_distance,
                seed + 2,
                args.bootstrap,
            ),
            "zero_accuracy": bootstrap_delta(
                frame.zero_accuracy.to_numpy() - baseline.zero_accuracy.to_numpy(), seed + 3, args.bootstrap
            ),
        }
        row["passes_development_gate"] = bool(
            row["order_changed_row_rate"] >= 0.01
            and row["selected_ready_rate"] >= 0.95
            and row["utility"]["ci95_low"] > 0.0
            and row["token_recovery"]["ci95_high"] >= 0.0
            and row["mae_benefit"]["ci95_high"] >= 0.0
            and row["zero_accuracy"]["ci95_high"] >= 0.0
        )
        candidates.append(row)

    if not candidates:
        raise RuntimeError(f"no sweep summaries found under {root}")
    passing = [row for row in candidates if row["passes_development_gate"]]
    ranked = sorted(
        passing or candidates,
        key=lambda row: (row["passes_development_gate"], row["utility"]["mean"]),
        reverse=True,
    )
    selected = ranked[0]
    report = {
        "selection_split": "first 128 validation cells",
        "weights": [0.45, 0.35, 0.20],
        "mae_benefit": f"1 - MAE/{args.max_bin_distance:g}",
        "gate": (
            "order-change >= 0.01, selected-ready >= 0.95, utility paired CI lower bound > 0, "
            "and no component has a paired CI wholly below zero"
        ),
        "selected": selected,
        "confirmation_allowed": bool(selected["passes_development_gate"]),
        "candidates": candidates,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    pd.json_normalize(candidates, sep=".").to_csv(output.with_suffix(".csv"), index=False)
    env_path = output.with_suffix(".env")
    env_path.write_text(
        f"SELECTED_METHOD='{selected['method']}'\n"
        f"SELECTED_GUIDANCE='{selected['guidance']}'\n"
        f"CONFIRMATION_ALLOWED='{int(selected['passes_development_gate'])}'\n"
    )


if __name__ == "__main__":
    main()
