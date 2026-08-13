#!/usr/bin/env python3
"""Build the archived Omni DPRM-BoN mechanism diagnostic figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = ["random", "progressive_confidence", "dprm_softbon_n2", "dprm_softbon_n4"]
LABELS = ["Random", "Confidence", "DPRM-BoN-2", "DPRM-BoN-4"]
COLORS = ["#777777", "#2f7d32", "#d28b16", "#6846a5"]


def comparison(rows: list[dict], method: str, metric: str) -> dict:
    return next(
        row
        for row in rows
        if row["baseline"] == "progressive_confidence"
        and row["method"] == method
        and row["metric"] == metric
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-summary", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paired = json.loads(args.paired_summary.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    records = json.loads(Path(paired["records"]).read_text(encoding="utf-8"))
    reports = {row["method"]: row for row in selection["reports"]}

    means = {
        metric: [
            float(np.mean([float(row[metric]) for row in records[method]]))
            for method in METHODS
        ]
        for metric in ("clip_cosine", "clip_b32_cosine")
    }

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.55))
    x = np.arange(len(METHODS))
    axes[0].bar(x, means["clip_cosine"], color=COLORS, width=0.72)
    axes[0].set_xticks(x, LABELS, rotation=18, ha="right")
    axes[0].set_ylabel("CLIP-L/14 cosine")
    axes[0].set_ylim(0.25, 0.295)
    axes[0].set_title("(a) Formal 96-prompt score")
    for idx, value in enumerate(means["clip_cosine"]):
        axes[0].text(idx, value + 0.001, f"{value:.3f}", ha="center", fontsize=9)

    metrics = [("clip_cosine", "CLIP-L/14"), ("clip_b32_cosine", "CLIP-B/32")]
    offsets = [-0.12, 0.12]
    for offset, (method, label, color) in zip(
        offsets,
        [("dprm_softbon_n2", "DPRM-BoN-2", COLORS[2]), ("dprm_softbon_n4", "DPRM-BoN-4", COLORS[3])],
    ):
        rows = [comparison(paired["comparisons_by_metric"][metric], method, metric) for metric, _ in metrics]
        delta = np.array([row["mean_delta"] for row in rows])
        low = np.array([row["ci95_low"] for row in rows])
        high = np.array([row["ci95_high"] for row in rows])
        axes[1].errorbar(
            np.arange(len(metrics)) + offset,
            delta,
            yerr=np.vstack([delta - low, high - delta]),
            fmt="o",
            color=color,
            capsize=4,
            label=label,
        )
    axes[1].axhline(0, color="#444444", linewidth=1)
    axes[1].set_xticks(np.arange(len(metrics)), [label for _, label in metrics])
    axes[1].set_ylabel("Paired gain over confidence")
    axes[1].set_title("(b) Independent encoder check")
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")

    branch_rates = [reports[method]["selected_branch_rate"] for method in METHODS[2:]]
    confidence_gaps = [reports[method]["mean_selected_confidence_gap"] for method in METHODS[2:]]
    mech_x = np.arange(2)
    axes[2].bar(mech_x, branch_rates, color=COLORS[2:], width=0.62)
    axes[2].set_xticks(mech_x, LABELS[2:], rotation=18, ha="right")
    axes[2].set_ylim(0, 1.0)
    axes[2].set_ylabel("Non-confidence action rate")
    axes[2].set_title("(c) Confidence is rejected")
    for idx, (rate, gap) in enumerate(zip(branch_rates, confidence_gaps)):
        axes[2].text(idx, rate + 0.025, f"{100 * rate:.1f}%", ha="center", fontsize=9)
        axes[2].text(idx, 0.07, f"mean conf. gap\n{gap:.3f}", ha="center", fontsize=8, color="white")

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", alpha=0.2, linewidth=0.6)
    fig.tight_layout(w_pad=2.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved {args.output} and {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
