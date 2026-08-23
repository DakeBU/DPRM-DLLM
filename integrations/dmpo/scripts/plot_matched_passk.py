#!/usr/bin/env python3
"""Plot matched Progressive-DMPO and DPRM pass@K from retained matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


KS = np.asarray([1, 2, 4, 8, 16, 32])
TASKS = {
    "math": {
        "levels": [1, 2, 3, 4, 5],
        "titles": ["Level 1", "Level 2", "Level 3", "Level 4 (hard)", "Level 5 (OOD)"],
    },
    "countdown": {
        "levels": [0, 1, 2, 3, 4],
        "titles": ["2 operands", "3 operands", "4 operands", "5 operands (hard)", "6 operands (OOD)"],
    },
}


def passk(matrix: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.asarray([matrix[mask, :k].any(axis=1).mean() for k in KS])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-map", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    payload = json.loads(args.source_map.read_text(encoding="utf-8"))
    sources = payload["sources"][args.task]
    arrays = {}
    levels = None
    for policy in ("confidence", "dprm_confidence"):
        root = Path(sources[policy]["path"])
        if not root.is_absolute():
            if args.artifact_root is None:
                raise ValueError("relative record paths require --artifact-root")
            root = args.artifact_root / root
        arrays[policy] = np.load(root / "success_matrix.npy").astype(bool)
        current_levels = np.load(root / "levels.npy")
        if levels is None:
            levels = current_levels
        elif not np.array_equal(levels, current_levels):
            raise ValueError("paired policies use different level arrays")

    spec = TASKS[args.task]
    fig, axes = plt.subplots(1, 5, figsize=(13.0, 2.55), sharey=True)
    styles = {
        "confidence": ("Progressive DMPO", "#4C78A8", "o"),
        "dprm_confidence": ("DMPO-DPRM", "#E45756", "s"),
    }
    for axis, level, title in zip(axes, spec["levels"], spec["titles"]):
        mask = levels == level
        if not mask.any():
            raise ValueError(f"no examples for stored level {level}")
        if "hard" in title or "OOD" in title:
            axis.set_facecolor("#F5F7FA")
        for policy, (label, color, marker) in styles.items():
            axis.plot(
                KS,
                100.0 * passk(arrays[policy], mask),
                color=color,
                marker=marker,
                linewidth=2.0,
                markersize=4.4,
                label=label,
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks(KS, [str(k) for k in KS])
        axis.set_title(f"{title}\n$n={int(mask.sum())}$", fontsize=9.2)
        axis.set_xlabel("K")
        axis.grid(axis="y", alpha=0.22, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("pass@K (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=0.8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
