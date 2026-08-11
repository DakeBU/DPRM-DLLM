"""Summarize DCM ordering evaluations with bootstrap uncertainty.

The evaluator writes one ``*_per_cell.csv`` file per method. This script reads
those per-cell outputs, recomputes deterministic bootstrap intervals, and emits
CSV, LaTeX, and PNG artifacts for paper reporting.
"""

import argparse
import csv
import json
import zlib
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


METRICS = {
    "token_recovery": ("Token recovery", True, 100.0),
    "mae": ("MAE", False, 1.0),
    "zero_accuracy": ("Zero accuracy", True, 100.0),
}

DEFAULT_ORDER = [
    "DCM-random",
    "Progressive-DCM",
    "DPRM-random",
    "DPRM-confidence",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--baseline", default="DCM-random")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--title", default="Dentate Gyrus DCM Ordering Evaluation")
    return parser.parse_args()


def stable_seed(base: int, *parts: str) -> int:
    payload = "::".join(str(p) for p in parts).encode("utf-8")
    return int((base + zlib.crc32(payload)) % (2**32 - 1))


def read_series(input_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    series = {}
    for csv_path in sorted(input_dir.glob("*_per_cell.csv")):
        label = csv_path.name[: -len("_per_cell.csv")]
        rows = []
        with csv_path.open() as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(row)
        if not rows:
            continue
        series[label] = {
            metric: np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
            for metric in METRICS
        }
    return series


def bootstrap(values: np.ndarray, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = values[idx].mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def summarize(series: Dict[str, Dict[str, np.ndarray]], baseline: str, n_boot: int, seed: int):
    summary = {"metrics": {}, "paired_deltas": {}, "num_cells": None}
    for label, metrics in series.items():
        summary["num_cells"] = len(next(iter(metrics.values())))
        summary["metrics"][label] = {}
        for metric, values in metrics.items():
            mean, lo, hi = bootstrap(values, n_boot, stable_seed(seed, label, metric))
            summary["metrics"][label][metric] = {
                "mean": mean,
                "ci95_low": lo,
                "ci95_high": hi,
                "higher_is_better": METRICS[metric][1],
            }
    if baseline in series:
        for label, metrics in series.items():
            if label == baseline:
                continue
            key = f"{label}_minus_{baseline}"
            summary["paired_deltas"][key] = {}
            for metric, values in metrics.items():
                delta = values - series[baseline][metric]
                mean, lo, hi = bootstrap(delta, n_boot, stable_seed(seed, "delta", label, metric))
                summary["paired_deltas"][key][metric] = {
                    "delta_mean": mean,
                    "ci95_low": lo,
                    "ci95_high": hi,
                }
    return summary


def ordered_labels(series: Dict[str, Dict[str, np.ndarray]]):
    labels = [label for label in DEFAULT_ORDER if label in series]
    labels.extend(label for label in series if label not in labels)
    return labels


def write_csv(summary, labels, output_dir: Path):
    with (output_dir / "dcm_dentate_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "metric", "mean", "ci95_low", "ci95_high", "higher_is_better"])
        for label in labels:
            for metric, stats in summary["metrics"][label].items():
                writer.writerow([label, metric, stats["mean"], stats["ci95_low"], stats["ci95_high"], stats["higher_is_better"]])
    with (output_dir / "dcm_dentate_paired_deltas.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["delta", "metric", "mean", "ci95_low", "ci95_high"])
        for key, metrics in summary["paired_deltas"].items():
            for metric, stats in metrics.items():
                writer.writerow([key, metric, stats["delta_mean"], stats["ci95_low"], stats["ci95_high"]])


def fmt(metric: str, value: float) -> str:
    scale = METRICS[metric][2]
    if scale == 100.0:
        return f"{100.0 * value:.2f}"
    return f"{value:.3f}"


def write_latex(summary, labels, output_dir: Path):
    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{Method} & \\textbf{Token recovery (\\%) $\\uparrow$} & \\textbf{MAE $\\downarrow$} & \\textbf{Zero accuracy (\\%) $\\uparrow$} \\\\",
        "\\midrule",
    ]
    for label in labels:
        cells = []
        for metric in ["token_recovery", "mae", "zero_accuracy"]:
            s = summary["metrics"][label][metric]
            cells.append(f"{fmt(metric, s['mean'])} [{fmt(metric, s['ci95_low'])}, {fmt(metric, s['ci95_high'])}]")
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (output_dir / "dcm_dentate_latex_table.tex").write_text("\n".join(lines) + "\n")


def plot(summary, labels, output_dir: Path, title: str):
    colors = {
        "DCM-random": "#4C78A8",
        "Progressive-DCM": "#F58518",
        "DPRM-random": "#54A24B",
        "DPRM-confidence": "#B279A2",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for ax, metric in zip(axes, ["token_recovery", "mae", "zero_accuracy"]):
        name, higher, scale = METRICS[metric]
        means = np.array([summary["metrics"][label][metric]["mean"] * scale for label in labels])
        lows = np.array([summary["metrics"][label][metric]["ci95_low"] * scale for label in labels])
        highs = np.array([summary["metrics"][label][metric]["ci95_high"] * scale for label in labels])
        x = np.arange(len(labels))
        yerr = np.vstack([means - lows, highs - means])
        ax.bar(x, means, yerr=yerr, capsize=4, color=[colors.get(label, "#777777") for label in labels], alpha=0.9)
        ax.set_title(f"{name} {'↑' if higher else '↓'}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        if scale == 100.0:
            ax.set_ylabel("Percent")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "dcm_dentate_metrics.png", dpi=300)
    fig.savefig(output_dir / "dcm_dentate_metrics.pdf")
    plt.close(fig)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir or args.input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    series = read_series(input_dir)
    if not series:
        raise RuntimeError(f"No *_per_cell.csv files found under {input_dir}")
    labels = ordered_labels(series)
    summary = summarize(series, args.baseline, args.bootstrap, args.seed)
    with (output_dir / "summary_stable.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    write_csv(summary, labels, output_dir)
    write_latex(summary, labels, output_dir)
    plot(summary, labels, output_dir, args.title)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
