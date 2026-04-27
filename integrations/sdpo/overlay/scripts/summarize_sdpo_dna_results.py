#!/usr/bin/env python3
"""Summarize SDPO-DNA ordering experiments with bootstrap intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METHODS = [
    ("SDPO-DNA", "sdpo-dna-baseline"),
    ("Progressive-SDPO-DNA", "sdpo-dna-progressive"),
    ("DPRM-SDPO-DNA", "sdpo-dna-dprm-confidence"),
    ("DPRM(random)-SDPO-DNA", "sdpo-dna-dprm-random"),
]

METRICS = [
    ("total_metric", "Total metric", True),
    ("hepg2_mean", "HepG2", True),
    ("atac_acc", "ATAC acc.", True),
    ("kmer_pearson", "K-mer Pearson", True),
    ("log_lik_mean", "Log likelihood", True),
]


def load_result(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def collect(output_root: Path) -> pd.DataFrame:
    rows = []
    missing = []
    for label, run_name in METHODS:
        result_path = output_root / run_name / "eval_bootstrap.json"
        if not result_path.exists():
            missing.append(str(result_path))
            continue
        payload = load_result(result_path)
        row = {"method": label, "run_name": run_name, "n_samples": payload.get("n_samples")}
        for metric, _, _ in METRICS:
            stats = payload.get(metric, {})
            row[f"{metric}_mean"] = stats.get("mean")
            row[f"{metric}_ci_low"] = stats.get("ci_low")
            row[f"{metric}_ci_high"] = stats.get("ci_high")
        rows.append(row)
    if missing:
        raise FileNotFoundError("Missing SDPO-DNA eval files:\n" + "\n".join(missing))
    return pd.DataFrame(rows)


def write_latex(df: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{SDPO-DNA ordering comparison. Metrics are computed from the same number of generated DNA samples per method. Brackets are 95\% bootstrap intervals over generated samples. Higher is better for all metrics shown except where stated otherwise.}",
        r"\label{tab:sdpo_dna_ordering_results}",
        r"\small",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Total} & \textbf{HepG2} & \textbf{ATAC} & \textbf{K-mer Pearson} & \textbf{Log lik.} \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        cells = [row["method"]]
        for metric, _, _ in METRICS:
            cells.append(
                f"{row[f'{metric}_mean']:.4f} "
                f"[{row[f'{metric}_ci_low']:.4f}, {row[f'{metric}_ci_high']:.4f}]"
            )
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def plot(df: pd.DataFrame, path: Path, dpi: int) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, len(METRICS), figsize=(19, 4.2), dpi=dpi)
    colors = ["#4C78A8", "#E45756", "#9467BD", "#59A14F"]
    x = range(len(df))
    for ax, (metric, label, _) in zip(axes, METRICS):
        means = df[f"{metric}_mean"].astype(float)
        lows = df[f"{metric}_ci_low"].astype(float)
        highs = df[f"{metric}_ci_high"].astype(float)
        yerr = [means - lows, highs - means]
        ax.bar(x, means, yerr=yerr, color=colors, capsize=3, alpha=0.86)
        ax.set_title(label)
        ax.set_xticks(list(x))
        ax.set_xticklabels(df["method"], rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("data_and_model/dprm_sdpo_outputs"))
    parser.add_argument("--summary-dir", type=Path, default=Path("eval_outputs/sdpo_dna_ordering"))
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    args.summary_dir.mkdir(parents=True, exist_ok=True)
    df = collect(args.output_root)
    df.to_csv(args.summary_dir / "sdpo_dna_ordering_summary.csv", index=False)
    write_latex(df, args.summary_dir / "sdpo_dna_ordering_table.tex")
    plot(df, args.summary_dir / "sdpo_dna_ordering_metrics.png", args.dpi)
    print(df.to_string(index=False))
    print(f"Wrote SDPO-DNA summaries to {args.summary_dir}")


if __name__ == "__main__":
    main()
