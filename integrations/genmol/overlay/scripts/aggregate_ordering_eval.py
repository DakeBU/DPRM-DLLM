# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_method(spec: str) -> tuple[str, dict]:
    if "=" not in spec:
        raise ValueError(f"method spec must be METHOD=SUMMARY_JSON, got {spec}")
    method, path = spec.split("=", 1)
    with Path(path).open("r") as handle:
        return method, json.load(handle)


def interval_text(stat: dict, scale: float = 1.0, digits: int = 3) -> str:
    return f"{stat['mean'] * scale:.{digits}f} [{stat['lo'] * scale:.{digits}f}, {stat['hi'] * scale:.{digits}f}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", action="append", required=True, help="METHOD=summary.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline", default="GenMol-V2")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = dict(load_method(spec) for spec in args.method)
    with (out_dir / "comparison_summary.json").open("w") as handle:
        json.dump({"methods": methods}, handle, indent=2, sort_keys=True)

    denovo_rows = []
    for method, payload in methods.items():
        if "denovo" not in payload:
            continue
        for metric, stat in payload["denovo"]["bootstrap"].items():
            denovo_rows.append({"method": method, "metric": metric, **stat})
    denovo = pd.DataFrame(denovo_rows)
    denovo.to_csv(out_dir / "denovo_bootstrap_summary.csv", index=False)

    if len(denovo):
        metrics = ["validity", "uniqueness", "quality", "diversity"]
        fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4), constrained_layout=True)
        for ax, metric in zip(axes, metrics):
            sub = denovo[denovo["metric"] == metric]
            x = np.arange(len(sub))
            y = sub["mean"].to_numpy()
            yerr = np.vstack([y - sub["lo"].to_numpy(), sub["hi"].to_numpy() - y])
            yerr = np.nan_to_num(np.maximum(yerr, 0.0), nan=0.0)
            ax.bar(x, y, yerr=yerr, capsize=4)
            ax.set_title(metric)
            ax.set_xticks(x)
            ax.set_xticklabels(sub["method"], rotation=30, ha="right")
            ax.grid(axis="y", alpha=0.3)
        fig.savefig(out_dir / "denovo_bootstrap_metrics.png", dpi=240)
        plt.close(fig)

    fragment_rows = []
    for method, payload in methods.items():
        if "fragment" not in payload:
            continue
        for task, task_payload in payload["fragment"]["bootstrap"].items():
            for metric, stat in task_payload.items():
                fragment_rows.append({"method": method, "task": task, "metric": metric, **stat})
    fragment = pd.DataFrame(fragment_rows)
    fragment.to_csv(out_dir / "fragment_bootstrap_summary.csv", index=False)

    if len(fragment):
        for metric in ["validity", "uniqueness", "quality", "diversity", "distance"]:
            subm = fragment[fragment["metric"] == metric]
            tasks = sorted(subm["task"].unique())
            method_names = list(methods.keys())
            fig, ax = plt.subplots(figsize=(max(9, 1.1 * len(tasks) * len(method_names)), 4.8), constrained_layout=True)
            width = 0.8 / max(len(method_names), 1)
            base = np.arange(len(tasks))
            for j, method in enumerate(method_names):
                sub = subm[subm["method"] == method].set_index("task").reindex(tasks)
                y = sub["mean"].to_numpy()
                yerr = np.vstack([y - sub["lo"].to_numpy(), sub["hi"].to_numpy() - y])
                yerr = np.nan_to_num(np.maximum(yerr, 0.0), nan=0.0)
                ax.bar(base + (j - (len(method_names) - 1) / 2) * width, y, width=width, yerr=yerr, capsize=3, label=method)
            ax.set_title(f"Fragment-constrained {metric}")
            ax.set_xticks(base)
            ax.set_xticklabels(tasks, rotation=25, ha="right")
            ax.grid(axis="y", alpha=0.3)
            ax.legend(fontsize=8)
            fig.savefig(out_dir / f"fragment_{metric}_bootstrap.png", dpi=240)
            plt.close(fig)

    lines = []
    lines.append("# GenMol V2 ordering comparison\n")
    if len(denovo):
        lines.append("## De novo generation\n")
        pivot = denovo.pivot(index="method", columns="metric", values="mean")
        lines.append(pivot.to_markdown(floatfmt=".4f"))
        lines.append("\n")
    if len(fragment):
        lines.append("## Fragment-constrained generation\n")
        for metric in ["validity", "quality", "diversity", "distance"]:
            lines.append(f"### {metric}\n")
            pivot = fragment[fragment["metric"] == metric].pivot(index="method", columns="task", values="mean")
            lines.append(pivot.to_markdown(floatfmt=".4f"))
            lines.append("\n")
    (out_dir / "comparison_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
