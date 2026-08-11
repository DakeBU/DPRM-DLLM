#!/usr/bin/env python3
"""Analyze reward selection against a compute-matched uniform action policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


SHORTLISTS = {
    "n2": (0.3, 0.7),
    "n4": (0.15, 0.3, 0.7, 0.85),
}
METRICS = {
    "clip_cosine": "CLIP-L/14",
    "clip_b32_cosine": "CLIP-B/32",
    "aesthetic_score": "Aesthetic",
    "terminal_utility": "Terminal utility",
}


def candidate_metric(row: dict[str, Any], metric: str) -> float:
    if metric == "clip_b32_cosine":
        return float(row[metric])
    if row["kind"] == "baseline":
        return float(row[metric])
    branch_key = {
        "clip_cosine": "branch_clip",
        "aesthetic_score": "branch_aesthetic",
        "terminal_utility": "branch_utility",
    }[metric]
    return float(row[branch_key])


def paired_interval(values: np.ndarray, rng: np.random.Generator, iters: int) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(iters, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()

    payload = json.loads(args.records.read_text(encoding="utf-8"))
    baselines = {(str(row["prompt"]), int(row["seed"])): {**row, "kind": "baseline"} for row in payload["baseline"]}
    branches: dict[tuple[str, int], dict[float, dict[str, Any]]] = {key: {} for key in baselines}
    for raw in payload["branches"]:
        key = str(raw["prompt"]), int(raw["seed"])
        if key in branches and int(raw["step"]) == 96:
            branches[key][float(raw["requested_quantile"])] = {**raw, "kind": "branch"}

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    prompt_records: list[dict[str, Any]] = []
    for name, quantiles in SHORTLISTS.items():
        per_metric = {metric: {"confidence": [], "uniform": [], "dprm": []} for metric in METRICS}
        selected_branch = []
        selected_confidence_gap = []
        for key, baseline in baselines.items():
            alternatives = [branches[key][quantile] for quantile in quantiles]
            candidates = [baseline, *alternatives]
            selected = max(candidates, key=lambda row: candidate_metric(row, "terminal_utility"))
            selected_branch.append(selected["kind"] == "branch")
            if selected["kind"] == "branch":
                selected_confidence_gap.append(float(selected["confidence_gap_from_default"]))

            record = {
                "shortlist": name,
                "prompt": key[0],
                "seed": key[1],
                "selected_kind": selected["kind"],
                "selected_quantile": selected.get("requested_quantile"),
            }
            for metric in METRICS:
                values = np.asarray([candidate_metric(row, metric) for row in candidates], dtype=np.float64)
                confidence = float(values[0])
                uniform = float(values.mean())
                dprm = candidate_metric(selected, metric)
                per_metric[metric]["confidence"].append(confidence)
                per_metric[metric]["uniform"].append(uniform)
                per_metric[metric]["dprm"].append(dprm)
                record[f"confidence_{metric}"] = confidence
                record[f"uniform_{metric}"] = uniform
                record[f"dprm_{metric}"] = dprm
            prompt_records.append(record)

        for metric, values in per_metric.items():
            arrays = {key: np.asarray(value, dtype=np.float64) for key, value in values.items()}
            for baseline_name in ("confidence", "uniform"):
                delta = arrays["dprm"] - arrays[baseline_name]
                low, high = paired_interval(delta, rng, args.bootstrap)
                rows.append(
                    {
                        "shortlist": name,
                        "alternative_actions": len(quantiles),
                        "total_rollouts": len(quantiles) + 1,
                        "metric": metric,
                        "metric_label": METRICS[metric],
                        "reference": baseline_name,
                        "reference_mean": float(arrays[baseline_name].mean()),
                        "dprm_mean": float(arrays["dprm"].mean()),
                        "mean_delta": float(delta.mean()),
                        "ci95_low": low,
                        "ci95_high": high,
                        "prompt_wins": int(np.sum(delta > 1e-12)),
                        "prompt_losses": int(np.sum(delta < -1e-12)),
                    }
                )
        rows.append(
            {
                "shortlist": name,
                "diagnostic": "selection_behavior",
                "selected_branch_rate": float(np.mean(selected_branch)),
                "mean_selected_confidence_gap": float(np.mean(selected_confidence_gap)),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "design": "same shared state and same candidate rollouts; uniform expectation is the exact mean over candidate actions; DPRM selects maximum terminal utility",
        "prompt_count": len(baselines),
        "bootstrap_iterations": args.bootstrap,
        "comparisons": rows,
        "records": prompt_records,
    }
    (args.output_dir / "compute_matched_action_control.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    summary_lines = [
        "# Omni compute-matched action control",
        "",
        "Uniform action uses the exact candidate-average score at the same 3/5-rollout budget; DPRM uses the same candidates and selects by terminal utility.",
        "",
        "| Shortlist | Metric | Reference | Reference mean | DPRM mean | Delta [95% CI] |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        if "metric" not in row:
            continue
        summary_lines.append(
            f"| {row['shortlist'].upper()} | {row['metric_label']} | {row['reference']} | "
            f"{row['reference_mean']:.5f} | {row['dprm_mean']:.5f} | "
            f"{row['mean_delta']:+.5f} [{row['ci95_low']:+.5f}, {row['ci95_high']:+.5f}] |"
        )
    (args.output_dir / "compute_matched_action_control.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    methods = ["Confidence", "Uniform-3", "DPRM-3", "Uniform-5", "DPRM-5"]
    colors = ["#3b7f3b", "#9a9a9a", "#d28b16", "#707070", "#6948a6"]
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.45))
    for axis, metric in zip(axes, ("clip_cosine", "clip_b32_cosine")):
        by_key = {(row.get("shortlist"), row.get("metric"), row.get("reference")): row for row in rows}
        n2 = by_key[("n2", metric, "uniform")]
        n4 = by_key[("n4", metric, "uniform")]
        values = [
            by_key[("n2", metric, "confidence")]["reference_mean"],
            n2["reference_mean"],
            n2["dprm_mean"],
            n4["reference_mean"],
            n4["dprm_mean"],
        ]
        x = np.arange(len(values))
        axis.bar(x, values, color=colors, width=0.72)
        axis.set_xticks(x, methods, rotation=24, ha="right")
        axis.set_ylabel(METRICS[metric] + " cosine")
        axis.set_title("Selection" if metric == "clip_cosine" else "Independent check")
        margin = max(max(values) - min(values), 0.005)
        axis.set_ylim(min(values) - 0.35 * margin, max(values) + 0.35 * margin)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", alpha=0.2, linewidth=0.6)
        for idx, value in enumerate(values):
            axis.text(idx, value + 0.035 * margin, f"{value:.3f}", ha="center", fontsize=8)
    fig.tight_layout(w_pad=1.8)
    fig.savefig(args.output_dir / "omni_compute_matched_action_control.pdf", bbox_inches="tight")
    fig.savefig(args.output_dir / "omni_compute_matched_action_control.png", dpi=220, bbox_inches="tight")

    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
