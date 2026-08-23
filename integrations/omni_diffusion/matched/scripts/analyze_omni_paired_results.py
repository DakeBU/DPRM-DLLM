#!/usr/bin/env python3
"""Compute paired uncertainty for the formal matched Omni evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


COMPARISONS = (
    ("random", "progressive_confidence"),
    ("random", "dprm_confidence_warmup"),
    ("progressive_confidence", "dprm_confidence_warmup"),
)


def parse_comparisons(values: list[str] | None) -> tuple[tuple[str, str], ...]:
    if not values:
        return COMPARISONS
    parsed: list[tuple[str, str]] = []
    for value in values:
        parts = value.split(":", maxsplit=1)
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                f"invalid comparison {value!r}; expected BASELINE:METHOD"
            )
        parsed.append((parts[0], parts[1]))
    return tuple(parsed)


def load_scores(records_path: Path, metric: str) -> dict[str, dict[str, float]]:
    payload = json.loads(records_path.read_text(encoding="utf-8"))
    scores: dict[str, dict[str, float]] = {}
    for order, records in payload.items():
        scores[order] = {
            str(record["prompt_id"]): float(record[metric])
            for record in records
            if record.get(metric) is not None
        }
    return scores


def paired_summary(
    baseline: str,
    method: str,
    scores: dict[str, dict[str, float]],
    *,
    bootstrap_iters: int,
    seed: int,
) -> tuple[dict, list[dict]]:
    prompt_ids = sorted(set(scores[baseline]) & set(scores[method]))
    if not prompt_ids:
        raise ValueError(f"no matched prompts for {baseline} and {method}")
    base = np.asarray([scores[baseline][prompt_id] for prompt_id in prompt_ids])
    candidate = np.asarray([scores[method][prompt_id] for prompt_id in prompt_ids])
    delta = candidate - base
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(bootstrap_iters, len(delta)))
    bootstrap = delta[indices].mean(axis=1)
    tolerance = 1e-12
    summary = {
        "baseline": baseline,
        "method": method,
        "matched_prompts": len(prompt_ids),
        "baseline_mean": float(base.mean()),
        "method_mean": float(candidate.mean()),
        "mean_delta": float(delta.mean()),
        "median_delta": float(np.median(delta)),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "wins": int((delta > tolerance).sum()),
        "ties": int((np.abs(delta) <= tolerance).sum()),
        "losses": int((delta < -tolerance).sum()),
    }
    rows = [
        {
            "baseline": baseline,
            "method": method,
            "prompt_id": prompt_id,
            "baseline_clip": float(base[idx]),
            "method_clip": float(candidate[idx]),
            "delta": float(delta[idx]),
        }
        for idx, prompt_id in enumerate(prompt_ids)
    ]
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-iters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument(
        "--comparisons",
        nargs="+",
        default=None,
        metavar="BASELINE:METHOD",
        help="Paired comparisons to compute; defaults to the formal three-order set.",
    )
    args = parser.parse_args()
    comparisons = parse_comparisons(args.comparisons)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries_by_metric: dict[str, list[dict]] = {}
    rows: list[dict] = []
    payload = json.loads(args.records.read_text(encoding="utf-8"))
    candidate_metrics = ("clip_cosine", "clip_b32_cosine")
    metrics = [
        metric
        for metric in candidate_metrics
        if any(
            record.get(metric) is not None
            for records in payload.values()
            for record in records
        )
    ]
    for metric_idx, metric in enumerate(metrics):
        scores = load_scores(args.records, metric)
        summaries: list[dict] = []
        for comparison_idx, (baseline, method) in enumerate(comparisons):
            if baseline not in scores or method not in scores:
                continue
            summary, comparison_rows = paired_summary(
                baseline,
                method,
                scores,
                bootstrap_iters=args.bootstrap_iters,
                seed=args.seed + metric_idx * 100 + comparison_idx,
            )
            summary["metric"] = metric
            for row in comparison_rows:
                row["metric"] = metric
            summaries.append(summary)
            rows.extend(comparison_rows)
        summaries_by_metric[metric] = summaries

    payload = {
        "records": str(args.records),
        "bootstrap_iters": args.bootstrap_iters,
        "seed": args.seed,
        "comparisons": summaries_by_metric.get("clip_cosine", []),
        "comparisons_by_metric": summaries_by_metric,
    }
    (args.output_dir / "paired_clip_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "paired_clip_per_prompt.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Omni paired CLIP analysis",
        "",
        f"Matched prompt bootstrap with {args.bootstrap_iters:,} resamples.",
        "",
        "| Baseline | Method | N | Baseline | Method | Delta [95% CI] | W/T/L |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for metric, summaries in summaries_by_metric.items():
        lines.extend(["", f"## {metric}", ""])
        for item in summaries:
            lines.append(
                f"| `{item['baseline']}` | `{item['method']}` | {item['matched_prompts']} "
                f"| {item['baseline_mean']:.5f} | {item['method_mean']:.5f} "
                f"| {item['mean_delta']:+.5f} [{item['ci95_low']:+.5f}, {item['ci95_high']:+.5f}] "
                f"| {item['wins']}/{item['ties']}/{item['losses']} |"
            )
    (args.output_dir / "paired_clip_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
