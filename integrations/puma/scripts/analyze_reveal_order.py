#!/usr/bin/env python3
"""Compute paired, content-only reveal-order diagnostics from PUMA traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


METRICS = (
    "same_step_span",
    "same_step_nonlocal_rate",
    "same_step_adjacency",
    "backfill_step_rate",
    "first_numeric_step_fraction",
)


def load_rows(paths: list[Path]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                index = int(row["index"])
                if index in rows:
                    raise ValueError(f"duplicate example index {index} in {path}")
                rows[index] = row
    return rows


def content_steps(
    row: dict[str, Any], excluded_tokens: set[str]
) -> list[tuple[int, list[tuple[int, str]]]]:
    steps = []
    for trace in row["trace_steps"]:
        selected = [
            (int(position), str(token))
            for position, token in zip(
                trace.get("selected_positions", []),
                trace.get("selected_token_texts", []),
            )
            if str(token) not in excluded_tokens
        ]
        if selected:
            steps.append((int(trace["step"]), selected))
    if not steps:
        raise ValueError(f"trace for example {row['index']} has no content actions")
    return steps


def order_metrics(row: dict[str, Any], excluded_tokens: set[str]) -> dict[str, float]:
    steps = content_steps(row, excluded_tokens)
    spans: list[float] = []
    adjacency: list[float] = []
    centroids: list[float] = []
    numeric_steps: list[int] = []
    for step, selected in steps:
        positions = sorted(position for position, _ in selected)
        span = float(max(positions) - min(positions)) if len(positions) > 1 else 0.0
        spans.append(span)
        adjacency.append(
            float(np.mean(np.diff(positions) <= 1)) if len(positions) > 1 else 0.0
        )
        centroids.append(float(np.mean(positions)))
        if any(any(character.isdigit() for character in token) for _, token in selected):
            numeric_steps.append(step)
    max_step = max(step for step, _ in steps)
    return {
        "same_step_span": float(np.mean(spans)),
        "same_step_nonlocal_rate": float(np.mean(np.asarray(spans) >= 8)),
        "same_step_adjacency": float(np.mean(adjacency)),
        "backfill_step_rate": (
            float(np.mean(np.diff(centroids) < 0)) if len(centroids) > 1 else 0.0
        ),
        "first_numeric_step_fraction": (
            float(min(numeric_steps) / max_step)
            if numeric_steps and max_step > 0
            else 1.0
        ),
    }


def bootstrap_mean(
    values: np.ndarray, rng: np.random.Generator, iterations: int
) -> list[float]:
    sampled = values[rng.integers(0, values.size, size=(iterations, values.size))]
    means = sampled.mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confidence", type=Path, nargs="+", required=True)
    parser.add_argument("--dprm", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-output", type=Path)
    parser.add_argument("--exclude-token", action="append", default=["<|endoftext|>"])
    parser.add_argument("--bootstrap-iters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()

    confidence = load_rows(args.confidence)
    dprm = load_rows(args.dprm)
    if set(confidence) != set(dprm):
        raise ValueError(
            "trace ids are not paired: "
            f"confidence-only={len(set(confidence) - set(dprm))}, "
            f"DPRM-only={len(set(dprm) - set(confidence))}"
        )

    excluded_tokens = set(args.exclude_token)
    indices = sorted(confidence)
    rng = np.random.default_rng(args.seed)
    per_example: list[dict[str, Any]] = []
    for index in indices:
        baseline_metrics = order_metrics(confidence[index], excluded_tokens)
        dprm_metrics = order_metrics(dprm[index], excluded_tokens)
        per_example.append(
            {
                "index": index,
                "confidence_correct": bool(confidence[index]["correct"]),
                "dprm_correct": bool(dprm[index]["correct"]),
                "confidence_metrics": baseline_metrics,
                "dprm_metrics": dprm_metrics,
            }
        )

    confidence_correct = np.asarray(
        [row["confidence_correct"] for row in per_example], dtype=float
    )
    dprm_correct = np.asarray([row["dprm_correct"] for row in per_example], dtype=float)
    accuracy_delta = dprm_correct - confidence_correct
    summary: dict[str, Any] = {
        "paired_examples": len(indices),
        "excluded_tokens": sorted(excluded_tokens),
        "bootstrap_iterations": args.bootstrap_iters,
        "bootstrap_seed": args.seed,
        "accuracy": {
            "confidence": float(confidence_correct.mean()),
            "dprm": float(dprm_correct.mean()),
            "mean_delta": float(accuracy_delta.mean()),
            "delta_ci95": bootstrap_mean(accuracy_delta, rng, args.bootstrap_iters),
            "dprm_only_wins": int(np.sum(accuracy_delta > 0)),
            "confidence_only_wins": int(np.sum(accuracy_delta < 0)),
        },
        "order_metrics": {},
    }
    for metric in METRICS:
        baseline = np.asarray(
            [row["confidence_metrics"][metric] for row in per_example]
        )
        method = np.asarray([row["dprm_metrics"][metric] for row in per_example])
        delta = method - baseline
        summary["order_metrics"][metric] = {
            "confidence": float(baseline.mean()),
            "dprm": float(method.mean()),
            "mean_delta": float(delta.mean()),
            "delta_ci95": bootstrap_mean(delta, rng, args.bootstrap_iters),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.case_output:
        args.case_output.parent.mkdir(parents=True, exist_ok=True)
        with args.case_output.open("w", encoding="utf-8") as handle:
            for row in per_example:
                if not row["dprm_correct"] or row["confidence_correct"]:
                    continue
                index = row["index"]
                case = {
                    **row,
                    "prompt": dprm[index].get("prompt"),
                    "gold_answer": dprm[index].get("gold_answer"),
                    "confidence_code": confidence[index].get("code"),
                    "dprm_code": dprm[index].get("code"),
                    "confidence_trace_steps": confidence[index]["trace_steps"],
                    "dprm_trace_steps": dprm[index]["trace_steps"],
                }
                handle.write(json.dumps(case) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
