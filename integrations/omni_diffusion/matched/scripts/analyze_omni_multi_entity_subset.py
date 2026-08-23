#!/usr/bin/env python3
"""Analyze a prompt-only, predeclared multi-entity Omni subset."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def _term_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![A-Za-z]){escaped}(?![A-Za-z])", re.IGNORECASE)


def classify_prompt(prompt: str, rule: dict) -> list[str]:
    triggers: list[str] = []
    for family in (
        "explicit_quantity_terms",
        "collective_terms",
        "animate_plural_terms",
    ):
        for term in rule[family]:
            if _term_pattern(term).search(prompt):
                triggers.append(f"{family}:{term}")
    return triggers


def paired_interval(
    baseline: np.ndarray,
    method: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> dict:
    delta = method - baseline
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(iterations, len(delta)))
    bootstrap = delta[indices].mean(axis=1)
    return {
        "matched_prompts": int(len(delta)),
        "baseline_mean": float(baseline.mean()),
        "method_mean": float(method.mean()),
        "mean_delta": float(delta.mean()),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "wins": int((delta > 1e-12).sum()),
        "ties": int((np.abs(delta) <= 1e-12).sum()),
        "losses": int((delta < -1e-12).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--rule", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline", default="progressive_confidence")
    parser.add_argument("--method", default="dprm_confidence_warmup")
    parser.add_argument("--bootstrap-iters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()

    records = json.loads(args.records.read_text(encoding="utf-8"))
    rule = json.loads(args.rule.read_text(encoding="utf-8"))
    by_order = {
        order: {str(row["prompt_id"]): row for row in rows}
        for order, rows in records.items()
    }
    prompt_ids = sorted(set(by_order[args.baseline]) & set(by_order[args.method]))
    selected: list[dict] = []
    for prompt_id in prompt_ids:
        base_row = by_order[args.baseline][prompt_id]
        method_row = by_order[args.method][prompt_id]
        prompt = str(base_row["prompt"])
        if prompt != str(method_row["prompt"]):
            raise ValueError(f"prompt mismatch for {prompt_id}")
        triggers = classify_prompt(prompt, rule)
        if triggers:
            selected.append(
                {"prompt_id": prompt_id, "prompt": prompt, "triggers": triggers}
            )
    if not selected:
        raise SystemExit("the prompt-only rule selected no matched examples")

    metrics: dict[str, dict] = {}
    for metric_idx, metric in enumerate(("clip_cosine", "clip_b32_cosine")):
        if not all(
            by_order[order][row["prompt_id"]].get(metric) is not None
            for order in (args.baseline, args.method)
            for row in selected
        ):
            continue
        baseline = np.asarray(
            [by_order[args.baseline][row["prompt_id"]][metric] for row in selected]
        )
        method = np.asarray(
            [by_order[args.method][row["prompt_id"]][metric] for row in selected]
        )
        metrics[metric] = paired_interval(
            baseline,
            method,
            iterations=args.bootstrap_iters,
            seed=args.seed + metric_idx,
        )

    output = {
        "role": "prompt-only development diagnostic",
        "records": str(args.records),
        "rule": str(args.rule),
        "baseline": args.baseline,
        "method": args.method,
        "selected_prompt_count": len(selected),
        "selection_fraction": len(selected) / len(prompt_ids),
        "metrics": metrics,
        "selected_prompts": selected,
        "confirmation_required": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
