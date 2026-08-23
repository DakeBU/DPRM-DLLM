#!/usr/bin/env python3
"""Analyze frozen prompt-only subsets for the Omni online DPRM controller."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_omni_multi_entity_subset import classify_prompt


def summarize(rows: list[dict], metric: str, *, seed: int, resamples: int) -> dict:
    delta = np.asarray(
        [float(row[f"dprm_{metric}"]) - float(row[f"confidence_{metric}"]) for row in rows]
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(resamples, len(delta)))
    means = delta[indices].mean(axis=1)
    return {
        "n": len(rows),
        "confidence_mean": float(np.mean([row[f"confidence_{metric}"] for row in rows])),
        "dprm_mean": float(np.mean([row[f"dprm_{metric}"] for row in rows])),
        "mean_delta": float(delta.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "wins": int(np.sum(delta > 1e-12)),
        "ties": int(np.sum(np.abs(delta) <= 1e-12)),
        "losses": int(np.sum(delta < -1e-12)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--rule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    prompt_rows = [
        json.loads(line)
        for line in args.prompt_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    prompt_ids = {str(row["prompt"]): str(row.get("prompt_id", index)) for index, row in enumerate(prompt_rows)}
    rule = json.loads(args.rule.read_text(encoding="utf-8"))
    selected, complement = [], []
    for row in summary["records"]:
        prompt = str(row["prompt"])
        if prompt not in prompt_ids:
            raise ValueError("controller summary contains a prompt outside the frozen split")
        item = {**row, "prompt_id": prompt_ids[prompt], "triggers": classify_prompt(prompt, rule)}
        (selected if item["triggers"] else complement).append(item)
    if not selected or not complement:
        raise SystemExit("prompt rule must produce nonempty selected and complement subsets")

    groups = {}
    for group_index, (name, rows) in enumerate((("multi_entity", selected), ("other", complement))):
        groups[name] = {
            "prompt_count": len(rows),
            "override_fraction": float(np.mean([row["selected_index"] != 0 for row in rows])),
            "selected_action_counts": dict(Counter(row["selected_method"] for row in rows)),
            "metrics": {
                metric: summarize(
                    rows,
                    metric,
                    seed=args.seed + 2 * group_index + metric_index,
                    resamples=args.bootstrap,
                )
                for metric_index, metric in enumerate(("clip_cosine", "clip_b32_cosine"))
            },
        }
    output = {
        "format": "omni_online_prompt_subset_diagnostic_v1",
        "role": "frozen prompt-only confirmation diagnostic",
        "rule_sha256": hashlib.sha256(args.rule.read_bytes()).hexdigest(),
        "prompt_file_sha256": hashlib.sha256(args.prompt_file.read_bytes()).hexdigest(),
        "groups": groups,
        "selected_prompt_ids": [row["prompt_id"] for row in selected],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
