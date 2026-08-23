#!/usr/bin/env python3
"""Select a sparse Omni DPRM controller on a declared development split."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def intervention_stats(root: Path, label: str) -> dict[str, float]:
    overrides_per_prompt: list[int] = []
    for trace in sorted((root / label).glob("prompt_*/*_order_trace.jsonl")):
        overrides = 0
        with trace.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                selected = row.get("selected_candidate_indices", [])
                default = row.get("confidence_default_candidate_index")
                if len(selected) == 1 and default is not None:
                    overrides += int(int(selected[0]) != int(default))
        overrides_per_prompt.append(overrides)
    if not overrides_per_prompt:
        raise ValueError(f"no order traces for {label}")
    count = len(overrides_per_prompt)
    return {
        "traced_prompts": float(count),
        "mean_direct_overrides": sum(overrides_per_prompt) / count,
        "prompt_fraction_with_override": sum(
            value > 0 for value in overrides_per_prompt
        )
        / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--candidate-map", required=True, type=Path)
    parser.add_argument("--selected-output", required=True, type=Path)
    parser.add_argument("--decision-output", required=True, type=Path)
    parser.add_argument("--min-prompt-override-fraction", type=float, default=0.20)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    candidates: dict[str, dict] = {}
    with args.candidate_map.open(encoding="utf-8") as handle:
        for line in handle:
            label, path = line.rstrip("\n").split("\t", maxsplit=1)
            metrics = summary["methods"][label]
            activation = intervention_stats(args.root, label)
            passed = (
                float(metrics["mean_delta_vs_confidence"]) > 0.0
                and activation["prompt_fraction_with_override"]
                >= args.min_prompt_override_fraction
                and activation["mean_direct_overrides"] > 0.0
            )
            candidates[label] = {
                "path": path,
                "metrics": metrics,
                "interventions": activation,
                "passed": passed,
            }

    eligible = [
        (float(row["metrics"]["mean_delta_vs_confidence"]), label)
        for label, row in candidates.items()
        if row["passed"]
    ]
    selected = max(eligible)[1] if eligible else None
    if selected is not None:
        args.selected_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidates[selected]["path"], args.selected_output)
    decision = {
        "design": "disjoint development selection before matched continuation training",
        "passed": selected is not None,
        "selection_metric": "mean paired CLIP-L/14 delta versus Omni confidence",
        "min_prompt_override_fraction": args.min_prompt_override_fraction,
        "selected": selected,
        "candidates": candidates,
    }
    args.decision_output.parent.mkdir(parents=True, exist_ok=True)
    args.decision_output.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))
    raise SystemExit(0 if selected is not None else 2)


if __name__ == "__main__":
    main()
