#!/usr/bin/env python3
"""Select an Omni controller using paired results from two CLIP encoders."""

from __future__ import annotations

import argparse
import json
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
        "prompt_fraction_with_override": sum(value > 0 for value in overrides_per_prompt)
        / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--paired-summary", required=True, type=Path)
    parser.add_argument("--candidate-map", required=True, type=Path)
    parser.add_argument("--selected-output", required=True, type=Path)
    parser.add_argument("--decision-output", required=True, type=Path)
    parser.add_argument("--min-prompt-override-fraction", type=float, default=0.10)
    parser.add_argument("--primary-metric", default="clip_cosine")
    parser.add_argument("--secondary-metric", default="clip_b32_cosine")
    parser.add_argument(
        "--require-positive-primary-ci",
        action="store_true",
        help="Require the paired 95% bootstrap lower bound to exceed zero.",
    )
    args = parser.parse_args()

    payload = json.loads(args.paired_summary.read_text(encoding="utf-8"))
    by_metric = payload.get("comparisons_by_metric", {})
    metric_rows: dict[str, dict[str, dict]] = {}
    for metric in (args.primary_metric, args.secondary_metric):
        rows = by_metric.get(metric, [])
        metric_rows[metric] = {str(row["method"]): row for row in rows}
        if not rows:
            raise ValueError(f"paired summary has no {metric} comparisons")

    candidates: dict[str, dict] = {}
    with args.candidate_map.open(encoding="utf-8") as handle:
        for line in handle:
            label, path = line.rstrip("\n").split("\t", maxsplit=1)
            controller = json.loads(Path(path).read_text(encoding="utf-8"))
            primary = metric_rows[args.primary_metric].get(label)
            secondary = metric_rows[args.secondary_metric].get(label)
            if primary is None or secondary is None:
                raise ValueError(f"missing paired comparison for {label}")
            activation = intervention_stats(args.root, label)
            passed = (
                float(primary["mean_delta"]) > 0.0
                and float(secondary["mean_delta"]) >= 0.0
                and (
                    not args.require_positive_primary_ci
                    or float(primary["ci95_low"]) > 0.0
                )
                and activation["prompt_fraction_with_override"]
                >= args.min_prompt_override_fraction
                and activation["mean_direct_overrides"] > 0.0
            )
            candidates[label] = {
                "path": path,
                "config": controller.get("config", {}),
                "selection_score": (
                    float(primary["mean_delta"]) + float(secondary["mean_delta"])
                )
                / 2.0,
                "metrics": {
                    args.primary_metric: primary,
                    args.secondary_metric: secondary,
                },
                "primary": primary,
                "secondary": secondary,
                "interventions": activation,
                "passed": passed,
            }

    eligible = [
        (
            float(row["selection_score"]),
            float(row["primary"]["mean_delta"]),
            label,
        )
        for label, row in candidates.items()
        if row["passed"]
    ]
    selected = max(eligible)[2] if eligible else None
    if selected is not None:
        args.selected_output.parent.mkdir(parents=True, exist_ok=True)
        selected_payload = json.loads(
            Path(candidates[selected]["path"]).read_text(encoding="utf-8")
        )
        metadata = dict(selected_payload.get("metadata", {}))
        metadata["development_selection"] = {
            "selected_label": selected,
            "selection_metric": (
                f"equal-weight mean of {args.primary_metric} and "
                f"{args.secondary_metric} paired mean deltas"
            ),
            "selected_metrics": candidates[selected]["metrics"],
            "selected_interventions": candidates[selected]["interventions"],
            "min_prompt_override_fraction": args.min_prompt_override_fraction,
            "required_positive_primary_ci": args.require_positive_primary_ci,
        }
        selected_payload["metadata"] = metadata
        args.selected_output.write_text(
            json.dumps(selected_payload, indent=2) + "\n", encoding="utf-8"
        )
    decision = {
        "design": "disjoint two-encoder controller development before matched training",
        "passed": selected is not None,
        "selection_metric": (
            f"equal-weight mean of {args.primary_metric} and {args.secondary_metric} "
            "paired mean deltas"
        ),
        "non_regression_metrics": [args.primary_metric, args.secondary_metric],
        "min_prompt_override_fraction": args.min_prompt_override_fraction,
        "require_positive_primary_ci": args.require_positive_primary_ci,
        "selected": selected,
        "candidates": candidates,
    }
    args.decision_output.parent.mkdir(parents=True, exist_ok=True)
    args.decision_output.write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))
    raise SystemExit(0 if selected is not None else 2)


if __name__ == "__main__":
    main()
