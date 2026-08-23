#!/usr/bin/env python3
"""Cross-fit low-dimensional Omni action buckets on paired rollouts.

Each held-out prompt is scored by bucket means computed from other prompts.
The resulting policy estimate therefore measures whether a bucket definition
transfers, rather than whether it can describe the records used to fit it.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable

import numpy as np


BucketKey = tuple[int, ...]


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("branches", payload if isinstance(payload, list) else [])
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"no action records in {path}")
    return [dict(row) for row in rows]


def spatial_bin(row: dict[str, Any], bins: int = 4) -> int:
    visual_index = int(row["visual_index"])
    image_side = 16
    side_bins = int(round(bins**0.5))
    if side_bins * side_bins != bins:
        return min(bins - 1, visual_index * bins // (image_side * image_side))
    image_row, image_column = divmod(visual_index, image_side)
    row_bin = min(side_bins - 1, image_row * side_bins // image_side)
    column_bin = min(side_bins - 1, image_column * side_bins // image_side)
    return row_bin * side_bins + column_bin


def score_gap(row: dict[str, Any]) -> float:
    return max(
        0.0,
        float(row["default_raw_order_score"]) - float(row["raw_order_score"]),
    )


def visual_code_bin(row: dict[str, Any], bins: int) -> int:
    code = int(row["provisional_token_id"]) - 168072
    return min(bins - 1, max(0, code * bins // 8192))


def bin_from_edges(value: float, edges: np.ndarray) -> int:
    return int(np.searchsorted(edges, value, side="right"))


def prompt_fold(prompt: str, folds: int) -> int:
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def term_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![A-Za-z]){escaped}(?![A-Za-z])", re.IGNORECASE)


def is_multi_entity(prompt: str, rule: dict[str, Any] | None) -> int:
    if rule is None:
        return 0
    for family in (
        "explicit_quantity_terms",
        "collective_terms",
        "animate_plural_terms",
    ):
        if any(term_pattern(str(term)).search(prompt) for term in rule[family]):
            return 1
    return 0


def bootstrap_prompt_interval(
    prompt_values: np.ndarray,
    *,
    seed: int,
    resamples: int,
) -> tuple[float, float]:
    if prompt_values.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, prompt_values.shape[0], size=(resamples, prompt_values.shape[0])
    )
    means = prompt_values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def metric_summary(
    values: np.ndarray,
    *,
    seed: int,
    resamples: int,
) -> dict[str, float]:
    low, high = bootstrap_prompt_interval(values, seed=seed, resamples=resamples)
    return {
        "mean": float(values.mean()) if values.size else 0.0,
        "ci95_low": low,
        "ci95_high": high,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--min-count", type=int, default=12)
    parser.add_argument("--gap-bins", type=int, default=2)
    parser.add_argument("--prompt-rule", type=Path)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    if args.folds < 2 or args.gap_bins < 1 or args.min_count < 1:
        raise SystemExit("folds must be >=2 and bucket settings must be positive")

    rows = [row for row in load_rows(args.records) if bool(row.get("applied", True))]
    prompt_rule = (
        json.loads(args.prompt_rule.read_text(encoding="utf-8"))
        if args.prompt_rule is not None
        else None
    )
    prompts = sorted({str(row["prompt"]) for row in rows})
    if len(prompts) < args.folds:
        raise SystemExit("fewer prompts than requested folds")

    specifications: dict[str, Callable[[dict[str, Any], np.ndarray], BucketKey]] = {
        "stage": lambda row, edges: (int(row["step"]),),
        "stage_spatial4": lambda row, edges: (
            int(row["step"]),
            spatial_bin(row, 4),
        ),
        "stage_gap": lambda row, edges: (
            int(row["step"]),
            bin_from_edges(score_gap(row), edges),
        ),
        "stage_spatial4_gap": lambda row, edges: (
            int(row["step"]),
            spatial_bin(row, 4),
            bin_from_edges(score_gap(row), edges),
        ),
        "stage_neighbor3": lambda row, edges: (
            int(row["step"]),
            min(2, int(float(row["local_revealed_fraction"]) * 3)),
        ),
        "stage_code4": lambda row, edges: (
            int(row["step"]),
            visual_code_bin(row, 4),
        ),
        "stage_code8": lambda row, edges: (
            int(row["step"]),
            visual_code_bin(row, 8),
        ),
        "stage_center3": lambda row, edges: (
            int(row["step"]),
            min(2, int(float(row["center_distance"]) * 3 / 11.0)),
        ),
        "stage_spatial4_code4": lambda row, edges: (
            int(row["step"]),
            spatial_bin(row, 4),
            visual_code_bin(row, 4),
        ),
    }
    if prompt_rule is not None:
        specifications.update(
            {
                "stage_prompt_multi": lambda row, edges: (
                    int(row["step"]),
                    is_multi_entity(str(row["prompt"]), prompt_rule),
                ),
                "stage_prompt_multi_spatial4": lambda row, edges: (
                    int(row["step"]),
                    is_multi_entity(str(row["prompt"]), prompt_rule),
                    spatial_bin(row, 4),
                ),
                "stage_prompt_multi_gap": lambda row, edges: (
                    int(row["step"]),
                    is_multi_entity(str(row["prompt"]), prompt_rule),
                    bin_from_edges(score_gap(row), edges),
                ),
            }
        )

    results = []
    for spec_index, (name, key_fn) in enumerate(specifications.items()):
        decisions: list[dict[str, Any]] = []
        fold_metadata = []
        for fold in range(args.folds):
            train = [
                row
                for row in rows
                if prompt_fold(str(row["prompt"]), args.folds) != fold
            ]
            test = [
                row
                for row in rows
                if prompt_fold(str(row["prompt"]), args.folds) == fold
            ]
            if args.gap_bins == 1 or not train:
                edges = np.asarray([], dtype=float)
            else:
                quantiles = np.arange(1, args.gap_bins, dtype=float) / args.gap_bins
                edges = np.quantile(
                    np.asarray([score_gap(row) for row in train], dtype=float),
                    quantiles,
                )

            grouped: dict[BucketKey, list[dict[str, Any]]] = defaultdict(list)
            for row in train:
                grouped[key_fn(row, edges)].append(row)
            active: dict[BucketKey, dict[str, float]] = {}
            for key, group in grouped.items():
                if len(group) < args.min_count:
                    continue
                clip_l = float(np.mean([float(row["clip_advantage"]) for row in group]))
                clip_b = float(
                    np.mean([float(row["clip_b32_advantage"]) for row in group])
                )
                # The action replaces confidence only when both independent
                # text-image encoders favor that bucket on training prompts.
                if clip_l > 0.0 and clip_b > 0.0:
                    active[key] = {"clip_l": clip_l, "clip_b": clip_b}

            for row in test:
                key = key_fn(row, edges)
                selected = key in active
                decisions.append(
                    {
                        "prompt": str(row["prompt"]),
                        "step": int(row["step"]),
                        "bucket": list(key),
                        "selected": selected,
                        "clip_advantage": float(row["clip_advantage"]) if selected else 0.0,
                        "clip_b32_advantage": (
                            float(row["clip_b32_advantage"]) if selected else 0.0
                        ),
                    }
                )
            fold_metadata.append(
                {
                    "fold": fold,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "gap_edges": edges.tolist(),
                    "active_buckets": [list(key) for key in sorted(active)],
                }
            )

        # One policy can intervene at more than one stage for a prompt. We
        # report the additive one-action estimate and bootstrap by prompt so
        # both stages from the same prompt remain paired.
        per_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for decision in decisions:
            per_prompt[decision["prompt"]].append(decision)
        prompt_l = np.asarray(
            [sum(item["clip_advantage"] for item in group) for group in per_prompt.values()]
        )
        prompt_b = np.asarray(
            [
                sum(item["clip_b32_advantage"] for item in group)
                for group in per_prompt.values()
            ]
        )
        selected = [decision for decision in decisions if decision["selected"]]
        by_step = []
        for step_index, step in enumerate(sorted({item["step"] for item in decisions})):
            step_rows = [item for item in decisions if item["step"] == step]
            step_l = np.asarray([item["clip_advantage"] for item in step_rows])
            step_b = np.asarray([item["clip_b32_advantage"] for item in step_rows])
            by_step.append(
                {
                    "step": step,
                    "selected_rows": sum(item["selected"] for item in step_rows),
                    "clip_l_delta": metric_summary(
                        step_l,
                        seed=args.seed + 100 + 20 * spec_index + 2 * step_index,
                        resamples=args.resamples,
                    ),
                    "clip_b32_delta": metric_summary(
                        step_b,
                        seed=args.seed + 101 + 20 * spec_index + 2 * step_index,
                        resamples=args.resamples,
                    ),
                }
            )
        results.append(
            {
                "specification": name,
                "prompts": len(per_prompt),
                "rows": len(decisions),
                "selected_rows": len(selected),
                "selected_prompt_fraction": float(
                    np.mean(
                        [any(item["selected"] for item in group) for group in per_prompt.values()]
                    )
                ),
                "additive_clip_l_delta_per_prompt": metric_summary(
                    prompt_l,
                    seed=args.seed + 2 * spec_index,
                    resamples=args.resamples,
                ),
                "additive_clip_b32_delta_per_prompt": metric_summary(
                    prompt_b,
                    seed=args.seed + 2 * spec_index + 1,
                    resamples=args.resamples,
                ),
                "by_step": by_step,
                "folds": fold_metadata,
            }
        )

    result = {
        "design": "prompt-disjoint cross-fit of offline action buckets",
        "records": str(args.records),
        "records_sha256": hashlib.sha256(args.records.read_bytes()).hexdigest(),
        "fold_count": args.folds,
        "minimum_training_count": args.min_count,
        "gap_bins": args.gap_bins,
        "prompt_rule": str(args.prompt_rule) if args.prompt_rule else None,
        "prompt_rule_sha256": (
            hashlib.sha256(args.prompt_rule.read_bytes()).hexdigest()
            if args.prompt_rule
            else None
        ),
        "selection_rule": "training mean advantage is positive for CLIP-L/14 and CLIP-B/32",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            [
                {
                    "specification": row["specification"],
                    "selected_rows": row["selected_rows"],
                    "clip_l": row["additive_clip_l_delta_per_prompt"],
                    "clip_b32": row["additive_clip_b32_delta_per_prompt"],
                }
                for row in results
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
