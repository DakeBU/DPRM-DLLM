#!/usr/bin/env python3
"""Measure how much DPRM changes a matched Omni reveal order."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


COMPARISONS = (
    ("progressive_confidence", "dprm_confidence_warmup"),
)
EARLY_FRACTIONS = (0.1, 0.25, 0.5)


def load_order(path: Path) -> list[int]:
    events: list[tuple[int, int]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            events.extend(
                (int(sequence_position), int(visual_index))
                for sequence_position, visual_index in zip(
                    record.get("selected_sequence_positions", []),
                    record.get("selected_visual_indices", []),
                    strict=True,
                )
            )
    if not events:
        raise ValueError(f"empty trace: {path}")
    visual_offset = Counter(
        sequence_position - visual_index
        for sequence_position, visual_index in events
    ).most_common(1)[0][0]
    order = [
        visual_index
        for sequence_position, visual_index in events
        if sequence_position - visual_index == visual_offset
    ]
    if len(order) != len(set(order)):
        raise ValueError(f"trace reveals a visual position more than once: {path}")
    return order


def trace_map(formal_root: Path, policy: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted((formal_root / policy).glob("prompt_*/*_order_trace.jsonl")):
        result[path.parent.name] = path
    return result


def compare_orders(reference: list[int], method: list[int]) -> dict[str, float]:
    if set(reference) != set(method):
        raise ValueError("matched traces do not reveal the same visual positions")
    length = len(reference)
    ref_rank = {position: rank for rank, position in enumerate(reference)}
    method_rank = {position: rank for rank, position in enumerate(method)}
    displacement = np.asarray(
        [abs(method_rank[position] - ref_rank[position]) for position in reference],
        dtype=np.float64,
    )
    metrics: dict[str, float] = {
        "positions": float(length),
        "same_step_fraction": float(
            np.mean(np.asarray(reference) == np.asarray(method))
        ),
        "moved_position_fraction": float(np.mean(displacement > 0)),
        "mean_absolute_rank_displacement": float(displacement.mean()),
        "normalized_mean_rank_displacement": float(displacement.mean() / max(length - 1, 1)),
        "max_rank_displacement": float(displacement.max(initial=0.0)),
    }
    for fraction in EARLY_FRACTIONS:
        width = max(1, int(round(length * fraction)))
        ref_early = set(reference[:width])
        method_early = set(method[:width])
        metrics[f"early_{fraction:g}_jaccard"] = float(
            len(ref_early & method_early) / len(ref_early | method_early)
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | float]] = []
    summaries: list[dict] = []
    for reference_policy, method_policy in COMPARISONS:
        reference_traces = trace_map(args.formal_root, reference_policy)
        method_traces = trace_map(args.formal_root, method_policy)
        prompt_ids = sorted(set(reference_traces) & set(method_traces))
        if not prompt_ids:
            continue
        comparison_rows: list[dict[str, str | float]] = []
        for prompt_id in prompt_ids:
            metrics = compare_orders(
                load_order(reference_traces[prompt_id]),
                load_order(method_traces[prompt_id]),
            )
            row: dict[str, str | float] = {
                "reference": reference_policy,
                "method": method_policy,
                "prompt_id": prompt_id,
                **metrics,
            }
            rows.append(row)
            comparison_rows.append(row)

        numeric_keys = [
            key
            for key in comparison_rows[0]
            if key not in {"reference", "method", "prompt_id"}
        ]
        summaries.append(
            {
                "reference": reference_policy,
                "method": method_policy,
                "matched_prompts": len(comparison_rows),
                **{
                    key: float(np.mean([float(row[key]) for row in comparison_rows]))
                    for key in numeric_keys
                },
            }
        )

    if not rows:
        raise SystemExit("no matched order traces found")
    with (args.output_dir / "paired_order_divergence.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {"formal_root": str(args.formal_root), "comparisons": summaries}
    (args.output_dir / "paired_order_divergence.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
