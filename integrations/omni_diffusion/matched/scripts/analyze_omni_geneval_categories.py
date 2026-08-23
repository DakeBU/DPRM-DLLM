#!/usr/bin/env python3
"""Report paired Omni order effects by the official GenEval categories."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_interval(
    values: list[float], *, samples: int, seed: int
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap requires at least one paired delta")
    if len(values) == 1:
        return values[0], values[0]
    generator = random.Random(seed)
    count = len(values)
    means = [
        statistics.fmean(values[generator.randrange(count)] for _ in range(count))
        for _ in range(samples)
    ]
    return percentile(means, 0.025), percentile(means, 0.975)


def prompt_metadata(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = str(row["prompt"]).strip()
        if prompt in rows:
            raise ValueError(f"duplicate GenEval prompt: {prompt}")
        rows[prompt] = row
    return rows


def has_direct_override(record: dict[str, Any]) -> bool:
    trace = Path(str(record.get("order_trace_path", "")))
    if not trace.is_file():
        return False
    for line in trace.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        selected = row.get("selected_candidate_indices", [])
        default = row.get("confidence_default_candidate_index")
        if len(selected) == 1 and default is not None:
            if int(selected[0]) != int(default):
                return True
    return False


def summarize(
    deltas: list[float], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    low, high = bootstrap_interval(deltas, samples=bootstrap_samples, seed=seed)
    return {
        "matched_prompts": len(deltas),
        "mean_delta": statistics.fmean(deltas),
        "median_delta": statistics.median(deltas),
        "ci95_low": low,
        "ci95_high": high,
        "wins": sum(value > 0 for value in deltas),
        "ties": sum(value == 0 for value in deltas),
        "losses": sum(value < 0 for value in deltas),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tsv-output", type=Path)
    parser.add_argument("--baseline", default="progressive_confidence")
    parser.add_argument("--method", default="dprm_confidence_warmup")
    parser.add_argument(
        "--metrics", nargs="+", default=["clip_cosine", "clip_b32_cosine"]
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()

    records = json.loads(args.records.read_text(encoding="utf-8"))
    if args.baseline not in records or args.method not in records:
        raise ValueError("records do not contain the requested comparison")
    metadata = prompt_metadata(args.metadata)
    baseline = {str(row["prompt_id"]): row for row in records[args.baseline]}
    method = {str(row["prompt_id"]): row for row in records[args.method]}
    shared = sorted(set(baseline) & set(method))
    if not shared:
        raise ValueError("comparison has no matched prompt ids")

    examples: list[dict[str, Any]] = []
    for prompt_id in shared:
        base_row = baseline[prompt_id]
        method_row = method[prompt_id]
        prompt = str(base_row["prompt"]).strip()
        if prompt != str(method_row["prompt"]).strip():
            raise ValueError(f"prompt mismatch for {prompt_id}")
        if prompt not in metadata:
            raise ValueError(f"prompt is absent from GenEval metadata: {prompt}")
        examples.append(
            {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "tag": str(metadata[prompt]["tag"]),
                "direct_override": has_direct_override(method_row),
                "baseline": base_row,
                "method": method_row,
            }
        )

    groups: list[tuple[str, str, list[dict[str, Any]]]] = [("all", "all", examples)]
    for tag in sorted({row["tag"] for row in examples}):
        groups.append(("tag", tag, [row for row in examples if row["tag"] == tag]))
    groups.extend(
        [
            ("intervention", "direct_override", [row for row in examples if row["direct_override"]]),
            ("intervention", "confidence_fallback", [row for row in examples if not row["direct_override"]]),
        ]
    )

    rows: list[dict[str, Any]] = []
    for metric_index, metric in enumerate(args.metrics):
        for group_index, (group_type, group, members) in enumerate(groups):
            if not members:
                continue
            deltas = [
                float(row["method"][metric]) - float(row["baseline"][metric])
                for row in members
            ]
            rows.append(
                {
                    "metric": metric,
                    "group_type": group_type,
                    "group": group,
                    **summarize(
                        deltas,
                        bootstrap_samples=args.bootstrap_samples,
                        seed=args.seed + 1000 * metric_index + group_index,
                    ),
                }
            )

    payload = {
        "baseline": args.baseline,
        "method": args.method,
        "matched_prompts": len(examples),
        "prompt_fraction_with_direct_override": statistics.fmean(
            float(row["direct_override"]) for row in examples
        ),
        "bootstrap_samples": args.bootstrap_samples,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.tsv_output:
        args.tsv_output.parent.mkdir(parents=True, exist_ok=True)
        columns = [
            "metric",
            "group_type",
            "group",
            "matched_prompts",
            "mean_delta",
            "ci95_low",
            "ci95_high",
            "wins",
            "ties",
            "losses",
        ]
        lines = ["\t".join(columns)]
        lines.extend("\t".join(str(row[column]) for column in columns) for row in rows)
        args.tsv_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
