#!/usr/bin/env python3
"""Select terminally scored Omni action branches from fixed Soft-BoN shortlists."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def bootstrap(values: np.ndarray, seed: int, iterations: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(iterations, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def parse_spec(text: str) -> tuple[str, set[float]]:
    name, values = text.split("=", 1)
    return name, {float(value) for value in values.split(",")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=96)
    parser.add_argument(
        "--shortlist",
        action="append",
        default=["N2=0.3,0.7", "N4=0.15,0.3,0.7,0.85"],
    )
    parser.add_argument("--bootstrap-iters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()

    payload = json.loads(args.records.read_text(encoding="utf-8"))
    baselines = {
        (str(row["prompt"]), int(row["seed"])): row for row in payload["baseline"]
    }
    branches: dict[tuple[str, int], list[dict[str, Any]]] = {
        key: [] for key in baselines
    }
    for row in payload["branches"]:
        key = str(row["prompt"]), int(row["seed"])
        if key in branches and int(row["step"]) == args.step:
            branches[key].append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    selected_records = []
    audit_records: dict[str, list[dict[str, Any]]] = {
        "progressive_confidence": [
            {
                "order": "progressive_confidence",
                "prompt_id": str(row["seed"]),
                "prompt": str(row["prompt"]),
                "seed": int(row["seed"]),
                "has_image": True,
                "image_path": str(row["image_path"]),
                "clip_cosine": float(row["clip_cosine"]),
                "aesthetic_score": float(row["aesthetic_score"]),
                "terminal_utility": float(row["terminal_utility"]),
            }
            for row in baselines.values()
        ]
    }
    for name, quantiles in map(parse_spec, args.shortlist):
        metric_deltas = {
            "terminal_utility": [],
            "clip_cosine": [],
            "aesthetic_score": [],
        }
        choices = []
        for key, baseline in baselines.items():
            candidates = [
                row
                for row in branches[key]
                if any(
                    abs(float(row["requested_quantile"]) - quantile) < 1e-8
                    for quantile in quantiles
                )
            ]
            if len(candidates) != len(quantiles):
                raise SystemExit(
                    f"{name}: expected {len(quantiles)} branches for seed {key[1]}, "
                    f"found {len(candidates)}"
                )
            best = max(candidates, key=lambda row: float(row["branch_utility"]))
            use_branch = float(best["branch_utility"]) > float(
                baseline["terminal_utility"]
            )
            selected = best if use_branch else baseline
            selected_utility = (
                float(best["branch_utility"])
                if use_branch
                else float(baseline["terminal_utility"])
            )
            selected_clip = (
                float(best["branch_clip"])
                if use_branch
                else float(baseline["clip_cosine"])
            )
            selected_aesthetic = (
                float(best["branch_aesthetic"])
                if use_branch
                else float(baseline["aesthetic_score"])
            )
            metric_deltas["terminal_utility"].append(
                selected_utility - float(baseline["terminal_utility"])
            )
            metric_deltas["clip_cosine"].append(
                selected_clip - float(baseline["clip_cosine"])
            )
            metric_deltas["aesthetic_score"].append(
                selected_aesthetic - float(baseline["aesthetic_score"])
            )
            choice = {
                "method": f"dprm_softbon_{name.lower()}",
                "prompt": key[0],
                "seed": key[1],
                "selected_branch": bool(use_branch),
                "selected_quantile": (
                    float(best["requested_quantile"]) if use_branch else None
                ),
                "image_path": str(selected["image_path"]),
                "baseline_image_path": str(baseline["image_path"]),
                "clip_cosine": selected_clip,
                "aesthetic_score": selected_aesthetic,
                "terminal_utility": selected_utility,
                "selected_action_confidence": (
                    float(best["confidence"]) if use_branch else None
                ),
                "default_action_confidence": (
                    float(best["default_confidence"])
                    if use_branch and best.get("default_confidence") is not None
                    else None
                ),
                "confidence_gap_from_default": (
                    float(best["confidence_gap_from_default"])
                    if use_branch and best.get("confidence_gap_from_default") is not None
                    else None
                ),
                "selected_visual_index": (
                    int(best["visual_index"]) if use_branch else None
                ),
                "selected_aux_bin": int(best["aux_bin"]) if use_branch else None,
                "selected_rank_bin": int(best["rank_bin"]) if use_branch else None,
            }
            choices.append(choice)
            selected_records.append(choice)

        metrics = {}
        for metric, raw_values in metric_deltas.items():
            values = np.asarray(raw_values, dtype=np.float64)
            low, high = bootstrap(values, args.seed, args.bootstrap_iters)
            metrics[metric] = {
                "mean_delta": float(values.mean()),
                "ci95_low": low,
                "ci95_high": high,
                "wins": int(np.sum(values > 1e-12)),
                "losses": int(np.sum(values < -1e-12)),
                "ties": int(np.sum(np.abs(values) <= 1e-12)),
            }
        reports.append(
            {
                "method": f"dprm_softbon_{name.lower()}",
                "step": args.step,
                "action_shortlist_size": len(quantiles),
                "total_rollouts_per_prompt": len(quantiles) + 1,
                "quantiles": sorted(quantiles),
                "metrics": metrics,
                "selected_branch_count": int(sum(row["selected_branch"] for row in choices)),
                "selected_branch_rate": float(
                    np.mean([row["selected_branch"] for row in choices])
                ),
                "selected_quantile_counts": {
                    str(quantile): int(
                        sum(row["selected_quantile"] == quantile for row in choices)
                    )
                    for quantile in sorted(quantiles)
                },
                "mean_selected_confidence_gap": float(
                    np.mean(
                        [
                            row["confidence_gap_from_default"]
                            for row in choices
                            if row["confidence_gap_from_default"] is not None
                        ]
                    )
                )
                if any(row["confidence_gap_from_default"] is not None for row in choices)
                else None,
                "choices": choices,
            }
        )
        audit_records[f"dprm_softbon_{name.lower()}"] = [
            {
                **choice,
                "order": f"dprm_softbon_{name.lower()}",
                "prompt_id": str(choice["seed"]),
                "has_image": True,
            }
            for choice in choices
        ]

    output = {
        "design": "shared-state action shortlist; confidence completion; hard terminal-utility selection with confidence fallback",
        "paper_name": "DPRM-BoN (hard reward-tilt limit)",
        "selection_reward": "CLIP-L/14 cosine + 0.01 * LAION aesthetic score",
        "prompt_count": len(baselines),
        "reports": reports,
    }
    (args.output_dir / "softbon_action_selection.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "selected_records.json").write_text(
        json.dumps(selected_records, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "audit_records.json").write_text(
        json.dumps(audit_records, indent=2) + "\n", encoding="utf-8"
    )
    audit_summary = {
        "clip": {
            "orders": {
                order: {
                    "count": len(records),
                    "mean_clip_cosine": float(
                        np.mean([row["clip_cosine"] for row in records])
                    ),
                    "median_clip_cosine": float(
                        np.median([row["clip_cosine"] for row in records])
                    ),
                }
                for order, records in audit_records.items()
            }
        }
    }
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(audit_summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({**output, "reports": [{k: v for k, v in row.items() if k != "choices"} for row in reports]}, indent=2))


if __name__ == "__main__":
    main()
