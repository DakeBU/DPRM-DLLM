#!/usr/bin/env python3
"""Fit a stage/rank/spatial DPRM controller from action-conditioned rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dprm.omni_order import OmniStageRankSpatialDPRM, spatial_bin_ids


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("records", payload.get("branches", []))
    else:
        rows = []
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"no action-conditioned records in {path}")
    return [dict(row) for row in rows]


def spatial_bin(visual_index: int, bins: int) -> int:
    import torch

    value = spatial_bin_ids(torch.tensor([visual_index]), spatial_bins=bins)
    return int(value.item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reward-field", default="advantage")
    parser.add_argument("--active-steps", type=int, nargs="+", required=True)
    parser.add_argument("--rank-bins", type=int, default=8)
    parser.add_argument("--spatial-bins", type=int, default=4)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument(
        "--shrinkage",
        type=float,
        default=4.0,
        help="Pseudo-count on zero action advantage in each bucket.",
    )
    args = parser.parse_args()

    if args.beta <= 0 or args.rank_bins < 1 or args.spatial_bins < 1:
        raise SystemExit("beta and bucket counts must be positive")
    active_steps = tuple(sorted(set(args.active_steps)))
    stage_index = {step: index for index, step in enumerate(active_steps)}
    shape = (len(active_steps), args.rank_bins, args.spatial_bins)
    counts = [
        [[0 for _ in range(shape[2])] for _ in range(shape[1])]
        for _ in range(shape[0])
    ]
    exp_sums = [
        [[0.0 for _ in range(shape[2])] for _ in range(shape[1])]
        for _ in range(shape[0])
    ]

    source_rows = load_rows(args.records)
    accepted: list[dict[str, Any]] = []
    for row in source_rows:
        if not bool(row.get("applied", True)):
            continue
        step = int(row.get("step", -1))
        if step not in stage_index or row.get(args.reward_field) is None:
            continue
        if row.get("rank_quantile") is not None:
            rank = min(
                int(args.rank_bins) - 1,
                max(0, int(float(row["rank_quantile"]) * int(args.rank_bins))),
            )
        else:
            rank = int(row["rank_bin"])
        if not 0 <= rank < args.rank_bins:
            raise ValueError(f"rank_bin={rank} is outside [0,{args.rank_bins})")
        visual_index = int(row["visual_index"])
        spatial = spatial_bin(visual_index, args.spatial_bins)
        reward = float(row[args.reward_field])
        if not math.isfinite(reward):
            raise ValueError("action reward must be finite")
        stage = stage_index[step]
        counts[stage][rank][spatial] += 1
        exp_sums[stage][rank][spatial] += math.exp(
            args.beta * max(-20.0, min(20.0, reward))
        )
        accepted.append(
            {
                "step": step,
                "rank_bin": rank,
                "spatial_bin": spatial,
                "reward": reward,
            }
        )

    if not accepted:
        raise SystemExit("no records match the requested action steps and reward field")
    shrinkage = max(float(args.shrinkage), 0.0)
    values = [
        [[0.0 for _ in range(shape[2])] for _ in range(shape[1])]
        for _ in range(shape[0])
    ]
    for stage in range(shape[0]):
        for rank in range(shape[1]):
            for spatial in range(shape[2]):
                count = counts[stage][rank][spatial]
                if count == 0:
                    continue
                mean_exp = (exp_sums[stage][rank][spatial] + shrinkage) / (
                    count + shrinkage
                )
                values[stage][rank][spatial] = math.log(max(mean_exp, 1e-12)) / args.beta

    controller = OmniStageRankSpatialDPRM(
        active_steps=active_steps,
        rank_bins=args.rank_bins,
        spatial_bins=args.spatial_bins,
        reward_values=tuple(
            tuple(tuple(cell for cell in rank) for rank in stage) for stage in values
        ),
        counts=tuple(
            tuple(tuple(cell for cell in rank) for rank in stage) for stage in counts
        ),
        beta=float(args.guidance_scale),
        min_count=int(args.min_count),
    )
    metadata = {
        "design": "offline action-conditioned DPRM; one forced reveal followed by confidence completion",
        "records": str(args.records),
        "records_sha256": hashlib.sha256(args.records.read_bytes()).hexdigest(),
        "reward_field": args.reward_field,
        "reward_temperature": args.beta,
        "shrinkage": shrinkage,
        "accepted_records": len(accepted),
        "nonempty_buckets": sum(
            count > 0 for stage in counts for rank in stage for count in rank
        ),
        "total_buckets": math.prod(shape),
        "source_summary": {
            "action_fit_prompt_count": len(
                {str(row.get("prompt", "")) for row in source_rows}
            ),
            "fixed_visual_canvas": True,
            "action_conditioned_continuations": True,
        },
        "score_contract": {
            "base_order_score": "negative_token_entropy",
            "bucket_coordinate": "within_state_confidence_rank",
            "position_selection_rule": "single_path_top1_adjusted_order_score",
        },
        "stagewise_order_contract": {
            "reward_action_steps": list(active_steps),
            "fallback": "native confidence order",
        },
        "deployment_contract": {
            "paths_per_prompt": 1,
            "terminal_reward_calls_at_test": 0,
            "complete_image_selection": False,
            "fixed_t2i_scaffold": True,
            "ordered_visual_positions": 256,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    controller.save_artifact(args.output, metadata)
    print(json.dumps({"config": controller.__dict__, "metadata": metadata}, indent=2))


if __name__ == "__main__":
    main()
