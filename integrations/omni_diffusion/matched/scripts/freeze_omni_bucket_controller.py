#!/usr/bin/env python3
"""Freeze an existing development bucket estimate as a shared Omni scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dprm.omni_order import OmniBucketTableDPRM


def nested_tuple(values, cast):
    if isinstance(values, list):
        return tuple(nested_tuple(value, cast) for value in values)
    return cast(values)


def prompt_hashes(records: list[dict]) -> list[str]:
    prompts = {
        str(record.get("prompt", "")).strip()
        for record in records
        if str(record.get("prompt", "")).strip()
    }
    return [hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in sorted(prompts)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--guidance-scale", type=float, required=True)
    parser.add_argument("--ready-count", type=int, required=True)
    parser.add_argument("--policy-warmup-steps", type=int, default=32)
    parser.add_argument("--reward-action-steps", type=int, nargs="*", default=())
    parser.add_argument("--max-base-score-gap", type=float, default=None)
    parser.add_argument("--max-reward-confidence-bin", type=int, default=None)
    args = parser.parse_args()

    payload = json.loads(args.source_table.read_text(encoding="utf-8"))
    cfg = payload["cfg"]
    controller = OmniBucketTableDPRM(
        num_phases=int(cfg["num_phases"]),
        confidence_bins=int(cfg["confidence_bins"]),
        spatial_bins=int(cfg.get("aux_bins", 1)),
        reward_temperature=float(cfg["reward_temperature"]),
        guidance_scale=float(args.guidance_scale),
        warmup_steps=int(cfg.get("warmup_steps", 0)),
        switch_steps=int(cfg.get("switch_steps", 0)),
        ready_count=int(args.ready_count),
        counts=nested_tuple(payload["counts"], float),
        exp_reward_sums=nested_tuple(payload["exp_reward_sums"], float),
        confidence_bin_edges=tuple(
            float(value) for value in cfg.get("confidence_bin_edges", ())
        ),
        policy_warmup_steps=int(args.policy_warmup_steps),
        reward_action_steps=tuple(int(step) for step in args.reward_action_steps),
        max_base_score_gap=args.max_base_score_gap,
        max_reward_confidence_bin=args.max_reward_confidence_bin,
    )
    source_metadata = payload.get("metadata", {})
    source_score_contract = source_metadata.get("score_contract", {})
    base_order_score = cfg.get(
        "base_order_score",
        source_score_contract.get("base_order_score", "negative_token_entropy"),
    )
    bucket_coordinate = cfg.get(
        "bucket_coordinate",
        source_score_contract.get(
            "bucket_coordinate", "exp_negative_token_entropy"
        ),
    )
    if base_order_score != "negative_token_entropy":
        raise ValueError(
            f"Omni controller requires negative-token-entropy base score, got {base_order_score}"
        )
    if bucket_coordinate != "exp_negative_token_entropy":
        raise ValueError(
            "Omni controller requires exp-negative-token-entropy bucket coordinate, "
            f"got {bucket_coordinate}"
        )
    metadata = {
        "source_table_file": args.source_table.name,
        "source_table_sha256": hashlib.sha256(args.source_table.read_bytes()).hexdigest(),
        "source_prompt_text_sha256": prompt_hashes(source_metadata.get("records", [])),
        "source_summary": {
            key: source_metadata.get(key)
            for key in (
                "orders",
                "clip_model",
                "secondary_clip_model",
                "secondary_metric_name",
                "reward_normalization",
                "reward_stats",
                "num_rollouts",
                "num_rollouts_before_prompt_deduplication",
                "prompt_text_deduplicated",
                "fixed_visual_canvas",
                "nonempty_buckets",
                "total_buckets",
                "bucket_coverage",
                "mean_clip_cosine",
            )
        },
        "configuration_selection": "development-only dual-CLIP sweep before matched branch training",
        "stagewise_order_contract": {
            "reward_action_steps": list(controller.reward_action_steps),
            "max_base_score_gap": controller.max_base_score_gap,
            "max_reward_confidence_bin": controller.max_reward_confidence_bin,
            "fallback": "native confidence order",
        },
        "score_contract": {
            "base_order_score": "negative_token_entropy",
            "base_order_formula": "-H[p_theta(.|s,i)]",
            "bucket_coordinate": "exp_negative_token_entropy",
            "bucket_coordinate_formula": "exp(-H[p_theta(.|s,i)])",
            "token_value_rule": "host_argmax_token",
            "position_selection_rule": "single_path_top1_adjusted_order_score",
        },
        "deployment_contract": {
            "paths_per_prompt": 1,
            "positions_per_order_action": 1,
            "terminal_reward_calls_at_test": 0,
            "complete_image_selection": False,
            "fixed_t2i_scaffold": True,
            "ordered_visual_positions": 256,
        },
        "test_time_terminal_rollouts": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    controller.save_artifact(args.output, metadata)
    print(json.dumps({"config": controller.__dict__, "metadata": metadata}, indent=2))


if __name__ == "__main__":
    main()
