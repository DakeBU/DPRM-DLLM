#!/usr/bin/env python3
"""Freeze the semantics of a selected Omni controller for matched train/test use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SCORE_CONTRACT = {
    "base_order_score": "negative_token_entropy",
    "base_order_formula": "-H[p_theta(.|s,i)]",
    "bucket_coordinate": "exp_negative_token_entropy",
    "bucket_coordinate_formula": "exp(-H[p_theta(.|s,i)])",
    "token_value_rule": "host_argmax_token",
    "position_selection_rule": "single_path_top1_adjusted_order_score",
}

DEPLOYMENT_CONTRACT = {
    "paths_per_prompt": 1,
    "positions_per_order_action": 1,
    "terminal_reward_calls_at_test": 0,
    "complete_image_selection": False,
    "fixed_t2i_scaffold": True,
    "ordered_visual_positions": 256,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if payload.get("format") != "omni_bucket_table_dprm_v1":
        raise SystemExit("formal Omni controller must be an omni_bucket_table_dprm_v1 artifact")
    config = payload.get("config", {})
    if not config.get("confidence_bin_edges"):
        raise SystemExit("formal Omni controller requires frozen development quantile edges")
    source = payload.get("metadata", {}).get("source_summary", {})
    if source.get("prompt_text_deduplicated") is not True:
        raise SystemExit("formal Omni controller requires prompt-text-deduplicated source rollouts")
    if source.get("fixed_visual_canvas") is not True:
        raise SystemExit("formal Omni controller requires fixed 256-position source rollouts")

    metadata = payload.setdefault("metadata", {})
    existing_score = metadata.get("score_contract", {})
    for key, expected in SCORE_CONTRACT.items():
        observed = existing_score.get(key, expected)
        if observed != expected:
            raise SystemExit(
                f"incompatible Omni score contract for {key}: {observed!r} != {expected!r}"
            )
    metadata["score_contract"] = SCORE_CONTRACT
    metadata["deployment_contract"] = DEPLOYMENT_CONTRACT
    metadata["train_test_order_match"] = {
        "training_state_policy": "same_frozen_bucket_controller",
        "training_state_construction": "teacher_forced_deployed_sampler_trajectory",
        "training_current_model_action": "same_policy_refresh_from_cached_canvas",
        "inference_policy": "same_frozen_bucket_controller",
        "host_sampler": "entropy-penalty",
        "ordered_action_space": "256 visual-code positions",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
