#!/usr/bin/env python3
"""Freeze the semantics of a selected Omni controller for matched train/test use."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


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


def prompt_text_hashes(path: Path) -> list[str]:
    """Hash normalized unique prompt texts for split-overlap auditing."""
    seen: set[str] = set()
    hashes: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        prompt = raw.strip()
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        hashes.append(hashlib.sha256(prompt.encode("utf-8")).hexdigest())
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-decision", type=Path)
    parser.add_argument(
        "--source-table-root",
        type=Path,
        help="Directory containing the checksummed source table named in the controller.",
    )
    parser.add_argument("--table-prompt-range", nargs=2, type=int, metavar=("START", "END"))
    parser.add_argument("--selection-prompt-range", nargs=2, type=int, metavar=("START", "END"))
    parser.add_argument("--selection-prompt-file", type=Path)
    parser.add_argument(
        "--controller-only-host",
        action="store_true",
        help="Record that the public host is frozen and only the order controller is fitted.",
    )
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if payload.get("format") != "omni_bucket_table_dprm_v1":
        raise SystemExit("formal Omni controller must be an omni_bucket_table_dprm_v1 artifact")
    config = payload.get("config", {})
    if not config.get("confidence_bin_edges"):
        raise SystemExit("formal Omni controller requires frozen development quantile edges")
    if not config.get("reward_action_steps"):
        raise SystemExit("formal Omni controller requires fixed reward-action steps")
    if config.get("max_base_score_gap") is None:
        raise SystemExit("formal Omni controller requires an ambiguity score gap")
    source = payload.get("metadata", {}).get("source_summary", {})
    if source.get("prompt_text_deduplicated") is not True:
        raise SystemExit("formal Omni controller requires prompt-text-deduplicated source rollouts")
    if source.get("fixed_visual_canvas") is not True:
        raise SystemExit("formal Omni controller requires fixed 256-position source rollouts")

    metadata = payload.setdefault("metadata", {})
    source_table = metadata.pop("source_table", None)
    if source_table:
        metadata["source_table_file"] = Path(source_table).name
    source_summary = metadata.get("source_summary")
    if isinstance(source_summary, dict):
        source_summary.pop("rollout_root", None)
    if args.source_table_root is not None:
        source_name = metadata.get("source_table_file")
        expected_digest = metadata.get("source_table_sha256")
        if not source_name or not expected_digest:
            raise SystemExit("controller does not identify its source table")
        source_path = args.source_table_root / str(source_name)
        if not source_path.is_file():
            raise SystemExit(f"missing source table: {source_path}")
        observed_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if observed_digest != expected_digest:
            raise SystemExit("source table checksum does not match selected controller")
        source_payload = json.loads(source_path.read_text(encoding="utf-8"))
        source_metadata = source_payload.get("metadata", {})
        metadata["source_summary"] = {
            key: source_metadata.get(key)
            for key in (
                "orders",
                "clip_model",
                "secondary_clip_model",
                "secondary_metric_name",
                "reward_normalization",
                "reward_stats",
                "num_rollouts",
                "prompt_text_deduplicated",
                "fixed_visual_canvas",
                "nonempty_buckets",
                "total_buckets",
                "bucket_coverage",
            )
        }
    existing_score = metadata.get("score_contract", {})
    for key, expected in SCORE_CONTRACT.items():
        observed = existing_score.get(key, expected)
        if observed != expected:
            raise SystemExit(
                f"incompatible Omni score contract for {key}: {observed!r} != {expected!r}"
            )
    metadata["score_contract"] = SCORE_CONTRACT
    metadata["deployment_contract"] = DEPLOYMENT_CONTRACT
    metadata["stagewise_order_contract"] = {
        "reward_action_steps": [int(step) for step in config["reward_action_steps"]],
        "max_base_score_gap": float(config["max_base_score_gap"]),
        "max_reward_confidence_bin": config.get("max_reward_confidence_bin"),
        "fallback": "native confidence order",
    }
    if args.controller_only_host:
        metadata["train_test_order_match"] = {
            "shared_initialization": "public Omni-Diffusion checkpoint",
            "controller_development_states": "fixed-scaffold public-host rollouts",
            "controller_frozen_after_development": True,
            "training_policy": "the frozen controller constructs DPRM branch states",
            "inference_policy": "the same frozen controller orders the DPRM branch",
            "comparison_training": "matched confidence and DPRM branches",
            "host_sampler": "entropy-penalty",
            "ordered_action_space": "256 visual-code positions",
        }
    else:
        metadata["train_test_order_match"] = {
            "training_state_policy": "same_frozen_bucket_controller",
            "training_state_construction": "teacher_forced_deployed_sampler_trajectory",
            "training_current_model_action": "same_policy_refresh_from_cached_canvas",
            "inference_policy": "same_frozen_bucket_controller",
            "host_sampler": "entropy-penalty",
            "ordered_action_space": "256 visual-code positions",
        }
    if args.selection_decision is not None:
        decision = json.loads(args.selection_decision.read_text(encoding="utf-8"))
        if decision.get("passed") is not True:
            raise SystemExit("formal Omni controller requires a passed development decision")
        selected = decision.get("selected")
        selected_record = decision.get("candidates", {}).get(selected)
        if not selected or not isinstance(selected_record, dict):
            raise SystemExit("development decision does not identify a selected controller")
        selected_config = selected_record.get("config", {})
        for key in (
            "guidance_scale",
            "ready_count",
            "reward_action_steps",
            "max_base_score_gap",
            "max_reward_confidence_bin",
        ):
            if selected_config.get(key) != config.get(key):
                raise SystemExit(
                    f"selected controller mismatch for {key}: "
                    f"{selected_config.get(key)!r} != {config.get(key)!r}"
                )
        metadata["development_selection"] = {
            "design": decision.get("design"),
            "selection_metric": decision.get("selection_metric"),
            "selected_label": selected,
            "candidate_count": len(decision.get("candidates", {})),
            "table_prompt_range": args.table_prompt_range,
            "selection_prompt_range": args.selection_prompt_range,
            "selection_prompt_file": args.selection_prompt_file.name
            if args.selection_prompt_file
            else None,
            "selection_prompt_file_sha256": hashlib.sha256(
                args.selection_prompt_file.read_bytes()
            ).hexdigest()
            if args.selection_prompt_file
            else None,
            "selection_prompt_text_sha256": prompt_text_hashes(
                args.selection_prompt_file
            )
            if args.selection_prompt_file
            else None,
            "selected_metrics": selected_record.get("metrics", {}),
            "selected_interventions": selected_record.get("interventions", {}),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=args.output.parent,
        prefix=f".{args.output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
