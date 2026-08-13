#!/usr/bin/env python3
"""Apply the prespecified manuscript gate to matched Omni evaluation outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


BASELINE = "progressive_confidence"
METHOD = "dprm_confidence_warmup"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def comparison(payload: dict, metric: str) -> dict:
    rows = payload.get("comparisons_by_metric", {}).get(metric, [])
    for row in rows:
        if row.get("baseline") == BASELINE and row.get("method") == METHOD:
            return row
    raise ValueError(f"missing {metric} comparison for {BASELINE} vs {METHOD}")


def divergence(payload: dict) -> dict:
    for row in payload.get("comparisons", []):
        if row.get("reference") == BASELINE and row.get("method") == METHOD:
            return row
    raise ValueError(f"missing order divergence for {BASELINE} vs {METHOD}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--divergence", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-prompts", type=int, default=96)
    args = parser.parse_args()

    paired = json.loads(args.paired.read_text(encoding="utf-8"))
    order = json.loads(args.divergence.read_text(encoding="utf-8"))
    controller = json.loads(args.controller.read_text(encoding="utf-8"))
    run_manifest = json.loads(args.run_manifest.read_text(encoding="utf-8"))
    training_audit_path = Path(run_manifest.get("training_contract_audit", ""))
    training_audit = (
        json.loads(training_audit_path.read_text(encoding="utf-8"))
        if training_audit_path.is_file()
        else {}
    )
    training_audit_hash = (
        sha256(training_audit_path) if training_audit_path.is_file() else ""
    )
    clip_l = comparison(paired, "clip_cosine")
    clip_b = comparison(paired, "clip_b32_cosine")
    order_delta = divergence(order)
    deployment = controller.get("metadata", {}).get("deployment_contract", {})

    checks = {
        "all_prompts_paired_clip_l14": clip_l.get("matched_prompts") == args.expected_prompts,
        "all_prompts_paired_clip_b32": clip_b.get("matched_prompts") == args.expected_prompts,
        "clip_l14_paired_ci_low_positive": float(clip_l.get("ci95_low", 0.0)) > 0.0,
        "clip_b32_mean_delta_positive": float(clip_b.get("mean_delta", 0.0)) > 0.0,
        "order_changes_measurably": float(order_delta.get("moved_position_fraction", 0.0)) >= 0.02,
        "single_path_deployment": deployment.get("paths_per_prompt") == 1,
        "no_test_reward_calls": deployment.get("terminal_reward_calls_at_test") == 0,
        "no_complete_image_selection": deployment.get("complete_image_selection") is False,
        "controller_fixes_t2i_scaffold": deployment.get("fixed_t2i_scaffold") is True,
        "controller_orders_256_visual_positions": deployment.get("ordered_visual_positions") == 256,
        "evaluation_uses_one_path": run_manifest.get("paths_per_prompt") == 1,
        "evaluation_has_no_test_reward": run_manifest.get("test_time_terminal_rollouts") == 0,
        "evaluation_has_no_image_selection": run_manifest.get("complete_image_selection") is False,
        "evaluation_has_no_outcome_ranked_visuals": run_manifest.get(
            "outcome_ranked_visual_selection"
        )
        is False,
        "evaluation_has_no_aesthetic_scoring": run_manifest.get("aesthetic_scoring") is False,
        "evaluation_metrics_are_fixed": run_manifest.get("evaluation_metrics")
        == ["CLIP-L/14", "CLIP-B/32"],
        "main_figure_prompt_is_preregistered": run_manifest.get("main_figure_prompt_id")
        == f"prompt_{int(run_manifest.get('prompt_offset', -1)):04d}",
        "supplement_prompts_are_preregistered": run_manifest.get(
            "supplement_figure_prompt_ids"
        )
        == [
            f"prompt_{int(run_manifest.get('prompt_offset', -1)) + offset:04d}"
            for offset in (1, 2, 3)
        ],
        "evaluation_fixes_t2i_scaffold": run_manifest.get("fixed_t2i_scaffold") is True,
        "evaluation_orders_visual_positions_only": run_manifest.get("ordered_action_space")
        == "256 visual-code positions; four T2I format tokens fixed",
        "evaluation_prompt_count_matches": run_manifest.get("prompt_count") == args.expected_prompts,
        "matched_training_contract_passed": training_audit.get("passed") is True,
        "matched_training_contract_hash_intact": training_audit_hash
        == run_manifest.get("training_contract_audit_sha256"),
    }
    passed = all(checks.values())
    report = {
        "passed": passed,
        "role": "prespecified manuscript promotion gate",
        "checks": checks,
        "clip_l14": clip_l,
        "clip_b32": clip_b,
        "order_divergence": order_delta,
        "evidence_sha256": {
            "paired": sha256(args.paired),
            "divergence": sha256(args.divergence),
            "controller": sha256(args.controller),
            "run_manifest": sha256(args.run_manifest),
            "training_contract_audit": training_audit_hash,
        },
        "failure_policy": (
            "Keep this untouched test result. Any follow-up changes must be selected "
            "on development data and evaluated on a new untouched confirmation split."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    ready_marker = args.output.parent / "MANUSCRIPT_PROMOTION_READY"
    failed_marker = args.output.parent / "RESULT_COMPLETE_NOT_PROMOTED"
    for stale in (ready_marker, failed_marker):
        if stale.exists():
            stale.unlink()
    marker = ready_marker if passed else failed_marker
    marker.write_text("\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
