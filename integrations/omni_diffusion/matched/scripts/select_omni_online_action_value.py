#!/usr/bin/env python3
"""Select an Omni visual-token action with online terminal-value estimates."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np


def paired_summary(values: np.ndarray, *, rng: np.random.Generator, resamples: int) -> dict[str, Any]:
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean_delta": float(values.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "wins": int(np.sum(values > 1e-12)),
        "ties": int(np.sum(np.abs(values) <= 1e-12)),
        "losses": int(np.sum(values < -1e-12)),
    }


def action_metadata(row: dict[str, Any]) -> dict[str, Any]:
    override = dict(row.get("counterfactual_override") or {})
    actions = override.get("actions") or []
    if actions:
        return dict(actions[0])
    return override


def candidate_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["prompt"]), int(row["seed"])


def index_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    indexed = {candidate_key(row): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("duplicate prompt/seed records")
    return indexed


def shared_state_digest(rows: list[dict[str, Any]], key: tuple[str, int]) -> tuple[str, str]:
    """Verify that all forced actions branch from the same provisional canvas."""
    actions = [action_metadata(row) for row in rows]
    token_arrays = [action.get("shared_provisional_visual_token_ids") for action in actions]
    available_token_arrays = [tokens for tokens in token_arrays if tokens is not None]
    default_fields = (
        "default_candidate_index",
        "default_sequence_position",
        "default_visual_index",
        "default_confidence",
        "default_raw_order_score",
    )
    reference_default = {field: actions[0].get(field) for field in default_fields}
    for action in actions[1:]:
        current_default = {field: action.get(field) for field in default_fields}
        if current_default != reference_default:
            raise ValueError(f"forced branches disagree on the default action for {key}")

    if available_token_arrays:
        reference = available_token_arrays[0]
        if any(tokens != reference for tokens in available_token_arrays[1:]):
            raise ValueError(f"forced branches do not share the same action state for {key}")
        digest_payload = reference
        verification = f"full_canvas:{len(available_token_arrays)}/{len(rows)}"
    else:
        digest_payload = reference_default
        verification = "default_action_only"
    serialized = json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest(), verification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--branch-methods", nargs="+", required=True)
    parser.add_argument(
        "--guidance-grid",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    )
    parser.add_argument("--fixed-guidance", type=float)
    parser.add_argument("--reward-scale", type=float, default=0.03)
    parser.add_argument("--selection-metric", default="clip_cosine")
    parser.add_argument("--check-metric", default="clip_b32_cosine")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    if args.reward_scale <= 0:
        raise SystemExit("--reward-scale must be positive")

    payload = json.loads(args.records.read_text(encoding="utf-8"))
    confidence = index_rows(payload["confidence"])
    branches = {method: index_rows(payload[method]) for method in args.branch_methods}
    if any(set(rows) != set(confidence) for rows in branches.values()):
        raise SystemExit("confidence and branch prompt/seed keys do not match")

    rng = np.random.default_rng(args.seed)
    metric_names = (args.selection_metric, args.check_metric)
    guidance_values = [args.fixed_guidance] if args.fixed_guidance is not None else args.guidance_grid
    evaluations: list[dict[str, Any]] = []
    per_guidance_records: dict[str, list[dict[str, Any]]] = {}

    for guidance in guidance_values:
        records: list[dict[str, Any]] = []
        method_counts: Counter[str] = Counter()
        metric_values = {
            metric: {"confidence": [], "uniform": [], "reward_only": [], "dprm": []}
            for metric in metric_names
        }
        confidence_gaps: list[float] = []
        for key in sorted(confidence):
            baseline = confidence[key]
            candidates = [("confidence", baseline)] + [
                (method, branches[method][key]) for method in args.branch_methods
            ]
            state_digest, state_verification = shared_state_digest(
                [row for _, row in candidates[1:]], key
            )
            baseline_action = action_metadata(candidates[1][1])
            base_scores = [float(baseline_action["default_raw_order_score"])]
            for _, row in candidates[1:]:
                action = action_metadata(row)
                if not action.get("applied", True):
                    raise ValueError(f"forced action was not applied for {key}")
                base_scores.append(float(action["raw_order_score"]))
            rewards = [float(row[args.selection_metric]) for _, row in candidates]
            reward_advantages = np.asarray(rewards) - rewards[0]
            adjusted = np.asarray(base_scores) + float(guidance) * reward_advantages / args.reward_scale
            selected_index = int(np.argmax(adjusted))
            reward_only_index = int(np.argmax(rewards))
            selected_method, selected = candidates[selected_index]
            candidate_actions = [
                {
                    "method": "confidence",
                    "visual_index": int(baseline_action["default_visual_index"]),
                    "confidence": float(baseline_action["default_confidence"]),
                    "raw_order_score": float(baseline_action["default_raw_order_score"]),
                    "rank_quantile": 1.0,
                }
            ]
            candidate_actions.extend(
                {"method": method, **action_metadata(row)}
                for method, row in candidates[1:]
            )
            method_counts[selected_method] += 1
            if selected_index:
                confidence_gaps.append(
                    float(action_metadata(selected)["confidence_gap_from_default"])
                )

            record: dict[str, Any] = {
                "prompt": key[0],
                "seed": key[1],
                "selected_method": selected_method,
                "selected_index": selected_index,
                "reward_only_index": reward_only_index,
                "reward_only_method": candidates[reward_only_index][0],
                "confidence_image_path": baseline["image_path"],
                "selected_image_path": selected["image_path"],
                "confidence_history_frame_paths": baseline.get("history_frame_paths", []),
                "selected_history_frame_paths": selected.get("history_frame_paths", []),
                "shared_action_canvas_path": candidates[1][1].get("shared_action_canvas_path"),
                "shared_action_state_sha256": state_digest,
                "shared_action_state_verification": state_verification,
                "candidate_actions": candidate_actions,
                "selected_action": candidate_actions[selected_index],
                "base_order_scores": base_scores,
                "terminal_rewards": rewards,
                "adjusted_order_scores": adjusted.tolist(),
            }
            for metric in metric_names:
                values = np.asarray([float(row[metric]) for _, row in candidates])
                metric_values[metric]["confidence"].append(float(values[0]))
                metric_values[metric]["uniform"].append(float(values.mean()))
                metric_values[metric]["reward_only"].append(float(values[reward_only_index]))
                metric_values[metric]["dprm"].append(float(values[selected_index]))
                record[f"confidence_{metric}"] = float(values[0])
                record[f"uniform_{metric}"] = float(values.mean())
                record[f"reward_only_{metric}"] = float(values[reward_only_index])
                record[f"dprm_{metric}"] = float(values[selected_index])
            records.append(record)

        metrics: dict[str, Any] = {}
        for metric in metric_names:
            arrays = {
                name: np.asarray(values, dtype=np.float64)
                for name, values in metric_values[metric].items()
            }
            metrics[metric] = {
                "confidence_mean": float(arrays["confidence"].mean()),
                "uniform_mean": float(arrays["uniform"].mean()),
                "reward_only_mean": float(arrays["reward_only"].mean()),
                "dprm_mean": float(arrays["dprm"].mean()),
                "dprm_minus_confidence": paired_summary(
                    arrays["dprm"] - arrays["confidence"],
                    rng=rng,
                    resamples=args.bootstrap,
                ),
                "dprm_minus_uniform": paired_summary(
                    arrays["dprm"] - arrays["uniform"],
                    rng=rng,
                    resamples=args.bootstrap,
                ),
                "dprm_minus_reward_only": paired_summary(
                    arrays["dprm"] - arrays["reward_only"],
                    rng=rng,
                    resamples=args.bootstrap,
                ),
            }
        tag = f"g{guidance:g}"
        evaluations.append(
            {
                "guidance": float(guidance),
                "selected_action_counts": dict(method_counts),
                "override_fraction": float(1.0 - method_counts["confidence"] / len(records)),
                "mean_selected_confidence_gap": (
                    float(np.mean(confidence_gaps)) if confidence_gaps else 0.0
                ),
                "metrics": metrics,
            }
        )
        per_guidance_records[tag] = records

    if args.fixed_guidance is None:
        eligible = [
            row
            for row in evaluations
            if row["metrics"][args.selection_metric]["dprm_minus_confidence"]["mean_delta"] > 0
            and row["metrics"][args.check_metric]["dprm_minus_confidence"]["mean_delta"] >= 0
        ]
        if not eligible:
            raise SystemExit("no guidance improves the selection metric without lowering the check metric")
        selected_eval = max(
            eligible,
            key=lambda row: row["metrics"][args.selection_metric]["dprm_minus_confidence"]["mean_delta"],
        )
    else:
        selected_eval = evaluations[0]
    selected_tag = f"g{selected_eval['guidance']:g}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_root = args.output_dir / "selected_images"
    selected_root.mkdir(exist_ok=True)
    for index, row in enumerate(per_guidance_records[selected_tag]):
        destination = selected_root / f"prompt_{index:04d}.png"
        shutil.copy2(row["selected_image_path"], destination)
        row["published_image_path"] = str(destination)

    result = {
        "format": "omni_online_rank_bucket_dprm_v1",
        "design": (
            "one action-conditioned confidence continuation per predeclared phase/rank bucket; "
            "DPRM selects by base order score plus normalized terminal advantage"
        ),
        "prompt_count": len(confidence),
        "candidate_methods": ["confidence", *args.branch_methods],
        "candidate_rollouts_per_prompt": 1 + len(args.branch_methods),
        "complete_candidate_path_selection": True,
        "human_selection": False,
        "action_phase_step": 96,
        "selection_metric": args.selection_metric,
        "independent_check_metric": args.check_metric,
        "reward_scale": args.reward_scale,
        "bootstrap_iterations": args.bootstrap,
        "evaluations": evaluations,
        "selected_guidance": selected_eval["guidance"],
        "selected_evaluation": selected_eval,
        "records": per_guidance_records[selected_tag],
    }
    (args.output_dir / "online_action_value_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Omni online action-value controller",
        "",
        f"Selected guidance: `{selected_eval['guidance']:g}`",
        "",
        "| Guidance | Metric | Confidence | Uniform | Reward-only | DPRM | DPRM - confidence [95% CI] |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in evaluations:
        for metric in metric_names:
            values = row["metrics"][metric]
            delta = values["dprm_minus_confidence"]
            lines.append(
                f"| {row['guidance']:g} | {metric} | {values['confidence_mean']:.5f} | "
                f"{values['uniform_mean']:.5f} | {values['reward_only_mean']:.5f} | "
                f"{values['dprm_mean']:.5f} | "
                f"{delta['mean_delta']:+.5f} [{delta['ci95_low']:+.5f}, {delta['ci95_high']:+.5f}] |"
            )
    (args.output_dir / "online_action_value_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
