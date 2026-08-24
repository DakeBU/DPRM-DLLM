#!/usr/bin/env python3
"""Publish claim-eligible Omni online action-value results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--scored-records", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-tex", type=Path, required=True)
    parser.add_argument("--expected-prompts", type=int, default=512)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    scored = json.loads(args.scored_records.read_text(encoding="utf-8"))
    manifest = json.loads(args.run_manifest.read_text(encoding="utf-8"))
    if summary.get("format") != "omni_online_rank_bucket_dprm_v1":
        raise SystemExit("unexpected online DPRM summary format")
    if int(summary.get("prompt_count", -1)) != args.expected_prompts:
        raise SystemExit("online DPRM prompt count does not match the release contract")
    if int(summary.get("candidate_rollouts_per_prompt", -1)) != 5:
        raise SystemExit("the release contract requires the frozen five-path shortlist")
    if int(manifest.get("prompt_count", -1)) != args.expected_prompts:
        raise SystemExit("run manifest prompt count mismatch")
    if int(manifest.get("candidate_rollouts_per_prompt", -1)) != 5:
        raise SystemExit("run manifest shortlist mismatch")
    if float(manifest.get("fixed_guidance", -1)) != float(summary["selected_guidance"]):
        raise SystemExit("confirmation guidance is not bound to the selected summary")
    if manifest.get("random_control") is not True:
        raise SystemExit("formal Omni records require the random-order control")
    if len(scored.get("random", [])) != args.expected_prompts:
        raise SystemExit("missing random-order records")
    verification = [row.get("shared_action_state_verification") for row in summary["records"]]
    if verification != ["full_canvas:4/4"] * args.expected_prompts:
        raise SystemExit("formal Omni records require full shared-canvas verification")
    required_methods = list(summary.get("candidate_methods", []))
    if len(required_methods) != 5 or any(
        len(scored.get(method, [])) != args.expected_prompts for method in required_methods
    ):
        raise SystemExit("missing scored shortlist records")
    expected_keys = {
        (str(row["prompt"]), int(row["seed"])) for row in scored["confidence"]
    }
    for method in ["random", *required_methods[1:]]:
        keys = {(str(row["prompt"]), int(row["seed"])) for row in scored[method]}
        if keys != expected_keys:
            raise SystemExit(f"prompt/seed mismatch for {method}")

    selected = summary["selected_evaluation"]
    metrics = selected["metrics"]
    for metric in ("clip_cosine", "clip_b32_cosine"):
        scored_confidence = float(
            np.mean([float(row[metric]) for row in scored["confidence"]])
        )
        if not np.isclose(scored_confidence, metrics[metric]["confidence_mean"]):
            raise SystemExit(f"summary/record confidence mean mismatch for {metric}")
    random_means = {
        metric: float(np.mean([float(row[metric]) for row in scored["random"]]))
        for metric in ("clip_cosine", "clip_b32_cosine")
    }
    methods = {
        "random": {
            "label": "Random",
            "clip_l14": random_means["clip_cosine"],
            "clip_b32": random_means["clip_b32_cosine"],
            "paths": 1,
        },
        "confidence": {
            "label": "Omni default",
            "clip_l14": float(metrics["clip_cosine"]["confidence_mean"]),
            "clip_b32": float(metrics["clip_b32_cosine"]["confidence_mean"]),
            "paths": 1,
        },
        "uniform": {
            "label": "Uniform order",
            "clip_l14": float(metrics["clip_cosine"]["uniform_mean"]),
            "clip_b32": float(metrics["clip_b32_cosine"]["uniform_mean"]),
            "paths": 5,
        },
        "reward_only": {
            "label": "Reward-only BoN",
            "clip_l14": float(metrics["clip_cosine"]["reward_only_mean"]),
            "clip_b32": float(metrics["clip_b32_cosine"]["reward_only_mean"]),
            "paths": 5,
        },
        "dprm": {
            "label": "DPRM",
            "clip_l14": float(metrics["clip_cosine"]["dprm_mean"]),
            "clip_b32": float(metrics["clip_b32_cosine"]["dprm_mean"]),
            "paths": 5,
        },
    }
    if methods["dprm"]["clip_l14"] <= methods["confidence"]["clip_l14"]:
        raise SystemExit("DPRM does not improve the declared terminal reward")

    artifact: dict[str, Any] = {
        "format": "omni_online_action_value_release_v1",
        "prompt_count": args.expected_prompts,
        "checkpoint": f"omni_diffusion/{Path(manifest['checkpoint']).name}",
        "prompt_file_sha256": manifest["prompt_file_sha256"],
        "action_step": manifest["action_step"],
        "confidence_rank_quantiles": manifest["confidence_rank_quantiles"],
        "reward_scale": summary["reward_scale"],
        "guidance": summary["selected_guidance"],
        "candidate_paths": summary["candidate_rollouts_per_prompt"],
        "terminal_reward": summary["selection_metric"],
        "independent_check": summary["independent_check_metric"],
        "methods": methods,
        "paired_deltas": {
            metric: values["dprm_minus_confidence"] for metric, values in metrics.items()
        },
        "relative_improvements_percent": {
            "clip_l14": 100.0
            * (methods["dprm"]["clip_l14"] - methods["confidence"]["clip_l14"])
            / methods["confidence"]["clip_l14"],
            "clip_b32": 100.0
            * (methods["dprm"]["clip_b32"] - methods["confidence"]["clip_b32"])
            / methods["confidence"]["clip_b32"],
        },
        "selected_action_counts": selected["selected_action_counts"],
        "override_fraction": selected["override_fraction"],
        "source_hashes": {
            "summary": sha256(args.summary),
            "scored_records": sha256(args.scored_records),
            "run_manifest": sha256(args.run_manifest),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    lines = ["\\newcommand{\\omnimatchedrows}{%"]
    for key in ("random", "confidence", "uniform", "dprm"):
        row = methods[key]
        l14 = f"{row['clip_l14']:.5f}"
        b32 = f"{row['clip_b32']:.5f}"
        if key == "dprm":
            l14, b32 = f"\\bestcell{{{l14}}}", f"\\bestcell{{{b32}}}"
        lines.append(f"{row['label']} & {l14} & {b32} & {row['paths']} \\\\")
    lines.append("}")
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
