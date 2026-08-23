#!/usr/bin/env python3
"""Select an Omni training endpoint using only frozen development results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


BASELINE = "progressive_confidence"
METHOD = "dprm_confidence_warmup"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparison(payload: dict, metric: str) -> dict:
    for row in payload.get("comparisons_by_metric", {}).get(metric, []):
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
    parser.add_argument("--candidate", nargs=3, action="append", metavar=("STEP", "PAIRED", "DIVERGENCE"), required=True)
    parser.add_argument("--expected-prompts", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidates = []
    for step_text, paired_text, divergence_text in args.candidate:
        step = int(step_text)
        paired_path = Path(paired_text)
        divergence_path = Path(divergence_text)
        paired = json.loads(paired_path.read_text(encoding="utf-8"))
        order = json.loads(divergence_path.read_text(encoding="utf-8"))
        clip_l = comparison(paired, "clip_cosine")
        clip_b = comparison(paired, "clip_b32_cosine")
        order_delta = divergence(order)
        clip_l_delta = float(clip_l["mean_delta"])
        clip_b_delta = float(clip_b["mean_delta"])
        paired_complete = (
            clip_l.get("matched_prompts") == args.expected_prompts
            and clip_b.get("matched_prompts") == args.expected_prompts
        )
        measurable_order_change = (
            float(order_delta.get("direct_override_fraction", 0.0)) >= 0.001
            and float(order_delta.get("has_direct_override", 0.0)) >= 0.20
        )
        eligible = (
            paired_complete
            and measurable_order_change
            and clip_l_delta > 0.0
            and clip_b_delta > 0.0
        )
        candidates.append(
            {
                "step": step,
                "eligible": eligible,
                "selection_score": (clip_l_delta + clip_b_delta) / 2.0,
                "clip_l14": clip_l,
                "clip_b32": clip_b,
                "order_divergence": order_delta,
                "evidence_sha256": {
                    "paired": sha256(paired_path),
                    "divergence": sha256(divergence_path),
                },
            }
        )

    eligible = [row for row in candidates if row["eligible"]]
    selected = max(eligible, key=lambda row: (row["selection_score"], -row["step"])) if eligible else None
    payload = {
        "protocol": "development-only Omni training-endpoint selection",
        "expected_prompts": args.expected_prompts,
        "eligibility": (
            "complete paired development results, positive CLIP-L/14 and CLIP-B/32 "
            "mean deltas, and measurable order divergence"
        ),
        "selection_rule": "largest equal-weight mean of the two CLIP deltas; earlier step breaks ties",
        "candidates": candidates,
        "selected_step": selected["step"] if selected else None,
        "confirmation_data_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
