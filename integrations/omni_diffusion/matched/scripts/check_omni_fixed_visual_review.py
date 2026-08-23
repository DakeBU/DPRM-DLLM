#!/usr/bin/env python3
"""Gate pre-registered Omni supplementary images on blinded human ratings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_score(value: str, field: str, prompt_id: str) -> float:
    try:
        score = float(value)
    except ValueError as error:
        raise ValueError(f"{prompt_id}: invalid {field}: {value!r}") from error
    if not 1.0 <= score <= 5.0:
        raise ValueError(f"{prompt_id}: {field} must be in [1, 5]")
    return score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-score", type=float, default=3.0)
    args = parser.parse_args()

    selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    fixed_ids = [str(value) for value in selection.get("supplement_fixed_prompt_ids", [])]
    if not fixed_ids:
        raise ValueError("selection manifest has no pre-registered supplementary prompts")

    with args.ratings.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames or []
        rows = {str(row.get("prompt_id", "")): row for row in reader}
    labels = sorted(
        field.removesuffix("_recognizable_yes_no")
        for field in fieldnames
        if field.endswith("_recognizable_yes_no")
    )
    if not labels:
        raise ValueError("ratings file has no blinded recognizability columns")

    checks: list[dict] = []
    passed = True
    for prompt_id in fixed_ids:
        if prompt_id not in rows:
            raise ValueError(f"ratings file is missing fixed prompt {prompt_id}")
        row = rows[prompt_id]
        for label in labels:
            recognizable = str(row[f"{label}_recognizable_yes_no"]).strip().lower()
            if recognizable not in {"yes", "no"}:
                raise ValueError(
                    f"{prompt_id}: {label}_recognizable_yes_no must be yes or no"
                )
            scores = {
                axis: parse_score(row[f"{label}_{axis}_1to5"], axis, prompt_id)
                for axis in ("semantic_alignment", "visual_coherence", "artifact_free")
            }
            item_passed = recognizable == "yes" and all(
                value >= args.minimum_score for value in scores.values()
            )
            passed = passed and item_passed
            checks.append(
                {
                    "prompt_id": prompt_id,
                    "blind_column": label,
                    "recognizable": recognizable == "yes",
                    "scores": scores,
                    "passed": item_passed,
                }
            )

    report = {
        "schema_version": 1,
        "passed": passed,
        "review_scope": "pre-registered supplementary prompts, blinded columns",
        "fixed_prompt_ids": fixed_ids,
        "minimum_score": args.minimum_score,
        "checks": checks,
        "method_identity_used_for_review": False,
        "outcome_ranked_replacement_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pending = args.output.parent / "VISUAL_REVIEW_PENDING"
    ready = args.output.parent / "SUPPLEMENT_VISUAL_READY"
    failed = args.output.parent / "SUPPLEMENT_VISUAL_FAILED"
    for marker in (pending, ready, failed):
        if marker.exists():
            marker.unlink()
    (ready if passed else failed).write_text("\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
