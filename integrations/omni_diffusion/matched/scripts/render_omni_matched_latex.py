#!/usr/bin/env python3
"""Render promoted matched Omni results into shared LaTeX macro fragments."""

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


def find_comparison(payload: dict, metric: str, baseline: str, method: str) -> dict:
    for row in payload.get("comparisons_by_metric", {}).get(metric, []):
        if row.get("baseline") == baseline and row.get("method") == method:
            return row
    raise ValueError(f"missing {metric}: {baseline} vs {method}")


def value(number: float, best: bool) -> str:
    formatted = f"{number:.5f}"
    return rf"\bestcell{{{formatted}}}" if best else formatted


def latex_row(*cells: str) -> str:
    return " & ".join(cells) + r" \\"


def macro(name: str, body: str) -> str:
    return rf"\newcommand{{\{name}}}{{%" + "\n" + body + "%\n}\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--promotion-dir", type=Path, required=True)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path, required=True)
    args = parser.parse_args()

    marker = args.promotion_dir / "MANUSCRIPT_PROMOTION_READY"
    if not marker.is_file():
        raise SystemExit("refusing to render unpromoted Omni results")
    report = json.loads((args.promotion_dir / "promotion_report.json").read_text())
    if report.get("passed") is not True:
        raise SystemExit("promotion report did not pass")
    expected_paired_hash = report.get("evidence_sha256", {}).get("paired")
    if expected_paired_hash != sha256(args.paired):
        raise SystemExit("paired result does not match the promoted evidence hash")

    paired = json.loads(args.paired.read_text(encoding="utf-8"))
    clip_l = find_comparison(paired, "clip_cosine", BASELINE, METHOD)
    clip_b = find_comparison(paired, "clip_b32_cosine", BASELINE, METHOD)
    random_l = find_comparison(paired, "clip_cosine", "random", BASELINE)
    random_b = find_comparison(paired, "clip_b32_cosine", "random", BASELINE)
    comparisons = (clip_l, clip_b, random_l, random_b)
    if not all(row.get("matched_prompts") == 96 for row in comparisons):
        raise SystemExit("formal Omni table requires 96 paired prompts")

    random_l_mean = float(random_l["baseline_mean"])
    random_b_mean = float(random_b["baseline_mean"])
    confidence_l = float(clip_l["baseline_mean"])
    confidence_b = float(clip_b["baseline_mean"])
    dprm_l = float(clip_l["method_mean"])
    dprm_b = float(clip_b["method_mean"])
    l_best = max(random_l_mean, confidence_l, dprm_l)
    b_best = max(random_b_mean, confidence_b, dprm_b)
    rows = [
        latex_row(
            "Random",
            value(random_l_mean, random_l_mean == l_best),
            value(random_b_mean, random_b_mean == b_best),
            "1",
        ),
        latex_row(
            "Omni default",
            value(confidence_l, confidence_l == l_best),
            value(confidence_b, confidence_b == b_best),
            "1",
        ),
        latex_row(
            "DPRM",
            value(dprm_l, dprm_l == l_best),
            value(dprm_b, dprm_b == b_best),
            "1",
        ),
    ]

    pairs = [
        (confidence_l, dprm_l),
        (0.658, 0.692),
        (0.4735, 0.4892),
        (0.696, 0.710),
    ]
    confidence_aggregate = sum(c / max(c, d) for c, d in pairs) / len(pairs)
    dprm_aggregate = sum(d / max(c, d) for c, d in pairs) / len(pairs)
    aggregate = latex_row(
        "Equal-weight mean after row-wise normalization",
        value(confidence_aggregate, confidence_aggregate >= dprm_aggregate),
        value(dprm_aggregate, dprm_aggregate >= confidence_aggregate),
    )

    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    args.rows_output.write_text(
        macro("omnimatchedrows", "\n".join(rows)), encoding="utf-8"
    )
    args.aggregate_output.write_text(
        macro("multimodalaggregaterow", aggregate), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
