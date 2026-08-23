#!/usr/bin/env python3
"""Render a numerically promoted frozen-host Omni comparison as LaTeX macros."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BASELINE = "progressive_confidence"
METHOD = "dprm_confidence_warmup"


def find_comparison(payload: dict, metric: str, baseline: str, method: str) -> dict:
    for row in payload.get("comparisons_by_metric", {}).get(metric, []):
        if row.get("baseline") == baseline and row.get("method") == method:
            return row
    raise ValueError(f"missing {metric}: {baseline} vs {method}")


def checked_match(observed: dict, expected: dict, label: str) -> None:
    keys = ("baseline", "method", "matched_prompts", "baseline_mean", "method_mean", "mean_delta")
    for key in keys:
        if observed.get(key) != expected.get(key):
            raise SystemExit(f"{label} promotion record differs at {key}")


def value(number: float, best: bool) -> str:
    text = f"{number:.5f}"
    return rf"\bestcell{{{text}}}" if best else text


def latex_row(*cells: str) -> str:
    return " & ".join(cells) + r" \\"


def macro(name: str, body: str) -> str:
    return rf"\newcommand{{\{name}}}{{%" + "\n" + body + "%\n}\n"


def llada_v_pairs(path: Path) -> list[tuple[float, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    llada_v = payload.get("llada_v")
    if not isinstance(llada_v, dict):
        raise SystemExit("multimodal summary has no llada_v result block")
    pairs: list[tuple[float, float]] = []
    for task in ("ai2d", "realworldqa", "chartqa_frozen_transfer"):
        row = llada_v.get(task)
        if not isinstance(row, dict) or int(row.get("n", 0)) <= 0:
            continue
        try:
            pairs.append((float(row["confidence"]), float(row["dprm_confidence"])))
        except (KeyError, TypeError, ValueError) as error:
            raise SystemExit(f"invalid confidence/DPRM values for {task}") from error
    realworld = llada_v.get("realworldqa", {})
    if not isinstance(realworld, dict) or int(realworld.get("n", 0)) <= 0:
        raise SystemExit("multimodal summary has no complete RealWorldQA result")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numeric-promotion", type=Path, required=True)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--multimodal-summary", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path, required=True)
    args = parser.parse_args()

    promotion = json.loads(args.numeric_promotion.read_text(encoding="utf-8"))
    if promotion.get("passed") is not True:
        raise SystemExit("refusing to render an unpromoted Omni comparison")
    paired = json.loads(args.paired.read_text(encoding="utf-8"))
    clip_l = find_comparison(paired, "clip_cosine", BASELINE, METHOD)
    clip_b = find_comparison(paired, "clip_b32_cosine", BASELINE, METHOD)
    checked_match(clip_l, promotion["primary"], "CLIP-L/14")
    checked_match(clip_b, promotion["secondary"], "CLIP-B/32")

    comparisons = (clip_l, clip_b)
    counts = {int(row.get("matched_prompts", -1)) for row in comparisons}
    if len(counts) != 1 or next(iter(counts)) <= 0:
        raise SystemExit("Omni rows do not share one complete prompt set")

    confidence_values = (float(clip_l["baseline_mean"]), float(clip_b["baseline_mean"]))
    dprm_values = (float(clip_l["method_mean"]), float(clip_b["method_mean"]))
    best = tuple(max(values) for values in zip(confidence_values, dprm_values))
    rows = []
    for label, values in (
        ("Omni default", confidence_values),
        ("DPRM", dprm_values),
    ):
        rows.append(latex_row(label, value(values[0], values[0] == best[0]), value(values[1], values[1] == best[1]), "1"))

    pairs = [(confidence_values[0], dprm_values[0]), *llada_v_pairs(args.multimodal_summary)]
    confidence_aggregate = sum(c / max(c, d) for c, d in pairs) / len(pairs)
    dprm_aggregate = sum(d / max(c, d) for c, d in pairs) / len(pairs)
    aggregate = latex_row(
        "Equal-weight mean after row-wise normalization",
        value(confidence_aggregate, confidence_aggregate >= dprm_aggregate),
        value(dprm_aggregate, dprm_aggregate >= confidence_aggregate),
    )

    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    args.rows_output.write_text(macro("omnimatchedrows", "\n".join(rows)), encoding="utf-8")
    args.aggregate_output.write_text(macro("multimodalaggregaterow", aggregate), encoding="utf-8")


if __name__ == "__main__":
    main()
