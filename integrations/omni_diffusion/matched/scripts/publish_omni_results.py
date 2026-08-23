#!/usr/bin/env python3
"""Publish a promoted Omni confirmation run to canonical release files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


BASELINE = "progressive_confidence"
METHOD = "dprm_confidence_warmup"
FIELDS = (
    "host", "task", "method", "variant", "metric", "direction", "value",
    "ci95_low", "ci95_high", "n", "protocol",
)


def comparison(payload: dict, metric: str, baseline: str, method: str) -> dict:
    for row in payload["comparisons_by_metric"][metric]:
        if row.get("baseline") == baseline and row.get("method") == method:
            return row
    raise ValueError(f"missing {metric}: {baseline} vs {method}")


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--promotion", type=Path, required=True)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--paper-results", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--multimodal-summary", type=Path, required=True)
    args = parser.parse_args()

    promotion = json.loads(args.promotion.read_text(encoding="utf-8"))
    if promotion.get("passed") is not True:
        raise SystemExit("refusing to publish an unpromoted Omni result")
    paired = json.loads(args.paired.read_text(encoding="utf-8"))
    metric_names = {
        "clip_cosine": "clip_l14_cosine",
        "clip_b32_cosine": "clip_b32_cosine",
    }
    canonical_rows: list[dict[str, object]] = []
    summary = {"protocol": "untouched_partiprompts_confirmation", "metrics": {}}
    expected_n = None
    for source_metric, metric in metric_names.items():
        dprm = comparison(paired, source_metric, BASELINE, METHOD)
        n = int(dprm["matched_prompts"])
        if n != 512:
            raise SystemExit(f"formal Omni result requires 512 paired prompts, got {n}")
        expected_n = n if expected_n is None else expected_n
        if expected_n != n:
            raise SystemExit("Omni metrics use different prompt counts")
        values = {
            "confidence": float(dprm["baseline_mean"]),
            "dprm_confidence": float(dprm["method_mean"]),
        }
        summary["metrics"][metric] = {
            **values,
            "paired_delta": float(dprm["mean_delta"]),
            "ci95_low": float(dprm["ci95_low"]),
            "ci95_high": float(dprm["ci95_high"]),
            "n": n,
        }
        labels = {
            "confidence": "Omni default",
            "dprm_confidence": "DPRM",
        }
        for variant, value in values.items():
            canonical_rows.append(
                {
                    "host": "Omni-Diffusion",
                    "task": "PartiPrompts",
                    "method": labels[variant],
                    "variant": variant,
                    "metric": metric,
                    "direction": "higher",
                    "value": value,
                    "ci95_low": "",
                    "ci95_high": "",
                    "n": n,
                    "protocol": "untouched_confirmation",
                }
            )
        canonical_rows.append(
            {
                "host": "Omni-Diffusion",
                "task": "PartiPrompts",
                "method": "DPRM",
                "variant": "dprm_confidence",
                "metric": f"{metric}_paired_delta",
                "direction": "higher",
                "value": float(dprm["mean_delta"]),
                "ci95_low": float(dprm["ci95_low"]),
                "ci95_high": float(dprm["ci95_high"]),
                "n": n,
                "protocol": "untouched_confirmation",
            }
        )

    with args.paper_results.open(newline="", encoding="utf-8") as handle:
        existing = [row for row in csv.DictReader(handle) if row["host"] != "Omni-Diffusion"]
    temporary_csv = args.paper_results.with_suffix(args.paper_results.suffix + ".tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(existing + canonical_rows)
    temporary_csv.replace(args.paper_results)

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    experiment = next(row for row in registry["experiments"] if row["id"] == "omni_diffusion")
    for variant in experiment["variants"]:
        if variant["id"] in {"confidence", "dprm_confidence"}:
            variant["status"] = "reported"
        elif variant["status"] == "formal_pending":
            variant["status"] = "implemented_control"
    write_json_atomic(args.registry, registry)

    multimodal = json.loads(args.multimodal_summary.read_text(encoding="utf-8"))
    multimodal["omni_diffusion"] = summary
    write_json_atomic(args.multimodal_summary, multimodal)


if __name__ == "__main__":
    main()
