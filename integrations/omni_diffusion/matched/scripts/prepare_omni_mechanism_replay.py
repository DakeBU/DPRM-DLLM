#!/usr/bin/env python3
"""Create deterministic replay jobs for frozen Omni mechanism cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--selection-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.case_manifest.read_text(encoding="utf-8"))
    summary = json.loads(args.selection_summary.read_text(encoding="utf-8"))
    records = summary["records"]
    jobs = []
    replay_cases = []
    for case in manifest["cases"]:
        record = records[int(case["record_index"])]
        method = str(record["selected_method"])
        if not method.startswith("step96_q"):
            raise ValueError(f"{case['id']} does not select a step-96 branch: {method}")
        quantile = method.rsplit("q", 1)[1]
        common = ["--fixed-t2i-scaffold", "--save-history-frames", "--history-frame-stride", "32", "--history-frame-limit", "12"]
        case_root = args.output_root / case["id"]
        jobs.append(
            {
                "output_dir": str(case_root / "confidence"),
                "prompt": record["prompt"],
                "order_policy": "progressive_confidence",
                "seed": int(record["seed"]),
                "steps": 260,
                "max_tokens": 260,
                "extra_args": common,
            }
        )
        jobs.append(
            {
                "output_dir": str(case_root / "dprm"),
                "prompt": record["prompt"],
                "order_policy": "progressive_confidence",
                "seed": int(record["seed"]),
                "steps": 260,
                "max_tokens": 260,
                "extra_args": common
                + [
                    "--force-order-step", "96",
                    "--force-confidence-quantile", quantile,
                    "--force-confidence-bins", "8",
                    "--force-rank-bins", "64",
                    "--force-aux-bins", "16",
                    "--require-forced-action",
                ],
            }
        )
        replay_cases.append(
            {
                **case,
                "prompt": record["prompt"],
                "seed": int(record["seed"]),
                "selected_method": method,
                "confidence_clip_l14": record["confidence_clip_cosine"],
                "dprm_clip_l14": record["dprm_clip_cosine"],
                "confidence_clip_b32": record["confidence_clip_b32_cosine"],
                "dprm_clip_b32": record["dprm_clip_b32_cosine"],
                "source_confidence_image_path": record["confidence_image_path"],
                "source_dprm_image_path": record["selected_image_path"],
                "confidence_dir": str(case_root / "confidence"),
                "dprm_dir": str(case_root / "dprm"),
            }
        )

    args.jobs.parent.mkdir(parents=True, exist_ok=True)
    with args.jobs.open("w", encoding="utf-8") as handle:
        for job in jobs:
            handle.write(json.dumps(job) + "\n")
    replay_manifest = {
        **{key: value for key, value in manifest.items() if key != "cases"},
        "cases": replay_cases,
    }
    (args.output_root / "replay_manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (args.output_root / "replay_manifest.json").write_text(
        json.dumps(replay_manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
