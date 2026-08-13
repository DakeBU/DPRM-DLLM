#!/usr/bin/env python3
"""Apply a predeclared acceptance rule to an Omni paired gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-mean-delta", type=float, default=0.0)
    parser.add_argument("--min-win-rate", type=float, default=0.5)
    parser.add_argument("--min-ci-low", type=float, default=-0.01)
    args = parser.parse_args()
    payload = json.loads(args.paired_summary.read_text(encoding="utf-8"))
    matches = [
        item
        for item in payload["comparisons_by_metric"]["clip_cosine"]
        if item["baseline"] == "progressive_confidence"
        and item["method"] == "dprm_confidence_warmup"
    ]
    if len(matches) != 1:
        raise RuntimeError("paired summary does not contain the required comparison")
    result = matches[0]
    denominator = result["wins"] + result["ties"] + result["losses"]
    win_rate = result["wins"] / max(denominator, 1)
    passed = (
        result["mean_delta"] > args.min_mean_delta
        and win_rate >= args.min_win_rate
        and result["ci95_low"] >= args.min_ci_low
    )
    decision = {
        "format": "omni_single_path_gate_v1",
        "passed": passed,
        "rule": {
            "mean_delta_strictly_greater_than": args.min_mean_delta,
            "win_rate_at_least": args.min_win_rate,
            "ci95_low_at_least": args.min_ci_low,
        },
        "observed": {**result, "win_rate": win_rate},
    }
    args.output.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
