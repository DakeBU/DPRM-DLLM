#!/usr/bin/env python3
"""Publish audited LLaDA-V summaries to the canonical result registry."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = ("host", "task", "method", "variant", "metric", "direction", "value", "ci95_low", "ci95_high", "n", "protocol")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--paper-results", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))["llada_v"]
    ai2d = payload["ai2d"]
    chartqa = payload["chartqa_frozen_transfer"]
    rows = list(csv.DictReader(args.paper_results.open(newline="", encoding="utf-8")))
    replaced = {"AI2D", "ChartQA"}
    rows = [row for row in rows if not (row["host"] == "LLaDA-V" and row["task"] in replaced)]

    def add_pair(task: str, data: dict, protocol: str) -> None:
        n = int(data["n"])
        for method, variant, key in (("Confidence", "confidence", "confidence"), ("DPRM-confidence", "dprm_confidence", "dprm_confidence")):
            rows.append(dict(zip(FIELDS, ("LLaDA-V", task, method, variant, "accuracy", "higher", f'{float(data[key]):.6f}', "", "", str(n), protocol))))
        delta = data["paired_delta"]
        rows.append(dict(zip(FIELDS, ("LLaDA-V", task, "DPRM-confidence", "dprm_confidence", "paired_delta", "higher", f'{float(delta["mean"]):.6f}', f'{float(delta["ci95"][0]):.6f}', f'{float(delta["ci95"][1]):.6f}', str(n), protocol))))

    add_pair("AI2D", ai2d, "heldout_after_dev")
    add_pair("ChartQA", chartqa, "frozen_rwqa_controller_transfer")
    with args.paper_results.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    experiments = registry["experiments"] if isinstance(registry, dict) else registry
    host = next(row for row in experiments if row["id"] == "llada_v")
    for variant in host["variants"]:
        variant["status"] = "reported" if variant["id"] in {"confidence", "dprm_confidence"} else "implemented_control"
    args.registry.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
