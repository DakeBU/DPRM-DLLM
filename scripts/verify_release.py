#!/usr/bin/env python3
"""Validate the public DPRM release manifest and canonical results."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "reproducibility" / "experiments.json"
RESULTS = ROOT / "results" / "paper_results.csv"
EXPECTED_HOSTS = {
    "PUMA",
    "DMPO",
    "Prism",
    "DPLM-2 Bit",
    "DCM",
    "GenMol V2",
    "SDPO",
    "Omni-Diffusion",
    "LLaDA-V",
}
TEXT_SUFFIXES = {
    ".cfg", ".cff", ".csv", ".json", ".md", ".py", ".sh", ".tex",
    ".toml", ".txt", ".yaml", ".yml",
}
FORBIDDEN = (
    "/home/" + "nitanda_sub",
    "/var/tmp/" + "nitanda_sub",
    "T" + "PAMI_",
    "t" + "pami_",
    "DPRM_" + "DPRM",
    "ICML " + "2026",
    "<" + "PATH",
    "/path" + "/to/",
    "your-entity-" + "here",
)
REQUIRED_REPRO_FILES = (
    "integrations/dcm/scripts/run_preference_sweep.sh",
    "integrations/dcm/scripts/run_terminal_calibration.sh",
    "integrations/genmol/scripts/run_preference_sweep.sh",
    "integrations/genmol/overlay/src/genmol/utils/utils_data.py",
    "reproducibility/scientific_preference_sweeps.json",
    "scripts/sync_scientific_results.py",
)


def fail(message: str) -> None:
    raise AssertionError(message)


def main() -> None:
    for relative_path in REQUIRED_REPRO_FILES:
        if not (ROOT / relative_path).is_file():
            fail(f"missing reproduction file: {relative_path}")

    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    experiments = payload.get("experiments", [])
    if len(experiments) != 9:
        fail(f"expected 9 experiments, found {len(experiments)}")

    ids = [row["id"] for row in experiments]
    if len(ids) != len(set(ids)):
        fail("experiment ids are not unique")
    hosts = {row["host"] for row in experiments}
    if hosts != EXPECTED_HOSTS:
        fail(f"registry hosts differ: {sorted(hosts ^ EXPECTED_HOSTS)}")

    for experiment in experiments:
        commit = experiment.get("upstream_commit", "")
        if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
            fail(f"{experiment['id']} has invalid upstream_commit")
        variants = experiment.get("variants", [])
        if len(variants) != 4:
            fail(f"{experiment['id']} has {len(variants)} variants, expected 4")
        variant_ids = [row["id"] for row in variants]
        if len(variant_ids) != len(set(variant_ids)):
            fail(f"{experiment['id']} has duplicate variant ids")
        for field in ("integration_readme", "entrypoint", "result_file"):
            path = ROOT / experiment[field]
            if not path.is_file():
                fail(f"{experiment['id']} missing {field}: {path}")
        for variant in variants:
            if not variant.get("command", "").strip():
                fail(f"{experiment['id']}/{variant['id']} has no command")
            if variant.get("status") not in {
                "reported",
                "reported_ai2d",
                "reported_development_gate",
                "implemented_control",
            }:
                fail(f"{experiment['id']}/{variant['id']} has invalid status")

    with RESULTS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result_hosts = {row["host"] for row in rows}
    if not EXPECTED_HOSTS.issubset(result_hosts):
        fail(f"results missing hosts: {sorted(EXPECTED_HOSTS - result_hosts)}")
    for row_number, row in enumerate(rows, start=2):
        try:
            float(row["value"])
        except ValueError as error:
            fail(f"invalid result value on CSV line {row_number}: {error}")
        if row["direction"] not in {"higher", "lower"}:
            fail(f"invalid metric direction on CSV line {row_number}")

    violations = []
    scan_roots = [
        ROOT / "README.md",
        ROOT / "src",
        ROOT / "integrations",
        ROOT / "reproducibility",
        ROOT / "results",
        ROOT / "scripts",
        ROOT / "tests",
    ]
    for scan_root in scan_roots:
        paths = [scan_root] if scan_root.is_file() else scan_root.rglob("*")
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if path.resolve() == Path(__file__).resolve():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for token in FORBIDDEN:
                if token in text:
                    violations.append(f"{path.relative_to(ROOT)}: {token}")
    if violations:
        fail("release text contains private/history residue:\n  " + "\n  ".join(violations))

    print(f"release audit passed: 9 hosts, 36 variants, {len(rows)} result rows")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, KeyError, json.JSONDecodeError) as error:
        print(f"release audit failed: {error}", file=sys.stderr)
        raise SystemExit(1)
