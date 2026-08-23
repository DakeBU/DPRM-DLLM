#!/usr/bin/env python3
"""Fail unless the public release and manuscript are ready for submission."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def reported(status: str) -> bool:
    return status.startswith("reported")


def pdf_pages(path: Path) -> int:
    result = subprocess.run(
        ["pdfinfo", str(path)], check=True, capture_output=True, text=True
    )
    match = re.search(r"^Pages:\s+(\d+)\s*$", result.stdout, re.MULTILINE)
    if match is None:
        raise ValueError(f"cannot read page count from {path}")
    return int(match.group(1))


def audit(
    registry_path: Path,
    results_path: Path,
    artifact_manifest_path: Path,
    paper_root: Path,
) -> dict:
    errors: list[str] = []
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    experiments = registry.get("experiments", [])
    if len(experiments) != 9:
        errors.append(f"registry has {len(experiments)} hosts, expected 9")

    host_by_name = {experiment["host"]: experiment for experiment in experiments}
    for experiment in experiments:
        variants = {variant["id"]: variant for variant in experiment.get("variants", [])}
        pending = sorted(
            variant_id
            for variant_id, variant in variants.items()
            if variant.get("status") == "formal_pending"
        )
        if pending:
            errors.append(f"{experiment['id']} pending variants: {', '.join(pending)}")
        confidence = variants.get("confidence", {})
        dprm_variants = [
            variant
            for variant_id, variant in variants.items()
            if variant_id.startswith("dprm")
        ]
        if not reported(confidence.get("status", "")):
            errors.append(f"{experiment['id']} has no reported confidence result")
        if not any(reported(variant.get("status", "")) for variant in dprm_variants):
            errors.append(f"{experiment['id']} has no reported DPRM result")

    with results_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result_hosts = {row["host"] for row in rows}
    missing_result_hosts = sorted(set(host_by_name) - result_hosts)
    if missing_result_hosts:
        errors.append("hosts missing canonical result rows: " + ", ".join(missing_result_hosts))

    artifacts = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
    artifact_hosts = artifacts.get("hosts", {})
    if len(artifact_hosts) != 9:
        errors.append(f"artifact manifest has {len(artifact_hosts)} hosts, expected 9")
    pending_artifacts = sorted(
        host for host, entry in artifact_hosts.items() if entry.get("status") != "complete"
    )
    if pending_artifacts:
        errors.append("incomplete artifact hosts: " + ", ".join(pending_artifacts))

    main_tex = paper_root / "main_tpami.tex"
    main_pdf = paper_root / "main_tpami.pdf"
    generated = paper_root / "generated"
    manuscript_text = main_tex.read_text(encoding="utf-8")
    generated_text = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(generated.glob("*.tex"))
    )
    for token in ("in progress", "formal_pending", "PLACEHOLDER"):
        if token.lower() in (manuscript_text + "\n" + generated_text).lower():
            errors.append(f"manuscript contains unresolved token: {token}")
    if re.search(r"\\section\*?\{Acknowledg", manuscript_text, re.IGNORECASE):
        errors.append("main manuscript still contains acknowledgements")
    if not main_pdf.is_file():
        errors.append("main_tpami.pdf is missing")
        pages = None
    else:
        pages = pdf_pages(main_pdf)
        if pages > 12:
            errors.append(f"main_tpami.pdf has {pages} pages, limit is 12")

    return {
        "ready": not errors,
        "hosts": len(experiments),
        "result_rows": len(rows),
        "pdf_pages": pages,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=ROOT / "reproducibility/experiments.json")
    parser.add_argument("--results", type=Path, default=ROOT / "results/paper_results.csv")
    parser.add_argument(
        "--artifact-manifest",
        type=Path,
        default=ROOT / "reproducibility/release_artifacts.json",
    )
    parser.add_argument("--paper-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.registry, args.results, args.artifact_manifest, args.paper_root)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
