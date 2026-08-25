#!/usr/bin/env python3
"""Render strict held-out RealWorldQA DPRM-only numeric/count wins."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from datasets import load_dataset

from build_dprm_table import response_text, target_normalized_match


def read_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[int(row["doc_id"])] = row
    return rows


def clean_question(row: dict[str, Any]) -> str:
    question = str(row.get("input") or row.get("doc", {}).get("question", ""))
    return question.split("Please answer directly", 1)[0].strip()


def render_page(cases: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(len(cases), 2, figsize=(11.8, 3.25 * len(cases)), gridspec_kw={"width_ratios": [1.05, 1.35]})
    if len(cases) == 1:
        axes = [axes]
    for row_index, (image_axis, text_axis) in enumerate(axes):
        case = cases[row_index]
        image_axis.imshow(case["image"])
        image_axis.set_xticks([])
        image_axis.set_yticks([])
        for spine in image_axis.spines.values():
            spine.set_color("#b7bdc2")
            spine.set_linewidth(0.8)

        text_axis.axis("off")
        question = textwrap.fill(case["question"], width=63)
        coordinates = text_axis.transAxes
        text_axis.text(0.0, 0.91, f"Document {case['doc_id']}  |  Numeric/count", fontsize=9, color="#687078", va="top", transform=coordinates)
        text_axis.text(0.0, 0.77, question, fontsize=11.2, fontweight="bold", va="top", linespacing=1.35, transform=coordinates)
        text_axis.text(0.0, 0.34, "Confidence order", fontsize=8.7, color="#737a81", va="top", transform=coordinates)
        text_axis.text(0.0, 0.25, case["confidence"], fontsize=14, color="#a7342d", va="top", transform=coordinates)
        text_axis.text(0.52, 0.34, "DPRM order", fontsize=8.7, color="#737a81", va="top", transform=coordinates)
        text_axis.text(0.52, 0.25, case["dprm"], fontsize=14, color="#176c59", fontweight="bold", va="top", transform=coordinates)
        text_axis.text(0.0, 0.06, f"Target: {case['target']}", fontsize=10, color="#22272b", va="bottom", transform=coordinates)
        text_axis.plot([0, 1], [0.0, 0.0], color="#d4d8dc", linewidth=0.8, transform=text_axis.transAxes)
    fig.subplots_adjust(left=0.02, right=0.985, top=0.98, bottom=0.02, hspace=0.11, wspace=0.06)
    fig.savefig(output, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confidence-records", type=Path, required=True)
    parser.add_argument("--dprm-records", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tex-output", type=Path)
    parser.add_argument("--dataset", default="lmms-lab/RealWorldQA")
    parser.add_argument("--exclude-doc-ids", nargs="*", type=int, default=[])
    parser.add_argument("--page-size", type=int, default=4)
    parser.add_argument("--output-prefix", default="lladav_realworldqa_gallery")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.page_size < 1:
        raise ValueError("--page-size must be positive")
    manifest_doc_ids = [int(doc_id) for doc_id in manifest["doc_ids"]]
    excluded = set(args.exclude_doc_ids)
    unknown_exclusions = sorted(excluded - set(manifest_doc_ids))
    if unknown_exclusions:
        raise ValueError(f"excluded document ids are not in the manifest: {unknown_exclusions}")
    selected_doc_ids = [doc_id for doc_id in manifest_doc_ids if doc_id not in excluded]
    if not selected_doc_ids:
        raise ValueError("all manifest documents were excluded")
    confidence = read_rows(args.confidence_records)
    dprm = read_rows(args.dprm_records)
    dataset = load_dataset(args.dataset, split="test")
    cases = []
    for doc_id in selected_doc_ids:
        left, right = confidence[int(doc_id)], dprm[int(doc_id)]
        if target_normalized_match(left) or not target_normalized_match(right):
            raise ValueError(f"document {doc_id} is not a DPRM-only win")
        source = dataset[int(doc_id)]
        cases.append(
            {
                "doc_id": int(doc_id),
                "image": source["image"].convert("RGB"),
                "question": clean_question(left),
                "target": str(left["target"]),
                "confidence": response_text(left),
                "dprm": response_text(right),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pages = [cases[start : start + args.page_size] for start in range(0, len(cases), args.page_size)]
    paths = []
    for index, page in enumerate(pages, start=1):
        path = args.output_dir / f"{args.output_prefix}_{index}.png"
        render_page(page, path)
        paths.append(path)
    public_cases = [{key: value for key, value in case.items() if key != "image"} for case in cases]
    (args.output_dir / f"{args.output_prefix}_records.json").write_text(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "excluded_doc_ids": sorted(excluded),
                "cases": public_cases,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if args.tex_output:
        if excluded:
            evidence_note = (
                f"Main-paper document(s) {', '.join(str(doc_id) for doc_id in sorted(excluded))} are omitted; "
                f"the gallery contains the remaining {len(cases)} DPRM-only wins."
            )
        else:
            evidence_note = (
                "The gallery contains all seven DPRM-only wins in this prompt-defined class; "
                "the class has no DPRM losses."
            )
        lines = []
        for index, path in enumerate(paths, start=1):
            lines.extend(
                [
                    "\\begin{figure}[H]",
                    "\\centering",
                    f"\\includegraphics[width=\\linewidth]{{figs/{path.name}}}",
                    "\\caption{RealWorldQA strict held-out numeric/count wins "
                    f"({index}/{len(paths)}). Each row shows the evaluation image, question, confidence-order answer, DPRM answer, and target. "
                    f"{evidence_note}}}",
                    f"\\label{{fig:{args.output_prefix}_{index}}}",
                    "\\end{figure}",
                    "",
                ]
            )
        args.tex_output.parent.mkdir(parents=True, exist_ok=True)
        args.tex_output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
