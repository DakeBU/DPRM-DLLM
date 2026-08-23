#!/usr/bin/env python3
"""Build the fixed-prompt Omni shared-state action case study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from PIL import Image


STEPS = (64, 96, 192, 259)


def load_record(path: Path, key: str, prompt_id: str) -> dict:
    payload = json.loads(path.read_text())
    if key not in payload and key == "dprm_direct":
        key = "dprm_softbon_n4"  # Backward-compatible archived artifact key.
    return next(row for row in payload[key] if row["prompt_id"] == prompt_id)


def frame(root: Path, step: int) -> Image.Image:
    return Image.open(root / "history_frames" / f"step_{step:04d}.png").convert("RGB")


def mark_action(
    ax, visual_index: int, color: str, image_size: tuple[int, int]
) -> None:
    row, column = divmod(visual_index, 16)
    image_width, image_height = image_size
    width = image_width / 16
    height = image_height / 16
    center = ((column + 0.5) * width, (row + 0.5) * height)
    ax.add_patch(
        Rectangle(
            (column * width, row * height),
            width,
            height,
            fill=False,
            edgecolor=color,
            linewidth=2.4,
        )
    )
    ax.add_patch(
        Circle(
            center,
            radius=0.72 * width,
            fill=False,
            edgecolor="white",
            linewidth=4.2,
        )
    )
    ax.add_patch(
        Circle(
            center,
            radius=0.72 * width,
            fill=False,
            edgecolor=color,
            linewidth=2.6,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confidence-dir", required=True, type=Path)
    parser.add_argument("--dprm-dir", required=True, type=Path)
    parser.add_argument("--formal-records", required=True, type=Path)
    parser.add_argument("--prompt-id", default="20270008")
    parser.add_argument("--case-name", default="Beach")
    parser.add_argument("--second-confidence-dir", type=Path)
    parser.add_argument("--second-dprm-dir", type=Path)
    parser.add_argument("--second-formal-records", type=Path)
    parser.add_argument("--second-prompt-id")
    parser.add_argument("--second-case-name", default="Second case")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    cases = [
        {
            "name": args.case_name,
            "confidence_dir": args.confidence_dir,
            "dprm_dir": args.dprm_dir,
            "records": args.formal_records,
            "prompt_id": args.prompt_id,
        }
    ]
    second_values = (
        args.second_confidence_dir,
        args.second_dprm_dir,
        args.second_formal_records,
        args.second_prompt_id,
    )
    if any(value is not None for value in second_values):
        if not all(value is not None for value in second_values):
            parser.error("all second-case arguments must be supplied together")
        cases.append(
            {
                "name": args.second_case_name,
                "confidence_dir": args.second_confidence_dir,
                "dprm_dir": args.second_dprm_dir,
                "records": args.second_formal_records,
                "prompt_id": args.second_prompt_id,
            }
        )

    for case in cases:
        case["confidence"] = load_record(
            case["records"], "progressive_confidence", case["prompt_id"]
        )
        case["dprm"] = load_record(
            case["records"], "dprm_direct", case["prompt_id"]
        )
        case["override"] = json.loads(
            (
                case["dprm_dir"] / "omni_t2i_progressive_confidence.json"
            ).read_text()
        )["counterfactual_override"]

    fig, axes = plt.subplots(
        2 * len(cases),
        len(STEPS),
        figsize=(12.8, 4.85 * len(cases)),
        squeeze=False,
        gridspec_kw={"left": 0.13, "right": 0.99, "top": 0.88, "bottom": 0.04,
                     "wspace": 0.035, "hspace": 0.09},
    )
    column_titles = (
        "Same trajectory\nstep 64",
        "One action differs\nstep 96",
        "Later denoising\nstep 192",
        "Completed image\nstep 259",
    )
    for column, (step, title) in enumerate(zip(STEPS, column_titles)):
        for case_index, case in enumerate(cases):
            for method_index, root in enumerate(
                (case["confidence_dir"], case["dprm_dir"])
            ):
                row = 2 * case_index + method_index
                axes[row, column].imshow(frame(root, step))
                axes[row, column].set_xticks(())
                axes[row, column].set_yticks(())
                for spine in axes[row, column].spines.values():
                    spine.set_linewidth(1.6)
                    spine.set_edgecolor(
                        "#66717C" if method_index == 0 else "#D95F0E"
                    )
        axes[0, column].set_title(title, fontsize=10.2, pad=7)

    for case_index, case in enumerate(cases):
        confidence_row = 2 * case_index
        dprm_row = confidence_row + 1
        override = case["override"]
        image_size = frame(case["confidence_dir"], 96).size
        if override.get("default_visual_index") is not None:
            mark_action(
                axes[confidence_row, 1],
                int(override["default_visual_index"]),
                "#2474B5",
                image_size,
            )
        mark_action(
            axes[dprm_row, 1],
            int(override["visual_index"]),
            "#D95F0E",
            image_size,
        )
        confidence = case["confidence"]
        dprm = case["dprm"]
        axes[confidence_row, 0].set_ylabel(
            f"{case['name']}\nConfidence\n"
            f"CLIP-L/B {confidence['clip_cosine']:.3f}/{confidence['clip_b32_cosine']:.3f}",
            fontsize=9.8,
            fontweight="semibold",
            rotation=0,
            ha="right",
            va="center",
            labelpad=16,
        )
        axes[dprm_row, 0].set_ylabel(
            f"{case['name']}\nDPRM ($N=4$)\n"
            f"CLIP-L/B {dprm['clip_cosine']:.3f}/{dprm['clip_b32_cosine']:.3f}",
            fontsize=9.8,
            fontweight="semibold",
            rotation=0,
            ha="right",
            va="center",
            labelpad=16,
        )
    if len(cases) > 1:
        fig.suptitle(
            "Lower-confidence actions establish useful global visual structure",
            fontsize=13.0,
            fontweight="bold",
            y=0.94,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
