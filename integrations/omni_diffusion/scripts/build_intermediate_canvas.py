#!/usr/bin/env python3
"""Build a shared-state Omni confidence-versus-DPRM canvas comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from PIL import Image


STEPS = (64, 96, 128, 192, 259)


def load_record(path: Path, key: str, prompt_id: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return next(row for row in payload[key] if str(row["prompt_id"]) == prompt_id)


def frame(root: Path, step: int) -> Image.Image:
    return Image.open(root / "history_frames" / f"step_{step:04d}.png").convert("RGB")


def mark_action(
    axis, visual_index: int, color: str, image_size: tuple[int, int]
) -> None:
    row, column = divmod(visual_index, 16)
    image_width, image_height = image_size
    width, height = image_width / 16, image_height / 16
    center = ((column + 0.5) * width, (row + 0.5) * height)
    axis.add_patch(
        Rectangle(
            (column * width, row * height),
            width,
            height,
            fill=False,
            edgecolor=color,
            linewidth=2.4,
        )
    )
    for edgecolor, linewidth in (("white", 4.2), (color, 2.6)):
        axis.add_patch(
            Circle(
                center,
                radius=0.72 * width,
                fill=False,
                edgecolor=edgecolor,
                linewidth=linewidth,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confidence-dir", required=True, type=Path)
    parser.add_argument("--dprm-dir", required=True, type=Path)
    parser.add_argument("--formal-records", required=True, type=Path)
    parser.add_argument("--confidence-key", default="progressive_confidence")
    parser.add_argument("--dprm-key", default="dprm_softbon_n4")
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    confidence = load_record(
        args.formal_records, args.confidence_key, args.prompt_id
    )
    dprm = load_record(args.formal_records, args.dprm_key, args.prompt_id)
    override = json.loads(
        (args.dprm_dir / "omni_t2i_progressive_confidence.json").read_text(
            encoding="utf-8"
        )
    )["counterfactual_override"]

    fig, axes = plt.subplots(
        2,
        len(STEPS),
        figsize=(13.8, 5.7),
        gridspec_kw={
            "left": 0.115,
            "right": 0.99,
            "top": 0.82,
            "bottom": 0.08,
            "wspace": 0.035,
            "hspace": 0.08,
        },
    )
    titles = (
        "Same trajectory\nstep 64",
        "One action differs\nstep 96",
        "Effects spread\nstep 128",
        "Global layout forms\nstep 192",
        "Completed image\nstep 259",
    )
    for column, (step, title) in enumerate(zip(STEPS, titles)):
        for row, root in enumerate((args.confidence_dir, args.dprm_dir)):
            axes[row, column].imshow(frame(root, step))
            axes[row, column].set_xticks(())
            axes[row, column].set_yticks(())
            for spine in axes[row, column].spines.values():
                spine.set_linewidth(1.6)
                spine.set_edgecolor("#66717C" if row == 0 else "#D95F0E")
        axes[0, column].set_title(title, fontsize=10.2, pad=7)

    image_size = frame(args.confidence_dir, 96).size
    mark_action(
        axes[0, 1],
        int(override["default_visual_index"]),
        "#2474B5",
        image_size,
    )
    mark_action(
        axes[1, 1], int(override["visual_index"]), "#D95F0E", image_size
    )

    axes[0, 0].set_ylabel(
        "Confidence\n"
        f"action conf. {dprm['default_action_confidence']:.3f}\n"
        f"CLIP-L/B {confidence['clip_cosine']:.3f}/"
        f"{confidence['clip_b32_cosine']:.3f}",
        fontsize=10.2,
        fontweight="semibold",
        rotation=0,
        ha="right",
        va="center",
        labelpad=16,
    )
    axes[1, 0].set_ylabel(
        "DPRM-BoN\n"
        f"action conf. {dprm['selected_action_confidence']:.3f}\n"
        f"CLIP-L/B {dprm['clip_cosine']:.3f}/{dprm['clip_b32_cosine']:.3f}",
        fontsize=10.2,
        fontweight="semibold",
        rotation=0,
        ha="right",
        va="center",
        labelpad=16,
    )
    fig.suptitle(
        "A lower-confidence visual action leads to a higher-value completion",
        fontsize=13.0,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.55,
        0.895,
        "Same prompt, seed, checkpoint, tokenizer, and 260-step sampler; "
        "only the highlighted step-96 position changes.",
        ha="center",
        fontsize=9.8,
        color="#44515C",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
