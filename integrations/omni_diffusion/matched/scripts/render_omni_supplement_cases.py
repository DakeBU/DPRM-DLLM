#!/usr/bin/env python3
"""Render pairs of frozen Omni token-order mechanism cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from PIL import Image


STEPS = (64, 96, 192, None)


def frame(root: Path, step: int | None) -> Image.Image:
    path = (
        root / "omni_t2i_progressive_confidence.png"
        if step is None
        else root / "history_frames" / f"step_{step:04d}.png"
    )
    return Image.open(path).convert("RGB")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mark(ax, visual_index: int, color: str, image_size: tuple[int, int]) -> None:
    row, column = divmod(visual_index, 16)
    width, height = image_size[0] / 16, image_size[1] / 16
    center = ((column + 0.5) * width, (row + 0.5) * height)
    ax.add_patch(Rectangle((column * width, row * height), width, height, fill=False, edgecolor=color, linewidth=2.3))
    ax.add_patch(Circle(center, 0.7 * width, fill=False, edgecolor="white", linewidth=4.0))
    ax.add_patch(Circle(center, 0.7 * width, fill=False, edgecolor=color, linewidth=2.5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-ids", nargs="*", default=[])
    parser.add_argument("--output-prefix", default="omni_mechanism_cases")
    parser.add_argument("--tex-name", default="omni_supplement_mechanism_figures.tex")
    args = parser.parse_args()
    payload = json.loads(args.replay_manifest.read_text(encoding="utf-8"))
    manifest_root = args.replay_manifest.parent

    def resolve(raw: str) -> Path:
        path = Path(raw)
        return path if path.is_absolute() else manifest_root / path

    cases = payload["cases"]
    if args.case_ids:
        requested = set(args.case_ids)
        cases = [case for case in cases if case["id"] in requested]
        missing = requested - {case["id"] for case in cases}
        if missing:
            raise ValueError(f"unknown case ids: {sorted(missing)}")
    if not cases:
        raise ValueError("no cases selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    latex = []
    for figure_index in range(0, len(cases), 2):
        pair = cases[figure_index : figure_index + 2]
        fig, axes = plt.subplots(2 * len(pair), 4, figsize=(12.8, 4.55 * len(pair)), squeeze=False,
            gridspec_kw={"left": 0.13, "right": 0.99, "top": 0.91, "bottom": 0.035, "wspace": 0.035, "hspace": 0.08})
        titles = ("Same trajectory\nstep 64", "One token order differs\nstep 96", "Later denoising\nstep 192", "Completed image")
        for column, title in enumerate(titles):
            axes[0, column].set_title(title, fontsize=10.2, pad=7)
        explanations = []
        for case_index, case in enumerate(pair):
            roots = (resolve(case["confidence_dir"]), resolve(case["dprm_dir"]))
            replay_images = tuple(root / "omni_t2i_progressive_confidence.png" for root in roots)
            source_images = (
                resolve(case["source_confidence_image_path"]),
                resolve(case["source_dprm_image_path"]),
            )
            for replay, source in zip(replay_images, source_images):
                if sha256(replay) != sha256(source):
                    raise ValueError(f"replay differs from frozen confirmation: {case['id']}")
            dprm_payload = json.loads((roots[1] / "omni_t2i_progressive_confidence.json").read_text(encoding="utf-8"))
            override = dprm_payload["counterfactual_override"]
            for method_index, root in enumerate(roots):
                row = 2 * case_index + method_index
                for column, step in enumerate(STEPS):
                    axes[row, column].imshow(frame(root, step))
                    axes[row, column].set_xticks(())
                    axes[row, column].set_yticks(())
                    for spine in axes[row, column].spines.values():
                        spine.set_linewidth(1.5)
                        spine.set_edgecolor("#66717C" if method_index == 0 else "#D95F0E")
                image_size = frame(root, 96).size
                position = override["default_visual_index"] if method_index == 0 else override["visual_index"]
                mark(axes[row, 1], int(position), "#2474B5" if method_index == 0 else "#D95F0E", image_size)
                label = "Confidence" if method_index == 0 else "DPRM"
                l14 = case["confidence_clip_l14"] if method_index == 0 else case["dprm_clip_l14"]
                b32 = case["confidence_clip_b32"] if method_index == 0 else case["dprm_clip_b32"]
                axes[row, 0].set_ylabel(f"{case['id'].replace('_', ' ').title()}\n{label}\nCLIP-L/B {l14:.3f}/{b32:.3f}", fontsize=9.3, fontweight="semibold", rotation=0, ha="right", va="center", labelpad=15)
            explanations.append(case["explanation"])
        output = args.output_dir / f"{args.output_prefix}_{figure_index // 2 + 1}.png"
        fig.savefig(output, dpi=240, bbox_inches="tight")
        plt.close(fig)
        latex.append((output.name, pair))

    tex = []
    for number, (filename, pair) in enumerate(latex, start=1):
        explanation = " ".join(case["explanation"] for case in pair)
        tex.extend([
            "\\begin{figure}[H]",
            "\\centering",
            f"\\includegraphics[width=0.98\\linewidth]{{figs/{filename}}}",
            f"\\caption{{Additional Omni-Diffusion token-order diagnostics from the frozen 512-prompt confirmation. \\textbf{{{explanation}}} Blue and orange mark the confidence and DPRM step-96 positions; all earlier states and all later continuation rules are shared. Both CLIP encoders improve in each case.}}",
            f"\\label{{fig:omni_supplement_mechanism_{number}}}",
            "\\end{figure}",
            "",
        ])
    (args.output_dir / args.tex_name).write_text("\n".join(tex), encoding="utf-8")


if __name__ == "__main__":
    main()
