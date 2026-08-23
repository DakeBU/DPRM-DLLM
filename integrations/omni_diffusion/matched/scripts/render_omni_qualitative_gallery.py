#!/usr/bin/env python3
"""Render the frozen four-policy Omni-Diffusion qualitative gallery."""

from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def record_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["prompt"]), int(row["seed"])


def uniform_method(prompt: str, seed: int, methods: list[str], salt: str) -> str:
    digest = hashlib.sha256(f"{prompt}|{seed}|{salt}".encode()).hexdigest()
    return methods[int(digest, 16) % len(methods)]


def short_method(method: str) -> str:
    return method.replace("step96_", "")


def resolve_image_path(item: dict[str, Any], confirmation_root: Path | None) -> Path:
    path = Path(item["image_path"])
    if path.is_file():
        return path
    if confirmation_root is None:
        raise FileNotFoundError(path)
    parts = path.parts
    for marker in ("baseline", "random", "branches"):
        if marker in parts:
            archived = confirmation_root.joinpath(*parts[parts.index(marker) :])
            if archived.is_file():
                return archived
    raise FileNotFoundError(
        f"could not resolve archived image for {path} under {confirmation_root}"
    )


def confirmation_relative_path(path: Path) -> str:
    for marker in ("baseline", "random", "branches"):
        if marker in path.parts:
            return Path(*path.parts[path.parts.index(marker) :]).as_posix()
    raise ValueError(f"image path is outside a recognized confirmation subtree: {path}")


def render_figure(rows: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(len(rows), 4, figsize=(12.2, 3.05 * len(rows)))
    headers = ["Random", "Omni default", "Uniform action", "DPRM"]
    for column, title in enumerate(headers):
        axes[0, column].set_title(title, fontsize=12, fontweight="bold", pad=10)

    for row_index, row in enumerate(rows):
        variants = row["variants"]
        for column, variant in enumerate(variants):
            axis = axes[row_index, column]
            with Image.open(variant["image_path"]) as image:
                axis.imshow(image.convert("RGB"))
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(2.1 if column == 3 else 0.7)
                spine.set_edgecolor("#17836f" if column == 3 else "#b9bec3")
            axis.set_xlabel(
                f"L/14 {variant['clip_l14']:.3f}  B/32 {variant['clip_b32']:.3f}",
                fontsize=8.2,
                color="#176a5c" if column == 3 else "#4f555b",
                labelpad=4,
            )
        prompt = textwrap.fill(row["prompt"], width=58)
        axes[row_index, 0].set_ylabel(
            prompt,
            fontsize=8.5,
            rotation=0,
            ha="right",
            va="center",
            labelpad=16,
        )
    fig.subplots_adjust(left=0.19, right=0.995, top=0.965, bottom=0.025, wspace=0.04, hspace=0.34)
    fig.savefig(output, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--scored-records", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirmation-root", type=Path)
    parser.add_argument("--tex-output", type=Path)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    summary = load_json(args.summary)["records"]
    scored = load_json(args.scored_records)
    methods = list(manifest["uniform_action"]["candidate_order"])
    salt = str(manifest["uniform_action"]["salt"])
    lookup = {
        method: {record_key(row): row for row in scored[method]}
        for method in scored
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rendered: list[dict[str, Any]] = []
    figure_paths: list[Path] = []
    for figure_index, indices in enumerate(manifest["figures"], start=1):
        figure_rows = []
        for index in indices:
            selected = summary[int(index)]
            key = record_key(selected)
            prompt, seed = key
            confidence = lookup["confidence"][key]
            random = lookup["random"][key]
            chosen = lookup[str(selected["selected_method"])][key]
            uniform_name = uniform_method(prompt, seed, methods, salt)
            uniform = lookup[uniform_name][key]
            if not (
                float(chosen["clip_cosine"]) > float(confidence["clip_cosine"])
                and float(chosen["clip_b32_cosine"]) > float(confidence["clip_b32_cosine"])
            ):
                raise ValueError(f"gallery record {index} does not improve both CLIP metrics")
            variants = []
            for label, method, item in (
                ("Random", "random", random),
                ("Omni default", "confidence", confidence),
                ("Uniform action", uniform_name, uniform),
                ("DPRM", str(selected["selected_method"]), chosen),
            ):
                image_path = resolve_image_path(item, args.confirmation_root)
                variants.append(
                    {
                        "label": label,
                        "method": method,
                        "image_path": str(image_path),
                        "clip_l14": float(item["clip_cosine"]),
                        "clip_b32": float(item["clip_b32_cosine"]),
                    }
                )
            row = {
                "record_index": int(index),
                "prompt": prompt,
                "seed": seed,
                "uniform_method": uniform_name,
                "dprm_method": str(selected["selected_method"]),
                "variants": variants,
            }
            rendered.append(row)
            figure_rows.append(row)
        figure_path = args.output_dir / f"omni_qualitative_gallery_{figure_index}.png"
        render_figure(figure_rows, figure_path)
        figure_paths.append(figure_path)

    public_records = []
    for row in rendered:
        public_row = dict(row)
        public_row["variants"] = [
            {
                **variant,
                "image_path": confirmation_relative_path(Path(variant["image_path"])),
            }
            for variant in row["variants"]
        ]
        public_records.append(public_row)
    with (args.output_dir / "omni_qualitative_gallery_records.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"manifest": args.manifest.name, "uniform_salt": salt, "records": public_records},
            handle,
            indent=2,
        )
        handle.write("\n")

    if args.tex_output:
        lines = []
        for index, path in enumerate(figure_paths, start=1):
            lines.extend(
                [
                    "\\begin{figure}[H]",
                    "\\centering",
                    f"\\includegraphics[width=\\linewidth]{{figs/{path.name}}}",
                    "\\caption{Post-evaluation Omni-Diffusion qualitative gallery. "
                    "Random, Omni default, a deterministic uniform action, and DPRM use matched prompts and seeds. "
                    "The displayed DPRM result improves both CLIP-L/14 and CLIP-B/32 over Omni default. "
                    "These examples visualize the frozen confirmation split and do not select the controller.}",
                    f"\\label{{fig:omni_qualitative_gallery_{index}}}",
                    "\\end{figure}",
                    "",
                ]
            )
        args.tex_output.parent.mkdir(parents=True, exist_ok=True)
        args.tex_output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
