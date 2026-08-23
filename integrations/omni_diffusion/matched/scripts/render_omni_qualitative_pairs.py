#!/usr/bin/env python3
"""Render the two preregistered Omni endpoint comparisons without score-based selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ORDERS = ("progressive_confidence", "dprm_confidence_warmup")
LABELS = ("Omni default", "DPRM")


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-ids", nargs=2, required=True)
    args = parser.parse_args()
    payload = json.loads(args.records.read_text(encoding="utf-8"))
    by_order = {
        order: {str(row["prompt_id"]): row for row in payload[order]} for order in ORDERS
    }

    tile = 420
    top = 52
    left = 42
    gap = 16
    canvas = Image.new("RGB", (left + 2 * tile + gap, top + 2 * tile + gap), "white")
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(LABELS):
        box = draw.textbbox((0, 0), label, font=font(24, True))
        width = box[2] - box[0]
        x = left + column * (tile + gap) + (tile - width) // 2
        draw.text((x, 12), label, fill=(20, 20, 20), font=font(24, True))
    row_labels = ("(a)", "(b)")
    manifest = {"selection": "fixed prompt ids before endpoint generation", "rows": []}
    for row_index, prompt_id in enumerate(args.prompt_ids):
        y = top + row_index * (tile + gap)
        draw.text((5, y + tile // 2 - 14), row_labels[row_index], fill=(25, 25, 25), font=font(22, True))
        prompt = None
        for column, order in enumerate(ORDERS):
            record = by_order[order][prompt_id]
            image = Image.open(record["image_path"]).convert("RGB")
            image.thumbnail((tile, tile))
            x = left + column * (tile + gap) + (tile - image.width) // 2
            image_y = y + (tile - image.height) // 2
            canvas.paste(image, (x, image_y))
            prompt = record["prompt"]
        manifest["rows"].append({"prompt_id": prompt_id, "prompt": prompt})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
