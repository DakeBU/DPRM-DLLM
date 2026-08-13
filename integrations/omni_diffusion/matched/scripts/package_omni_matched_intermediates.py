#!/usr/bin/env python3
"""Build fixed-index intermediate-canvas sheets for matched Omni policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


LABELS = {
    "random": "Random",
    "progressive_confidence": "Omni default",
    "dprm_confidence_warmup": "DPRM",
}


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixed-count", type=int, default=4)
    args = parser.parse_args()
    payload = json.loads(args.records.read_text(encoding="utf-8"))
    orders = [order for order in LABELS if order in payload]
    maps = {
        order: {str(row["prompt_id"]): row for row in payload[order]} for order in orders
    }
    common = sorted(set.intersection(*(set(rows) for rows in maps.values())))
    fixed = common[: args.fixed_count]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "selection": "first fixed prompt ids before score inspection",
        "main_figure_prompt_id": fixed[0] if fixed else None,
        "supplement_fixed_prompt_ids": fixed[1:],
        "prompts": [],
    }

    for prompt_id in fixed:
        rows = [maps[order][prompt_id] for order in orders]
        histories = [row.get("history_frame_paths", []) for row in rows]
        width_frames = min((len(paths) for paths in histories), default=0)
        if width_frames == 0:
            continue
        thumb = 150
        left = 155
        top = 74
        sheet = Image.new("RGB", (left + width_frames * thumb, top + len(orders) * thumb), "white")
        draw = ImageDraw.Draw(sheet)
        prompt = " ".join(str(rows[0].get("prompt", "")).split())
        draw.text((8, 8), f"{prompt_id}: {prompt[:130]}", fill=(20, 20, 20), font=font(16, True))
        for col in range(width_frames):
            frame_name = Path(histories[0][col]).stem.replace("step_", "step ")
            draw.text((left + col * thumb + 8, 48), frame_name, fill=(40, 40, 40), font=font(12))
        for row_idx, (order, paths) in enumerate(zip(orders, histories)):
            y = top + row_idx * thumb
            draw.text((8, y + 62), LABELS[order], fill=(20, 45, 80), font=font(14, True))
            for col, image_path in enumerate(paths[:width_frames]):
                image = Image.open(image_path).convert("RGB")
                image.thumbnail((thumb, thumb))
                x = left + col * thumb + (thumb - image.width) // 2
                sheet.paste(image, (x, y + (thumb - image.height) // 2))
        output = args.output_dir / f"{prompt_id}_intermediate_canvases.png"
        sheet.save(output)
        manifest["prompts"].append({"prompt_id": prompt_id, "prompt": prompt, "sheet": str(output)})

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
