#!/usr/bin/env python3
"""Render fixed Omni online action-value cases from saved controller records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BLUE = (33, 102, 172)
GREEN = (24, 142, 79)


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def frame_step(path: str) -> int:
    return int(Path(path).stem.removeprefix("step_"))


def nearest_frame(paths: list[str], step: int) -> Path:
    available = [(abs(frame_step(path) - step), frame_step(path), Path(path)) for path in paths]
    if not available:
        raise ValueError("case record has no saved history frames")
    return min(available)[2]


def centered_text(draw: ImageDraw.ImageDraw, text: str, center: int, y: int) -> None:
    face = font(25, bold=True)
    box = draw.textbbox((0, 0), text, font=face)
    draw.text((center - (box[2] - box[0]) // 2, y), text, fill=(20, 20, 20), font=face)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    requested = [
        json.loads(line)
        for line in args.prompt_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_prompt = {str(row["prompt"]): row for row in summary["records"]}
    rows = [by_prompt[str(item["prompt"])] for item in requested]

    tile = 390
    gap = 14
    left = 38
    top = 58
    canvas = Image.new("RGB", (left + 3 * tile + 2 * gap, top + len(rows) * (tile + gap)), "white")
    draw = ImageDraw.Draw(canvas)
    headers = (
        "Step 96: confidence / DPRM",
        "Omni default",
        "DPRM",
    )
    for column, header in enumerate(headers):
        centered_text(draw, header, left + column * (tile + gap) + tile // 2, 15)

    manifest = {
        "format": summary["format"],
        "selection": "fixed prompt file; no renderer-side image selection",
        "rows": [],
    }
    for row_index, (request, record) in enumerate(zip(requested, rows)):
        y = top + row_index * (tile + gap)
        draw.text((4, y + tile // 2 - 14), f"({chr(97 + row_index)})", fill=(20, 20, 20), font=font(22, bold=True))
        shared_path = record.get("shared_action_canvas_path")
        if shared_path:
            shared = Image.open(shared_path).convert("RGB")
        else:
            shared = Image.open(nearest_frame(record["confidence_history_frame_paths"], 96)).convert("RGB")
        confidence = Image.open(record["confidence_image_path"]).convert("RGB")
        selected = Image.open(record["selected_image_path"]).convert("RGB")
        images = [shared, confidence, selected]
        placements = []
        for column, image in enumerate(images):
            image.thumbnail((tile, tile))
            x = left + column * (tile + gap) + (tile - image.width) // 2
            image_y = y + (tile - image.height) // 2
            canvas.paste(image, (x, image_y))
            placements.append((x, image_y, image.width, image.height))

        action = record["selected_action"]
        default_visual = int(record["candidate_actions"][0]["visual_index"])
        selected_visual = int(action["visual_index"])
        x, image_y, width, height = placements[0]
        for visual_index, color in ((default_visual, BLUE), (selected_visual, GREEN)):
            grid_row, grid_column = divmod(visual_index, 16)
            x0 = x + round(grid_column * width / 16)
            y0 = image_y + round(grid_row * height / 16)
            x1 = x + round((grid_column + 1) * width / 16)
            y1 = image_y + round((grid_row + 1) * height / 16)
            draw.rectangle((x0, y0, x1, y1), outline=color, width=5)

        manifest["rows"].append(
            {
                "prompt_id": request.get("prompt_id"),
                "prompt": record["prompt"],
                "seed": record["seed"],
                "selected_method": record["selected_method"],
                "candidate_actions": record["candidate_actions"],
                "confidence_visual_index": default_visual,
                "dprm_visual_index": selected_visual,
                "base_order_scores": record["base_order_scores"],
                "terminal_rewards": record["terminal_rewards"],
                "adjusted_order_scores": record["adjusted_order_scores"],
                "confidence_image_path": record["confidence_image_path"],
                "dprm_image_path": record["selected_image_path"],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
