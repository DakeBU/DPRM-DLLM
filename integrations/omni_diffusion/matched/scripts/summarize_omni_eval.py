#!/usr/bin/env python3
"""Summarize Omni-Diffusion official-step eval outputs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def load_records(eval_root: Path, orders: list[str]) -> dict[str, list[dict[str, Any]]]:
    by_order: dict[str, list[dict[str, Any]]] = {}
    for order in orders:
        records: list[dict[str, Any]] = []
        order_root = eval_root / order
        for prompt_dir in sorted(order_root.glob("prompt_*")):
            json_files = sorted(prompt_dir.glob("*.json"))
            if not json_files:
                continue
            with json_files[0].open() as f:
                rec = json.load(f)
            rec["prompt_id"] = prompt_dir.name
            rec["json_path"] = str(json_files[0])
            image_path = Path(rec.get("image_path", ""))
            if not image_path.is_file():
                png_files = sorted(prompt_dir.glob("*.png"))
                image_path = png_files[0] if png_files else Path()
            rec["image_path"] = str(image_path)
            rec["has_image"] = image_path.is_file()
            records.append(rec)
        by_order[order] = records
    return by_order


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "count": len(records),
        "images": sum(1 for r in records if r.get("has_image")),
    }
    for key in ("generation_seconds", "total_seconds", "num_image_tokens"):
        vals = [float(r[key]) for r in records if r.get(key) is not None]
        if vals:
            out[key] = {
                "mean": statistics.fmean(vals),
                "median": statistics.median(vals),
                "min": min(vals),
                "max": max(vals),
            }
    return out


def short_text(text: str, max_len: int = 54) -> str:
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], lines: list[str]) -> None:
    x, y = xy
    for line in lines:
        draw.text((x, y), line, fill=(20, 20, 20))
        y += 13


def make_order_sheet(records: list[dict[str, Any]], order: str, output: Path, thumb: int = 160) -> None:
    valid = [r for r in records if r.get("has_image")]
    if not valid:
        return
    cols = 8
    label_h = 44
    rows = math.ceil(len(valid) / cols)
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for n, rec in enumerate(valid):
        row, col = divmod(n, cols)
        x = col * thumb
        y = row * (thumb + label_h)
        img = Image.open(rec["image_path"]).convert("RGB")
        img.thumbnail((thumb, thumb))
        paste_x = x + (thumb - img.width) // 2
        paste_y = y + (thumb - img.height) // 2
        sheet.paste(img, (paste_x, paste_y))
        draw_label(
            draw,
            (x + 3, y + thumb + 2),
            [f"{order} {rec['prompt_id']}", short_text(str(rec.get("prompt", "")), 42)],
        )
    sheet.save(output)


def make_paired_sheet(by_order: dict[str, list[dict[str, Any]]], orders: list[str], output: Path, thumb: int = 128) -> None:
    index: dict[str, dict[str, dict[str, Any]]] = {}
    for order, records in by_order.items():
        for rec in records:
            index.setdefault(rec["prompt_id"], {})[order] = rec
    prompt_ids = [pid for pid in sorted(index) if all(index[pid].get(o, {}).get("has_image") for o in orders)]
    if not prompt_ids:
        return
    group_w = thumb * len(orders)
    group_h = thumb + 46
    cols = 4
    rows = math.ceil(len(prompt_ids) / cols)
    sheet = Image.new("RGB", (cols * group_w, rows * group_h), "white")
    draw = ImageDraw.Draw(sheet)
    for n, pid in enumerate(prompt_ids):
        row, col = divmod(n, cols)
        x0 = col * group_w
        y0 = row * group_h
        prompt = ""
        for j, order in enumerate(orders):
            rec = index[pid][order]
            prompt = str(rec.get("prompt", prompt))
            img = Image.open(rec["image_path"]).convert("RGB")
            img.thumbnail((thumb, thumb))
            x = x0 + j * thumb + (thumb - img.width) // 2
            y = y0 + (thumb - img.height) // 2
            sheet.paste(img, (x, y))
            draw.text((x0 + j * thumb + 3, y0 + thumb + 2), order, fill=(20, 20, 20))
        draw.text((x0 + 3, y0 + thumb + 17), pid, fill=(20, 20, 20))
        draw.text((x0 + 3, y0 + thumb + 31), short_text(prompt, 70), fill=(20, 20, 20))
    sheet.save(output)


def add_clip_scores(
    by_order: dict[str, list[dict[str, Any]]],
    model_name: str,
    device: str,
    aesthetic_weights: Path | None,
    aesthetic_weight: float,
) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    from transformers import CLIPModel, CLIPProcessor

    class AestheticMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(768, 1024),
                nn.Dropout(0.2),
                nn.Linear(1024, 128),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.Dropout(0.1),
                nn.Linear(64, 16),
                nn.Linear(16, 1),
            )

        def forward(self, inputs):
            return self.layers(inputs)

    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    aesthetic = None
    if aesthetic_weights is not None:
        aesthetic = AestheticMLP().to(device)
        aesthetic.load_state_dict(
            torch.load(aesthetic_weights, map_location=device, weights_only=True)
        )
        aesthetic.eval()

    summary: dict[str, Any] = {"model": model_name, "orders": {}}
    with torch.no_grad():
        for order, records in by_order.items():
            scored: list[float] = []
            aesthetic_scored: list[float] = []
            utilities: list[float] = []
            for rec in records:
                if not rec.get("has_image"):
                    continue
                image = Image.open(rec["image_path"]).convert("RGB")
                inputs = processor(
                    text=[str(rec.get("prompt", ""))],
                    images=[image],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(device)
                image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
                text_features = model.get_text_features(
                    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
                )
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                score = (image_features * text_features).sum(dim=-1).item()
                rec["clip_cosine"] = score
                scored.append(score)
                if aesthetic is not None:
                    aesthetic_score = float(
                        aesthetic(image_features.float()).squeeze().item()
                    )
                    rec["aesthetic_score"] = aesthetic_score
                    rec["terminal_utility"] = score + aesthetic_weight * aesthetic_score
                    aesthetic_scored.append(aesthetic_score)
                    utilities.append(rec["terminal_utility"])
            order_summary = {
                "count": len(scored),
                "mean_clip_cosine": statistics.fmean(scored) if scored else None,
                "median_clip_cosine": statistics.median(scored) if scored else None,
            }
            if aesthetic is not None:
                order_summary.update(
                    {
                        "mean_aesthetic_score": statistics.fmean(aesthetic_scored)
                        if aesthetic_scored
                        else None,
                        "mean_terminal_utility": statistics.fmean(utilities)
                        if utilities
                        else None,
                    }
                )
            summary["orders"][order] = order_summary
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--orders", nargs="+", default=["random", "progressive_confidence"])
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--aesthetic-weights",
        type=Path,
        default=None,
    )
    parser.add_argument("--aesthetic-weight", type=float, default=0.01)
    parser.add_argument(
        "--no-aesthetic",
        action="store_true",
        help="Do not load or emit aesthetic/combined-utility scores.",
    )
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument(
        "--strict-clip",
        action="store_true",
        help="Fail instead of emitting a visual-only package when CLIP scoring fails.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or (args.eval_root / "summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    by_order = load_records(args.eval_root, args.orders)
    result: dict[str, Any] = {
        "eval_root": str(args.eval_root),
        "orders": {order: summarize(records) for order, records in by_order.items()},
    }

    for order, records in by_order.items():
        make_order_sheet(records, order, out_dir / f"{order}_contact_sheet.png")
    make_paired_sheet(by_order, args.orders, out_dir / "paired_contact_sheet.png")

    if not args.no_clip:
        try:
            result["clip"] = add_clip_scores(
                by_order,
                args.clip_model,
                args.device,
                None if args.no_aesthetic else args.aesthetic_weights,
                args.aesthetic_weight,
            )
        except Exception as exc:  # Keep visual summaries available if CLIP is unavailable.
            if args.strict_clip:
                raise
            result["clip_error"] = repr(exc)

    with (out_dir / "summary.json").open("w") as f:
        json.dump(result, f, indent=2)
    with (out_dir / "records.json").open("w") as f:
        json.dump(by_order, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
