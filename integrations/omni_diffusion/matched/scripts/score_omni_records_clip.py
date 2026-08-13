#!/usr/bin/env python3
"""Add a post-selection CLIP score to an Omni records payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--metric-name", default="clip_b32_cosine")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    payload = json.loads(args.records.read_text(encoding="utf-8"))
    model = CLIPModel.from_pretrained(args.model, local_files_only=True).to(args.device).eval()
    processor = CLIPProcessor.from_pretrained(args.model, local_files_only=True)

    with torch.inference_mode():
        for records in payload.values():
            for start in range(0, len(records), args.batch_size):
                batch = records[start : start + args.batch_size]
                images = [Image.open(row["image_path"]).convert("RGB") for row in batch]
                texts = [str(row["prompt"]) for row in batch]
                inputs = processor(
                    text=texts,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(args.device)
                image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
                text_features = model.get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                scores = (image_features * text_features).sum(dim=-1).cpu().tolist()
                for row, score in zip(batch, scores):
                    row[args.metric_name] = float(score)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: len(rows) for name, rows in payload.items()}, indent=2))


if __name__ == "__main__":
    main()
