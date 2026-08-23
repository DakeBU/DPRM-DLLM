#!/usr/bin/env python3
"""Score a predeclared Omni controller sweep with paired uncertainty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip-model", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    from transformers import CLIPModel, CLIPProcessor

    records = []
    for label in args.labels:
        for prompt_dir in sorted((args.root / label).glob("prompt_*")):
            files = sorted(prompt_dir.glob("omni_t2i_*.json"))
            if len(files) != 1:
                raise RuntimeError(f"expected one result in {prompt_dir}")
            row = json.loads(files[0].read_text(encoding="utf-8"))
            if row.get("fixed_t2i_scaffold") is not True:
                raise RuntimeError(f"fixed T2I scaffold required: {files[0]}")
            if int(row.get("order_trace_records", 0)) != 256:
                raise RuntimeError(f"256 visual actions required: {files[0]}")
            row.update(label=label, prompt_id=prompt_dir.name)
            records.append(row)

    model = CLIPModel.from_pretrained(args.clip_model).to(args.device).eval()
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    with torch.no_grad():
        for start in range(0, len(records), 16):
            batch = records[start : start + 16]
            inputs = processor(
                text=[row["prompt"] for row in batch],
                images=[Image.open(row["image_path"]).convert("RGB") for row in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(args.device)
            image = model.get_image_features(pixel_values=inputs["pixel_values"])
            text = model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            scores = (
                (image / image.norm(dim=-1, keepdim=True))
                * (text / text.norm(dim=-1, keepdim=True))
            ).sum(-1).cpu().tolist()
            for row, score in zip(batch, scores):
                row["clip_cosine"] = float(score)

    by_label = {label: {} for label in args.labels}
    for row in records:
        by_label[row["label"]][row["prompt_id"]] = row["clip_cosine"]
    prompt_ids = sorted(by_label["confidence"])
    expected = set(prompt_ids)
    for label in args.labels:
        if set(by_label[label]) != expected:
            raise RuntimeError(f"unpaired controller sweep for {label}")

    rng = np.random.default_rng(956)
    summary = {
        "clip_model": args.clip_model,
        "prompt_count": len(prompt_ids),
        "methods": {},
    }
    base = np.asarray([by_label["confidence"][key] for key in prompt_ids])
    for label in args.labels:
        values = np.asarray([by_label[label][key] for key in prompt_ids])
        delta = values - base
        bootstrap = np.asarray(
            [
                delta[rng.integers(0, len(delta), len(delta))].mean()
                for _ in range(5000)
            ]
        )
        summary["methods"][label] = {
            "mean_clip": float(values.mean()),
            "mean_delta_vs_confidence": float(delta.mean()),
            "ci95_delta": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
            "wins": int((delta > 0).sum()),
            "ties": int((delta == 0).sum()),
            "losses": int((delta < 0).sum()),
        }
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
