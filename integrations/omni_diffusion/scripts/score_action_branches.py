#!/usr/bin/env python3
"""Score shared-canvas Omni branches and write DPRM-BoN selection records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def load_runs(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline_pattern = "baseline/prompt_*/omni_t2i_progressive_confidence.json"
    branch_pattern = "branches/step*_q*/prompt_*/omni_t2i_progressive_confidence.json"
    for kind, pattern in (("baseline", baseline_pattern), ("branch", branch_pattern)):
        for path in sorted(root.glob(pattern)):
            row = json.loads(path.read_text(encoding="utf-8"))
            row["kind"] = kind
            row["tag"] = "baseline" if kind == "baseline" else path.parents[1].name
            rows.append(row)
    return rows


def score_rows(
    rows: list[dict[str, Any]],
    model_name: str,
    device: str,
    batch_size: int,
    aesthetic_weights: Path,
    aesthetic_weight: float,
) -> None:
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    aesthetic = AestheticMLP().to(device)
    aesthetic.load_state_dict(
        torch.load(aesthetic_weights, map_location=device, weights_only=True)
    )
    aesthetic.eval()
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        images = [Image.open(str(row["image_path"])).convert("RGB") for row in batch]
        inputs = processor(
            text=[str(row["prompt"]) for row in batch],
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        with torch.inference_mode():
            image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
            text_features = model.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            clip = (image_features * text_features).sum(dim=-1).float().cpu().tolist()
            aes = aesthetic(image_features.float()).squeeze(-1).float().cpu().tolist()
        for row, clip_score, aesthetic_score in zip(batch, clip, aes):
            row["clip_cosine"] = float(clip_score)
            row["aesthetic_score"] = float(aesthetic_score)
            row["terminal_utility"] = float(clip_score) + aesthetic_weight * float(
                aesthetic_score
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aesthetic-weights", type=Path, required=True)
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--aesthetic-weight", type=float, default=0.01)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    rows = load_runs(args.root)
    if not rows:
        raise SystemExit(f"no action-branch runs found under {args.root}")
    score_rows(
        rows,
        args.clip_model,
        args.device,
        args.batch_size,
        args.aesthetic_weights,
        args.aesthetic_weight,
    )
    baseline = {
        (str(row["prompt"]), int(row["seed"])): row
        for row in rows
        if row["kind"] == "baseline"
    }
    branches = []
    for row in rows:
        if row["kind"] != "branch":
            continue
        key = str(row["prompt"]), int(row["seed"])
        force = dict(row.get("counterfactual_override") or {})
        if key not in baseline or not force.get("applied"):
            continue
        branches.append(
            {
                "tag": row["tag"],
                "prompt": row["prompt"],
                "seed": row["seed"],
                "image_path": row["image_path"],
                "branch_clip": row["clip_cosine"],
                "branch_aesthetic": row["aesthetic_score"],
                "branch_utility": row["terminal_utility"],
                **force,
            }
        )
    output = {
        "utility": (
            f"{args.clip_model} cosine + {args.aesthetic_weight:g} * aesthetic score"
        ),
        "baseline": list(baseline.values()),
        "branches": branches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"baseline": len(baseline), "branches": len(branches)}, indent=2))


if __name__ == "__main__":
    main()
