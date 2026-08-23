#!/usr/bin/env python3
"""Evaluate a retained PUMA checkpoint with its checkpoint-local order state."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from eval.gsm8k_eval import evaluate_samples, get_tokenizer, test_gsm8k_tokenization
from model.transformer import MDMConfig, MDMTransformer
from sampling import mdm_sampling


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--confidence",
        choices=("top_k", "dprm_soft_bon"),
        required=True,
    )
    parser.add_argument("--unmasking-num", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.cfg).resolve()
    ckpt_path = Path(args.ckpt).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(cfg_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model = MDMTransformer(MDMConfig(**cfg.model)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    dprm_state = checkpoint.get("dprm_order_state")
    if args.confidence == "dprm_soft_bon" and dprm_state is None:
        raise SystemExit("DPRM evaluation requires checkpoint['dprm_order_state']")
    if dprm_state is not None:
        counts = dprm_state.get("counts")
        if counts is None or counts.ndim != 3:
            raise SystemExit("invalid checkpoint-local DPRM count table")

    sampling = deepcopy(cfg.validation.sampling)
    sampling.confidence = args.confidence
    sampling.unmasking_num = int(args.unmasking_num)
    mask_id = int(cfg.data.mask_id)
    inputs, answers = test_gsm8k_tokenization(mask_id)
    tokenizer = get_tokenizer()

    records = []
    correct = 0
    for start in range(0, len(inputs), int(args.batch_size)):
        end = min(start + int(args.batch_size), len(inputs))
        batch = torch.from_numpy(np.asarray(inputs[start:end], dtype=np.int64)).to(device)
        with torch.inference_mode():
            samples = mdm_sampling(
                model,
                batch,
                mask_id,
                sampling,
                device,
                arm_init=cfg.model.arm_init != "none",
                dprm_state=dprm_state if args.confidence == "dprm_soft_bon" else None,
            )
        samples = samples.masked_fill(samples == mask_id, tokenizer.pad_token_id)
        texts = tokenizer.batch_decode(samples.cpu().numpy(), skip_special_tokens=True)
        for offset, (text, answer) in enumerate(zip(texts, answers[start:end])):
            is_correct = bool(evaluate_samples(text, answer))
            correct += int(is_correct)
            records.append(
                {
                    "example_id": start + offset,
                    "index": start + offset,
                    "gold_answer": str(answer),
                    "correct": is_correct,
                    "sample_text": text,
                    "sampling_confidence": args.confidence,
                    "sampling_unmasking_num": int(args.unmasking_num),
                }
            )

    details_path = output_dir / "per_question.jsonl"
    with details_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    table_shape = None
    if dprm_state is not None:
        table_shape = list(dprm_state["counts"].shape)
    summary = {
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": sha256(ckpt_path),
        "checkpoint_step": checkpoint.get("global_step"),
        "config": str(cfg_path),
        "confidence": args.confidence,
        "unmasking_num": int(args.unmasking_num),
        "n": len(records),
        "correct": correct,
        "accuracy": correct / max(len(records), 1),
        "dprm_state_loaded": args.confidence == "dprm_soft_bon",
        "dprm_table_shape": table_shape,
        "per_question": str(details_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
