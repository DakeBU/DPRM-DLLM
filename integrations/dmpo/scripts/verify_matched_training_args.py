#!/usr/bin/env python3
"""Verify that confidence and DPRM checkpoints share the paper training setup."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SHARED_FIELDS = (
    "seed",
    "max_steps",
    "learning_rate",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "generation_batch_size",
    "num_generations",
    "num_iterations",
    "num_replicates",
    "compute_ref_log_prob_elbo_size",
    "alpha",
    "sampler_steps",
    "temperature",
    "loss_mask_sampler",
    "loss_progressive_k",
    "loss_progressive_phase_init",
    "loss_progressive_threshold",
)


def load_args(checkpoint: Path, source_root: Path):
    sys.path[:0] = [str(source_root), str(source_root / "DMPO")]
    import torch

    return torch.load(
        checkpoint / "training_args.bin", map_location="cpu", weights_only=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confidence", type=Path, required=True)
    parser.add_argument("--dprm", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    confidence = load_args(args.confidence, args.source_root)
    dprm = load_args(args.dprm, args.source_root)
    mismatches = {}
    shared = {}
    for field in SHARED_FIELDS:
        left = getattr(confidence, field, None)
        right = getattr(dprm, field, None)
        shared[field] = left
        if left != right:
            mismatches[field] = {"confidence": left, "dprm": right}

    policy = {
        "confidence": getattr(confidence, "loss_progressive_order_policy", None),
        "dprm": getattr(dprm, "loss_progressive_order_policy", None),
    }
    if policy != {"confidence": "confidence", "dprm": "dprm_soft_bon"}:
        mismatches["order_policy_contract"] = policy

    payload = {
        "schema_version": 1,
        "matched": not mismatches,
        "shared_fields": shared,
        "order_policy": policy,
        "mismatches": mismatches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
