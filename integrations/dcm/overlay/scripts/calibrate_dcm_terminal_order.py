#!/usr/bin/env python3
"""Calibrate DCM order tables with fixed-model, action-conditioned rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dprm import DPRMConfig, HostDPRMBatch, OnlineDPRMController, scalarize_benefits
from eval_dcm_ordering_bootstrap import (
    build_model,
    load_checkpoint,
    load_config,
    read_data,
    reveal_budget,
    topk_mask,
)
from sedd.data import train_val_split
from sedd.noise import LogLinearNoise


PREFERENCES = {
    "recovery": (0.90, 0.075, 0.025),
    "mae": (0.05, 0.90, 0.05),
    "balanced": (0.45, 0.35, 0.20),
    "zero": (0.025, 0.075, 0.90),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rnaseq_dprm_multiobjective_dentate.yaml")
    parser.add_argument(
        "--checkpoint",
        default="experiments/dcm_single_cell_real_progressive_orderfix/final.pt",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-path", default="datasets/dentate/dentate_5000_bins32.h5ad")
    parser.add_argument("--max-train-cells", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=32)
    parser.add_argument("--branch-steps", default="0,8,16,24")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num-phases", type=int, default=4)
    parser.add_argument("--confidence-bins", type=int, default=16)
    parser.add_argument("--ready-count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sample_tokens(model, x, step, num_steps, temperature, noise):
    t = torch.full(
        (x.shape[0],),
        1.0 - step / max(num_steps - 1, 1),
        device=x.device,
    )
    score = model.score(x, noise.total(t))
    probs = F.softmax(score[..., :-1] / max(temperature, 1e-6), dim=-1)
    sampled = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape_as(x)
    confidence = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1).clamp_(1e-6, 1.0)
    return sampled, confidence


@torch.no_grad()
def complete_decode(model, graph, initial, start_step, num_steps, temperature, seed):
    torch.manual_seed(seed)
    if initial.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    x = initial.clone()
    noise = LogLinearNoise(eps=1e-3)
    for step in range(start_step, num_steps):
        masked = x == graph.mask_index
        if not masked.any():
            break
        sampled, confidence = sample_tokens(model, x, step, num_steps, temperature, noise)
        reveal = topk_mask(
            torch.log(confidence), masked, reveal_budget(masked, step, num_steps)
        )
        x = torch.where(reveal, sampled, x)
    if (x == graph.mask_index).any():
        sigma = torch.full((x.shape[0],), 0.01, device=x.device)
        fill = model.score(x, sigma)[..., :-1].argmax(dim=-1)
        x = torch.where(x == graph.mask_index, fill, x)
    return x


def terminal_benefits(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    equal = predicted == target
    nonzero = target != 0
    nonzero_count = nonzero.sum(dim=1).clamp_min(1)
    recovery = (equal & nonzero).sum(dim=1).float() / nonzero_count.float()
    nonzero_mae = (
        ((predicted.float() - target.float()).abs() * nonzero).sum(dim=1)
        / (7.0 * nonzero_count.float())
    )
    mae_benefit = (1.0 - nonzero_mae).clamp(0.0, 1.0)
    zero = target == 0
    zero_count = zero.sum(dim=1).clamp_min(1)
    zero_accuracy = ((predicted == 0) & zero).sum(dim=1).float() / zero_count.float()
    return torch.stack((recovery, mae_benefit, zero_accuracy), dim=-1)


def action_sets(confidence, sampled, masked, budget):
    log_confidence = torch.log(confidence)
    return {
        "confidence": topk_mask(log_confidence, masked, budget),
        "nonzero": topk_mask(log_confidence + 20.0 * (sampled != 0), masked, budget),
        "zero": topk_mask(log_confidence + 20.0 * (sampled == 0), masked, budget),
        "random": topk_mask(torch.rand_like(confidence), masked, budget),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    cfg = load_config(args.config)
    data = read_data(args.data_path)
    train, _ = train_val_split(data, val_fraction=0.1, seed=42)
    train_tensor = data[list(train.indices)[: args.max_train_cells]]

    model, graph = build_model(cfg, data, device)
    base_payload = load_checkpoint(model, args.checkpoint, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    controllers = {
        name: OnlineDPRMController(
            DPRMConfig(
                num_phases=args.num_phases,
                confidence_bins=args.confidence_bins,
                aux_bins=2,
                reward_temperature=1.0,
                guidance_scale=1.0,
                warmup_steps=0,
                switch_steps=1,
                ready_count=args.ready_count,
                sampled_soft_bon=False,
            ),
            device=device,
        )
        for name in PREFERENCES
    }
    weights = {
        name: torch.tensor(value, dtype=torch.float32, device=device)
        for name, value in PREFERENCES.items()
    }
    branch_steps = {int(value) for value in args.branch_steps.split(",") if value.strip()}
    loader = DataLoader(train_tensor, batch_size=args.batch_size, shuffle=False)
    noise = LogLinearNoise(eps=1e-3)
    rollout_count = 0

    with torch.no_grad():
        for batch_index, clean in enumerate(tqdm(loader, desc="calibrate terminal order")):
            clean = clean.to(device)
            x_state = torch.full_like(clean, graph.mask_index)
            for step in range(args.num_steps):
                masked = x_state == graph.mask_index
                if not masked.any():
                    break
                local_seed = args.seed + 100003 * batch_index + step
                torch.manual_seed(local_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(local_seed)
                sampled, confidence = sample_tokens(
                    model, x_state, step, args.num_steps, args.temperature, noise
                )
                budget = reveal_budget(masked, step, args.num_steps)
                phase = OnlineDPRMController.phase_from_progress(
                    step, args.num_steps, args.num_phases, clean.shape[0], device
                )
                host = HostDPRMBatch(
                    confidence=confidence,
                    candidate_mask=masked,
                    phase_ids=phase,
                    aux_bin_ids=(sampled != 0).long(),
                    global_step=10**12,
                    force_full_dprm=True,
                )
                if step in branch_steps:
                    for action_index, action in enumerate(
                        action_sets(confidence, sampled, masked, budget).values()
                    ):
                        branch = torch.where(action, sampled, x_state)
                        terminal = complete_decode(
                            model,
                            graph,
                            branch,
                            step + 1,
                            args.num_steps,
                            args.temperature,
                            args.seed + 10_000_019 * batch_index + 1009 * step + action_index,
                        )
                        benefits = terminal_benefits(terminal, clean)
                        for name, controller in controllers.items():
                            reward = scalarize_benefits(
                                benefits,
                                weights[name],
                                method="smooth_tchebycheff",
                                temperature=0.05,
                                augmentation=0.05,
                            )
                            controller.observe(host, action, reward)
                        rollout_count += int(clean.shape[0])

                confidence_action = topk_mask(torch.log(confidence), masked, budget)
                x_state = torch.where(confidence_action, clean, x_state)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "base_checkpoint": str(Path(args.checkpoint).resolve()),
        "model_frozen": True,
        "train_cells": len(train_tensor),
        "split_seed": 42,
        "branch_steps": sorted(branch_steps),
        "action_proposals": ["confidence", "nonzero", "zero", "random"],
        "terminal_rollouts": rollout_count,
        "preferences": PREFERENCES,
        "seed": args.seed,
    }
    for name, controller in controllers.items():
        payload = dict(base_payload)
        payload["dprm_state_dict"] = controller.state_dict()
        payload["order_policy"] = "dprm_confidence"
        payload["dprm_reward_config"] = {
            "mode": "terminal_rollout_tchebycheff",
            "objective_names": [
                "nonzero_recovery",
                "nonzero_mae_benefit",
                "zero_accuracy",
            ],
            "objective_weights": list(PREFERENCES[name]),
            "tchebycheff_temperature": 0.05,
            "tchebycheff_augmentation": 0.05,
            "aux_mode": "predicted_zero",
            "model_frozen": True,
        }
        endpoint = args.output_dir / name
        endpoint.mkdir(parents=True, exist_ok=True)
        torch.save(payload, endpoint / "calibrated.pt")
    (args.output_dir / "calibration_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / ".complete").touch()


if __name__ == "__main__":
    main()
