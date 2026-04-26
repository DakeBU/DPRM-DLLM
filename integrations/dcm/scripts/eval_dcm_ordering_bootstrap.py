"""Evaluate DCM checkpoints with train-test aligned reveal ordering and bootstrap CIs."""

import argparse
import csv
import json
import zlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scanpy as sc
import torch
import torch.nn.functional as F
import yaml
from scipy import sparse
from tqdm import tqdm

from sedd.data import train_val_split
from sedd.graph import AbsorbingGraph
from sedd.model import SEDDTransformerSmall
from sedd.noise import LogLinearNoise

try:
    from dprm import DPRMConfig, HostDPRMBatch, OnlineDPRMController
except Exception:
    DPRMConfig = None
    HostDPRMBatch = None
    OnlineDPRMController = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", action="append", required=True,
                        help="label=config_path=checkpoint_path=policy")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-path", default="datasets/dentate/dentate_5000_bins32.h5ad")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def read_data(path: str) -> torch.Tensor:
    adata = sc.read_h5ad(path)
    x = adata.X
    if sparse.issparse(x):
        x = x.toarray()
    return torch.as_tensor(x).long()


def val_tensor(data: torch.Tensor, val_fraction: float, seed: int, max_cells: int) -> torch.Tensor:
    _, val_ds = train_val_split(data, val_fraction=val_fraction, seed=seed)
    indices = list(val_ds.indices)
    if max_cells and max_cells > 0:
        indices = indices[:max_cells]
    return data[indices]


def build_model(cfg: dict, data: torch.Tensor, device: torch.device) -> Tuple[SEDDTransformerSmall, AbsorbingGraph]:
    model_cfg = cfg.get("model", {})
    num_bins = int(data.max().item()) + 1
    model = SEDDTransformerSmall(
        num_genes=int(data.shape[1]),
        num_bins=num_bins,
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        num_layers=int(model_cfg.get("num_layers", 4)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        max_seq_len=int(data.shape[1]),
    ).to(device)
    return model, AbsorbingGraph(num_states=num_bins + 1)


def load_checkpoint(model, checkpoint_path: str, device: torch.device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    return payload


def make_controller(payload: dict, cfg: dict, device: torch.device):
    if "dprm_state_dict" not in payload:
        return None
    if OnlineDPRMController is None:
        raise ImportError("DPRM package is not importable; set PYTHONPATH to DPRM/src.")
    training = cfg.get("training", {})
    controller = OnlineDPRMController(
        DPRMConfig(
            num_phases=int(training.get("dprm_num_phases", 8)),
            confidence_bins=int(training.get("dprm_confidence_bins", 16)),
            reward_temperature=float(training.get("dprm_reward_temperature", 1.0)),
            guidance_scale=float(training.get("dprm_guidance_scale", 1.0)),
            warmup_steps=int(training.get("dprm_warmup_steps", 0)),
            switch_steps=int(training.get("dprm_switch_steps", 1)),
            ready_count=int(training.get("dprm_ready_count", 128)),
            sampled_soft_bon=False,
        ),
        device=device,
    )
    controller.load_state_dict(payload["dprm_state_dict"])
    controller.cfg.sampled_soft_bon = False
    return controller


def reveal_budget(masked: torch.Tensor, step: int, num_steps: int) -> torch.Tensor:
    remaining_steps = max(num_steps - step, 1)
    k = torch.ceil(masked.sum(dim=1).float() / remaining_steps).long()
    return torch.where(masked.any(dim=1), k.clamp_min(1), torch.zeros_like(k))


def topk_mask(scores: torch.Tensor, candidate_mask: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    picked = torch.zeros_like(candidate_mask, dtype=torch.bool)
    scores = scores.masked_fill(~candidate_mask, float("-inf"))
    for row in range(candidate_mask.shape[0]):
        active = torch.where(candidate_mask[row])[0]
        kk = min(int(k[row].item()), int(active.numel()))
        if kk <= 0:
            continue
        chosen = active[torch.topk(scores[row, active], kk, largest=True).indices]
        picked[row, chosen] = True
    return picked


@torch.no_grad()
def decode(
    model,
    graph,
    clean: torch.Tensor,
    policy: str,
    controller,
    num_steps: int,
    temperature: float,
    device: torch.device,
    sample_seed: int,
) -> torch.Tensor:
    torch.manual_seed(sample_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(sample_seed)
    batch_size, seq_len = clean.shape
    mask_idx = graph.mask_index
    x = torch.full((batch_size, seq_len), mask_idx, dtype=torch.long, device=device)
    noise = LogLinearNoise(eps=1e-3)

    for step in range(num_steps):
        masked = x == mask_idx
        if not masked.any():
            break
        t = torch.full((batch_size,), 1.0 - (step / max(num_steps - 1, 1)), device=device)
        sigma = noise.total(t)
        score = model.score(x, sigma)
        probs = F.softmax(score[..., :-1] / max(temperature, 1e-6), dim=-1)
        sampled = torch.multinomial(
            probs.reshape(-1, probs.shape[-1]),
            num_samples=1,
        ).reshape(batch_size, seq_len)
        confidence = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1).clamp_(1e-6, 1.0)
        k = reveal_budget(masked, step, num_steps)

        if policy in {"random", "random_ordered"}:
            select_scores = torch.rand_like(confidence)
            reveal = topk_mask(select_scores, masked, k)
        elif policy in {"confidence", "progressive"}:
            reveal = topk_mask(torch.log(confidence), masked, k)
        elif policy in {"dprm_random", "dprm_confidence", "dprm"}:
            if controller is None:
                raise ValueError(f"Policy {policy} requires a checkpoint with dprm_state_dict.")
            phase = OnlineDPRMController.phase_from_progress(
                step, num_steps, controller.cfg.num_phases, batch_size, device
            )
            host = HostDPRMBatch(
                confidence=confidence,
                candidate_mask=masked,
                phase_ids=phase,
                global_step=10**12,
                force_full_dprm=True,
            )
            summary = controller.summarize(host)
            if policy == "dprm_random":
                # Evaluation uses the learned DPRM-adjusted score after the
                # training warmup has completed, matching the paper algorithm.
                reveal = topk_mask(summary.score, masked, k)
            else:
                reveal = controller.select(host, k).selected_mask
        else:
            raise ValueError(f"Unknown eval policy: {policy}")

        x = torch.where(reveal, sampled, x)

    if (x == mask_idx).any():
        sigma = torch.full((batch_size,), 0.01, device=device)
        score = model.score(x, sigma)
        fill = score[..., :-1].argmax(dim=-1)
        x = torch.where(x == mask_idx, fill, x)
    return x


def evaluate_series(label: str, cfg_path: str, ckpt_path: str, policy: str, args, val: torch.Tensor):
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    cfg = load_config(cfg_path)
    model, graph = build_model(cfg, val, device)
    payload = load_checkpoint(model, ckpt_path, device)
    controller = make_controller(payload, cfg, device)
    model.eval()

    n = int(val.shape[0])
    sums = {
        "token_recovery": np.zeros(n, dtype=np.float64),
        "mae": np.zeros(n, dtype=np.float64),
        "zero_accuracy": np.zeros(n, dtype=np.float64),
    }
    loader = torch.utils.data.DataLoader(val, batch_size=args.batch_size, shuffle=False)
    offset = 0
    for batch in tqdm(loader, desc=f"eval {label}"):
        batch = batch.to(device)
        bsz = int(batch.shape[0])
        local = {key: np.zeros(bsz, dtype=np.float64) for key in sums}
        for sample_idx in range(args.num_samples):
            pred = decode(
                model=model,
                graph=graph,
                clean=batch,
                policy=policy,
                controller=controller,
                num_steps=args.num_steps,
                temperature=args.temperature,
                device=device,
                sample_seed=args.seed + 1009 * sample_idx + offset,
            )
            equal = (pred == batch)
            local["token_recovery"] += equal.float().mean(dim=1).cpu().numpy()
            local["mae"] += (pred.float() - batch.float()).abs().mean(dim=1).cpu().numpy()
            zero_mask = batch == 0
            zero_denom = zero_mask.sum(dim=1).clamp_min(1)
            local["zero_accuracy"] += ((pred == 0) & zero_mask).sum(dim=1).float().div(zero_denom).cpu().numpy()
        for key in sums:
            sums[key][offset:offset + bsz] = local[key] / float(args.num_samples)
        offset += bsz
    return sums


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int, higher_is_better: bool = True):
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = values[idx].mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "higher_is_better": bool(higher_is_better),
    }


def paired_bootstrap_delta(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    diff = b - a
    n = len(diff)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = diff[idx].mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"delta_mean": float(diff.mean()), "ci95_low": float(lo), "ci95_high": float(hi)}


def stable_seed(base: int, *parts: str) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return int((base + zlib.crc32(payload)) % (2**32 - 1))


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = read_data(args.data_path)
    val = val_tensor(data, args.val_fraction, args.seed, args.max_cells)

    parsed = []
    for raw in args.series:
        parts = raw.split("=", 3)
        if len(parts) != 4:
            raise ValueError("--series must be label=config=checkpoint=policy")
        parsed.append(tuple(parts))

    per_series: Dict[str, Dict[str, np.ndarray]] = {}
    for label, cfg_path, ckpt_path, policy in parsed:
        per_series[label] = evaluate_series(label, cfg_path, ckpt_path, policy, args, val)
        csv_path = out / f"{label}_per_cell.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["cell_index", "token_recovery", "mae", "zero_accuracy"])
            for i in range(len(val)):
                writer.writerow([i] + [per_series[label][metric][i] for metric in ["token_recovery", "mae", "zero_accuracy"]])

    summary = {
        "data_path": args.data_path,
        "num_val_cells": int(len(val)),
        "num_steps": int(args.num_steps),
        "num_samples": int(args.num_samples),
        "metrics": {},
        "paired_deltas": {},
    }
    higher = {"token_recovery": True, "mae": False, "zero_accuracy": True}
    for label, values in per_series.items():
        summary["metrics"][label] = {
            metric: bootstrap_ci(arr, args.bootstrap, stable_seed(args.seed, label, metric), higher[metric])
            for metric, arr in values.items()
        }
    if len(parsed) >= 2:
        base_label = parsed[0][0]
        for label, _, _, _ in parsed[1:]:
            summary["paired_deltas"][f"{label}_minus_{base_label}"] = {
                metric: paired_bootstrap_delta(
                    per_series[base_label][metric],
                    per_series[label][metric],
                    args.bootstrap,
                    stable_seed(args.seed, "delta", label, metric),
                )
                for metric in per_series[base_label]
            }
    with (out / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
