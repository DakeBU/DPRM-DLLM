import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


DEFAULT_DPRM_ESTIMATOR_FILENAME = "dprm_estimator.json"


def _masked_topk(scores: torch.Tensor, num_select: torch.Tensor) -> torch.Tensor:
    transfer_index = torch.zeros_like(scores, dtype=torch.bool)
    if scores.numel() == 0:
        return transfer_index

    for row in range(scores.shape[0]):
        k = int(num_select[row].item())
        if k <= 0:
            continue
        valid = torch.isfinite(scores[row])
        if not valid.any():
            continue
        k = min(k, int(valid.sum().item()))
        _, idx = torch.topk(scores[row], k=k, largest=True)
        transfer_index[row, idx] = True
    return transfer_index


@dataclass
class DPRMSelectionSummary:
    bins: torch.Tensor
    mix: torch.Tensor
    dprm_values: torch.Tensor
    adjusted_scores: torch.Tensor


def phase_tensor_from_step(
    step_index: int,
    total_steps: int,
    num_phases: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if total_steps <= 1:
        phase_index = 0
    else:
        progress = float(step_index) / float(max(total_steps - 1, 1))
        phase_index = int(math.floor(progress * num_phases))
    phase_index = max(0, min(num_phases - 1, phase_index))
    return torch.full((batch_size,), phase_index, dtype=torch.long, device=device)


class OnlineDPRMEstimator:
    """
    Online state-value estimator for DPRM-style adjusted sampling.

    We use confidence bins as a shared low-dimensional surrogate for the state
    value R_DPRM(o_l) = (1 / lambda_reward) log E[exp(lambda_reward * R_out) | o_l].
    The estimator is updated online from training rewards and is reused during
    decoding for train/test alignment.
    """

    def __init__(
        self,
        num_phases: int,
        num_bins: int = 16,
        reward_temperature: float = 1.0,
        dprm_lambda: float = 1.0,
        warmup_steps: int = 0,
        switch_steps: int = 1000,
        ready_count: int = 128,
        mode: str = "analytic",
        candidate_multiplier: int = 4,
        max_candidates: int = 32,
        min_candidates: int = 8,
        seed: int = 42,
        eps: float = 1e-6,
    ) -> None:
        if num_phases <= 0:
            raise ValueError("num_phases must be positive")
        if num_bins <= 1:
            raise ValueError("num_bins must be greater than 1")
        if reward_temperature <= 0.0:
            raise ValueError("reward_temperature must be positive")
        if dprm_lambda < 0.0:
            raise ValueError("dprm_lambda must be non-negative")
        if mode not in {"analytic", "sampled"}:
            raise ValueError(f"Unsupported DPRM mode: {mode}")

        self.num_phases = int(num_phases)
        self.num_bins = int(num_bins)
        self.reward_temperature = float(reward_temperature)
        self.dprm_lambda = float(dprm_lambda)
        self.warmup_steps = int(warmup_steps)
        self.switch_steps = int(max(switch_steps, warmup_steps))
        self.ready_count = int(max(1, ready_count))
        self.mode = mode
        self.candidate_multiplier = int(max(1, candidate_multiplier))
        self.max_candidates = int(max(1, max_candidates))
        self.min_candidates = int(max(1, min_candidates))
        self.seed = int(seed)
        self.eps = float(eps)

        shape = (self.num_phases, self.num_bins)
        self.counts = torch.zeros(shape, dtype=torch.long)
        self.reward_sum = torch.zeros(shape, dtype=torch.float64)
        self.reward_sq_sum = torch.zeros(shape, dtype=torch.float64)
        self.exp_reward_sum = torch.zeros(shape, dtype=torch.float64)
        self.global_updates = 0

    def state_dict(self) -> dict:
        return {
            "num_phases": self.num_phases,
            "num_bins": self.num_bins,
            "reward_temperature": self.reward_temperature,
            "dprm_lambda": self.dprm_lambda,
            "warmup_steps": self.warmup_steps,
            "switch_steps": self.switch_steps,
            "ready_count": self.ready_count,
            "mode": self.mode,
            "candidate_multiplier": self.candidate_multiplier,
            "max_candidates": self.max_candidates,
            "min_candidates": self.min_candidates,
            "seed": self.seed,
            "eps": self.eps,
            "global_updates": self.global_updates,
            "counts": self.counts.tolist(),
            "reward_sum": self.reward_sum.tolist(),
            "reward_sq_sum": self.reward_sq_sum.tolist(),
            "exp_reward_sum": self.exp_reward_sum.tolist(),
        }

    def load_state_dict(self, payload: dict) -> None:
        expected = (self.num_phases, self.num_bins)
        counts = torch.tensor(payload["counts"], dtype=torch.long)
        reward_sum = torch.tensor(payload["reward_sum"], dtype=torch.float64)
        reward_sq_sum = torch.tensor(payload["reward_sq_sum"], dtype=torch.float64)
        exp_reward_sum = torch.tensor(payload["exp_reward_sum"], dtype=torch.float64)
        if counts.shape != expected:
            raise RuntimeError(f"DPRM estimator shape mismatch: {counts.shape} != {expected}")
        self.counts.copy_(counts)
        self.reward_sum.copy_(reward_sum)
        self.reward_sq_sum.copy_(reward_sq_sum)
        self.exp_reward_sum.copy_(exp_reward_sum)
        self.global_updates = int(payload.get("global_updates", 0))

    def save_json(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.state_dict(), handle, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "OnlineDPRMEstimator":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        estimator = cls(
            num_phases=payload["num_phases"],
            num_bins=payload["num_bins"],
            reward_temperature=payload["reward_temperature"],
            dprm_lambda=payload["dprm_lambda"],
            warmup_steps=payload["warmup_steps"],
            switch_steps=payload["switch_steps"],
            ready_count=payload["ready_count"],
            mode=payload.get("mode", "analytic"),
            candidate_multiplier=payload.get("candidate_multiplier", 4),
            max_candidates=payload.get("max_candidates", 32),
            min_candidates=payload.get("min_candidates", 8),
            seed=payload.get("seed", 42),
            eps=payload.get("eps", 1e-6),
        )
        estimator.load_state_dict(payload)
        return estimator

    def _confidence_to_bins(self, probs: torch.Tensor) -> torch.Tensor:
        probs = probs.clamp(min=0.0, max=1.0 - self.eps)
        bins = torch.floor(probs * self.num_bins).long()
        return bins.clamp_(0, self.num_bins - 1)

    def _global_mix(self, global_step: int, force_full: bool = False) -> float:
        if force_full:
            return 1.0
        if global_step <= self.warmup_steps:
            return 0.0
        if self.switch_steps <= self.warmup_steps:
            return 1.0
        progress = (float(global_step) - float(self.warmup_steps)) / float(self.switch_steps - self.warmup_steps)
        return max(0.0, min(1.0, progress))

    def _counts_for_bins(self, phase: torch.Tensor, bins: torch.Tensor, device: torch.device) -> torch.Tensor:
        counts = self.counts.to(device=device)
        return counts.index_select(0, phase).gather(1, bins)

    def dprm_values(self, phase: torch.Tensor, bins: torch.Tensor, device: torch.device) -> torch.Tensor:
        counts = self.counts.to(device=device, dtype=torch.float32)
        exp_reward_sum = self.exp_reward_sum.to(device=device, dtype=torch.float32)
        phase_counts = counts.index_select(0, phase)
        phase_exp = exp_reward_sum.index_select(0, phase)
        gathered_counts = phase_counts.gather(1, bins)
        gathered_exp = phase_exp.gather(1, bins)
        mean_exp = gathered_exp / gathered_counts.clamp_min(1.0)
        dprm = torch.log(mean_exp.clamp_min(self.eps)) / self.reward_temperature
        return torch.where(gathered_counts > 0, dprm, torch.zeros_like(dprm))

    def selection_summary(
        self,
        probs: torch.Tensor,
        mask: torch.Tensor,
        phase: torch.Tensor,
        global_step: int,
        force_full: bool = False,
    ) -> DPRMSelectionSummary:
        bins = self._confidence_to_bins(probs)
        counts = self._counts_for_bins(phase, bins, probs.device).float()
        local_mix = (counts / float(self.ready_count)).clamp_(0.0, 1.0)
        mix = local_mix * self._global_mix(global_step, force_full=force_full)
        dprm_values = self.dprm_values(phase, bins, probs.device)
        base_scores = torch.log(probs.clamp_min(self.eps))
        adjusted_scores = base_scores + mix * self.dprm_lambda * dprm_values
        adjusted_scores = torch.where(mask, adjusted_scores, torch.full_like(adjusted_scores, float("-inf")))
        return DPRMSelectionSummary(
            bins=bins,
            mix=mix,
            dprm_values=dprm_values,
            adjusted_scores=adjusted_scores,
        )

    def select_positions(
        self,
        probs: torch.Tensor,
        mask: torch.Tensor,
        num_select: torch.Tensor,
        phase: torch.Tensor,
        global_step: int,
        force_full: bool = False,
    ) -> tuple[torch.Tensor, DPRMSelectionSummary]:
        summary = self.selection_summary(
            probs=probs,
            mask=mask,
            phase=phase,
            global_step=global_step,
            force_full=force_full,
        )

        if self.mode == "analytic" or (summary.mix.max().item() <= 0.0):
            return _masked_topk(summary.adjusted_scores, num_select), summary

        transfer_index = torch.zeros_like(mask, dtype=torch.bool)
        base_probs = torch.where(mask, probs.clamp_min(self.eps), torch.zeros_like(probs))
        generator = torch.Generator(device=probs.device)
        generator.manual_seed(self.seed + int(global_step))
        for row in range(mask.shape[0]):
            k = int(num_select[row].item())
            if k <= 0:
                continue
            valid_idx = mask[row].nonzero(as_tuple=False).squeeze(1)
            if valid_idx.numel() == 0:
                continue

            shortlist_size = max(self.min_candidates, k * self.candidate_multiplier)
            shortlist_size = min(shortlist_size, self.max_candidates, int(valid_idx.numel()))
            if shortlist_size <= k:
                shortlist = valid_idx
            else:
                proposal = base_probs[row, valid_idx]
                proposal = proposal / proposal.sum().clamp_min(self.eps)
                sampled = torch.multinomial(
                    proposal,
                    num_samples=shortlist_size,
                    replacement=False,
                    generator=generator,
                )
                shortlist = valid_idx[sampled]

            shortlist_scores = summary.adjusted_scores[row, shortlist]
            take = min(k, int(shortlist.numel()))
            _, idx = torch.topk(shortlist_scores, k=take, largest=True)
            transfer_index[row, shortlist[idx]] = True
        return transfer_index, summary

    def register_observations(
        self,
        phase: torch.Tensor,
        probs: torch.Tensor,
        transfer_index: torch.Tensor,
        rewards: torch.Tensor,
        global_step: Optional[int] = None,
    ) -> None:
        if transfer_index.numel() == 0 or not transfer_index.any():
            if global_step is not None:
                self.global_updates = max(self.global_updates, int(global_step))
            return

        bins = self._confidence_to_bins(probs)
        phase_grid = phase.unsqueeze(1).expand_as(bins)
        reward_grid = rewards.unsqueeze(1).expand_as(probs).to(torch.float64)
        reward_grid = reward_grid.clamp(min=-20.0, max=20.0)
        exp_reward_grid = torch.exp(self.reward_temperature * reward_grid)

        flat_keys = (phase_grid[transfer_index] * self.num_bins + bins[transfer_index]).cpu()
        if flat_keys.numel() == 0:
            if global_step is not None:
                self.global_updates = max(self.global_updates, int(global_step))
            return

        flat_rewards = reward_grid[transfer_index].cpu()
        flat_rewards_sq = flat_rewards.square()
        flat_exp_rewards = exp_reward_grid[transfer_index].cpu()

        counts_update = torch.bincount(flat_keys, minlength=self.num_phases * self.num_bins).reshape(self.num_phases, self.num_bins)
        reward_update = torch.bincount(flat_keys, weights=flat_rewards, minlength=self.num_phases * self.num_bins).reshape(self.num_phases, self.num_bins)
        reward_sq_update = torch.bincount(flat_keys, weights=flat_rewards_sq, minlength=self.num_phases * self.num_bins).reshape(self.num_phases, self.num_bins)
        exp_update = torch.bincount(flat_keys, weights=flat_exp_rewards, minlength=self.num_phases * self.num_bins).reshape(self.num_phases, self.num_bins)

        self.counts += counts_update.to(self.counts.dtype)
        self.reward_sum += reward_update.to(self.reward_sum.dtype)
        self.reward_sq_sum += reward_sq_update.to(self.reward_sq_sum.dtype)
        self.exp_reward_sum += exp_update.to(self.exp_reward_sum.dtype)
        if global_step is not None:
            self.global_updates = max(self.global_updates, int(global_step))


def resolve_dprm_estimator_path(checkpoint_path: str = "", explicit_path: str = "") -> str:
    if explicit_path:
        if os.path.isfile(explicit_path):
            return explicit_path
        raise FileNotFoundError(f"DPRM estimator file not found: {explicit_path}")

    if checkpoint_path:
        candidate = os.path.join(checkpoint_path, DEFAULT_DPRM_ESTIMATOR_FILENAME)
        if os.path.isfile(candidate):
            return candidate

    return ""


def load_dprm_estimator(checkpoint_path: str = "", explicit_path: str = "") -> Optional[OnlineDPRMEstimator]:
    path = resolve_dprm_estimator_path(checkpoint_path=checkpoint_path, explicit_path=explicit_path)
    if not path:
        return None
    return OnlineDPRMEstimator.load_json(path)
