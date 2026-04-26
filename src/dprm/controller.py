from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .contracts import DPRMSelection, HostDPRMBatch


@dataclass
class DPRMConfig:
    num_phases: int = 8
    confidence_bins: int = 16
    aux_bins: int = 1
    reward_temperature: float = 1.0
    guidance_scale: float = 1.0
    warmup_steps: int = 0
    switch_steps: int = 1000
    ready_count: int = 64
    sampled_soft_bon: bool = True
    candidate_multiplier: int = 4
    min_candidates: int = 8
    max_candidates: int = 64
    eps: float = 1e-6


class OnlineDPRMController:
    """Generic online DPRM controller.

    The controller is host-agnostic. The host is responsible for:
      * computing confidence or another base proposal score;
      * defining which positions are eligible at the current step;
      * mapping the current masked state to a phase bucket;
      * optionally providing an auxiliary bucket per position;
      * supplying a utility signal already produced by the host algorithm.
    """

    def __init__(self, cfg: DPRMConfig, device: Optional[torch.device] = None):
        self.cfg = cfg
        self.device = device or torch.device("cpu")

        shape = (
            int(cfg.num_phases),
            int(cfg.confidence_bins),
            int(max(cfg.aux_bins, 1)),
        )
        self.counts = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.exp_reward_sums = torch.zeros_like(self.counts)

    def to(self, device: torch.device) -> "OnlineDPRMController":
        self.device = device
        self.counts = self.counts.to(device)
        self.exp_reward_sums = self.exp_reward_sums.to(device)
        return self

    def state_dict(self) -> dict:
        return {
            "cfg": self.cfg.__dict__.copy(),
            "counts": self.counts.detach().cpu(),
            "exp_reward_sums": self.exp_reward_sums.detach().cpu(),
        }

    def load_state_dict(self, payload: dict) -> None:
        self.counts = payload["counts"].to(self.device)
        self.exp_reward_sums = payload["exp_reward_sums"].to(self.device)
        if self.counts.dim() != 3:
            raise ValueError(
                f"Expected DPRM counts tensor with 3 dims, got shape {tuple(self.counts.shape)}"
            )
        self.cfg.num_phases = int(self.counts.shape[0])
        self.cfg.confidence_bins = int(self.counts.shape[1])
        self.cfg.aux_bins = int(self.counts.shape[2])

    @staticmethod
    def phase_from_progress(
        step_index: int,
        total_steps: int,
        num_phases: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_steps = max(int(total_steps), 1)
        num_phases = max(int(num_phases), 1)
        phase = min((int(step_index) * num_phases) // total_steps, num_phases - 1)
        return torch.full((batch_size,), phase, dtype=torch.long, device=device)

    def _confidence_bins(self, confidence: torch.Tensor) -> torch.Tensor:
        conf = confidence.detach().clamp_(0.0, 1.0 - self.cfg.eps)
        bins = torch.floor(conf * self.cfg.confidence_bins).long()
        return bins.clamp_(0, self.cfg.confidence_bins - 1)

    def _aux_bins(
        self,
        aux_bin_ids: Optional[torch.Tensor],
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        if aux_bin_ids is None:
            return torch.zeros_like(candidate_mask, dtype=torch.long)
        return aux_bin_ids.long().clamp_(0, max(self.cfg.aux_bins - 1, 0))

    def _global_gate(self, global_step: int, force_full: bool) -> float:
        if force_full:
            return 1.0
        if global_step <= self.cfg.warmup_steps:
            return 0.0
        if self.cfg.switch_steps <= self.cfg.warmup_steps:
            return 1.0
        progress = (
            float(global_step) - float(self.cfg.warmup_steps)
        ) / float(max(self.cfg.switch_steps - self.cfg.warmup_steps, 1))
        return max(0.0, min(1.0, progress))

    def summarize(self, host: HostDPRMBatch) -> DPRMSelection:
        confidence = host.confidence.to(self.device)
        candidate_mask = host.candidate_mask.to(self.device).bool()
        phase_ids = host.phase_ids.to(self.device).long().clamp_(
            0, max(int(self.counts.shape[0]) - 1, 0)
        )
        conf_bins = self._confidence_bins(confidence)
        conf_bins = conf_bins.clamp_(0, max(int(self.counts.shape[1]) - 1, 0))
        aux_bins = self._aux_bins(host.aux_bin_ids, candidate_mask).to(self.device)
        aux_bins = aux_bins.clamp_(0, max(int(self.counts.shape[2]) - 1, 0))

        counts = self.counts[phase_ids[:, None], conf_bins, aux_bins]
        exp_rewards = self.exp_reward_sums[phase_ids[:, None], conf_bins, aux_bins]
        safe_mean = torch.where(
            counts > 0,
            exp_rewards / counts.clamp_min(1.0),
            torch.ones_like(exp_rewards),
        )

        dprm_value = torch.log(safe_mean.clamp_min(self.cfg.eps)) / max(
            self.cfg.reward_temperature, self.cfg.eps
        )
        local_gate = (counts / float(max(self.cfg.ready_count, 1))).clamp_(0.0, 1.0)
        gate = local_gate * self._global_gate(
            global_step=int(host.global_step),
            force_full=bool(host.force_full_dprm),
        )

        base_score = torch.log(confidence.clamp_min(self.cfg.eps))
        score = base_score + gate * self.cfg.guidance_scale * dprm_value
        score = torch.where(
            candidate_mask,
            score,
            torch.full_like(score, float("-inf")),
        )

        empty = torch.zeros_like(candidate_mask, dtype=torch.bool)
        return DPRMSelection(
            selected_mask=empty,
            score=score,
            base_score=base_score,
            dprm_value=dprm_value,
            gate=gate,
            confidence_bins=conf_bins,
        )

    def select(self, host: HostDPRMBatch, num_select: torch.Tensor) -> DPRMSelection:
        summary = self.summarize(host)
        confidence = host.confidence.to(self.device)
        candidate_mask = host.candidate_mask.to(self.device).bool()
        num_select = num_select.to(self.device)
        picked = torch.zeros_like(candidate_mask, dtype=torch.bool)

        for row in range(candidate_mask.size(0)):
            k = int(num_select[row].item())
            active = torch.nonzero(candidate_mask[row], as_tuple=False).squeeze(1)
            if active.numel() == 0:
                continue
            k = min(k, int(active.numel()))
            if k <= 0:
                continue

            shortlist = active
            if self.cfg.sampled_soft_bon and active.numel() > k:
                shortlist_size = max(
                    self.cfg.min_candidates,
                    self.cfg.candidate_multiplier * max(k, 1),
                )
                shortlist_size = min(
                    shortlist_size,
                    self.cfg.max_candidates,
                    int(active.numel()),
                )
                proposal = confidence[row, active].float().clamp_min(0.0)
                if proposal.sum().item() <= 0:
                    proposal = torch.ones_like(proposal)
                proposal = proposal / proposal.sum()
                sampled = torch.multinomial(
                    proposal,
                    num_samples=shortlist_size,
                    replacement=False,
                )
                shortlist = active[sampled]

            shortlist_scores = summary.score[row, shortlist]
            topk = min(k, int(shortlist.numel()))
            chosen = shortlist[torch.topk(shortlist_scores, k=topk, largest=True).indices]
            picked[row, chosen] = True

        summary.selected_mask = picked
        return summary

    def observe(
        self,
        host: HostDPRMBatch,
        revealed_mask: torch.Tensor,
        rewards: torch.Tensor,
    ) -> None:
        revealed_mask = revealed_mask.to(self.device).bool()
        if revealed_mask.numel() == 0 or not revealed_mask.any():
            return

        confidence = host.confidence.to(self.device)
        phase_ids = host.phase_ids.to(self.device).long()
        conf_bins = self._confidence_bins(confidence)
        aux_bins = self._aux_bins(host.aux_bin_ids, revealed_mask).to(self.device)
        phase_grid = phase_ids.unsqueeze(1).expand_as(conf_bins)
        reward_grid = rewards.to(self.device).float().unsqueeze(1).expand_as(confidence)
        exp_reward_grid = torch.exp(
            self.cfg.reward_temperature * reward_grid.clamp(min=-20.0, max=20.0)
        )

        phase_flat = phase_grid[revealed_mask]
        conf_flat = conf_bins[revealed_mask]
        aux_flat = aux_bins[revealed_mask]
        reward_flat = exp_reward_grid[revealed_mask]
        ones = torch.ones_like(reward_flat)

        self.counts.index_put_((phase_flat, conf_flat, aux_flat), ones, accumulate=True)
        self.exp_reward_sums.index_put_(
            (phase_flat, conf_flat, aux_flat),
            reward_flat,
            accumulate=True,
        )
