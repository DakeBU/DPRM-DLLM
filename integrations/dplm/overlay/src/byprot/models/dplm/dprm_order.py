# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DPLMOrderConfig:
    enable: bool = False
    train_order_policy: str = field(default="random_mask")
    decode_order_policy: str = field(default="confidence")
    num_phases: int = field(default=8)
    confidence_bins: int = field(default=16)
    structure_bins: int = field(default=1)
    use_structure_bins: bool = field(default=False)
    contact_threshold: float = field(default=10.0)
    reward: str = field(default="aar")
    reward_temperature: float = field(default=1.0)
    guidance_scale: float = field(default=1.0)
    warmup_steps: int = field(default=0)
    switch_steps: int = field(default=0)
    ready_count: int = field(default=64)
    sampled_soft_bon: bool = field(default=True)
    candidate_multiplier: int = field(default=4)
    min_candidates: int = field(default=8)
    max_candidates: int = field(default=32)
    confidence_threshold: float = field(default=0.0)


class DPRMOrderController(nn.Module):
    def __init__(self, cfg: DPLMOrderConfig):
        super().__init__()
        self.cfg = cfg
        structure_bins = (
            cfg.structure_bins if cfg.use_structure_bins else 1
        )
        stats_shape = (cfg.num_phases, cfg.confidence_bins, structure_bins)
        self.register_buffer(
            "counts", torch.zeros(stats_shape, dtype=torch.float32)
        )
        self.register_buffer(
            "exp_reward_sums", torch.zeros(stats_shape, dtype=torch.float32)
        )

    @property
    def structure_bins(self) -> int:
        return self.counts.shape[-1]

    def enabled_for_training(self) -> bool:
        return self.cfg.enable and self.cfg.train_order_policy in {
            "progressive_confidence",
            "progressive_dprm",
        }

    def enabled_for_decode(self) -> bool:
        return self.cfg.enable and self.cfg.decode_order_policy in {
            "confidence",
            "dprm_soft_bon",
        }

    def mask_counts_to_phase(
        self, masked_mask: torch.Tensor, design_mask: torch.Tensor
    ) -> torch.Tensor:
        total = design_mask.sum(dim=1).clamp_min(1)
        masked = (masked_mask & design_mask).sum(dim=1)
        revealed_ratio = 1.0 - masked.float() / total.float()
        phase = torch.floor(revealed_ratio * self.cfg.num_phases).long()
        return phase.clamp_(0, self.cfg.num_phases - 1)

    def decode_step_to_phase(
        self, step: int, max_step: int, batch_size: int, device
    ) -> torch.Tensor:
        if max_step <= 0:
            phase = 0
        else:
            phase = min(
                int((float(step) / float(max_step)) * self.cfg.num_phases),
                self.cfg.num_phases - 1,
            )
        return torch.full((batch_size,), phase, device=device, dtype=torch.long)

    def confidence_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        safe_logits = torch.nan_to_num(
            logits.float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        return F.softmax(safe_logits, dim=-1).amax(dim=-1)

    def confidence_bins(
        self, confidence: torch.Tensor, design_mask: torch.Tensor
    ) -> torch.Tensor:
        bins = torch.floor(
            confidence.clamp(0.0, 1.0 - 1e-8) * self.cfg.confidence_bins
        ).long()
        bins.clamp_(0, self.cfg.confidence_bins - 1)
        return bins.masked_fill(~design_mask, 0)

    def structure_bin_indices(
        self, batch: Optional[Dict], design_mask: torch.Tensor
    ) -> torch.Tensor:
        if (
            not self.cfg.use_structure_bins
            or self.structure_bins == 1
            or batch is None
            or "coords" not in batch
        ):
            return torch.zeros_like(design_mask, dtype=torch.long)

        coords = batch["coords"]
        if coords.ndim != 4 or coords.size(-2) < 2:
            return torch.zeros_like(design_mask, dtype=torch.long)

        ca_coords = torch.nan_to_num(coords[:, :, 1, :].float(), nan=0.0)
        valid = design_mask & torch.isfinite(coords[:, :, 1, :]).all(dim=-1)
        dists = torch.cdist(ca_coords, ca_coords)
        contacts = (
            (dists <= self.cfg.contact_threshold)
            & valid[:, :, None]
            & valid[:, None, :]
        ).float()
        contacts = contacts.sum(dim=-1) - valid.float()
        max_contacts = contacts.max(dim=1, keepdim=True).values.clamp_min(1.0)
        norm_contacts = (contacts / max_contacts).clamp(0.0, 1.0 - 1e-8)
        bins = torch.floor(norm_contacts * self.structure_bins).long()
        bins.clamp_(0, self.structure_bins - 1)
        return bins.masked_fill(~design_mask, 0)

    def estimate_terminal_reward(
        self,
        logits: torch.Tensor,
        current_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        design_mask: torch.Tensor,
        mask_id: int,
    ) -> torch.Tensor:
        if self.cfg.reward != "aar":
            raise NotImplementedError(
                f"Unsupported DPRM-DPLM utility: {self.cfg.reward}"
            )

        safe_logits = torch.nan_to_num(
            logits.float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        pred_tokens = safe_logits.argmax(dim=-1)
        provisional = torch.where(
            current_tokens.eq(mask_id), pred_tokens, current_tokens
        )
        correct = (provisional == target_tokens) & design_mask
        return correct.float().sum(dim=1) / design_mask.sum(dim=1).clamp_min(1)

    def _bucket_stats(
        self,
        phase_ids: torch.Tensor,
        conf_bins: torch.Tensor,
        struct_bins: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        counts = self.counts[phase_ids[:, None], conf_bins, struct_bins]
        exp_reward_sums = self.exp_reward_sums[
            phase_ids[:, None], conf_bins, struct_bins
        ]
        return counts, exp_reward_sums

    def dprm_values(
        self,
        phase_ids: torch.Tensor,
        conf_bins: torch.Tensor,
        struct_bins: torch.Tensor,
    ) -> torch.Tensor:
        counts, exp_reward_sums = self._bucket_stats(
            phase_ids, conf_bins, struct_bins
        )
        safe_mean = torch.where(
            exp_reward_sums > 0,
            exp_reward_sums / counts.clamp_min(1.0),
            torch.ones_like(exp_reward_sums),
        )
        beta = max(self.cfg.reward_temperature, 1e-6)
        return torch.log(safe_mean) / beta

    def dprm_gate(
        self,
        global_step: Optional[int],
        phase_ids: torch.Tensor,
        conf_bins: torch.Tensor,
        struct_bins: torch.Tensor,
    ) -> torch.Tensor:
        if global_step is None or self.cfg.switch_steps <= 0:
            global_gate = 1.0
        elif global_step <= self.cfg.warmup_steps:
            global_gate = 0.0
        elif global_step >= self.cfg.switch_steps:
            global_gate = 1.0
        else:
            denom = max(
                self.cfg.switch_steps - self.cfg.warmup_steps, 1
            )
            global_gate = float(global_step - self.cfg.warmup_steps) / float(
                denom
            )

        counts, _ = self._bucket_stats(phase_ids, conf_bins, struct_bins)
        if self.cfg.ready_count <= 0:
            local_gate = torch.ones_like(counts)
        else:
            local_gate = (counts / float(self.cfg.ready_count)).clamp(0.0, 1.0)
        return local_gate * float(global_gate)

    def selection_scores(
        self,
        *,
        confidence: torch.Tensor,
        phase_ids: torch.Tensor,
        conf_bins: torch.Tensor,
        struct_bins: torch.Tensor,
        policy: str,
        global_step: Optional[int] = None,
    ) -> torch.Tensor:
        base_score = torch.log(confidence.clamp_min(1e-8))
        if policy in {"confidence", "progressive_confidence"}:
            return base_score
        if policy not in {"dprm_soft_bon", "progressive_dprm"}:
            raise NotImplementedError(f"Unsupported ordering policy: {policy}")

        values = self.dprm_values(phase_ids, conf_bins, struct_bins)
        gates = self.dprm_gate(
            global_step, phase_ids, conf_bins, struct_bins
        )
        return base_score + gates * self.cfg.guidance_scale * values

    def shortlist_mask(
        self,
        candidate_mask: torch.Tensor,
        proposal_scores: torch.Tensor,
        target_counts: torch.Tensor,
    ) -> torch.Tensor:
        if not self.cfg.sampled_soft_bon:
            return candidate_mask

        shortlist = torch.zeros_like(candidate_mask, dtype=torch.bool)
        batch_size = candidate_mask.size(0)
        for row in range(batch_size):
            indices = torch.nonzero(candidate_mask[row], as_tuple=False).view(-1)
            if indices.numel() == 0:
                continue
            budget = int(target_counts[row].item())
            shortlist_size = max(
                self.cfg.min_candidates,
                self.cfg.candidate_multiplier * max(budget, 1),
            )
            shortlist_size = min(
                self.cfg.max_candidates, shortlist_size, indices.numel()
            )
            if shortlist_size >= indices.numel():
                shortlist[row, indices] = True
                continue

            weights = proposal_scores[row, indices].float()
            weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            weights = weights.clamp_min(0.0)

            if not torch.isfinite(weights).all() or weights.sum().item() <= 0.0:
                weights = torch.ones_like(weights, dtype=torch.float32)

            weights_cpu = weights.detach().cpu()
            chosen_local = None
            try:
                chosen_local = torch.multinomial(
                    weights_cpu, shortlist_size, replacement=False
                )
            except RuntimeError:
                # Fall back to deterministic shortlist selection if multinomial
                # sees a numerically degenerate proposal distribution.
                chosen_local = torch.topk(
                    weights_cpu, k=shortlist_size, largest=True
                ).indices

            chosen_local = chosen_local.to(indices.device, non_blocking=True)
            shortlist[row, indices[chosen_local]] = True
        return shortlist

    def select_positions(
        self,
        scores: torch.Tensor,
        candidate_mask: torch.Tensor,
        target_counts: torch.Tensor,
        proposal_scores: torch.Tensor,
    ) -> torch.Tensor:
        shortlist_mask = self.shortlist_mask(
            candidate_mask, proposal_scores, target_counts
        )
        picked = torch.zeros_like(candidate_mask, dtype=torch.bool)
        batch_size = candidate_mask.size(0)

        for row in range(batch_size):
            k = int(target_counts[row].item())
            if k <= 0:
                continue

            indices = torch.nonzero(
                shortlist_mask[row], as_tuple=False
            ).view(-1)
            if indices.numel() == 0:
                continue
            row_scores = scores[row, indices]
            topk = min(k, indices.numel())
            chosen_local = torch.topk(row_scores, k=topk, largest=True).indices
            picked[row, indices[chosen_local]] = True
        return picked

    def update_statistics(
        self,
        reveal_mask: torch.Tensor,
        phase_ids: torch.Tensor,
        conf_bins: torch.Tensor,
        struct_bins: torch.Tensor,
        rewards: torch.Tensor,
    ) -> None:
        reveal_idx = torch.nonzero(reveal_mask, as_tuple=False)
        if reveal_idx.numel() == 0:
            return

        batch_idx = reveal_idx[:, 0]
        token_idx = reveal_idx[:, 1]
        phase = phase_ids[batch_idx]
        conf = conf_bins[batch_idx, token_idx]
        struct = struct_bins[batch_idx, token_idx]

        flat_index = (
            phase * (self.cfg.confidence_bins * self.structure_bins)
            + conf * self.structure_bins
            + struct
        )
        exp_rewards = torch.exp(
            rewards[batch_idx].float() * float(self.cfg.reward_temperature)
        )

        flat_counts = self.counts.view(-1)
        flat_exp_sums = self.exp_reward_sums.view(-1)
        flat_counts.index_add_(
            0, flat_index, torch.ones_like(exp_rewards, dtype=flat_counts.dtype)
        )
        flat_exp_sums.index_add_(
            0, flat_index, exp_rewards.to(flat_exp_sums.dtype)
        )

    def build_progressive_states(
        self,
        *,
        target_tokens: torch.Tensor,
        design_mask: torch.Tensor,
        desired_mask_counts: torch.Tensor,
        score_fn: Callable[[torch.Tensor], torch.Tensor],
        batch: Optional[Dict],
        global_step: Optional[int],
        mask_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        current_tokens = target_tokens.masked_fill(design_mask, mask_id)
        current_mask = design_mask.clone()
        struct_bins = self.structure_bin_indices(batch, design_mask)
        total_maskable = design_mask.sum(dim=1)
        desired_mask_counts = torch.minimum(
            desired_mask_counts.long(), total_maskable
        ).clamp_min(1)

        for phase_step in range(self.cfg.num_phases):
            remaining_reveal = (
                current_mask.sum(dim=1).long() - desired_mask_counts
            ).clamp_min(0)
            if remaining_reveal.max().item() <= 0:
                break

            logits = score_fn(current_tokens)
            confidence = self.confidence_from_logits(logits)
            phase_ids = self.mask_counts_to_phase(current_mask, design_mask)
            conf_bins = self.confidence_bins(confidence, design_mask)
            scores = self.selection_scores(
                confidence=confidence,
                phase_ids=phase_ids,
                conf_bins=conf_bins,
                struct_bins=struct_bins,
                policy=self.cfg.train_order_policy,
                global_step=global_step,
            )
            scores = scores.masked_fill(~current_mask, -1e9)

            remaining_phases = max(self.cfg.num_phases - phase_step, 1)
            reveal_budget = torch.ceil(
                remaining_reveal.float() / float(remaining_phases)
            ).long()
            reveal_budget = torch.minimum(reveal_budget, remaining_reveal)

            reveal_mask = self.select_positions(
                scores=scores,
                candidate_mask=current_mask,
                target_counts=reveal_budget,
                proposal_scores=confidence,
            )

            if self.cfg.confidence_threshold > 0:
                extra_room = (
                    current_mask.sum(dim=1).long()
                    - desired_mask_counts
                    - reveal_mask.sum(dim=1).long()
                ).clamp_min(0)
                threshold_mask = (
                    current_mask
                    & ~reveal_mask
                    & (confidence >= self.cfg.confidence_threshold)
                )
                extra_mask = self.select_positions(
                    scores=confidence.masked_fill(~threshold_mask, -1e9),
                    candidate_mask=threshold_mask,
                    target_counts=extra_room,
                    proposal_scores=confidence,
                )
                reveal_mask |= extra_mask

            rewards = self.estimate_terminal_reward(
                logits=logits,
                current_tokens=current_tokens,
                target_tokens=target_tokens,
                design_mask=design_mask,
                mask_id=mask_id,
            )
            self.update_statistics(
                reveal_mask, phase_ids, conf_bins, struct_bins, rewards
            )
            current_tokens = torch.where(reveal_mask, target_tokens, current_tokens)
            current_mask = current_mask & ~reveal_mask

        leftover = (
            current_mask.sum(dim=1).long() - desired_mask_counts
        ).clamp_min(0)
        if leftover.max().item() > 0:
            logits = score_fn(current_tokens)
            confidence = self.confidence_from_logits(logits)
            phase_ids = self.mask_counts_to_phase(current_mask, design_mask)
            conf_bins = self.confidence_bins(confidence, design_mask)
            scores = self.selection_scores(
                confidence=confidence,
                phase_ids=phase_ids,
                conf_bins=conf_bins,
                struct_bins=struct_bins,
                policy=self.cfg.train_order_policy,
                global_step=global_step,
            ).masked_fill(~current_mask, -1e9)
            reveal_mask = self.select_positions(
                scores=scores,
                candidate_mask=current_mask,
                target_counts=leftover,
                proposal_scores=confidence,
            )
            rewards = self.estimate_terminal_reward(
                logits=logits,
                current_tokens=current_tokens,
                target_tokens=target_tokens,
                design_mask=design_mask,
                mask_id=mask_id,
            )
            struct_bins = self.structure_bin_indices(batch, design_mask)
            self.update_statistics(
                reveal_mask, phase_ids, conf_bins, struct_bins, rewards
            )
            current_tokens = torch.where(reveal_mask, target_tokens, current_tokens)
            current_mask = current_mask & ~reveal_mask

        return current_tokens, current_mask
