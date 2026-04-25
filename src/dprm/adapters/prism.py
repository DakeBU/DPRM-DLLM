from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..controller import DPRMConfig, OnlineDPRMController
from ..contracts import HostDPRMBatch


@dataclass
class PrismAdapterConfig:
    num_bins: int = 16
    num_phase_buckets: int = 8
    reward_beta: float = 1.0
    value_scale: float = 1.0
    warmup_pct: float = 0.2
    switch_pct: float = 0.7
    ready_count: int = 64
    candidate_multiplier: int = 4
    min_candidates: int = 8
    max_candidates: int = 64
    threshold_fallback_gate: float = 0.25


class OnlineDPRMSoftBON:
    """Compatibility adapter that maps Prism HTS ordering calls to DPRM core."""

    def __init__(self, total_steps: int, device: torch.device, cfg: Optional[dict] = None):
        cfg = cfg or {}
        parsed = PrismAdapterConfig(**{k: v for k, v in cfg.items() if k in PrismAdapterConfig.__annotations__})
        self.total_steps = max(int(total_steps), 1)
        self.device = device
        self.threshold_fallback_gate = float(parsed.threshold_fallback_gate)
        self._controller = OnlineDPRMController(
            DPRMConfig(
                num_phases=max(int(parsed.num_phase_buckets), 1),
                confidence_bins=int(parsed.num_bins),
                aux_bins=1,
                reward_temperature=float(parsed.reward_beta),
                guidance_scale=float(parsed.value_scale),
                warmup_steps=int(float(parsed.warmup_pct) * self.total_steps),
                switch_steps=max(
                    int(float(parsed.switch_pct) * self.total_steps),
                    int(float(parsed.warmup_pct) * self.total_steps) + 1,
                ),
                ready_count=max(int(parsed.ready_count), 1),
                sampled_soft_bon=True,
                candidate_multiplier=max(int(parsed.candidate_multiplier), 1),
                min_candidates=max(int(parsed.min_candidates), 1),
                max_candidates=max(int(parsed.max_candidates), max(int(parsed.min_candidates), 1)),
            ),
            device=device,
        )

    @property
    def counts(self) -> torch.Tensor:
        return self._controller.counts[..., 0]

    @property
    def reward_exp_sums(self) -> torch.Tensor:
        return self._controller.exp_reward_sums[..., 0]

    def state_dict(self) -> dict:
        return self._controller.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self._controller.load_state_dict(state_dict)

    def _phase_ids(self, confidence: torch.Tensor, step: int) -> torch.Tensor:
        return OnlineDPRMController.phase_from_progress(
            step_index=step,
            total_steps=self.total_steps,
            num_phases=self._controller.cfg.num_phases,
            batch_size=confidence.shape[0],
            device=confidence.device,
        )

    def _global_gate(self, step: int) -> float:
        return float(self._controller._global_gate(step, force_full=False))

    def score(self, confidence: torch.Tensor, valid_mask: torch.Tensor, step: int) -> torch.Tensor:
        host = HostDPRMBatch(
            confidence=confidence,
            candidate_mask=valid_mask,
            phase_ids=self._phase_ids(confidence, step),
            global_step=int(step),
        )
        return self._controller.summarize(host).score

    def select(
        self,
        confidence: torch.Tensor,
        valid_mask: torch.Tensor,
        num_select: torch.Tensor,
        step: int,
        threshold: float | None = None,
    ) -> torch.Tensor:
        selected = torch.zeros_like(valid_mask)
        global_gate = self._global_gate(step)

        for b in range(confidence.shape[0]):
            k = int(num_select[b].item())
            if k <= 0:
                continue
            active = torch.where(valid_mask[b])[0]
            if active.numel() == 0:
                continue
            if (
                threshold is not None
                and global_gate <= self.threshold_fallback_gate
                and (confidence[b, active] > threshold).sum().item() >= k
            ):
                chosen = active[torch.where(confidence[b, active] > threshold)[0][:k]]
                selected[b, chosen] = True
                continue

        remaining = (~selected) & valid_mask
        residual_select = (num_select - selected.sum(dim=1).long()).clamp_min(0)
        if residual_select.max().item() > 0:
            host = HostDPRMBatch(
                confidence=confidence,
                candidate_mask=remaining,
                phase_ids=self._phase_ids(confidence, step),
                global_step=int(step),
            )
            residual = self._controller.select(host, residual_select)
            selected |= residual.selected_mask
        return selected

    def select_low(
        self,
        confidence_row: torch.Tensor,
        valid_mask_row: torch.Tensor,
        k: int,
        step: int,
    ) -> torch.Tensor:
        if k <= 0:
            return torch.empty(0, device=confidence_row.device, dtype=torch.long)
        adjusted = self.score(
            confidence_row.unsqueeze(0),
            valid_mask_row.unsqueeze(0),
            step,
        )[0]
        active = torch.where(valid_mask_row)[0]
        if active.numel() == 0:
            return active
        k = min(k, active.numel())
        _, low_idx = torch.topk(adjusted[active], k=k, largest=False)
        return active[low_idx]

    def observe_visible(
        self,
        confidence: torch.Tensor,
        visible_mask: torch.Tensor,
        reward_per_seq: torch.Tensor,
        step: int,
    ):
        host = HostDPRMBatch(
            confidence=confidence,
            candidate_mask=visible_mask,
            phase_ids=self._phase_ids(confidence, step),
            global_step=int(step),
        )
        self._controller.observe(
            host=host,
            revealed_mask=visible_mask,
            rewards=reward_per_seq,
        )
