from __future__ import annotations

from typing import Optional

import torch

from ..controller import DPRMConfig, OnlineDPRMController
from ..contracts import HostDPRMBatch


class BucketizedDPRMController:
    """Compatibility adapter that maps the legacy PUMA fork API to DPRM core."""

    def __init__(
        self,
        num_phases: int,
        device: torch.device,
        cfg: Optional[dict] = None,
    ):
        cfg = cfg or {}
        shared_cfg = DPRMConfig(
            num_phases=int(num_phases),
            confidence_bins=int(cfg.get("num_bins", 16)),
            aux_bins=1,
            reward_temperature=float(cfg.get("reward_beta", 1.0)),
            guidance_scale=float(cfg.get("value_scale", 1.0)),
            warmup_steps=int(cfg.get("warmup_steps", 0)),
            switch_steps=int(cfg.get("switch_steps", 1000)),
            ready_count=max(int(cfg.get("ready_count", 64)), 1),
            sampled_soft_bon=bool(cfg.get("sampled_shortlist", False)),
            candidate_multiplier=max(int(cfg.get("candidate_multiplier", 4)), 1),
            min_candidates=max(int(cfg.get("min_candidates", 8)), 1),
            max_candidates=max(
                int(cfg.get("max_candidates", 64)),
                max(int(cfg.get("min_candidates", 8)), 1),
            ),
        )
        self._controller = OnlineDPRMController(shared_cfg, device=device)

    @property
    def num_phases(self) -> int:
        return int(self._controller.cfg.num_phases)

    @property
    def counts(self) -> torch.Tensor:
        return self._controller.counts[..., 0]

    @property
    def reward_exp_sums(self) -> torch.Tensor:
        return self._controller.exp_reward_sums[..., 0]

    def state_dict(self) -> dict:
        return self._controller.state_dict()

    def load_state_dict(self, state_dict: dict):
        self._controller.load_state_dict(state_dict)

    def select_mask(
        self,
        log_confidence: torch.Tensor,
        valid_mask: torch.Tensor,
        phases: torch.Tensor,
        step: int,
        num_select: torch.Tensor,
        force_full: bool = False,
    ) -> torch.Tensor:
        host = HostDPRMBatch(
            confidence=log_confidence.detach().exp().clamp_(0.0, 1.0 - 1e-6),
            candidate_mask=valid_mask,
            phase_ids=phases,
            global_step=int(step),
            force_full_dprm=bool(force_full),
        )
        return self._controller.select(host, num_select).selected_mask

    def observe(
        self,
        log_confidence: torch.Tensor,
        selected_mask: torch.Tensor,
        phases: torch.Tensor,
        reward_per_seq: torch.Tensor,
    ):
        host = HostDPRMBatch(
            confidence=log_confidence.detach().exp().clamp_(0.0, 1.0 - 1e-6),
            candidate_mask=selected_mask,
            phase_ids=phases,
        )
        self._controller.observe(
            host=host,
            revealed_mask=selected_mask,
            rewards=reward_per_seq,
        )


def confidence_phase_from_step(
    step: int,
    total_steps: int,
    num_phases: int,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    return OnlineDPRMController.phase_from_progress(
        step_index=step,
        total_steps=total_steps,
        num_phases=num_phases,
        batch_size=batch_size,
        device=device,
    )
