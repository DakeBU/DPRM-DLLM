import math
import os
import sys
from dataclasses import dataclass
from typing import Optional

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from dprm_guidance import OnlineDPRMEstimator


def build_intervals(k: int) -> list[tuple[float, float]]:
    if k <= 0:
        return []
    return [(j / k, (j + 1) / k) for j in range(k)]


def phase_initialize(batch_size: int, k: int, device: torch.device, mode: str = "random") -> torch.Tensor:
    if mode == "zero":
        return torch.zeros(batch_size, dtype=torch.long, device=device)
    if mode != "random":
        raise ValueError(f"Unsupported phase init mode: {mode}")
    base = torch.arange(k, device=device)
    repeats = math.ceil(batch_size / k)
    phases = base.repeat(repeats)[:batch_size]
    return phases[torch.randperm(batch_size, device=device)]


def unmask_from_scores(
    scores: torch.Tensor,
    num_unmask: torch.Tensor,
    x0: torch.Tensor,
    xt_format: torch.Tensor,
) -> torch.Tensor:
    batch_size = scores.shape[0]
    k_max = int(num_unmask.max().item())
    new_xt = xt_format.clone()
    if k_max == 0:
        return new_xt

    _, topk_idx = scores.topk(k=k_max, dim=1, largest=True)
    arange_k = torch.arange(k_max, device=scores.device).unsqueeze(0).expand(batch_size, k_max)
    unmask_idx = arange_k < num_unmask.unsqueeze(1)
    rows = torch.arange(batch_size, device=scores.device).unsqueeze(1).expand(batch_size, k_max)
    flat_r = rows[unmask_idx]
    flat_c = topk_idx[unmask_idx]
    new_xt[flat_r, flat_c] = x0[flat_r, flat_c]
    return new_xt


def build_loss_prompt_mask(
    prompt_length: int,
    completion_mask: torch.Tensor,
    loss_mask_non_eos: bool,
) -> torch.Tensor:
    completion_mask = completion_mask.bool()
    batch_size, gen_length = completion_mask.shape
    prompt_mask = torch.zeros(
        batch_size,
        prompt_length + gen_length,
        dtype=torch.bool,
        device=completion_mask.device,
    )
    prompt_mask[:, :prompt_length] = True
    if loss_mask_non_eos:
        prompt_mask[:, prompt_length:] = ~completion_mask
    return prompt_mask


@dataclass
class ProgressiveMaskingState:
    x0: torch.Tensor
    xt: torch.Tensor
    prompt_mask: torch.Tensor
    phase: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    L_eff: torch.Tensor
    mask_id: int
    confidence_threshold: Optional[float] = None
    order_policy: str = "confidence"
    dprm_estimator: Optional[OnlineDPRMEstimator] = None

    @property
    def device(self) -> torch.device:
        return self.x0.device

    @property
    def k(self) -> int:
        return int(self.lower.numel())

    @property
    def masked_index(self) -> torch.Tensor:
        return self.xt == self.mask_id

    def calculate_phase(self, xt: Optional[torch.Tensor] = None) -> torch.Tensor:
        if xt is None:
            xt = self.xt
        current_unmask = (~self.prompt_mask & (xt != self.mask_id)).sum(dim=1).long()
        ratio = current_unmask.float() / self.L_eff.clamp_min(1).float()
        boundaries = self.upper[:-1]
        stage = torch.bucketize(ratio, boundaries)
        return stage.clamp_(0, self.k - 1).long()

    def _sample_ratio(self, stages: torch.Tensor) -> torch.Tensor:
        lo = self.lower.index_select(0, stages)
        hi = self.upper.index_select(0, stages)
        return lo + torch.rand_like(lo) * (hi - lo)

    def _sample_target_unmasked(self, ratio: torch.Tensor, L_eff: Optional[torch.Tensor] = None) -> torch.Tensor:
        if L_eff is None:
            L_eff = self.L_eff
        num_unmask = torch.round(ratio * L_eff.float()).long()
        cap = (L_eff - 1).clamp_min(0)
        return torch.minimum(num_unmask, cap).clamp_min(0)

    def _initialize_xt(self, stages: torch.Tensor, row_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if row_mask is None:
            x0 = self.x0
            prompt_mask = self.prompt_mask
            L_eff = self.L_eff
        else:
            x0 = self.x0[row_mask]
            prompt_mask = self.prompt_mask[row_mask]
            L_eff = self.L_eff[row_mask]
        xt = torch.where(prompt_mask, x0, torch.full_like(x0, self.mask_id))
        ratio = self._sample_ratio(stages)
        num_unmask = self._sample_target_unmasked(ratio, L_eff=L_eff)
        if int(num_unmask.max().item()) == 0:
            return xt
        rand_score = torch.rand(x0.shape, device=self.device, dtype=torch.float32)
        rand_score = torch.where(
            prompt_mask,
            torch.full_like(rand_score, torch.finfo(rand_score.dtype).min),
            rand_score,
        )
        return unmask_from_scores(rand_score, num_unmask, x0, xt)

    @torch.no_grad()
    def reset_rows(self, row_mask: torch.Tensor, phase_value: int = 0) -> None:
        if not row_mask.any():
            return
        indices = row_mask.nonzero(as_tuple=False).squeeze(1)
        stages = torch.full(
            (indices.numel(),),
            phase_value,
            dtype=torch.long,
            device=self.device,
        )
        self.xt[indices] = self._initialize_xt(stages, row_mask=row_mask)
        self.phase[indices] = phase_value

    @torch.no_grad()
    def advance(
        self,
        log_probs: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
        reward_targets: Optional[torch.Tensor] = None,
        global_step: int = 0,
    ) -> None:
        if active_mask is None:
            active_mask = torch.ones_like(self.phase, dtype=torch.bool)
        if not active_mask.any():
            return

        phase_next = self.phase.clone()
        phase_next[active_mask] = (phase_next[active_mask] + 1) % self.k
        replace = active_mask & (phase_next == 0)

        mask_idx = self.masked_index
        if (mask_idx & self.prompt_mask).any():
            raise ValueError("Prompt positions should not be masked")

        ratio = self._sample_ratio(phase_next)
        target_unmask = self._sample_target_unmasked(ratio)
        current_unmask = (~mask_idx & ~self.prompt_mask).sum(dim=1).long()
        to_reveal = (target_unmask - current_unmask).clamp_min(0)
        to_reveal = torch.where(active_mask, to_reveal, torch.zeros_like(to_reveal))
        to_reveal = torch.where(replace, torch.zeros_like(to_reveal), to_reveal)

        xt = self.xt
        pmax = log_probs.max(dim=2).values.exp()
        if int(to_reveal.max().item()) > 0:
            selection_mask = mask_idx & (~self.prompt_mask) & active_mask[:, None]
            if self.order_policy == "dprm_soft_bon" and self.dprm_estimator is not None:
                transfer_index, _ = self.dprm_estimator.select_positions(
                    probs=pmax,
                    mask=selection_mask,
                    num_select=to_reveal,
                    phase=self.phase,
                    global_step=global_step,
                    force_full=False,
                )
                xt = torch.where(transfer_index, self.x0, xt)
                if reward_targets is not None:
                    self.dprm_estimator.register_observations(
                        phase=self.phase,
                        probs=pmax,
                        transfer_index=transfer_index,
                        rewards=reward_targets,
                        global_step=global_step,
                    )
            else:
                score_conf = torch.where(
                    selection_mask,
                    log_probs.max(dim=2).values,
                    torch.full_like(log_probs[..., 0], torch.finfo(log_probs.dtype).min),
                )
                xt = unmask_from_scores(score_conf, to_reveal, self.x0, xt)

        if self.confidence_threshold is not None:
            tau = math.log(self.confidence_threshold)
            collapse = (log_probs.max(dim=2).values > tau) & (xt == self.mask_id) & (~self.prompt_mask) & active_mask[:, None]
            xt = torch.where(collapse, self.x0, xt)
            phase_next = torch.where(active_mask, self.calculate_phase(xt), phase_next)

        self.xt = xt
        self.phase = phase_next
        self.reset_rows(replace, phase_value=0)


def initialize_progressive_state(
    input_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    mask_id: int,
    k: int,
    phase_init: str = "random",
    confidence_threshold: Optional[float] = None,
    order_policy: str = "confidence",
    dprm_estimator: Optional[OnlineDPRMEstimator] = None,
) -> ProgressiveMaskingState:
    device = input_ids.device
    batch_size = input_ids.shape[0]
    intervals = build_intervals(k)
    lower = torch.tensor([lo for lo, _ in intervals], device=device, dtype=torch.float32)
    upper = torch.tensor([hi for _, hi in intervals], device=device, dtype=torch.float32)
    phase = phase_initialize(batch_size, k, device, mode=phase_init)
    L_eff = (~prompt_mask).sum(dim=1).long()
    dummy_xt = torch.where(prompt_mask, input_ids, torch.full_like(input_ids, mask_id))
    state = ProgressiveMaskingState(
        x0=input_ids,
        xt=dummy_xt,
        prompt_mask=prompt_mask,
        phase=phase,
        lower=lower,
        upper=upper,
        L_eff=L_eff,
        mask_id=mask_id,
        confidence_threshold=confidence_threshold,
        order_policy=order_policy,
        dprm_estimator=dprm_estimator,
    )
    state.xt = state._initialize_xt(state.phase)
    return state


@torch.no_grad()
def teacher_forced_progressive_warm_start(
    model,
    input_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    mask_id: int,
    k: int,
    phase_init: str = "random",
    confidence_threshold: Optional[float] = None,
    order_policy: str = "confidence",
    dprm_estimator: Optional[OnlineDPRMEstimator] = None,
    global_step: int = 0,
) -> ProgressiveMaskingState:
    target_phase = phase_initialize(input_ids.shape[0], k, input_ids.device, mode=phase_init)
    state = initialize_progressive_state(
        input_ids=input_ids,
        prompt_mask=prompt_mask,
        mask_id=mask_id,
        k=k,
        phase_init="zero",
        confidence_threshold=confidence_threshold,
        order_policy=order_policy,
        dprm_estimator=dprm_estimator,
    )
    max_target = int(target_phase.max().item())
    for _ in range(max_target):
        active_mask = state.phase < target_phase
        if not active_mask.any():
            break
        log_probs = model(state.xt).logits.log_softmax(dim=-1)
        state.advance(log_probs=log_probs, active_mask=active_mask, global_step=global_step)
    return state
