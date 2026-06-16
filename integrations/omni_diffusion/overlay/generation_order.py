from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import torch

from dprm import append_trace_record, score_with_dprm_table, select_transfer_indices, trace_bucket_counts


def make_dprm_score_hook(
    *,
    dprm_table: str | Path,
    total_steps: int,
    prompt_length: int = 0,
    gen_length: int | None = None,
    guidance_scale: float | None = None,
    ready_count: int | None = None,
    warmup_steps: int | None = None,
    switch_steps: int | None = None,
    ablation: str = "normal",
) -> Callable[..., torch.Tensor]:
    """Return an Omni-style score hook.

    The hook expects the host to pass a one-dimensional `confidence` tensor for
    currently masked visual-token proposals and optional context containing
    `step` and `position_offset`.
    """

    def hook(*, confidence: torch.Tensor, **context) -> torch.Tensor:
        step = int(context.get("step", 0))
        position_offset = int(context.get("position_offset", 0))
        return score_with_dprm_table(
            base_score=confidence,
            bucket_confidence=confidence,
            table=dprm_table,
            step_index=step,
            total_steps=total_steps,
            prompt_length=prompt_length,
            gen_length=gen_length or confidence.numel(),
            position_offset=position_offset,
            guidance_scale=guidance_scale,
            ready_count=ready_count,
            warmup_steps=warmup_steps,
            switch_steps=switch_steps,
            ablation=ablation,
        ).score

    return hook


def choose_visual_tokens(
    *,
    confidence: torch.Tensor,
    number_transfer_tokens: int,
    order_policy: str,
    step: int,
    warmup_steps: int = 0,
    temperature: float | None = None,
    dprm_score_hook: Optional[Callable[..., torch.Tensor]] = None,
    hook_context: Optional[dict] = None,
) -> torch.Tensor:
    dprm_score = None
    if dprm_score_hook is not None:
        dprm_score = dprm_score_hook(confidence=confidence, **(hook_context or {}))
    return select_transfer_indices(
        confidence,
        number_transfer_tokens,
        order_policy=order_policy,
        temperature=temperature,
        step_index=step,
        warmup_steps=warmup_steps,
        dprm_score=dprm_score,
    )


def make_order_trace_observer(
    *,
    trace_path: str | Path,
    total_steps: int,
    num_phases: int = 8,
    confidence_bins: int = 16,
    aux_bins: int = 16,
    prompt_length: int = 0,
    gen_length: int | None = None,
) -> Callable[..., None]:
    """Return an observer compatible with Omni-Diffusion generation loops."""

    def observer(
        *,
        confidence: torch.Tensor,
        transfer_index: torch.Tensor,
        order_policy: str,
        step: int,
        block_idx: int = 0,
        sample_id: str | None = None,
        prompt_id: str | None = None,
        **context,
    ) -> None:
        selected = torch.zeros_like(confidence, dtype=torch.bool)
        selected[transfer_index] = True
        record = trace_bucket_counts(
            selected_mask=selected,
            bucket_confidence=confidence,
            step_index=step,
            total_steps=total_steps,
            prompt_length=prompt_length,
            gen_length=gen_length or confidence.numel(),
            position_offset=int(context.get("position_offset", 0)),
            num_phases=num_phases,
            confidence_bins=confidence_bins,
            aux_bins=aux_bins,
        )
        if record is None:
            return
        record.update(
            {
                "sample_id": sample_id,
                "prompt_id": prompt_id,
                "block_idx": int(block_idx),
                "order_policy": str(order_policy),
            }
        )
        append_trace_record(trace_path, record)

    return observer
