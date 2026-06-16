from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch

from dprm import append_trace_record, score_with_dprm_table, trace_bucket_counts


DPRM_REMASKING = {
    "dprm",
    "dprm_confidence",
    "dprm_confidence_warmup",
    "dprm_random",
    "dprm_random_warmup",
}


def is_dprm_remasking(remasking: str) -> bool:
    return str(remasking).lower() in DPRM_REMASKING


def resolve_dprm_table(path: Optional[str]) -> Optional[str]:
    if path:
        return str(path)
    for name in ("DPRM_LLADAV_TABLE", "DPRM_TABLE"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def apply_dprm_scores(
    *,
    remasking: str,
    base_score: torch.Tensor,
    bucket_confidence: torch.Tensor,
    dprm_table: Optional[str],
    step_index: int,
    total_steps: int,
    prompt_length: int,
    gen_length: int,
    position_offset: int = 0,
    dprm_guidance_scale: Optional[float] = None,
    dprm_ready_count: Optional[int] = None,
    dprm_switch_steps: Optional[int] = None,
    dprm_warmup_steps: Optional[int] = None,
    dprm_force_full: bool = False,
    dprm_ablation: str = "normal",
) -> torch.Tensor:
    table_path = resolve_dprm_table(dprm_table)
    if is_dprm_remasking(remasking) and not table_path:
        raise RuntimeError(
            f"{remasking} requires a DPRM table. Pass dprm_table or set DPRM_TABLE."
        )
    if not table_path:
        return base_score
    return score_with_dprm_table(
        base_score=base_score,
        bucket_confidence=bucket_confidence,
        table=table_path,
        step_index=step_index,
        total_steps=total_steps,
        prompt_length=prompt_length,
        gen_length=gen_length,
        position_offset=position_offset,
        guidance_scale=dprm_guidance_scale,
        ready_count=dprm_ready_count,
        warmup_steps=dprm_warmup_steps,
        switch_steps=dprm_switch_steps,
        force_full=dprm_force_full,
        ablation=dprm_ablation,
    ).score


def write_order_trace(
    *,
    trace_path: Optional[str],
    remasking: str,
    selected_mask: torch.Tensor,
    bucket_confidence: torch.Tensor,
    selected_token_ids: Optional[torch.Tensor] = None,
    eot_token_ids: Optional[list[int]] = None,
    step_index: int,
    total_steps: int,
    block_idx: int,
    prompt_length: int,
    gen_length: int,
    position_offset: int = 0,
    trace_sample_id: Optional[str] = None,
    trace_task: Optional[str] = None,
    trace_doc_id: Optional[str] = None,
    trace_num_phases: int = 8,
    trace_confidence_bins: int = 16,
    trace_aux_bins: int = 16,
) -> None:
    if not trace_path:
        return
    record = trace_bucket_counts(
        selected_mask=selected_mask,
        bucket_confidence=bucket_confidence,
        step_index=step_index,
        total_steps=total_steps,
        prompt_length=prompt_length,
        gen_length=gen_length,
        position_offset=position_offset,
        num_phases=trace_num_phases,
        confidence_bins=trace_confidence_bins,
        aux_bins=trace_aux_bins,
    )
    if record is None:
        return
    record.update(
        {
            "sample_id": trace_sample_id,
            "task": trace_task,
            "doc_id": trace_doc_id,
            "block_idx": int(block_idx),
            "order_policy": str(remasking),
        }
    )

    eot_ids = {int(x) for x in (eot_token_ids or []) if x is not None and int(x) >= 0}
    if selected_token_ids is not None and eot_ids:
        token_ids = selected_token_ids.detach()
        eot_mask = torch.zeros_like(token_ids, dtype=torch.bool)
        for token_id in eot_ids:
            eot_mask |= token_ids == int(token_id)
        selected = selected_mask.detach().bool()
        candidate_eot = eot_mask & torch.isfinite(bucket_confidence)
        selected_eot = selected & eot_mask
        record["candidate_eot_count"] = int(candidate_eot.sum().item())
        record["selected_eot_count"] = int(selected_eot.sum().item())
        if candidate_eot.any():
            record["max_eot_confidence"] = float(bucket_confidence[candidate_eot].max().item())
        if selected_eot.any():
            record["selected_eot_confidence_mean"] = float(
                bucket_confidence[selected_eot].mean().item()
            )

    append_trace_record(Path(trace_path), record)
