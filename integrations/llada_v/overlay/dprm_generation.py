import json
import os
import threading
from pathlib import Path
from typing import Optional

import torch


DPRM_REMASKING = {
    "dprm",
    "dprm_confidence",
    "dprm_confidence_warmup",
    "dprm_random",
    "dprm_random_warmup",
}

_TABLE_CACHE = {}
_TABLE_LOCK = threading.Lock()
_TRACE_LOCK = threading.Lock()


def is_dprm_remasking(remasking: str) -> bool:
    return str(remasking).lower() in DPRM_REMASKING


def _float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def resolve_dprm_table(path: Optional[str]) -> Optional[str]:
    if path:
        return str(path)
    for name in ("DPRM_LLADAV_DPRM_TABLE", "DPRM_LLADAV_TABLE", "DPRM_TABLE"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def _allow_dprm_fallback() -> bool:
    return os.environ.get("DPRM_LLADAV_ALLOW_DPRM_FALLBACK", "0") == "1"


def require_dprm_table(remasking: str, path: Optional[str]) -> Optional[str]:
    table_path = resolve_dprm_table(path)
    if is_dprm_remasking(remasking) and not table_path and not _allow_dprm_fallback():
        raise RuntimeError(
            f"{remasking} requires a real LLaDA-V DPRM table. "
            "Set DPRM_LLADAV_DPRM_TABLE or pass dprm_table. "
            "Use DPRM_LLADAV_ALLOW_DPRM_FALLBACK=1 only for explicit diagnostics."
        )
    return table_path


def _load_table(path: str, device: torch.device) -> dict:
    key = (str(Path(path).resolve()), str(device))
    with _TABLE_LOCK:
        cached = _TABLE_CACHE.get(key)
        if cached is not None:
            return cached
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        cfg = dict(payload.get("cfg", {}))
        counts = torch.tensor(payload["counts"], dtype=torch.float32, device=device)
        exp_reward_sums = torch.tensor(
            payload.get("exp_reward_sums", payload.get("exp_reward_sum")),
            dtype=torch.float32,
            device=device,
        )
        table = {
            "cfg": cfg,
            "counts": counts,
            "exp_reward_sums": exp_reward_sums,
            "path": str(path),
        }
        _TABLE_CACHE[key] = table
        return table


def _phase_id(step_index: int, total_steps: int, num_phases: int) -> int:
    total_steps = max(int(total_steps), 1)
    num_phases = max(int(num_phases), 1)
    return min((int(step_index) * num_phases) // total_steps, num_phases - 1)


def _aux_bins(
    seq_len: int,
    prompt_length: int,
    gen_length: int,
    aux_bins: int,
    device: torch.device,
    position_offset: int = 0,
) -> torch.Tensor:
    if aux_bins <= 1:
        return torch.zeros((seq_len,), dtype=torch.long, device=device)
    positions = torch.arange(seq_len, device=device, dtype=torch.long) + int(position_offset)
    rel = (positions - int(prompt_length)).clamp(min=0, max=max(int(gen_length) - 1, 0))
    bins = torch.div(rel * int(aux_bins), max(int(gen_length), 1), rounding_mode="floor")
    return bins.clamp_(0, int(aux_bins) - 1)


def _candidate_aux_bins(
    *,
    seq_len: int,
    prompt_length: int,
    gen_length: int,
    table_cfg: dict,
    table_aux_bins: int,
    device: torch.device,
    position_offset: int,
    predicted_token_ids: Optional[torch.Tensor],
    eot_token_ids: Optional[list[int]],
    context_bin: int = 0,
) -> torch.Tensor:
    mode = str(table_cfg.get("aux_mode", "position"))
    if mode == "position":
        return _aux_bins(
            seq_len, prompt_length, gen_length, table_aux_bins, device, position_offset
        )
    if mode not in {"eot_position", "format_eot_position"}:
        raise ValueError(f"unknown LLaDA-V DPRM aux_mode: {mode}")

    position_bins = int(table_cfg.get("position_bins", max(table_aux_bins // 2, 1)))
    format_bins = int(table_cfg.get("format_bins", 1)) if mode == "format_eot_position" else 1
    if table_aux_bins != 2 * position_bins * format_bins:
        raise ValueError(
            f"{mode} tables require aux_bins == 2 * position_bins * format_bins"
        )
    position = _aux_bins(
        seq_len, prompt_length, gen_length, position_bins, device, position_offset
    )
    if predicted_token_ids is None:
        return position
    is_eot = torch.zeros_like(predicted_token_ids, dtype=torch.bool)
    for token_id in eot_token_ids or []:
        is_eot |= predicted_token_ids == int(token_id)
    context = max(0, min(format_bins - 1, int(context_bin)))
    return position.unsqueeze(0).expand_as(predicted_token_ids) + position_bins * (
        is_eot.long() + 2 * context
    )


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
    predicted_token_ids: Optional[torch.Tensor] = None,
    eot_token_ids: Optional[list[int]] = None,
    dprm_context_bin: int = 0,
    diagnostics: Optional[dict] = None,
) -> torch.Tensor:
    table_path = require_dprm_table(remasking, dprm_table)
    if not table_path:
        return base_score

    table = _load_table(table_path, base_score.device)
    cfg = table["cfg"]
    counts = table["counts"]
    exp_reward_sums = table["exp_reward_sums"]

    num_phases = int(counts.shape[0])
    conf_bins_n = int(counts.shape[1])
    aux_bins_n = int(counts.shape[2])
    beta = float(cfg.get("reward_temperature", 1.0) or 1.0)
    guidance = _float_or_none(dprm_guidance_scale)
    if guidance is None:
        guidance = float(cfg.get("guidance_scale", 1.0) or 1.0)
    ready_count = _int_or_none(dprm_ready_count)
    if ready_count is None:
        ready_count = int(cfg.get("ready_count", 64) or 64)
    warmup_steps = _int_or_none(dprm_warmup_steps)
    if warmup_steps is None:
        warmup_steps = int(cfg.get("warmup_steps", 0) or 0)
    switch_steps = _int_or_none(dprm_switch_steps)
    if switch_steps is None:
        switch_steps = int(cfg.get("switch_steps", total_steps) or total_steps)

    candidate_mask = torch.isfinite(base_score)
    conf_prob = bucket_confidence.detach().float().clamp(1e-6, 1.0 - 1e-6)
    conf_bins = torch.floor(conf_prob * conf_bins_n).long().clamp_(0, conf_bins_n - 1)
    aux = _candidate_aux_bins(
        seq_len=base_score.shape[1],
        prompt_length=prompt_length,
        gen_length=gen_length,
        table_cfg=cfg,
        table_aux_bins=aux_bins_n,
        device=base_score.device,
        position_offset=position_offset,
        predicted_token_ids=predicted_token_ids,
        eot_token_ids=eot_token_ids,
        context_bin=dprm_context_bin,
    )
    if aux.dim() == 1:
        aux = aux.unsqueeze(0).expand_as(conf_bins)
    phase = _phase_id(step_index, total_steps, num_phases)

    bucket_counts = counts[phase, conf_bins, aux]
    bucket_exp = exp_reward_sums[phase, conf_bins, aux]
    safe_mean = torch.where(
        bucket_counts > 0,
        bucket_exp / bucket_counts.clamp_min(1.0),
        torch.ones_like(bucket_exp),
    )
    dprm_value = torch.log(safe_mean.clamp_min(1e-6)) / max(beta, 1e-6)
    local_gate = (bucket_counts / float(max(ready_count, 1))).clamp_(0.0, 1.0)
    if dprm_force_full:
        global_gate = 1.0
    elif int(step_index) <= warmup_steps:
        global_gate = 0.0
    elif switch_steps <= warmup_steps:
        global_gate = 1.0
    else:
        global_gate = (float(step_index) - float(warmup_steps)) / float(max(switch_steps - warmup_steps, 1))
        global_gate = max(0.0, min(1.0, global_gate))
    gate = local_gate * global_gate

    base_log = torch.log(base_score.detach().float().clamp_min(1e-6))
    score = base_log + gate * guidance * dprm_value
    score = torch.where(candidate_mask, score, torch.full_like(score, float("-inf")))
    if diagnostics is not None:
        diagnostics.update(
            {
                "phase": phase,
                "context_bin": int(dprm_context_bin),
                "aux": aux.detach(),
                "bucket_counts": bucket_counts.detach(),
                "gate": gate.detach(),
                "dprm_value": dprm_value.detach(),
                "base_log_score": base_log.detach(),
                "dprm_score": score.detach(),
            }
        )
    return score


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
    dprm_diagnostics: Optional[dict] = None,
) -> None:
    if not trace_path:
        return
    selected = selected_mask.detach().bool()
    if selected.dim() != 2 or selected.size(0) != 1 or not selected.any():
        return

    confidence = bucket_confidence.detach().float().clamp(1e-6, 1.0 - 1e-6)
    conf_bins = torch.floor(confidence * int(trace_confidence_bins)).long()
    conf_bins = conf_bins.clamp_(0, int(trace_confidence_bins) - 1)
    aux = _aux_bins(
        confidence.shape[1],
        prompt_length,
        gen_length,
        int(trace_aux_bins),
        confidence.device,
        position_offset=position_offset,
    ).unsqueeze(0)
    picked = selected & torch.isfinite(bucket_confidence)
    if not picked.any():
        return

    picked_conf = confidence[picked].detach().cpu()
    selected_positions = picked.nonzero(as_tuple=False)[:, 1].detach().cpu() + int(position_offset)
    eot_ids = {int(x) for x in (eot_token_ids or []) if x is not None and int(x) >= 0}
    selected_eot = None
    selected_eot_count = 0
    selected_eot_conf_mean = None
    candidate_eot_count = 0
    max_eot_confidence = None
    if selected_token_ids is not None and eot_ids:
        token_ids = selected_token_ids.detach()
        eot_mask = torch.zeros_like(token_ids, dtype=torch.bool)
        for token_id in eot_ids:
            eot_mask |= token_ids == int(token_id)
        candidate_eot = eot_mask & torch.isfinite(bucket_confidence)
        selected_eot = picked & eot_mask
        candidate_eot_count = int(candidate_eot.sum().item())
        selected_eot_count = int(selected_eot.sum().item())
        if candidate_eot_count > 0:
            max_eot_confidence = float(confidence[candidate_eot].max().item())
        if selected_eot_count > 0:
            selected_eot_conf_mean = float(confidence[selected_eot].mean().item())

    keys = torch.stack([conf_bins[picked], aux.expand_as(conf_bins)[picked]], dim=1).detach().cpu()
    unique, counts = torch.unique(keys, dim=0, return_counts=True)
    record = {
        "sample_id": trace_sample_id,
        "task": trace_task,
        "doc_id": trace_doc_id,
        "step": int(step_index),
        "block_idx": int(block_idx),
        "phase": int(_phase_id(step_index, total_steps, int(trace_num_phases))),
        "order_policy": str(remasking),
        "selected_count": int(picked.sum().item()),
        "selected_confidence_mean": float(picked_conf.mean().item()),
        "selected_confidence_min": float(picked_conf.min().item()),
        "selected_confidence_max": float(picked_conf.max().item()),
        "selected_entropy_proxy_mean": float((1.0 - picked_conf).mean().item()),
        "selected_position_min": int(selected_positions.min().item()),
        "selected_position_max": int(selected_positions.max().item()),
        "selected_positions": [int(x) for x in selected_positions.tolist()],
        "candidate_eot_count": candidate_eot_count,
        "selected_eot_count": selected_eot_count,
        "selected_eot_confidence_mean": selected_eot_conf_mean,
        "max_eot_confidence": max_eot_confidence,
        "bucket_counts": [
            {
                "confidence_bin": int(row[0].item()),
                "aux_bin": int(row[1].item()),
                "count": int(count.item()),
            }
            for row, count in zip(unique, counts)
        ],
    }
    if selected_token_ids is not None:
        picked_token_ids = selected_token_ids.detach()[picked].detach().cpu()
        record["selected_token_ids"] = [int(x) for x in picked_token_ids.tolist()]
    if dprm_diagnostics:
        diag_mask = torch.isfinite(bucket_confidence)
        k = int(picked.sum().item())
        confidence_top = torch.zeros_like(picked)
        for row in range(picked.shape[0]):
            valid_count = int(diag_mask[row].sum().item())
            if valid_count:
                _, indices = torch.topk(bucket_confidence[row], k=min(k, valid_count))
                confidence_top[row, indices] = True
        selected_aux = dprm_diagnostics["aux"][picked].detach().cpu()
        selected_counts = dprm_diagnostics["bucket_counts"][picked].detach().float().cpu()
        selected_gate = dprm_diagnostics["gate"][picked].detach().float().cpu()
        selected_value = dprm_diagnostics["dprm_value"][picked].detach().float().cpu()
        selected_base = dprm_diagnostics["base_log_score"][picked].detach().float().cpu()
        selected_score = dprm_diagnostics["dprm_score"][picked].detach().float().cpu()
        record.update(
            {
                "dprm_table_phase": int(dprm_diagnostics["phase"]),
                "dprm_context_bin": int(dprm_diagnostics["context_bin"]),
                "dprm_selected_aux_bins": [int(x) for x in selected_aux.tolist()],
                "dprm_selected_bucket_count_mean": float(selected_counts.mean().item()),
                "dprm_selected_bucket_count_min": float(selected_counts.min().item()),
                "dprm_selected_gate_mean": float(selected_gate.mean().item()),
                "dprm_selected_value_mean": float(selected_value.mean().item()),
                "dprm_selected_base_log_score_mean": float(selected_base.mean().item()),
                "dprm_selected_score_mean": float(selected_score.mean().item()),
                "order_changed_vs_confidence": bool(not torch.equal(picked, confidence_top)),
            }
        )
        candidate_rows = []
        valid_positions = diag_mask.nonzero(as_tuple=False)
        for row, column in valid_positions.tolist():
            candidate_rows.append(
                {
                    "position": int(column + position_offset),
                    "confidence": float(bucket_confidence[row, column].item()),
                    "aux_bin": int(dprm_diagnostics["aux"][row, column].item()),
                    "bucket_count": float(dprm_diagnostics["bucket_counts"][row, column].item()),
                    "gate": float(dprm_diagnostics["gate"][row, column].item()),
                    "dprm_value": float(dprm_diagnostics["dprm_value"][row, column].item()),
                    "base_log_score": float(dprm_diagnostics["base_log_score"][row, column].item()),
                    "dprm_score": float(dprm_diagnostics["dprm_score"][row, column].item()),
                }
            )
        record["dprm_candidates"] = candidate_rows
    path = Path(trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _TRACE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
