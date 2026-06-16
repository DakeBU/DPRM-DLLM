from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import torch


_TABLE_CACHE: dict[tuple[str, str], "DPRMTable"] = {}
_TABLE_LOCK = threading.Lock()
_TRACE_LOCK = threading.Lock()


@dataclass
class DPRMTable:
    """Offline bucket table used by decode-time DPRM adapters.

    The table stores the sufficient statistics for
    ``(1 / beta) log E[exp(beta * R) | phase, confidence_bin, aux_bin]``.
    """

    cfg: dict[str, Any]
    counts: torch.Tensor
    exp_reward_sums: torch.Tensor
    metadata: dict[str, Any] | None = None
    path: str | None = None


@dataclass
class DPRMScoreComponents:
    score: torch.Tensor
    base_log_score: torch.Tensor
    dprm_value: torch.Tensor
    gate: torch.Tensor
    confidence_bins: torch.Tensor
    aux_bins: torch.Tensor
    phase_id: int


def phase_from_step(step_index: int, total_steps: int, num_phases: int) -> int:
    total_steps = max(int(total_steps), 1)
    num_phases = max(int(num_phases), 1)
    return min((int(step_index) * num_phases) // total_steps, num_phases - 1)


def position_aux_bins(
    seq_len: int,
    *,
    prompt_length: int = 0,
    gen_length: int | None = None,
    aux_bins: int = 1,
    device: torch.device | None = None,
    position_offset: int = 0,
) -> torch.Tensor:
    if aux_bins <= 1:
        return torch.zeros((seq_len,), dtype=torch.long, device=device)
    gen_length = int(gen_length if gen_length is not None else max(seq_len - prompt_length, 1))
    positions = torch.arange(seq_len, device=device, dtype=torch.long) + int(position_offset)
    rel = (positions - int(prompt_length)).clamp(min=0, max=max(gen_length - 1, 0))
    bins = torch.div(rel * int(aux_bins), max(gen_length, 1), rounding_mode="floor")
    return bins.clamp_(0, int(aux_bins) - 1)


def load_dprm_table(path: str | Path, device: torch.device | str | None = None) -> DPRMTable:
    device = torch.device(device or "cpu")
    resolved = str(Path(path).expanduser().resolve())
    key = (resolved, str(device))
    with _TABLE_LOCK:
        cached = _TABLE_CACHE.get(key)
        if cached is not None:
            return cached
        payload = json.loads(Path(resolved).read_text(encoding="utf-8"))
        counts = torch.tensor(payload["counts"], dtype=torch.float32, device=device)
        exp_key = "exp_reward_sums" if "exp_reward_sums" in payload else "exp_reward_sum"
        exp_reward_sums = torch.tensor(payload[exp_key], dtype=torch.float32, device=device)
        table = DPRMTable(
            cfg=dict(payload.get("cfg", {})),
            counts=counts,
            exp_reward_sums=exp_reward_sums,
            metadata=dict(payload.get("metadata", {})),
            path=resolved,
        )
        _TABLE_CACHE[key] = table
        return table


def _global_gate(step_index: int, warmup_steps: int, switch_steps: int, force_full: bool) -> float:
    if force_full:
        return 1.0
    if int(step_index) <= int(warmup_steps):
        return 0.0
    if int(switch_steps) <= int(warmup_steps):
        return 1.0
    progress = (float(step_index) - float(warmup_steps)) / float(
        max(int(switch_steps) - int(warmup_steps), 1)
    )
    return max(0.0, min(1.0, progress))


def _bucket_values(
    table: DPRMTable,
    *,
    beta: float,
    ablation: str = "normal",
    shuffle_seed: int = 0,
) -> torch.Tensor:
    counts = table.counts
    safe_mean = torch.where(
        counts > 0,
        table.exp_reward_sums / counts.clamp_min(1.0),
        torch.ones_like(table.exp_reward_sums),
    )
    values = torch.log(safe_mean.clamp_min(1e-6)) / max(float(beta), 1e-6)
    ablation = str(ablation or "normal").lower()
    if ablation == "normal":
        return values
    if ablation == "gate_only":
        return torch.zeros_like(values)
    if ablation == "count_only":
        total_count = counts.sum().clamp_min(1.0)
        global_exp = table.exp_reward_sums.sum() / total_count
        global_value = torch.log(global_exp.clamp_min(1e-6)) / max(float(beta), 1e-6)
        return torch.where(counts > 0, torch.ones_like(values) * global_value, torch.zeros_like(values))
    if ablation == "shuffled_bucket":
        flat = values.flatten().clone()
        nonzero = (counts.flatten() > 0).nonzero(as_tuple=False).squeeze(1)
        if nonzero.numel() > 1:
            generator = torch.Generator(device=values.device)
            generator.manual_seed(int(shuffle_seed))
            perm = nonzero[torch.randperm(nonzero.numel(), generator=generator, device=values.device)]
            flat[nonzero] = flat[perm]
        return flat.view_as(values)
    raise ValueError(f"unknown DPRM ablation mode: {ablation}")


def score_with_dprm_table(
    *,
    base_score: torch.Tensor,
    bucket_confidence: torch.Tensor,
    table: DPRMTable | str | Path,
    step_index: int,
    total_steps: int,
    prompt_length: int = 0,
    gen_length: int | None = None,
    position_offset: int = 0,
    guidance_scale: float | None = None,
    ready_count: int | None = None,
    warmup_steps: int | None = None,
    switch_steps: int | None = None,
    force_full: bool = False,
    ablation: str = "normal",
    shuffle_seed: int = 0,
) -> DPRMScoreComponents:
    if not isinstance(table, DPRMTable):
        table = load_dprm_table(table, device=base_score.device)

    counts = table.counts
    cfg = table.cfg
    num_phases, num_bins, num_aux = [int(x) for x in counts.shape]
    beta = float(cfg.get("reward_temperature", 1.0) or 1.0)
    guidance = float(guidance_scale if guidance_scale is not None else cfg.get("guidance_scale", 1.0))
    ready = int(ready_count if ready_count is not None else cfg.get("ready_count", 64))
    warmup = int(warmup_steps if warmup_steps is not None else cfg.get("warmup_steps", 0))
    switch = int(switch_steps if switch_steps is not None else cfg.get("switch_steps", total_steps))

    valid = torch.isfinite(base_score)
    conf = bucket_confidence.detach().float().clamp(1e-6, 1.0 - 1e-6)
    conf_bins = torch.floor(conf * num_bins).long().clamp_(0, num_bins - 1)
    aux = position_aux_bins(
        base_score.shape[-1],
        prompt_length=prompt_length,
        gen_length=gen_length,
        aux_bins=num_aux,
        device=base_score.device,
        position_offset=position_offset,
    )
    while aux.dim() < conf_bins.dim():
        aux = aux.unsqueeze(0)
    aux = aux.expand_as(conf_bins).long().clamp_(0, num_aux - 1)
    phase = phase_from_step(step_index, total_steps, num_phases)

    values = _bucket_values(table, beta=beta, ablation=ablation, shuffle_seed=shuffle_seed)
    bucket_counts = counts[phase, conf_bins, aux]
    dprm_value = values[phase, conf_bins, aux]
    local_gate = (bucket_counts / float(max(ready, 1))).clamp_(0.0, 1.0)
    gate = local_gate * _global_gate(step_index, warmup, switch, force_full)
    base_log = torch.log(base_score.detach().float().clamp_min(1e-6))
    score = base_log + gate * guidance * dprm_value
    score = torch.where(valid, score, torch.full_like(score, float("-inf")))
    return DPRMScoreComponents(
        score=score,
        base_log_score=base_log,
        dprm_value=dprm_value,
        gate=gate,
        confidence_bins=conf_bins,
        aux_bins=aux,
        phase_id=phase,
    )


def select_topk_or_sample(
    scores: torch.Tensor,
    valid_mask: torch.Tensor,
    k: int,
    *,
    temperature: float | None = None,
) -> torch.Tensor:
    active = torch.where(valid_mask.bool())[0]
    if active.numel() == 0 or int(k) <= 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    k = min(int(k), int(active.numel()))
    local_scores = scores[active].float()
    if temperature is None or float(temperature) == 0.0:
        return active[torch.topk(local_scores, k=k, largest=True).indices]
    probs = torch.softmax(local_scores / float(temperature), dim=-1)
    return active[torch.multinomial(probs, num_samples=k, replacement=False)]


def select_transfer_indices(
    confidence: torch.Tensor,
    number_transfer_tokens: int,
    *,
    order_policy: str = "confidence",
    temperature: float | None = None,
    step_index: int = 0,
    warmup_steps: int = 0,
    dprm_score: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select a subset of currently eligible positions under a named policy.

    This mirrors the common MaskGIT/DLM hook used by Omni-Diffusion and LLaDA-V:
    token values are still sampled by the host model; DPRM changes only the
    position reveal order.
    """

    valid = torch.isfinite(confidence)
    policy = str(order_policy or "confidence").lower()
    if policy in {"random", "baseline_random", "random_ordered"}:
        active = torch.where(valid)[0]
        if active.numel() == 0:
            return active
        k = min(int(number_transfer_tokens), int(active.numel()))
        return active[torch.randperm(active.numel(), device=active.device)[:k]]
    if policy in {"confidence", "progressive", "progressive_confidence", "low_confidence"}:
        return select_topk_or_sample(confidence, valid, number_transfer_tokens, temperature=temperature)
    if policy == "entropy":
        return select_topk_or_sample(1.0 - confidence, valid, number_transfer_tokens, temperature=temperature)
    if policy in {"dprm", "dprm_confidence", "dprm_confidence_warmup"}:
        if int(step_index) < int(warmup_steps) or dprm_score is None:
            return select_topk_or_sample(confidence, valid, number_transfer_tokens, temperature=temperature)
        return select_topk_or_sample(dprm_score, valid, number_transfer_tokens, temperature=temperature)
    if policy in {"dprm_random", "dprm_random_warmup"}:
        if int(step_index) < int(warmup_steps) or dprm_score is None:
            return select_transfer_indices(
                confidence,
                number_transfer_tokens,
                order_policy="random",
                temperature=temperature,
            )
        return select_topk_or_sample(dprm_score, valid, number_transfer_tokens, temperature=temperature)
    raise ValueError(f"unknown order_policy: {order_policy}")


def trace_bucket_counts(
    *,
    selected_mask: torch.Tensor,
    bucket_confidence: torch.Tensor,
    step_index: int,
    total_steps: int,
    prompt_length: int = 0,
    gen_length: int | None = None,
    position_offset: int = 0,
    num_phases: int = 8,
    confidence_bins: int = 16,
    aux_bins: int = 1,
) -> dict[str, Any] | None:
    selected = selected_mask.detach().bool()
    if selected.numel() == 0 or not selected.any():
        return None
    confidence = bucket_confidence.detach().float().clamp(1e-6, 1.0 - 1e-6)
    conf_bins = torch.floor(confidence * int(confidence_bins)).long().clamp_(0, int(confidence_bins) - 1)
    aux = position_aux_bins(
        confidence.shape[-1],
        prompt_length=prompt_length,
        gen_length=gen_length,
        aux_bins=aux_bins,
        device=confidence.device,
        position_offset=position_offset,
    )
    while aux.dim() < conf_bins.dim():
        aux = aux.unsqueeze(0)
    aux = aux.expand_as(conf_bins)
    picked = selected & torch.isfinite(bucket_confidence)
    if not picked.any():
        return None
    keys = torch.stack([conf_bins[picked], aux[picked]], dim=1).detach().cpu()
    unique, counts = torch.unique(keys, dim=0, return_counts=True)
    picked_conf = confidence[picked].detach().cpu()
    return {
        "step": int(step_index),
        "phase": phase_from_step(step_index, total_steps, num_phases),
        "selected_count": int(picked.sum().item()),
        "selected_confidence_mean": float(picked_conf.mean().item()),
        "selected_entropy_proxy_mean": float((1.0 - picked_conf).mean().item()),
        "bucket_counts": [
            {
                "confidence_bin": int(row[0].item()),
                "aux_bin": int(row[1].item()),
                "count": int(count.item()),
            }
            for row, count in zip(unique, counts)
        ],
    }


def append_trace_record(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _TRACE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def build_bucket_table_from_trace_records(
    records: Iterable[dict[str, Any]],
    reward_by_key: dict[str, float],
    *,
    key_fields: tuple[str, ...] = ("sample_id",),
    num_phases: int = 8,
    confidence_bins: int = 16,
    aux_bins: int = 1,
    reward_temperature: float = 1.0,
    guidance_scale: float = 1.0,
    warmup_steps: int = 0,
    switch_steps: int = 1000,
    ready_count: int = 64,
) -> dict[str, Any]:
    shape = (int(num_phases), int(confidence_bins), int(aux_bins))
    counts = torch.zeros(shape, dtype=torch.float64)
    exp_reward_sums = torch.zeros(shape, dtype=torch.float64)
    reward_sums = torch.zeros(shape, dtype=torch.float64)
    reward_sq_sums = torch.zeros(shape, dtype=torch.float64)
    used_records = 0
    missing_reward = 0
    selected_total = 0

    for record in records:
        key = "|".join(str(record.get(field, "")) for field in key_fields)
        if key not in reward_by_key:
            missing_reward += 1
            continue
        reward = float(reward_by_key[key])
        exp_reward = math.exp(float(reward_temperature) * max(min(reward, 20.0), -20.0))
        phase = max(0, min(int(num_phases) - 1, int(record["phase"])))
        for bucket in record.get("bucket_counts", []):
            conf_bin = max(0, min(int(confidence_bins) - 1, int(bucket["confidence_bin"])))
            aux_bin = max(0, min(int(aux_bins) - 1, int(bucket.get("aux_bin", 0))))
            count = float(bucket["count"])
            counts[phase, conf_bin, aux_bin] += count
            exp_reward_sums[phase, conf_bin, aux_bin] += count * exp_reward
            reward_sums[phase, conf_bin, aux_bin] += count * reward
            reward_sq_sums[phase, conf_bin, aux_bin] += count * reward * reward
            selected_total += int(count)
        used_records += 1

    nonempty = int((counts > 0).sum().item())
    return {
        "cfg": {
            "num_phases": int(num_phases),
            "confidence_bins": int(confidence_bins),
            "aux_bins": int(aux_bins),
            "reward_temperature": float(reward_temperature),
            "guidance_scale": float(guidance_scale),
            "warmup_steps": int(warmup_steps),
            "switch_steps": int(switch_steps),
            "ready_count": int(ready_count),
            "sampled_soft_bon": False,
        },
        "counts": counts.tolist(),
        "exp_reward_sums": exp_reward_sums.tolist(),
        "reward_sums": reward_sums.tolist(),
        "reward_sq_sums": reward_sq_sums.tolist(),
        "metadata": {
            "used_records": used_records,
            "missing_reward_records": missing_reward,
            "selected_total": selected_total,
            "nonempty_buckets": nonempty,
            "total_buckets": int(counts.numel()),
            "bucket_coverage": nonempty / max(int(counts.numel()), 1),
        },
    }
