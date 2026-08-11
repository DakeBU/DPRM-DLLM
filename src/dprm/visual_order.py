from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import torch

from .controller import DPRMConfig, OnlineDPRMController
from .contracts import HostDPRMBatch


def confidence_to_probability(confidence: torch.Tensor, transform: str = "auto") -> torch.Tensor:
    confidence = confidence.detach().float()
    if transform == "neg_entropy":
        return confidence.exp().clamp(1e-6, 1.0 - 1e-6)
    if transform == "probability":
        return confidence.clamp(1e-6, 1.0 - 1e-6)
    finite = confidence[torch.isfinite(confidence)]
    if finite.numel() and float(finite.min().item()) < 0.0:
        return confidence.exp().clamp(1e-6, 1.0 - 1e-6)
    return confidence.clamp(1e-6, 1.0 - 1e-6)


def load_visual_dprm_controller(
    path: str | Path,
    *,
    device: torch.device,
    guidance_scale: Optional[float] = None,
    ready_count: Optional[int] = None,
    switch_steps: Optional[int] = None,
    controller_warmup_steps: Optional[int] = None,
) -> OnlineDPRMController:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    cfg_data = dict(payload.get("cfg", {}))
    valid = set(DPRMConfig.__dataclass_fields__)
    cfg = DPRMConfig(**{key: value for key, value in cfg_data.items() if key in valid})
    if guidance_scale is not None:
        cfg.guidance_scale = float(guidance_scale)
    if ready_count is not None:
        cfg.ready_count = int(ready_count)
    if switch_steps is not None:
        cfg.switch_steps = int(switch_steps)
    if controller_warmup_steps is not None:
        cfg.warmup_steps = int(controller_warmup_steps)

    controller = OnlineDPRMController(cfg, device=device)
    controller.load_state_dict(
        {
            "counts": torch.tensor(payload["counts"], dtype=torch.float32, device=device),
            "exp_reward_sums": torch.tensor(
                payload.get("exp_reward_sums", payload.get("exp_reward_sum")),
                dtype=torch.float32,
                device=device,
            ),
        }
    )
    controller.cfg = cfg
    return controller


def _as_batch(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(0) if x.dim() == 1 else x


def _square_grid(num_tokens: int) -> tuple[int, int]:
    width = max(1, int(round(math.sqrt(max(num_tokens, 1)))))
    height = max(1, int(math.ceil(max(num_tokens, 1) / width)))
    return height, width


def _token_indices_from_context(confidence: torch.Tensor, context: dict[str, Any]) -> torch.Tensor:
    conf = _as_batch(confidence)
    batch, length = conf.shape
    device = conf.device

    if "flat_idx" in context:
        positions = context["flat_idx"].to(device).long().view(-1)
        code_start = int(context.get("code_start") or 0)
        rel = (positions - code_start).clamp_min(0)
        newline_every = int(context.get("newline_every") or 0)
        if newline_every > 0:
            row_width = newline_every + 1
            row = torch.div(rel, row_width, rounding_mode="floor")
            col = (rel % row_width).clamp_max(newline_every - 1)
            token_idx = row * newline_every + col
        else:
            token_idx = rel
        token_idx = token_idx[:length]
        return token_idx.unsqueeze(0).expand(batch, -1)

    if "position_ids" in context:
        positions = context["position_ids"].to(device).long()
        return _as_batch(positions).expand(batch, -1)

    return torch.arange(length, device=device, dtype=torch.long).unsqueeze(0).expand(batch, -1)


def visual_aux_bins(
    confidence: torch.Tensor,
    context: dict[str, Any],
    aux_bins: int,
) -> torch.Tensor:
    conf = _as_batch(confidence)
    batch, length = conf.shape
    device = conf.device
    if aux_bins <= 1:
        return torch.zeros((batch, length), dtype=torch.long, device=device)

    token_idx = _token_indices_from_context(confidence, context)
    num_tokens = int(context.get("seq_len") or context.get("num_vq_tokens") or max(length, int(token_idx.max().item()) + 1))
    grid_h, grid_w = _square_grid(num_tokens)
    if "newline_every" in context and int(context.get("newline_every") or 0) > 0:
        grid_w = int(context["newline_every"])
        grid_h = max(1, int(math.ceil(num_tokens / grid_w)))

    row = torch.div(token_idx, grid_w, rounding_mode="floor").clamp(0, grid_h - 1)
    col = (token_idx % grid_w).clamp(0, grid_w - 1)
    side = int(round(math.sqrt(aux_bins)))
    if side * side == aux_bins:
        row_bin = torch.div(row * side, grid_h, rounding_mode="floor").clamp(0, side - 1)
        col_bin = torch.div(col * side, grid_w, rounding_mode="floor").clamp(0, side - 1)
        return (row_bin * side + col_bin).long()

    return torch.div(token_idx * aux_bins, max(num_tokens, 1), rounding_mode="floor").clamp(0, aux_bins - 1).long()


def make_visual_dprm_score_hook(
    controller: OnlineDPRMController,
    *,
    confidence_transform: str = "auto",
    force_full_dprm: bool = False,
    stats: Optional[dict[str, float]] = None,
):
    if stats is None:
        stats = {
            "calls": 0,
            "candidate_count": 0,
            "ready_candidate_count": 0,
            "gate_sum": 0.0,
            "dprm_abs_sum": 0.0,
        }

    def hook(confidence: torch.Tensor, *, candidate_mask: Optional[torch.Tensor] = None, **context) -> torch.Tensor:
        conf_prob = _as_batch(confidence_to_probability(confidence, confidence_transform))
        candidate = _as_batch(candidate_mask if candidate_mask is not None else torch.isfinite(confidence)).to(conf_prob.device).bool()
        aux_bins = visual_aux_bins(conf_prob, context, controller.cfg.aux_bins).to(conf_prob.device)
        phase = OnlineDPRMController.phase_from_progress(
            int(context.get("step", 0)),
            max(int(context.get("total_steps", 1)), 1),
            controller.cfg.num_phases,
            conf_prob.shape[0],
            conf_prob.device,
        )
        host = HostDPRMBatch(
            confidence=conf_prob,
            candidate_mask=candidate,
            phase_ids=phase,
            aux_bin_ids=aux_bins,
            global_step=int(context.get("step", 0)),
            force_full_dprm=force_full_dprm,
        )
        summary = controller.summarize(host)
        ready = candidate & (summary.gate > 0)
        stats["calls"] = int(stats.get("calls", 0)) + 1
        stats["candidate_count"] = int(stats.get("candidate_count", 0)) + int(candidate.sum().item())
        stats["ready_candidate_count"] = int(stats.get("ready_candidate_count", 0)) + int(ready.sum().item())
        stats["gate_sum"] = float(stats.get("gate_sum", 0.0)) + float(summary.gate[candidate].sum().item()) if candidate.any() else float(stats.get("gate_sum", 0.0))
        stats["dprm_abs_sum"] = float(stats.get("dprm_abs_sum", 0.0)) + float(summary.dprm_value[candidate].abs().sum().item()) if candidate.any() else float(stats.get("dprm_abs_sum", 0.0))
        return summary.score if confidence.dim() == 2 else summary.score[0]

    return hook


def make_visual_order_observer(
    trace_records: list[dict[str, Any]],
    *,
    num_phases: int,
    confidence_bins: int,
    aux_bins: int,
    confidence_transform: str = "auto",
):
    def observer(confidence: torch.Tensor, selected: torch.Tensor, *, candidate_mask: Optional[torch.Tensor] = None, **context) -> None:
        conf_prob = _as_batch(confidence_to_probability(confidence, confidence_transform))
        selected_batch = _as_batch(selected).to(conf_prob.device)
        if selected_batch.dtype != torch.bool:
            mask = torch.zeros_like(conf_prob, dtype=torch.bool)
            idx = selected_batch.long().view(-1)
            idx = idx[(idx >= 0) & (idx < conf_prob.shape[-1])]
            if idx.numel():
                mask[0, idx] = True
            selected_batch = mask
        else:
            selected_batch = selected_batch.bool()
        if candidate_mask is not None:
            selected_batch = selected_batch & _as_batch(candidate_mask).to(conf_prob.device).bool()
        if not selected_batch.any():
            return

        aux = visual_aux_bins(conf_prob, context, max(aux_bins, 1)).to(conf_prob.device)
        conf_bin = torch.floor(conf_prob * int(confidence_bins)).long().clamp(0, int(confidence_bins) - 1)
        phase = min(
            (int(context.get("step", 0)) * int(num_phases)) // max(int(context.get("total_steps", 1)), 1),
            int(num_phases) - 1,
        )
        for batch_idx in range(conf_prob.shape[0]):
            idx = torch.where(selected_batch[batch_idx])[0]
            if idx.numel() == 0:
                continue
            keys = torch.stack([conf_bin[batch_idx, idx], aux[batch_idx, idx]], dim=1).detach().cpu()
            unique, counts = torch.unique(keys, dim=0, return_counts=True)
            trace_records.append(
                {
                    "step": int(context.get("step", 0)),
                    "phase": int(phase),
                    "order_policy": str(context.get("order_policy", "")),
                    "batch_index": int(batch_idx),
                    "selected_count": int(idx.numel()),
                    "bucket_counts": [
                        {
                            "confidence_bin": int(row[0].item()),
                            "aux_bin": int(row[1].item()),
                            "count": int(count.item()),
                        }
                        for row, count in zip(unique, counts)
                    ],
                }
            )

    return observer


def summarize_hook_stats(stats: dict[str, float]) -> dict[str, float]:
    denom = max(int(stats.get("candidate_count", 0)), 1)
    return {
        **stats,
        "ready_fraction": float(stats.get("ready_candidate_count", 0)) / denom,
        "mean_gate": float(stats.get("gate_sum", 0.0)) / denom,
        "mean_abs_dprm_value": float(stats.get("dprm_abs_sum", 0.0)) / denom,
    }
