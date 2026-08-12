#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, GenerationConfig

from omni_diffusion.data.processor.image_processor import ImageProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dprm import DPRMConfig, HostDPRMBatch, OnlineDPRMController
from dprm.omni_action_value import action_feature_dict


def _cfg_from_payload(payload: dict, args: argparse.Namespace) -> DPRMConfig:
    cfg_data = dict(payload.get("cfg", {}))
    valid = set(DPRMConfig.__dataclass_fields__)
    cfg = DPRMConfig(**{k: v for k, v in cfg_data.items() if k in valid})
    if args.dprm_guidance_scale is not None:
        cfg.guidance_scale = float(args.dprm_guidance_scale)
    if args.dprm_ready_count is not None:
        cfg.ready_count = int(args.dprm_ready_count)
    if args.dprm_switch_steps is not None:
        cfg.switch_steps = int(args.dprm_switch_steps)
    if args.dprm_controller_warmup_steps is not None:
        cfg.warmup_steps = int(args.dprm_controller_warmup_steps)
    return cfg


def load_dprm_controller(path: str, args: argparse.Namespace, device: torch.device) -> OnlineDPRMController:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    cfg = _cfg_from_payload(payload, args)
    controller = OnlineDPRMController(cfg, device=device)
    counts = torch.tensor(payload["counts"], dtype=torch.float32, device=device)
    exp_reward_sums = torch.tensor(
        payload.get("exp_reward_sums", payload.get("exp_reward_sum")),
        dtype=torch.float32,
        device=device,
    )
    controller.load_state_dict({"counts": counts, "exp_reward_sums": exp_reward_sums})
    controller.cfg = cfg
    return controller


def confidence_to_probability(confidence: torch.Tensor, transform: str) -> torch.Tensor:
    confidence = confidence.detach().float()
    if transform == "neg_entropy":
        return confidence.exp().clamp(1e-6, 1.0 - 1e-6)
    if transform == "probability":
        return confidence.clamp(1e-6, 1.0 - 1e-6)
    if torch.isfinite(confidence).any() and float(confidence[torch.isfinite(confidence)].min().item()) < 0.0:
        return confidence.exp().clamp(1e-6, 1.0 - 1e-6)
    return confidence.clamp(1e-6, 1.0 - 1e-6)


def image_aux_bins(mask_index: torch.Tensor, block_mask: torch.Tensor, aux_bins: int) -> torch.Tensor:
    masked_positions = torch.where(mask_index[0])[0]
    block_positions = torch.where(block_mask)[0]
    if aux_bins <= 1 or block_positions.numel() == 0:
        return torch.zeros_like(masked_positions, dtype=torch.long)

    # T2I's 260-token block is <begin_image>, 255 visual codes, and four
    # trailing special tokens. The decoder pads the final visual code to 256.
    # Shift by one so spatial bins describe visual positions rather than the
    # begin-image token.
    rel = (masked_positions - int(block_positions[0].item()) - 1).clamp(min=0, max=255)
    if aux_bins == 16:
        row = torch.div(rel, 16, rounding_mode="floor").clamp(0, 15)
        col = (rel % 16).clamp(0, 15)
        return (torch.div(row, 4, rounding_mode="floor") * 4 + torch.div(col, 4, rounding_mode="floor")).long()
    if aux_bins == 4:
        row = torch.div(rel, 16, rounding_mode="floor").clamp(0, 15)
        col = (rel % 16).clamp(0, 15)
        return (torch.div(row, 8, rounding_mode="floor") * 2 + torch.div(col, 8, rounding_mode="floor")).long()
    return torch.div(rel * aux_bins, 256, rounding_mode="floor").clamp(0, aux_bins - 1).long()


def image_candidate_mask(mask_index: torch.Tensor, block_mask: torch.Tensor) -> torch.Tensor:
    masked_positions = torch.where(mask_index[0])[0]
    block_positions = torch.where(block_mask)[0]
    if block_positions.numel() == 0:
        return torch.zeros_like(masked_positions, dtype=torch.bool)
    rel = masked_positions - int(block_positions[0].item())
    return (rel >= 1) & (rel <= 255)


def relative_rank_features(
    scores: torch.Tensor,
    candidates: torch.Tensor,
    bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ascending within-state rank quantiles and bins for candidates."""
    quantiles = torch.zeros_like(scores, dtype=torch.float32)
    rank_bins = torch.zeros_like(scores, dtype=torch.long)
    if candidates.numel() == 0:
        return quantiles, rank_bins
    order = candidates[torch.argsort(scores[candidates])]
    denominator = max(int(order.numel()) - 1, 1)
    ordered_quantiles = torch.arange(
        order.numel(), dtype=torch.float32, device=scores.device
    ) / float(denominator)
    quantiles[order] = ordered_quantiles
    rank_bins[order] = torch.floor(ordered_quantiles * max(int(bins), 1)).long().clamp(
        0, max(int(bins), 1) - 1
    )
    return quantiles, rank_bins


def visual_context_features(
    visual_index: int,
    masked_visual_indices: set[int],
) -> dict[str, float | int]:
    """Summarize spatial context around one 16x16 visual-token action."""
    row, column = divmod(int(visual_index), 16)
    valid_indices = set(range(255))
    revealed = valid_indices - masked_visual_indices
    local_total = 0
    local_revealed = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, column + dc
            neighbor = nr * 16 + nc
            if 0 <= nr < 16 and 0 <= nc < 16 and neighbor in valid_indices:
                local_total += 1
                local_revealed += int(neighbor in revealed)
    nearest_revealed = 32
    if revealed:
        nearest_revealed = min(
            abs(row - (idx // 16)) + abs(column - (idx % 16)) for idx in revealed
        )
    return {
        "row": row,
        "column": column,
        "row_normalized": row / 15.0,
        "column_normalized": column / 15.0,
        "center_distance": float(((row - 7.5) ** 2 + (column - 7.5) ** 2) ** 0.5),
        "local_neighbor_count": local_total,
        "local_revealed_neighbor_count": local_revealed,
        "local_revealed_fraction": local_revealed / max(local_total, 1),
        "nearest_revealed_manhattan": int(nearest_revealed),
    }


def make_dprm_score_hook(
    controller: OnlineDPRMController,
    args: argparse.Namespace,
    stats: dict,
):
    def hook(confidence: torch.Tensor, **context) -> torch.Tensor:
        conf_prob = confidence_to_probability(confidence, args.dprm_confidence_transform)
        candidate_mask = torch.isfinite(confidence).unsqueeze(0)
        visual_candidate_mask = image_candidate_mask(
            context["mask_index"], context["block_mask"]
        ).to(confidence.device).unsqueeze(0)
        aux_bins = image_aux_bins(
            context["mask_index"],
            context["block_mask"],
            controller.cfg.aux_bins,
        ).to(confidence.device)
        phase = OnlineDPRMController.phase_from_progress(
            int(context["step"]),
            max(int(args.steps), 1),
            controller.cfg.num_phases,
            1,
            confidence.device,
        )
        host = HostDPRMBatch(
            confidence=conf_prob.unsqueeze(0),
            candidate_mask=candidate_mask,
            phase_ids=phase,
            aux_bin_ids=aux_bins.unsqueeze(0),
            global_step=int(context["step"]),
            force_full_dprm=bool(args.dprm_force_full),
        )
        summary = controller.summarize(host)
        finite = candidate_mask & visual_candidate_mask
        ready = finite & (summary.gate > 0)
        stats["calls"] += 1
        stats["candidate_count"] += int(finite.sum().item())
        stats["ready_candidate_count"] += int(ready.sum().item())
        stats["gate_sum"] += float(summary.gate[finite].sum().item()) if finite.any() else 0.0
        stats["dprm_abs_sum"] += float(summary.dprm_value[finite].abs().sum().item()) if finite.any() else 0.0
        score = torch.where(visual_candidate_mask, summary.score, summary.base_score)
        return score[0]

    return hook


def make_action_value_score_hook(
    model,
    args: argparse.Namespace,
    stats: dict,
    prompt: str,
):
    """Add a learned shared-state terminal-advantage estimate to confidence."""
    active_steps = {int(step) for step in args.dprm_action_steps}

    def hook(confidence: torch.Tensor, **context) -> torch.Tensor:
        step = int(context["step"])
        if step not in active_steps:
            return confidence
        conf_prob = confidence_to_probability(confidence, args.dprm_confidence_transform)
        visual = image_candidate_mask(context["mask_index"], context["block_mask"]).to(
            confidence.device
        )
        candidates = torch.where(torch.isfinite(confidence) & visual)[0]
        if candidates.numel() < 2:
            return confidence

        rank_quantiles, rank_bins = relative_rank_features(
            conf_prob, candidates, int(args.dprm_action_rank_bins)
        )
        aux_bins = image_aux_bins(
            context["mask_index"], context["block_mask"], int(args.dprm_action_aux_bins)
        ).to(confidence.device)
        default_choice = candidates[torch.argmax(conf_prob[candidates])]
        default_confidence = conf_prob[default_choice]
        masked_positions = torch.where(context["mask_index"][0])[0]
        block_positions = torch.where(context["block_mask"])[0]
        block_start = int(block_positions[0].item())
        candidate_visual_indices = masked_positions[candidates] - block_start - 1
        masked_visual_indices = {
            int(idx) for idx in candidate_visual_indices.detach().cpu().tolist() if 0 <= int(idx) < 255
        }

        feature_rows = []
        for candidate, visual_index_tensor in zip(candidates, candidate_visual_indices):
            visual_index = int(visual_index_tensor.item())
            spatial = visual_context_features(visual_index, masked_visual_indices)
            feature_rows.append(
                action_feature_dict(
                    step=step,
                    rank_quantile=float(rank_quantiles[candidate].item()),
                    rank_bin=int(rank_bins[candidate].item()),
                    aux_bin=int(aux_bins[candidate].item()),
                    row_normalized=float(spatial["row_normalized"]),
                    column_normalized=float(spatial["column_normalized"]),
                    center_distance=float(spatial["center_distance"]),
                    local_revealed_fraction=float(spatial["local_revealed_fraction"]),
                    nearest_revealed_manhattan=float(spatial["nearest_revealed_manhattan"]),
                    candidate_count=int(candidates.numel()),
                    confidence=float(conf_prob[candidate].item()),
                    confidence_gap_from_default=float(
                        (conf_prob[candidate] - default_confidence).item()
                    ),
                    provisional_token_id=int(context["x0"][candidate].item()),
                    prompt=prompt,
                    total_steps=int(args.steps),
                    num_phases=int(args.trace_num_phases),
                )
            )
        predictions = np.asarray(model.predict(feature_rows), dtype=np.float32)
        predicted = torch.from_numpy(predictions).to(confidence.device)
        default_offset = int(torch.where(candidates == default_choice)[0][0].item())
        predicted[default_offset] = 0.0
        adjusted = confidence.clone()
        adjusted[candidates] = (
            confidence[candidates]
            + float(args.dprm_action_guidance_scale) * predicted
        )
        stats["calls"] += 1
        stats["candidate_count"] += int(candidates.numel())
        stats["predicted_advantage_sum"] += float(predicted.sum().item())
        stats["predicted_advantage_abs_sum"] += float(predicted.abs().sum().item())
        stats["selected_nonconfidence_count"] += int(
            candidates[torch.argmax(adjusted[candidates])].item() != int(default_choice.item())
        )
        return adjusted

    return hook


def make_order_observer(
    args: argparse.Namespace,
    trace_records: list[dict],
    image_token_offset: int,
):
    provisional_phases_seen: set[int] = set()

    def observer(confidence: torch.Tensor, transfer_index: torch.Tensor, **context) -> None:
        if transfer_index.numel() == 0:
            return
        conf_prob = confidence_to_probability(confidence, args.dprm_confidence_transform)
        aux_bins = image_aux_bins(
            context["mask_index"],
            context["block_mask"],
            max(int(args.trace_aux_bins), 1),
        ).to(confidence.device)
        conf_bins = torch.floor(conf_prob * int(args.trace_confidence_bins)).long()
        conf_bins = conf_bins.clamp(0, int(args.trace_confidence_bins) - 1)
        selected = transfer_index.detach().long()
        selected = selected[(selected >= 0) & (selected < confidence.numel())]
        if selected.numel() == 0:
            return
        masked_positions = torch.where(context["mask_index"][0])[0].to(selected.device)
        block_positions = torch.where(context["block_mask"])[0].to(selected.device)
        sequence_positions = masked_positions[selected]
        block_start = int(block_positions[0].item()) if block_positions.numel() else 0
        visual_selected = (sequence_positions - block_start >= 1) & (sequence_positions - block_start <= 255)
        selected = selected[visual_selected]
        sequence_positions = sequence_positions[visual_selected]
        if selected.numel() == 0:
            return
        visual_indices = (sequence_positions - block_start - 1).clamp(min=0, max=255)
        selected_confidence = conf_prob[selected]
        selected_conf_bins = conf_bins[selected]
        selected_aux_bins = aux_bins[selected]
        provisional_token_ids = context["x0"][selected].detach().long()
        keys = torch.stack([conf_bins[selected], aux_bins[selected]], dim=1).detach().cpu()
        unique, counts = torch.unique(keys, dim=0, return_counts=True)
        phase = min(
            (int(context["step"]) * int(args.trace_num_phases)) // max(int(args.steps), 1),
            int(args.trace_num_phases) - 1,
        )
        record = {
                "step": int(context["step"]),
                "block_idx": int(context["block_idx"]),
                "phase": int(phase),
                "order_policy": str(context["order_policy"]),
                "selected_count": int(selected.numel()),
                "selected_candidate_indices": selected.detach().cpu().tolist(),
                "selected_sequence_positions": sequence_positions.detach().cpu().tolist(),
                "selected_visual_indices": visual_indices.detach().cpu().tolist(),
                "selected_rows": torch.div(visual_indices, 16, rounding_mode="floor").detach().cpu().tolist(),
                "selected_columns": (visual_indices % 16).detach().cpu().tolist(),
                "selected_confidence": selected_confidence.detach().cpu().tolist(),
                "selected_confidence_bins": selected_conf_bins.detach().cpu().tolist(),
                "selected_aux_bins": selected_aux_bins.detach().cpu().tolist(),
                "selected_provisional_token_ids": provisional_token_ids.detach().cpu().tolist(),
                "bucket_counts": [
                    {
                        "confidence_bin": int(row[0].item()),
                        "aux_bin": int(row[1].item()),
                        "count": int(count.item()),
                    }
                    for row, count in zip(unique, counts)
                ],
            }
        if args.trace_provisional_phases and phase not in provisional_phases_seen:
            provisional = context["x"][0].detach().clone()
            provisional[masked_positions] = context["x0"].detach().long()
            image_slots = block_positions[1:256]
            slot_tokens = provisional[image_slots]
            valid_image = (slot_tokens >= image_token_offset) & (
                slot_tokens < image_token_offset + 8192
            )
            image_logits = context["logits"][0, image_slots, image_token_offset : image_token_offset + 8192]
            projected = image_logits.argmax(dim=-1) + image_token_offset
            visual_tokens = torch.where(valid_image, slot_tokens, projected)
            if visual_tokens.numel() == 255:
                visual_tokens = torch.cat([visual_tokens, visual_tokens[-1:]], dim=0)
            if visual_tokens.numel() == 256:
                record["provisional_visual_token_ids"] = (
                    visual_tokens - image_token_offset
                ).detach().cpu().tolist()
                provisional_phases_seen.add(phase)
        trace_records.append(record)

    return observer


def make_counterfactual_order_override(args: argparse.Namespace, stats: dict):
    """Replace one confidence action at a shared state, then resume confidence decoding."""

    def override(
        confidence: torch.Tensor,
        transfer_index: torch.Tensor,
        number_transfer_tokens: int,
        **context,
    ) -> torch.Tensor:
        if int(context["step"]) != int(args.force_order_step):
            return transfer_index

        conf_prob = confidence_to_probability(confidence, args.dprm_confidence_transform)
        visual = image_candidate_mask(context["mask_index"], context["block_mask"]).to(
            confidence.device
        )
        valid = torch.isfinite(confidence) & visual
        candidates = torch.where(valid)[0]
        if candidates.numel() == 0:
            stats["failure"] = "no_visual_candidates"
            return transfer_index

        rank_quantiles, rank_bins = relative_rank_features(
            conf_prob, candidates, int(args.force_rank_bins)
        )
        ordered = candidates[torch.argsort(conf_prob[candidates])]
        quantile = float(np.clip(args.force_confidence_quantile, 0.0, 1.0))
        rank = int(round(quantile * max(int(ordered.numel()) - 1, 0)))
        chosen = ordered[rank]
        default = transfer_index.detach().long().view(-1)
        if bool((default == chosen).any()) and ordered.numel() > 1:
            neighbor = rank - 1 if rank > 0 else rank + 1
            chosen = ordered[neighbor]

        target_count = int(number_transfer_tokens)
        selected = [int(chosen.item())]
        if len(selected) < target_count:
            for idx in default.detach().cpu().tolist():
                idx = int(idx)
                if idx not in selected:
                    selected.append(idx)
                if len(selected) >= target_count:
                    break
        if len(selected) < target_count:
            for idx in reversed(ordered.detach().cpu().tolist()):
                idx = int(idx)
                if idx not in selected:
                    selected.append(idx)
                if len(selected) >= target_count:
                    break

        masked_positions = torch.where(context["mask_index"][0])[0]
        block_positions = torch.where(context["block_mask"])[0]
        sequence_position = int(masked_positions[chosen].item())
        block_start = int(block_positions[0].item())
        visual_index = sequence_position - block_start - 1
        candidate_sequence_positions = masked_positions[candidates]
        candidate_visual_indices = candidate_sequence_positions - block_start - 1
        masked_visual_indices = {
            int(idx) for idx in candidate_visual_indices.detach().cpu().tolist() if 0 <= int(idx) < 255
        }
        default_visual = default[(default >= 0) & (default < confidence.numel())]
        default_visual = default_visual[visual[default_visual]]
        default_choice = int(default_visual[0].item()) if default_visual.numel() else None
        default_sequence_position = (
            int(masked_positions[default_choice].item())
            if default_choice is not None
            else None
        )
        default_visual_index = (
            default_sequence_position - block_start - 1
            if default_sequence_position is not None
            else None
        )
        aux = image_aux_bins(
            context["mask_index"], context["block_mask"], int(args.force_aux_bins)
        ).to(confidence.device)
        conf_bin = int(
            torch.floor(conf_prob[chosen] * int(args.force_confidence_bins))
            .long()
            .clamp(0, int(args.force_confidence_bins) - 1)
            .item()
        )
        stats.update(
            {
                "applied": True,
                "step": int(context["step"]),
                "requested_quantile": quantile,
                "candidate_count": int(candidates.numel()),
                "candidate_index": int(chosen.item()),
                "sequence_position": sequence_position,
                "visual_index": visual_index,
                "confidence": float(conf_prob[chosen].item()),
                "raw_order_score": float(confidence[chosen].item()),
                "confidence_bin": conf_bin,
                "rank_quantile": float(rank_quantiles[chosen].item()),
                "rank_bin": int(rank_bins[chosen].item()),
                "aux_bin": int(aux[chosen].item()),
                "phase": min(
                    (int(context["step"]) * int(args.trace_num_phases))
                    // max(int(args.steps), 1),
                    int(args.trace_num_phases) - 1,
                ),
                "provisional_token_id": int(context["x0"][chosen].item()),
                "default_candidate_indices": default.detach().cpu().tolist(),
                "default_candidate_index": default_choice,
                "default_sequence_position": default_sequence_position,
                "default_visual_index": default_visual_index,
                "default_confidence": (
                    float(conf_prob[default_choice].item()) if default_choice is not None else None
                ),
                "default_raw_order_score": (
                    float(confidence[default_choice].item()) if default_choice is not None else None
                ),
                "default_rank_quantile": (
                    float(rank_quantiles[default_choice].item()) if default_choice is not None else None
                ),
                "confidence_gap_from_default": (
                    float((conf_prob[chosen] - conf_prob[default_choice]).item())
                    if default_choice is not None
                    else None
                ),
                "overridden_candidate_indices": selected,
                **visual_context_features(visual_index, masked_visual_indices),
            }
        )
        return torch.tensor(selected, dtype=torch.long, device=confidence.device)

    return override


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-tokenizer-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", default="A small red bird sitting on a branch, watercolor style.")
    parser.add_argument("--steps", type=int, default=260)
    parser.add_argument("--max-tokens", type=int, default=260)
    parser.add_argument("--alg", default="entropy-penalty")
    parser.add_argument("--order-policy", default="progressive_confidence")
    parser.add_argument("--dprm-warmup-steps", type=int, default=None)
    parser.add_argument("--dprm-table", default=None)
    parser.add_argument("--dprm-action-value-model", default=None)
    parser.add_argument(
        "--dprm-action-steps",
        type=int,
        nargs="+",
        default=[96, 128, 160, 192, 224],
    )
    parser.add_argument("--dprm-action-guidance-scale", type=float, default=20.0)
    parser.add_argument("--dprm-action-rank-bins", type=int, default=8)
    parser.add_argument("--dprm-action-aux-bins", type=int, default=16)
    parser.add_argument("--dprm-guidance-scale", type=float, default=None)
    parser.add_argument("--dprm-ready-count", type=int, default=None)
    parser.add_argument("--dprm-switch-steps", type=int, default=None)
    parser.add_argument("--dprm-controller-warmup-steps", type=int, default=None)
    parser.add_argument("--dprm-force-full", action="store_true")
    parser.add_argument(
        "--dprm-confidence-transform",
        choices=["auto", "neg_entropy", "probability"],
        default="auto",
    )
    parser.add_argument(
        "--trace-order-stats",
        nargs="?",
        const="auto",
        default=None,
        help="Write selected phase/confidence/aux bucket counts for reward-table construction.",
    )
    parser.add_argument("--trace-num-phases", type=int, default=8)
    parser.add_argument("--trace-confidence-bins", type=int, default=16)
    parser.add_argument("--trace-aux-bins", type=int, default=16)
    parser.add_argument(
        "--trace-provisional-phases",
        action="store_true",
        help="Store a full provisional visual-token canvas at the first decision in each trace phase.",
    )
    parser.add_argument("--save-history-frames", action="store_true")
    parser.add_argument("--history-frame-stride", type=int, default=32)
    parser.add_argument("--history-frame-limit", type=int, default=12)
    parser.add_argument(
        "--force-order-step",
        type=int,
        default=None,
        help="At this decode step, replace the confidence action by a shared-state counterfactual action.",
    )
    parser.add_argument(
        "--force-confidence-quantile",
        type=float,
        default=0.5,
        help="Ascending confidence quantile of the visual candidate forced at --force-order-step.",
    )
    parser.add_argument("--force-confidence-bins", type=int, default=8)
    parser.add_argument("--force-rank-bins", type=int, default=8)
    parser.add_argument("--force-aux-bins", type=int, default=16)
    parser.add_argument(
        "--require-forced-action",
        action="store_true",
        help="Fail unless the requested counterfactual action was applied.",
    )
    parser.add_argument(
        "--allow-dprm-diagnostic-fallback",
        action="store_true",
        help="Permit DPRM-labeled policies to fall back to confidence. Do not use for formal runs.",
    )
    parser.add_argument("--seed", type=int, default=20260522)
    args = parser.parse_args()

    if str(args.order_policy).startswith("dprm"):
        if args.dprm_table or args.dprm_action_value_model:
            pass
        elif not args.allow_dprm_diagnostic_fallback:
            raise SystemExit(
                f"{args.order_policy} needs a real DPRM score hook for formal Omni eval; "
                "pass --dprm-table, or use --allow-dprm-diagnostic-fallback only for explicit diagnostics."
            )
        else:
            os.environ["DPRM_ALLOW_DPRM_CONFIDENCE_FALLBACK"] = "1"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(args.model_path, trust_remote_code=True)

    image_processor = ImageProcessor(
        args.image_tokenizer_path,
        "dynamic",
        image_size=512,
        normalize_type="imagenet",
        min_patch_grid=1,
        max_patch_grid=12,
    )
    image_processor.image_tokenizer.rank = 0
    image_processor.load_model()

    dprm_hook = None
    dprm_hook_stats = {
        "calls": 0,
        "candidate_count": 0,
        "ready_candidate_count": 0,
        "gate_sum": 0.0,
        "dprm_abs_sum": 0.0,
    }
    action_value_stats = {
        "calls": 0,
        "candidate_count": 0,
        "predicted_advantage_sum": 0.0,
        "predicted_advantage_abs_sum": 0.0,
        "selected_nonconfidence_count": 0,
    }
    if args.dprm_table and args.dprm_action_value_model:
        raise SystemExit("choose either --dprm-table or --dprm-action-value-model")
    if args.dprm_table:
        controller = load_dprm_controller(args.dprm_table, args, torch.device("cuda:0"))
        dprm_hook = make_dprm_score_hook(controller, args, dprm_hook_stats)
    elif args.dprm_action_value_model:
        action_value_model = joblib.load(args.dprm_action_value_model)
        dprm_hook = make_action_value_score_hook(
            action_value_model, args, action_value_stats, args.prompt
        )

    image_offset = tokenizer.convert_tokens_to_ids("<|image_0|>")
    trace_records: list[dict] = []
    observer = (
        make_order_observer(args, trace_records, image_offset)
        if args.trace_order_stats
        else None
    )
    counterfactual_stats = {"applied": False}
    order_override = (
        make_counterfactual_order_override(args, counterfactual_stats)
        if args.force_order_step is not None
        else None
    )

    messages = [
        {
            "role": "user",
            "content": "Generate an image based on the provided text description.\n" + args.prompt,
        }
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    input_ids = torch.tensor([input_ids], dtype=torch.long, device="cuda:0")

    gen_started = time.time()
    with torch.inference_mode():
        outputs, _histories = model.generate(
            input_ids,
            temperature=0.0,
            top_p=0.9,
            steps=args.steps,
            max_new_tokens=args.max_tokens,
            alg=args.alg,
            cfg=0.0,
            tokenizer=tokenizer,
            max_position_penalty=2.0,
            repeat_penalty=1.2,
            output_text_only=False,
            task="T2I",
            order_policy=args.order_policy,
            dprm_warmup_steps=args.dprm_warmup_steps,
            generation_order_score_hook_func=dprm_hook,
            generation_order_observer_func=observer,
            generation_order_override_func=order_override,
        )
    generation_seconds = time.time() - gen_started

    generated = outputs[0][input_ids.shape[1] :]
    image_tokens = []
    text_tokens = []
    for token_id in generated:
        if token_id >= image_offset:
            image_tokens.append(token_id - image_offset)
        else:
            text_tokens.append(token_id)

    image_path = None
    if image_tokens:
        if len(image_tokens) < 256:
            image_tokens += [image_tokens[-1]] * (256 - len(image_tokens))
        gen_token_ids = torch.stack(image_tokens, dim=0).unsqueeze(0)
        gen_token_ids = torch.clamp(gen_token_ids, max=8192 - 1, min=0)
        image = image_processor.image_tokenizer.image_tokenizer.decode_code(gen_token_ids[:, :256])
        image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
        image = (image * 255.0).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        image = image[:, :, :, [2, 1, 0]][0]
        image_path = out_dir / f"omni_t2i_{args.order_policy}.png"
        cv2.imwrite(str(image_path), image)

    history_frame_paths = []
    if args.save_history_frames and _histories:
        frame_dir = out_dir / "history_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        stride = max(int(args.history_frame_stride), 1)
        history_indices = list(range(0, len(_histories), stride))
        if len(_histories) - 1 not in history_indices:
            history_indices.append(len(_histories) - 1)
        if args.history_frame_limit > 0 and len(history_indices) > args.history_frame_limit:
            picked = np.linspace(0, len(history_indices) - 1, args.history_frame_limit).round().astype(int).tolist()
            history_indices = [history_indices[i] for i in sorted(set(picked))]
        for hist_idx in history_indices:
            seq = _histories[hist_idx][0][input_ids.shape[1]:]
            hist_image_tokens = []
            for token_id in seq:
                token_int = int(token_id.item())
                if token_int >= int(image_offset) and token_int < int(image_offset) + 8192:
                    hist_image_tokens.append(token_int - int(image_offset))
            if not hist_image_tokens:
                continue
            if len(hist_image_tokens) < 256:
                hist_image_tokens += [hist_image_tokens[-1]] * (256 - len(hist_image_tokens))
            gen_token_ids = torch.tensor(hist_image_tokens[:256], dtype=torch.long, device="cuda:0").unsqueeze(0)
            gen_token_ids = torch.clamp(gen_token_ids, max=8192 - 1, min=0)
            frame = image_processor.image_tokenizer.image_tokenizer.decode_code(gen_token_ids)
            frame = torch.clamp((frame + 1.0) / 2.0, min=0.0, max=1.0)
            frame = (frame * 255.0).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            frame = frame[:, :, :, [2, 1, 0]][0]
            frame_path = frame_dir / f"step_{hist_idx:04d}.png"
            cv2.imwrite(str(frame_path), frame)
            history_frame_paths.append(str(frame_path))

    payload = {
        "model_path": args.model_path,
        "image_tokenizer_path": args.image_tokenizer_path,
        "prompt": args.prompt,
        "steps": args.steps,
        "max_tokens": args.max_tokens,
        "alg": args.alg,
        "order_policy": args.order_policy,
        "dprm_warmup_steps": args.dprm_warmup_steps,
        "dprm_table": args.dprm_table,
        "dprm_action_value_model": args.dprm_action_value_model,
        "dprm_action_steps": args.dprm_action_steps,
        "dprm_action_guidance_scale": args.dprm_action_guidance_scale,
        "seed": args.seed,
        "generation_seconds": generation_seconds,
        "total_seconds": time.time() - started,
        "num_image_tokens": len(image_tokens),
        "text_output": tokenizer.decode(text_tokens, skip_special_tokens=True),
        "image_path": str(image_path) if image_path else None,
        "history_frame_paths": history_frame_paths,
    }
    if args.dprm_table:
        denom = max(dprm_hook_stats["candidate_count"], 1)
        payload["dprm_hook_stats"] = {
            **dprm_hook_stats,
            "ready_fraction": dprm_hook_stats["ready_candidate_count"] / denom,
            "mean_gate": dprm_hook_stats["gate_sum"] / denom,
            "mean_abs_dprm_value": dprm_hook_stats["dprm_abs_sum"] / denom,
        }
    if args.dprm_action_value_model:
        denom = max(action_value_stats["candidate_count"], 1)
        payload["dprm_action_value_stats"] = {
            **action_value_stats,
            "mean_predicted_advantage": action_value_stats["predicted_advantage_sum"] / denom,
            "mean_abs_predicted_advantage": action_value_stats[
                "predicted_advantage_abs_sum"
            ]
            / denom,
        }
    if args.force_order_step is not None:
        payload["counterfactual_override"] = counterfactual_stats
        if args.require_forced_action and not counterfactual_stats.get("applied"):
            raise RuntimeError(
                f"counterfactual action was not applied at step {args.force_order_step}: "
                f"{counterfactual_stats}"
            )
    if args.trace_order_stats:
        trace_path = (
            out_dir / f"omni_t2i_{args.order_policy}_order_trace.jsonl"
            if args.trace_order_stats == "auto"
            else Path(args.trace_order_stats)
        )
        with trace_path.open("w", encoding="utf-8") as handle:
            for rec in trace_records:
                handle.write(json.dumps(rec) + "\n")
        payload["order_trace_path"] = str(trace_path)
        payload["order_trace_records"] = len(trace_records)
    (out_dir / f"omni_t2i_{args.order_policy}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
