#!/usr/bin/env python3
"""Build a bucketized DPRM table from traced Omni-Diffusion T2I rollouts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image


def iter_rollout_records(rollout_root: Path, orders: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for order in orders:
        order_root = rollout_root / order
        for prompt_dir in sorted(order_root.glob("prompt_*")):
            json_files = sorted(prompt_dir.glob(f"omni_t2i_{order}.json"))
            if not json_files:
                json_files = sorted(prompt_dir.glob("*.json"))
            if not json_files:
                continue
            with json_files[0].open(encoding="utf-8") as handle:
                rec = json.load(handle)
            trace_path = Path(rec.get("order_trace_path", ""))
            if not trace_path.is_file():
                traces = sorted(prompt_dir.glob("*order_trace.jsonl"))
                trace_path = traces[0] if traces else Path()
            image_path = Path(rec.get("image_path", ""))
            if not image_path.is_file():
                pngs = sorted(prompt_dir.glob("*.png"))
                image_path = pngs[0] if pngs else Path()
            rec.update(
                {
                    "order": order,
                    "prompt_id": prompt_dir.name,
                    "json_path": str(json_files[0]),
                    "trace_path": str(trace_path),
                    "image_path": str(image_path),
                    "has_trace": trace_path.is_file(),
                    "has_image": image_path.is_file(),
                }
            )
            if rec["has_trace"] and rec["has_image"]:
                records.append(rec)
    return records


def deduplicate_prompt_text(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one matched rollout per order for each distinct prompt text."""
    kept: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        key = (str(record.get("prompt", "")).strip(), str(record.get("order", "")))
        if key in seen:
            continue
        seen.add(key)
        kept.append(record)
    return kept


def add_clip_scores(
    records: list[dict[str, Any]],
    model_name: str,
    device: str,
    batch_size: int,
    metric_name: str = "clip_cosine",
) -> None:
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)

    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            images = [Image.open(rec["image_path"]).convert("RGB") for rec in batch]
            prompts = [str(rec.get("prompt", "")) for rec in batch]
            inputs = processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
            text_features = model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            scores = (image_features * text_features).sum(dim=-1).detach().cpu().tolist()
            for rec, score in zip(batch, scores):
                rec[metric_name] = float(score)


def reuse_clip_scores(
    records: list[dict[str, Any]], table_path: Path, metric_name: str = "clip_cosine"
) -> int:
    with table_path.open(encoding="utf-8") as handle:
        table = json.load(handle)
    source_records = table.get("metadata", {}).get("records", [])
    scores = {
        (str(rec.get("order", "")), str(rec.get("prompt_id", ""))): float(rec[metric_name])
        for rec in source_records
        if rec.get(metric_name) is not None
    }
    reused = 0
    for rec in records:
        key = (str(rec.get("order", "")), str(rec.get("prompt_id", "")))
        if key in scores:
            rec[metric_name] = scores[key]
            reused += 1
    return reused


def normalized_metric_rewards(
    records: list[dict[str, Any]], mode: str, metric_name: str
) -> tuple[dict[str, float], dict[str, Any]]:
    scores = [float(rec[metric_name]) for rec in records]
    mean = statistics.fmean(scores) if scores else 0.0
    std = statistics.pstdev(scores) if len(scores) > 1 else 1.0
    std = max(float(std), 1e-6)
    paired_advantages: dict[str, float] = {}
    paired_groups = 0
    paired_std = None
    if mode == "paired_prompt_advantage":
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rec in records:
            groups[str(rec["prompt_id"])].append(rec)
        invalid = {
            prompt_id: sorted(str(rec.get("order", "")) for rec in group)
            for prompt_id, group in groups.items()
            if len(group) < 2 or len({str(rec.get("order", "")) for rec in group}) < 2
        }
        if invalid:
            examples = list(invalid.items())[:5]
            raise ValueError(
                "paired_prompt_advantage requires at least two distinct orders per prompt; "
                f"invalid groups include {examples}"
            )
        raw_advantages: list[float] = []
        for prompt_id, group in groups.items():
            baseline = statistics.fmean(float(rec[metric_name]) for rec in group)
            for rec in group:
                advantage = float(rec[metric_name]) - baseline
                paired_advantages[str(rec["trace_path"])] = advantage
                rec[f"prompt_{metric_name}_baseline"] = baseline
                rec[f"raw_{metric_name}_paired_advantage"] = advantage
                raw_advantages.append(advantage)
            paired_groups += 1
        paired_std = max(float(statistics.pstdev(raw_advantages)), 1e-6)

    rewards: dict[str, float] = {}
    for rec in records:
        score = float(rec[metric_name])
        if mode == "raw":
            reward = score
        elif mode == "centered":
            reward = score - mean
        elif mode == "zscore":
            reward = (score - mean) / std
        elif mode == "paired_prompt_advantage":
            reward = paired_advantages[str(rec["trace_path"])] / float(paired_std)
        else:
            raise ValueError(f"unknown reward normalization: {mode}")
        rewards[str(rec["trace_path"])] = float(reward)
    return rewards, {
        "metric": metric_name,
        "mean": mean,
        "std": std,
        "paired_prompt_groups": paired_groups,
        "paired_advantage_std": paired_std,
    }


def normalized_rewards(
    records: list[dict[str, Any]],
    mode: str,
    metric_weights: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Build one terminal utility after normalizing each metric separately."""
    metric_weights = metric_weights or {"clip_cosine": 1.0}
    if not metric_weights or sum(abs(weight) for weight in metric_weights.values()) <= 0:
        raise ValueError("at least one non-zero terminal-utility weight is required")

    metric_rewards: dict[str, dict[str, float]] = {}
    metric_stats: dict[str, dict[str, Any]] = {}
    for metric_name, weight in metric_weights.items():
        if any(record.get(metric_name) is None for record in records):
            raise ValueError(f"missing {metric_name} for terminal-utility construction")
        rewards, stats = normalized_metric_rewards(records, mode, metric_name)
        metric_rewards[metric_name] = rewards
        metric_stats[metric_name] = stats | {"weight": float(weight)}

    rewards: dict[str, float] = {}
    for record in records:
        trace_path = str(record["trace_path"])
        reward = sum(
            float(weight) * metric_rewards[metric_name][trace_path]
            for metric_name, weight in metric_weights.items()
        )
        rewards[trace_path] = reward
        record["dprm_reward_components"] = {
            metric_name: metric_rewards[metric_name][trace_path]
            for metric_name in metric_weights
        }
        record["dprm_reward"] = reward
    return rewards, {
        "mode": mode,
        "metric_weights": {key: float(value) for key, value in metric_weights.items()},
        "metrics": metric_stats,
    }


def read_trace(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def active_trace_rows(
    rows: list[dict[str, Any]], active_steps: set[int]
) -> list[dict[str, Any]]:
    """Restrict process-value credit to the deployment decision stages."""
    if not active_steps:
        return rows
    return [row for row in rows if int(row.get("step", -1)) in active_steps]


def spatial_bin(visual_index: int, aux_bins: int, image_side: int = 16) -> int:
    """Map a saved visual index to the requested row-major spatial grid."""
    if aux_bins <= 1:
        return 0
    side_bins = int(round(aux_bins**0.5))
    index = max(0, min(int(visual_index), image_side * image_side - 1))
    if side_bins * side_bins == aux_bins:
        row, column = divmod(index, image_side)
        row_bin = row * side_bins // image_side
        column_bin = column * side_bins // image_side
        return min(aux_bins - 1, row_bin * side_bins + column_bin)
    return min(aux_bins - 1, index * aux_bins // (image_side * image_side))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--orders", nargs="+", default=["random", "progressive_confidence"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--secondary-clip-model", default=None)
    parser.add_argument("--secondary-metric-name", default="clip_b32_cosine")
    parser.add_argument("--primary-reward-weight", type=float, default=1.0)
    parser.add_argument("--secondary-reward-weight", type=float, default=0.0)
    parser.add_argument(
        "--deduplicate-prompt-text",
        action="store_true",
        help="give every distinct prompt text one rollout per order",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clip-batch-size", type=int, default=8)
    parser.add_argument(
        "--reuse-clip-from-table",
        type=Path,
        default=None,
        help="Reuse CLIP scores keyed by (order, prompt_id) from an existing DPRM table.",
    )
    parser.add_argument(
        "--reuse-secondary-clip-from-table",
        type=Path,
        default=None,
        help="Reuse secondary CLIP scores from an existing DPRM table.",
    )
    parser.add_argument(
        "--reward-normalization",
        choices=["raw", "centered", "zscore", "paired_prompt_advantage"],
        default="zscore",
    )
    parser.add_argument("--num-phases", type=int, default=8)
    parser.add_argument(
        "--phase-source",
        choices=["trace", "step"],
        default="trace",
        help="Use the stored trace phase or derive phase=floor(step*num_phases/steps).",
    )
    parser.add_argument("--confidence-bins", type=int, default=16)
    parser.add_argument(
        "--confidence-binning",
        choices=["equal_width", "development_quantile"],
        default="equal_width",
    )
    parser.add_argument("--aux-bins", type=int, default=16)
    parser.add_argument("--reward-temperature", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--ready-count", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--switch-steps", type=int, default=64)
    parser.add_argument(
        "--active-steps",
        type=int,
        nargs="*",
        default=(),
        help=(
            "Accumulate terminal-return statistics only for these reveal steps. "
            "Omit to use every traced action."
        ),
    )
    parser.add_argument(
        "--require-fixed-visual-canvas",
        action="store_true",
        help="Require one trace action for each visual index 0..255 and no format-token actions.",
    )
    args = parser.parse_args()

    records = iter_rollout_records(args.rollout_root, args.orders)
    records_before_deduplication = len(records)
    if args.deduplicate_prompt_text:
        records = deduplicate_prompt_text(records)
    if not records:
        raise SystemExit(f"no traced rollout records found under {args.rollout_root}")

    reused_clip = 0
    if args.reuse_clip_from_table is not None:
        reused_clip = reuse_clip_scores(records, args.reuse_clip_from_table)
    missing_clip = [rec for rec in records if rec.get("clip_cosine") is None]
    if missing_clip:
        add_clip_scores(missing_clip, args.clip_model, args.device, args.clip_batch_size)

    reused_secondary_clip = 0
    metric_weights = {"clip_cosine": args.primary_reward_weight}
    if args.secondary_clip_model is not None or args.secondary_reward_weight != 0.0:
        if args.secondary_clip_model is None:
            raise ValueError("--secondary-clip-model is required for a secondary utility")
        if args.reuse_secondary_clip_from_table is not None:
            reused_secondary_clip = reuse_clip_scores(
                records,
                args.reuse_secondary_clip_from_table,
                args.secondary_metric_name,
            )
        missing_secondary = [
            rec for rec in records if rec.get(args.secondary_metric_name) is None
        ]
        if missing_secondary:
            add_clip_scores(
                missing_secondary,
                args.secondary_clip_model,
                args.device,
                args.clip_batch_size,
                args.secondary_metric_name,
            )
        metric_weights[args.secondary_metric_name] = args.secondary_reward_weight
    rewards, reward_stats = normalized_rewards(
        records, args.reward_normalization, metric_weights
    )

    traces = {
        str(rec["trace_path"]): read_trace(Path(rec["trace_path"])) for rec in records
    }
    active_steps = {int(step) for step in args.active_steps}
    credited_traces = {
        path: active_trace_rows(rows, active_steps) for path, rows in traces.items()
    }
    if active_steps and not any(credited_traces.values()):
        raise ValueError(
            f"no trace decisions match active steps {sorted(active_steps)}"
        )
    if args.require_fixed_visual_canvas:
        for rec in records:
            if rec.get("fixed_t2i_scaffold") is not True:
                raise ValueError(
                    f"formal visual-canvas table requires fixed_t2i_scaffold: {rec['json_path']}"
                )
            selected_visual = [
                int(index)
                for row in traces[str(rec["trace_path"])]
                for index in row.get("selected_visual_indices", [])
            ]
            if sorted(selected_visual) != list(range(256)):
                raise ValueError(
                    "formal visual-canvas trace must select each index 0..255 exactly once: "
                    f"{rec['trace_path']} has {len(selected_visual)} actions and "
                    f"{len(set(selected_visual))} unique indices"
                )
    confidence_bin_edges: list[float] = []
    if args.confidence_binning == "development_quantile":
        selected_confidence = [
            float(value)
            for rows in credited_traces.values()
            for row in rows
            for value in row.get("selected_confidence", [])
        ]
        if not selected_confidence:
            raise ValueError(
                "development_quantile binning requires selected_confidence in traces"
            )
        values = torch.tensor(selected_confidence, dtype=torch.float64)
        quantiles = torch.arange(1, args.confidence_bins, dtype=torch.float64)
        quantiles = quantiles / float(args.confidence_bins)
        confidence_bin_edges = torch.quantile(values, quantiles).tolist()

    def confidence_bin(value: float) -> int:
        if confidence_bin_edges:
            return min(
                args.confidence_bins - 1,
                sum(float(value) >= edge for edge in confidence_bin_edges),
            )
        return max(
            0,
            min(args.confidence_bins - 1, int(float(value) * args.confidence_bins)),
        )

    shape = (args.num_phases, args.confidence_bins, args.aux_bins)
    counts = torch.zeros(shape, dtype=torch.float64)
    exp_reward_sums = torch.zeros(shape, dtype=torch.float64)
    reward_sums = torch.zeros(shape, dtype=torch.float64)
    reward_sq_sums = torch.zeros(shape, dtype=torch.float64)

    trace_rows = 0
    selected_total = 0
    for rec in records:
        reward = rewards[str(rec["trace_path"])]
        exp_reward = math.exp(args.reward_temperature * max(min(reward, 20.0), -20.0))
        rows = credited_traces[str(rec["trace_path"])]
        total_steps = max(int(rec.get("steps", len(rows))), 1)
        for row in rows:
            trace_rows += 1
            if args.phase_source == "step":
                step = max(0, int(row.get("step", 0)))
                phase = min(args.num_phases - 1, step * args.num_phases // total_steps)
            else:
                phase = max(0, min(args.num_phases - 1, int(row["phase"])))
            selected_confidence = row.get("selected_confidence", [])
            selected_aux = row.get("selected_aux_bins", [])
            selected_visual = row.get("selected_visual_indices", [])
            if selected_visual and len(selected_visual) == len(selected_confidence):
                selected_aux = [
                    spatial_bin(index, args.aux_bins) for index in selected_visual
                ]
            if selected_confidence and len(selected_confidence) == len(selected_aux):
                buckets = [
                    {
                        "confidence_bin": confidence_bin(confidence),
                        "aux_bin": aux,
                        "count": 1,
                    }
                    for confidence, aux in zip(selected_confidence, selected_aux)
                ]
            else:
                if confidence_bin_edges:
                    raise ValueError(
                        "quantile binning cannot reconstruct actions without selected_confidence"
                    )
                buckets = row.get("bucket_counts", [])
            for bucket in buckets:
                conf_bin = max(0, min(args.confidence_bins - 1, int(bucket["confidence_bin"])))
                aux_bin = max(0, min(args.aux_bins - 1, int(bucket.get("aux_bin", 0))))
                count = float(bucket["count"])
                counts[phase, conf_bin, aux_bin] += count
                exp_reward_sums[phase, conf_bin, aux_bin] += count * exp_reward
                reward_sums[phase, conf_bin, aux_bin] += count * reward
                reward_sq_sums[phase, conf_bin, aux_bin] += count * reward * reward
                selected_total += int(count)

    nonempty = int((counts > 0).sum().item())
    total_buckets = int(counts.numel())
    payload = {
        "cfg": {
            "num_phases": args.num_phases,
            "phase_source": args.phase_source,
            "base_order_score": "negative_token_entropy",
            "bucket_coordinate": "exp_negative_token_entropy",
            "confidence_bins": args.confidence_bins,
            "confidence_binning": args.confidence_binning,
            "confidence_bin_edges": confidence_bin_edges,
            "aux_bins": args.aux_bins,
            "reward_temperature": args.reward_temperature,
            "guidance_scale": args.guidance_scale,
            "warmup_steps": args.warmup_steps,
            "switch_steps": args.switch_steps,
            "ready_count": args.ready_count,
            "sampled_soft_bon": False,
        },
        "counts": counts.tolist(),
        "exp_reward_sums": exp_reward_sums.tolist(),
        "reward_sums": reward_sums.tolist(),
        "reward_sq_sums": reward_sq_sums.tolist(),
        "metadata": {
            "score_contract": {
                "base_order_score": "negative_token_entropy",
                "base_order_formula": "-H[p_theta(.|s,i)]",
                "bucket_coordinate": "exp_negative_token_entropy",
                "bucket_coordinate_formula": "exp(-H[p_theta(.|s,i)])",
                "token_value_rule": "host_argmax_token",
            },
            "rollout_root": str(args.rollout_root),
            "orders": args.orders,
            "clip_model": args.clip_model,
            "secondary_clip_model": args.secondary_clip_model,
            "secondary_metric_name": args.secondary_metric_name,
            "reused_clip_from_table": str(args.reuse_clip_from_table) if args.reuse_clip_from_table else None,
            "num_reused_clip_scores": reused_clip,
            "reused_secondary_clip_from_table": str(args.reuse_secondary_clip_from_table) if args.reuse_secondary_clip_from_table else None,
            "num_reused_secondary_clip_scores": reused_secondary_clip,
            "reward_normalization": args.reward_normalization,
            "reward_stats": reward_stats,
            "num_rollouts": len(records),
            "num_rollouts_before_prompt_deduplication": records_before_deduplication,
            "prompt_text_deduplicated": bool(args.deduplicate_prompt_text),
            "fixed_visual_canvas": bool(args.require_fixed_visual_canvas),
            "trace_rows": trace_rows,
            "source_trace_rows": sum(len(rows) for rows in traces.values()),
            "active_steps": sorted(active_steps),
            "selected_total": selected_total,
            "nonempty_buckets": nonempty,
            "total_buckets": total_buckets,
            "bucket_coverage": nonempty / max(total_buckets, 1),
            "mean_clip_cosine": statistics.fmean(float(r["clip_cosine"]) for r in records),
            "records": records,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload["metadata"], indent=2))


if __name__ == "__main__":
    main()
