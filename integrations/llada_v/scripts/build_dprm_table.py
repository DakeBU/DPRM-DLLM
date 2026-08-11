#!/usr/bin/env python3
"""Build a bucketized DPRM table from traced LLaDA-V lmms-eval rollouts."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

import torch


DEFAULT_ORDERS = ["random", "progressive_confidence"]
DEFAULT_TASKS = ["ai2d_lite", "realworldqa"]


ANSWER_FORMATS = {"choice": 0, "numeric": 1, "open": 2}


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def first_choice(text: str) -> str | None:
    match = re.search(r"\b([A-D])\b", str(text).upper())
    return match.group(1) if match else None


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9.]+", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def normalize_number(text: str) -> str | None:
    raw = str(text)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", raw)
    if not match:
        if re.search(r"\b(?:no|none|zero)\b", raw, flags=re.IGNORECASE):
            return "0"
        return None
    value = match.group(0)
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value


def target_normalized_match(row: dict[str, Any]) -> bool:
    target = str(row.get("target", "")).strip()
    target_upper = target.upper()
    response = response_text(row)
    if target_upper in {"A", "B", "C", "D"}:
        return first_choice(response) == target_upper
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", target):
        return normalize_number(response) == normalize_number(target)
    target_norm = normalize_text(target)
    response_norm = normalize_text(response)
    if not target_norm or not response_norm:
        return False
    return response_norm == target_norm or target_norm in response_norm


def response_text(row: dict[str, Any]) -> str:
    filtered = row.get("filtered_resps")
    if isinstance(filtered, list) and filtered:
        return str(filtered[0])
    resps = row.get("resps")
    if isinstance(resps, list) and resps:
        first = resps[0]
        if isinstance(first, list) and first:
            return str(first[0])
        return str(first)
    return ""


def classify_answer_format(prompt: str) -> str:
    """Classify from the visible question only; never inspect the target answer."""
    text = str(prompt)
    choices = re.findall(r"(?:^|\n)\s*[A-D][\.)]\s+", text, flags=re.IGNORECASE)
    if len(choices) >= 2 or "only the letter" in text.lower():
        return "choice"
    numeric_patterns = (
        r"\bhow many\b",
        r"\bhow much\b",
        r"\bnumber of\b",
        r"\bwhat (?:number|percentage|percent|fraction|time|year|age|distance|height|length)\b",
    )
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in numeric_patterns):
        return "numeric"
    return "open"


def latest_sample_file(root: Path, order: str, task: str) -> Path | None:
    task_dir = root / order / task
    search_dir = task_dir if task_dir.exists() else root / order
    candidates = sorted(
        search_dir.rglob(f"*_samples_{task}.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def collect_rewards(root: Path, orders: list[str], tasks: list[str], reward_mode: str) -> dict[tuple[str, str, str], float]:
    rewards: dict[tuple[str, str, str], float] = {}
    for order in orders:
        for task in tasks:
            sample_file = latest_sample_file(root, order, task)
            if sample_file is None:
                continue
            for row in iter_jsonl(sample_file):
                doc_id = str(row.get("doc_id"))
                if reward_mode == "exact":
                    reward = float(row.get("exact_match", 0.0) or 0.0)
                elif reward_mode == "choice":
                    target = str(row.get("target", "")).strip().upper()
                    pred = first_choice(response_text(row))
                    reward = 1.0 if pred is not None and pred == target else 0.0
                elif reward_mode == "target":
                    reward = 1.0 if target_normalized_match(row) else 0.0
                else:
                    raise ValueError(f"unknown reward mode: {reward_mode}")
                # DPRM estimates E[R | selected bucket] for the trajectory that
                # produced the trace. Collapsing random/progressive rewards by
                # doc_id leaks one order's success into the other order's trace.
                rewards[(order, task, doc_id)] = max(rewards.get((order, task, doc_id), 0.0), reward)
    return rewards


def collect_answer_formats(
    root: Path, orders: list[str], tasks: list[str]
) -> dict[tuple[str, str, str], int]:
    formats: dict[tuple[str, str, str], int] = {}
    for order in orders:
        for task in tasks:
            sample_file = latest_sample_file(root, order, task)
            if sample_file is None:
                continue
            for row in iter_jsonl(sample_file):
                prompt = row.get("input") or row.get("doc", {}).get("question", "")
                label = classify_answer_format(str(prompt))
                formats[(order, task, str(row.get("doc_id")))] = ANSWER_FORMATS[label]
    return formats


def normalize_rewards(rewards: dict[tuple[str, str, str], float], mode: str) -> dict[tuple[str, str, str], float]:
    values = list(rewards.values())
    if not values:
        return {}
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 1.0
    std = max(std, 1e-6)
    out: dict[tuple[str, str, str], float] = {}
    for key, value in rewards.items():
        if mode == "raw":
            out[key] = float(value)
        elif mode == "centered":
            out[key] = float(value - mean)
        elif mode == "zscore":
            out[key] = float((value - mean) / std)
        else:
            raise ValueError(f"unknown normalization: {mode}")
    return out


def trace_files(root: Path, orders: list[str], tasks: list[str]) -> list[Path]:
    files: list[Path] = []
    for order in orders:
        for task in tasks:
            preferred = root / "order_traces" / order / task / "order_trace.jsonl"
            if preferred.is_file():
                files.append(preferred)
    if files:
        return files
    return sorted(root.rglob("*trace*.jsonl"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--orders", nargs="+", default=DEFAULT_ORDERS)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--reward-mode", choices=["target", "choice", "exact"], default="target")
    parser.add_argument("--reward-normalization", choices=["raw", "centered", "zscore"], default="centered")
    parser.add_argument("--num-phases", type=int, default=8)
    parser.add_argument("--confidence-bins", type=int, default=16)
    parser.add_argument("--aux-bins", type=int, default=16)
    parser.add_argument(
        "--aux-mode",
        choices=["position", "eot_position", "format_eot_position"],
        default="position",
    )
    parser.add_argument("--position-bins", type=int, default=4)
    parser.add_argument("--format-bins", type=int, default=len(ANSWER_FORMATS))
    parser.add_argument("--source-num-phases", type=int, default=8)
    parser.add_argument("--source-confidence-bins", type=int, default=16)
    parser.add_argument("--source-aux-bins", type=int, default=16)
    parser.add_argument("--max-docs-per-task", type=int, default=None)
    parser.add_argument("--reward-temperature", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--ready-count", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--switch-steps", type=int, default=4)
    args = parser.parse_args()

    allowed_docs: dict[tuple[str, str], set[str]] = {}
    if args.max_docs_per_task is not None:
        for order in args.orders:
            for task in args.tasks:
                sample_file = latest_sample_file(args.eval_root, order, task)
                if sample_file is None:
                    continue
                doc_ids = []
                for sample in iter_jsonl(sample_file):
                    doc_ids.append(str(sample.get("doc_id")))
                    if len(doc_ids) >= args.max_docs_per_task:
                        break
                allowed_docs[(order, task)] = set(doc_ids)

    rewards = collect_rewards(args.eval_root, args.orders, args.tasks, args.reward_mode)
    if allowed_docs:
        rewards = {
            key: value
            for key, value in rewards.items()
            if key[2] in allowed_docs.get((key[0], key[1]), set())
        }
    rewards = normalize_rewards(rewards, args.reward_normalization)
    answer_formats = collect_answer_formats(args.eval_root, args.orders, args.tasks)
    if not rewards:
        raise SystemExit(f"no rewards found under {args.eval_root}")

    output_aux_bins = args.aux_bins
    if args.aux_mode == "eot_position":
        output_aux_bins = 2 * args.position_bins
    elif args.aux_mode == "format_eot_position":
        output_aux_bins = args.format_bins * 2 * args.position_bins
    shape = (args.num_phases, args.confidence_bins, output_aux_bins)
    counts = torch.zeros(shape, dtype=torch.float64)
    exp_reward_sums = torch.zeros(shape, dtype=torch.float64)
    reward_sums = torch.zeros(shape, dtype=torch.float64)
    reward_sq_sums = torch.zeros(shape, dtype=torch.float64)

    trace_rows = 0
    used_trace_rows = 0
    selected_total = 0
    missing_reward = 0
    for path in trace_files(args.eval_root, args.orders, args.tasks):
        for row in iter_jsonl(path):
            trace_rows += 1
            task = row.get("task")
            doc_id = None if row.get("doc_id") is None else str(row.get("doc_id"))
            order_policy = str(row.get("order_policy", ""))
            if allowed_docs and doc_id not in allowed_docs.get((order_policy, str(task)), set()):
                continue
            key = (order_policy, str(task), doc_id)
            if key not in rewards:
                missing_reward += 1
                continue
            reward = rewards[key]
            exp_reward = math.exp(args.reward_temperature * max(min(reward, 20.0), -20.0))
            source_phase = max(0, int(row["phase"]))
            phase = min(
                args.num_phases - 1,
                (source_phase * args.num_phases) // max(args.source_num_phases, 1),
            )
            for bucket in row.get("bucket_counts", []):
                source_conf_bin = max(0, int(bucket["confidence_bin"]))
                conf_bin = min(
                    args.confidence_bins - 1,
                    (source_conf_bin * args.confidence_bins)
                    // max(args.source_confidence_bins, 1),
                )
                source_aux = int(bucket.get("aux_bin", 0))
                if args.aux_mode == "position":
                    aux_bin = max(0, min(output_aux_bins - 1, source_aux))
                else:
                    position_bin = min(
                        args.position_bins - 1,
                        max(0, (source_aux * args.position_bins) // max(args.source_aux_bins, 1)),
                    )
                    if int(row.get("selected_count", 0)) != 1:
                        raise ValueError(
                            "eot_position table construction currently requires one selected token per trace row"
                        )
                    eot_bit = 1 if int(row.get("selected_eot_count", 0)) > 0 else 0
                    if args.aux_mode == "eot_position":
                        aux_bin = position_bin + eot_bit * args.position_bins
                    else:
                        format_bin = answer_formats.get(key)
                        if format_bin is None:
                            missing_reward += 1
                            continue
                        format_bin = max(0, min(args.format_bins - 1, int(format_bin)))
                        aux_bin = position_bin + args.position_bins * (
                            eot_bit + 2 * format_bin
                        )
                count = float(bucket["count"])
                counts[phase, conf_bin, aux_bin] += count
                exp_reward_sums[phase, conf_bin, aux_bin] += count * exp_reward
                reward_sums[phase, conf_bin, aux_bin] += count * reward
                reward_sq_sums[phase, conf_bin, aux_bin] += count * reward * reward
                selected_total += int(count)
            used_trace_rows += 1

    nonempty = int((counts > 0).sum().item())
    total_buckets = int(counts.numel())
    payload = {
        "cfg": {
            "num_phases": args.num_phases,
            "confidence_bins": args.confidence_bins,
            "aux_bins": output_aux_bins,
            "aux_mode": args.aux_mode,
            "position_bins": args.position_bins,
            "format_bins": args.format_bins,
            "answer_format_names": ANSWER_FORMATS,
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
            "eval_root": str(args.eval_root),
            "orders": args.orders,
            "tasks": args.tasks,
            "reward_mode": args.reward_mode,
            "reward_normalization": args.reward_normalization,
            "reward_count": len(rewards),
            "trace_rows": trace_rows,
            "used_trace_rows": used_trace_rows,
            "missing_reward_trace_rows": missing_reward,
            "selected_total": selected_total,
            "nonempty_buckets": nonempty,
            "total_buckets": total_buckets,
            "bucket_coverage": nonempty / max(total_buckets, 1),
            "max_docs_per_task": args.max_docs_per_task,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["metadata"], indent=2))


if __name__ == "__main__":
    main()
