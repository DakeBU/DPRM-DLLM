#!/usr/bin/env python3
"""Select or audit LLaDA-V RealWorldQA DPRM runs on a fixed document interval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from build_dprm_table import classify_answer_format, target_normalized_match


def _target_match(row):
    return target_normalized_match(row)


def parse_candidate(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("candidate must have LABEL=PATH form")
    return label, Path(path)


def bootstrap_delta(
    baseline: np.ndarray, candidate: np.ndarray, seed: int, draws: int
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=float)
    paired = candidate - baseline
    for draw in range(draws):
        indices = rng.integers(0, len(paired), size=len(paired))
        estimates[draw] = paired[indices].mean()
    return {
        "delta": float(paired.mean()),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
    }


def _latest_sample_file(root: Path, order: str, task: str) -> Path:
    candidates = sorted(
        (root / order / task).rglob(f"*_samples_{task}.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no {task} samples under {root / order / task}")
    return candidates[0]


def load_samples(root: Path, order: str, task: str) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with _latest_sample_file(root, order, task).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[int(row["doc_id"])] = row
    return rows


def load_trace(root: Path, order: str, task: str) -> dict[int, list[dict[str, Any]]]:
    path = root / order / task / "order_trace.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing order trace: {path}")
    grouped: dict[int, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            grouped.setdefault(int(row["doc_id"]), []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["step"]))
    return grouped


def _position_sequence(rows: list[dict[str, Any]]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(position) for position in row.get("selected_positions", []))
        for row in rows
    )


def trace_summary(
    candidate: dict[int, list[dict[str, Any]]],
    baseline: dict[int, list[dict[str, Any]]],
    doc_ids: list[int],
) -> dict[str, Any]:
    missing = [doc_id for doc_id in doc_ids if doc_id not in baseline or doc_id not in candidate]
    if missing:
        raise RuntimeError(f"missing paired traces for {len(missing)} documents")
    changed = [
        _position_sequence(candidate[doc_id]) != _position_sequence(baseline[doc_id])
        for doc_id in doc_ids
    ]
    candidate_rows = [row for doc_id in doc_ids for row in candidate[doc_id]]
    gates = [
        float(row["dprm_selected_gate_mean"])
        for row in candidate_rows
        if row.get("dprm_selected_gate_mean") is not None
    ]
    corrections = [
        abs(
            float(row["dprm_selected_score_mean"])
            - float(row["dprm_selected_base_log_score_mean"])
        )
        for row in candidate_rows
        if row.get("dprm_selected_score_mean") is not None
        and row.get("dprm_selected_base_log_score_mean") is not None
    ]
    return {
        "documents": len(doc_ids),
        "rows": len(candidate_rows),
        "order_changed_documents": int(sum(changed)),
        "order_changed_rate": float(np.mean(changed)),
        "selected_ready_rate": (
            float(np.mean(np.asarray(gates) >= 1.0)) if gates else None
        ),
        "mean_abs_score_correction": float(np.mean(corrections)) if corrections else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--task", default="realworldqa")
    parser.add_argument("--doc-min", type=int, required=True)
    parser.add_argument("--doc-max", type=int, required=True)
    parser.add_argument("--min-order-change", type=float, default=0.01)
    parser.add_argument(
        "--require-positive-delta", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_order = "progressive_confidence"
    dprm_order = "dprm_confidence_warmup"
    baseline = load_samples(args.baseline_root, baseline_order, args.task)
    baseline_trace = load_trace(args.baseline_root, baseline_order, args.task)
    requested = set(range(args.doc_min, args.doc_max))
    payload = {
        "protocol": {
            "doc_interval": [args.doc_min, args.doc_max],
            "selection_metric": "target-normalized accuracy",
            "minimum_order_changed_rate": args.min_order_change,
            "require_positive_development_delta": args.require_positive_delta,
            "tie_break": "input candidate order",
        },
        "baseline_root": str(args.baseline_root),
        "candidates": {},
        "selected": None,
    }

    ranked: list[tuple[float, int, str]] = []
    for index, (label, root) in enumerate(args.candidate):
        candidate = load_samples(root, dprm_order, args.task)
        doc_ids = sorted(requested & set(baseline) & set(candidate))
        if len(doc_ids) != len(requested):
            raise RuntimeError(
                f"{label}: expected {len(requested)} shared documents, found {len(doc_ids)}"
            )
        base_values = np.asarray([_target_match(baseline[i]) for i in doc_ids], dtype=float)
        candidate_values = np.asarray([_target_match(candidate[i]) for i in doc_ids], dtype=float)
        trace = trace_summary(load_trace(root, dprm_order, args.task), baseline_trace, doc_ids)
        categories = {}
        for category in ("choice", "numeric", "open"):
            ids = [
                doc_id
                for doc_id in doc_ids
                if classify_answer_format(
                    str(
                        candidate[doc_id].get("input")
                        or candidate[doc_id].get("doc", {}).get("question", "")
                    )
                )
                == category
            ]
            categories[category] = {
                "n": len(ids),
                "accuracy": (
                    float(np.mean([_target_match(candidate[i]) for i in ids])) if ids else None
                ),
            }
        accuracy = float(candidate_values.mean())
        active = (
            trace["order_changed_rate"] is not None
            and trace["order_changed_rate"] >= args.min_order_change
        )
        positive = accuracy > float(base_values.mean())
        eligible = active and (positive or not args.require_positive_delta)
        paired_delta = bootstrap_delta(
            base_values, candidate_values, args.seed + index, args.bootstrap
        )
        payload["candidates"][label] = {
            "root": str(root),
            "documents": len(doc_ids),
            "accuracy": accuracy,
            "baseline_accuracy": float(base_values.mean()),
            "paired_delta": paired_delta,
            "categories": categories,
            "trace": trace,
            "active_controller": active,
            "positive_point_delta": positive,
            "paired_ci_excludes_zero": paired_delta["ci_low"] > 0.0,
            "strict_promotion_eligible": (
                active and positive and paired_delta["ci_low"] > 0.0
            ),
            "eligible": eligible,
        }
        if eligible:
            ranked.append((accuracy, -index, label))

    if ranked:
        payload["selected"] = max(ranked)[2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
