#!/usr/bin/env python3
"""Paired audit for frozen-table LLaDA-V transfer to ChartQA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from build_dprm_table import classify_answer_format


def latest_sample(root: Path, order: str) -> Path:
    candidates = sorted(
        (root / order / "chartqa_lite").rglob("*_samples_chartqa_lite.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no ChartQA samples under {root / order}")
    return candidates[0]


def response(row: dict[str, Any]) -> str:
    filtered = row.get("filtered_resps")
    if isinstance(filtered, list) and filtered:
        return str(filtered[0]).strip()
    resps = row.get("resps")
    if isinstance(resps, list) and resps:
        value = resps[0]
        if isinstance(value, list) and value:
            return str(value[0]).strip()
        return str(value).strip()
    return ""


def relaxed_correct(prediction: str, target: str) -> bool:
    def as_float(value: str) -> float | None:
        value = value.strip()
        try:
            if value.endswith("%"):
                return float(value[:-1]) / 100.0
            return float(value)
        except ValueError:
            return None

    prediction_float = as_float(prediction)
    target_float = as_float(target)
    if prediction_float is not None and target_float not in (None, 0.0):
        return abs(prediction_float - target_float) / abs(target_float) <= 0.05
    return prediction.strip().lower() == target.strip().lower()


def load_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            rows[int(row["doc_id"])] = row
    return rows


def saved_correct(row: dict[str, Any]) -> float:
    if "relaxed_overall" not in row:
        raise KeyError(f"missing official relaxed_overall for doc {row.get('doc_id')}")
    return float(row["relaxed_overall"])


def load_trace(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    rows = {}
    if not path.exists():
        return rows
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            rows[(int(row["doc_id"]), int(row["step"]))] = row
    return rows


def question(row: dict[str, Any]) -> str:
    doc = row.get("doc")
    if isinstance(doc, dict) and doc.get("question") is not None:
        return str(doc["question"])
    return str(row.get("input", ""))


def bootstrap_delta(
    baseline: np.ndarray, candidate: np.ndarray, seed: int, draws: int
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    deltas = np.empty(draws, dtype=float)
    for draw in range(draws):
        indices = rng.integers(0, len(baseline), size=len(baseline))
        deltas[draw] = np.mean(candidate[indices] - baseline[indices])
    return {
        "delta": float(np.mean(candidate - baseline)),
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    baseline_path = latest_sample(args.root, "progressive_confidence")
    dprm_path = latest_sample(args.root, "dprm_confidence_warmup")
    baseline_rows = load_rows(baseline_path)
    dprm_rows = load_rows(dprm_path)
    doc_ids = sorted(set(baseline_rows) & set(dprm_rows))
    if len(doc_ids) != 500:
        raise RuntimeError(f"expected 500 paired ChartQA examples, found {len(doc_ids)}")

    baseline = np.asarray(
        [
            relaxed_correct(response(baseline_rows[doc_id]), str(baseline_rows[doc_id]["target"]))
            for doc_id in doc_ids
        ],
        dtype=float,
    )
    dprm = np.asarray(
        [
            relaxed_correct(response(dprm_rows[doc_id]), str(dprm_rows[doc_id]["target"]))
            for doc_id in doc_ids
        ],
        dtype=float,
    )
    baseline_saved = np.asarray(
        [saved_correct(baseline_rows[doc_id]) for doc_id in doc_ids], dtype=float
    )
    dprm_saved = np.asarray(
        [saved_correct(dprm_rows[doc_id]) for doc_id in doc_ids], dtype=float
    )
    baseline_mismatches = int(np.sum(baseline != baseline_saved))
    dprm_mismatches = int(np.sum(dprm != dprm_saved))
    if baseline_mismatches or dprm_mismatches:
        raise RuntimeError(
            "recomputed relaxed accuracy disagrees with lmms-eval: "
            f"confidence={baseline_mismatches}, dprm={dprm_mismatches}"
        )
    formats = np.asarray(
        [classify_answer_format(question(baseline_rows[doc_id])) for doc_id in doc_ids]
    )
    by_prompt_format = {}
    for answer_format in ("choice", "numeric", "open"):
        mask = formats == answer_format
        if not np.any(mask):
            continue
        by_prompt_format[answer_format] = {
            "documents": int(np.sum(mask)),
            "baseline_accuracy": float(baseline[mask].mean()),
            "dprm_accuracy": float(dprm[mask].mean()),
            "paired_delta": bootstrap_delta(
                baseline[mask], dprm[mask], args.seed + len(by_prompt_format) + 1, args.bootstrap
            ),
            "wins": int(np.sum((baseline[mask] == 0) & (dprm[mask] == 1))),
            "losses": int(np.sum((baseline[mask] == 1) & (dprm[mask] == 0))),
        }
    trace_path = (
        args.root
        / "order_traces"
        / "dprm_confidence_warmup"
        / "chartqa_lite"
        / "order_trace.jsonl"
    )
    trace = load_trace(trace_path)
    trace_rows = [row for (doc_id, _), row in trace.items() if doc_id in set(doc_ids)]
    payload = {
        "protocol": {
            "task": "chartqa_lite",
            "documents": 500,
            "source_table": "RealWorldQA refit; no ChartQA fitting",
            "metric": "task-native relaxed correctness recomputed from saved outputs",
            "bootstrap_draws": args.bootstrap,
            "official_score_mismatches": {
                "confidence": baseline_mismatches,
                "dprm": dprm_mismatches,
            },
        },
        "baseline_accuracy": float(baseline.mean()),
        "dprm_accuracy": float(dprm.mean()),
        "paired_delta": bootstrap_delta(baseline, dprm, args.seed, args.bootstrap),
        "wins": int(np.sum((baseline == 0) & (dprm == 1))),
        "losses": int(np.sum((baseline == 1) & (dprm == 0))),
        "by_prompt_format": by_prompt_format,
        "trace": {
            "rows": len(trace_rows),
            "order_changed_rate": (
                float(np.mean([bool(row.get("order_changed_vs_confidence")) for row in trace_rows]))
                if trace_rows
                else None
            ),
            "selected_ready_rate": (
                float(np.mean([float(row.get("dprm_selected_gate_mean", 0.0)) >= 1.0 for row in trace_rows]))
                if trace_rows
                else None
            ),
            "mean_abs_score_correction": (
                float(
                    np.mean(
                        [
                            abs(
                                float(row.get("dprm_selected_score_mean", 0.0))
                                - float(row.get("dprm_selected_base_log_score_mean", 0.0))
                            )
                            for row in trace_rows
                        ]
                    )
                )
                if trace_rows
                else None
            ),
        },
        "sample_paths": {
            "baseline": str(baseline_path),
            "dprm": str(dprm_path),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")

    delta = payload["paired_delta"]
    lines = [
        "# Frozen-table LLaDA-V ChartQA transfer",
        "",
        "The DPRM table and guidance were frozen from RealWorldQA before opening ChartQA outputs.",
        "",
        "| confidence | DPRM | paired delta [95% CI] | wins/losses | order changed | ready |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {payload['baseline_accuracy']:.4f} | {payload['dprm_accuracy']:.4f} | "
        f"{delta['delta']:+.4f} [{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}] | "
        f"{payload['wins']}/{payload['losses']} | "
        f"{payload['trace']['order_changed_rate']} | {payload['trace']['selected_ready_rate']} |",
        "",
        "Prompt-format strata are fixed by the visible question only; they do not inspect the target.",
        "",
        "| prompt format | n | confidence | DPRM | paired delta [95% CI] | wins/losses |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for answer_format, summary in by_prompt_format.items():
        format_delta = summary["paired_delta"]
        lines.append(
            f"| {answer_format} | {summary['documents']} | "
            f"{summary['baseline_accuracy']:.4f} | {summary['dprm_accuracy']:.4f} | "
            f"{format_delta['delta']:+.4f} "
            f"[{format_delta['ci_low']:+.4f}, {format_delta['ci_high']:+.4f}] | "
            f"{summary['wins']}/{summary['losses']} |"
        )
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
