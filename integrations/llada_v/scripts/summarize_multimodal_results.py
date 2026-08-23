#!/usr/bin/env python3
"""Rebuild LLaDA-V task results and paired intervals from saved samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from build_dprm_table import classify_answer_format, target_normalized_match


def read_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            doc_id = int(row["doc_id"])
            if doc_id in rows:
                raise ValueError(f"duplicate doc_id {doc_id} in {path}")
            rows[doc_id] = row
    if not rows:
        raise ValueError(f"no sample rows in {path}")
    return rows


def target_score(row: dict[str, Any]) -> float:
    return float(target_normalized_match(row))


def chartqa_score(row: dict[str, Any]) -> float:
    if "relaxed_overall" not in row:
        raise KeyError(
            f"ChartQA row {row.get('doc_id')} is missing lmms-eval relaxed_overall"
        )
    return float(row["relaxed_overall"])


def question(row: dict[str, Any]) -> str:
    doc = row.get("doc")
    if isinstance(doc, dict) and doc.get("question") is not None:
        return str(doc["question"])
    return str(row.get("input", ""))


def paired_interval(
    baseline: np.ndarray,
    method: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, float]:
    if baseline.shape != method.shape or baseline.ndim != 1 or not len(baseline):
        raise ValueError("paired arrays must be nonempty one-dimensional arrays")
    rng = np.random.default_rng(seed)
    deltas = np.empty(draws, dtype=float)
    for draw in range(draws):
        indices = rng.integers(0, len(baseline), size=len(baseline))
        deltas[draw] = np.mean(method[indices] - baseline[indices])
    return {
        "mean": float(np.mean(method - baseline)),
        "ci95": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
    }


def summarize_pair(
    baseline_path: Path,
    method_path: Path,
    *,
    scorer: Callable[[dict[str, Any]], float],
    doc_min: int | None,
    doc_max: int | None,
    draws: int,
    seed: int,
) -> tuple[dict[str, Any], list[int], np.ndarray, np.ndarray, dict[int, dict[str, Any]]]:
    baseline_rows = read_rows(baseline_path)
    method_rows = read_rows(method_path)
    shared = sorted(set(baseline_rows) & set(method_rows))
    if doc_min is not None:
        shared = [doc_id for doc_id in shared if doc_id >= doc_min]
    if doc_max is not None:
        shared = [doc_id for doc_id in shared if doc_id < doc_max]
    expected = None
    if doc_min is not None and doc_max is not None:
        expected = doc_max - doc_min
    if expected is not None and len(shared) != expected:
        raise ValueError(
            f"expected {expected} paired documents in [{doc_min}, {doc_max}), "
            f"found {len(shared)}"
        )
    if not shared:
        raise ValueError("no paired documents after applying the interval")
    baseline = np.asarray([scorer(baseline_rows[i]) for i in shared], dtype=float)
    method = np.asarray([scorer(method_rows[i]) for i in shared], dtype=float)
    summary = {
        "n": len(shared),
        "confidence": float(baseline.mean()),
        "dprm_confidence": float(method.mean()),
        "paired_delta": paired_interval(
            baseline, method, draws=draws, seed=seed
        ),
        "wins": int(np.sum((baseline == 0.0) & (method == 1.0))),
        "losses": int(np.sum((baseline == 1.0) & (method == 0.0))),
    }
    return summary, shared, baseline, method, baseline_rows


def optional_ai2d_orders(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "random": args.ai2d_random,
        "confidence": args.ai2d_confidence,
        "dprm_confidence": args.ai2d_dprm_confidence,
        "dprm_random": args.ai2d_dprm_random,
    }
    if not any(paths.values()):
        return {}
    required = ("confidence", "dprm_confidence")
    missing = [name for name in required if paths[name] is None]
    if missing:
        raise ValueError(f"AI2D requires confidence and DPRM files; missing {missing}")
    for pair in (("random",), ("dprm_random",)):
        name = pair[0]
        if paths[name] is None:
            paths.pop(name)
    rows = {name: read_rows(path) for name, path in paths.items()}
    shared = sorted(set.intersection(*(set(value) for value in rows.values())))
    doc_min = getattr(args, "ai2d_doc_min", None)
    doc_max = getattr(args, "ai2d_doc_max", None)
    if doc_min is not None:
        shared = [doc_id for doc_id in shared if doc_id >= doc_min]
    if doc_max is not None:
        shared = [doc_id for doc_id in shared if doc_id < doc_max]
    if args.ai2d_expected is not None and len(shared) != args.ai2d_expected:
        raise ValueError(
            f"expected {args.ai2d_expected} paired AI2D documents, found {len(shared)}"
        )
    output: dict[str, Any] = {"n": len(shared)}
    confidence = np.asarray(
        [target_score(rows["confidence"][i]) for i in shared], dtype=float
    )
    for index, (name, values) in enumerate(rows.items()):
        scores = np.asarray([target_score(values[i]) for i in shared], dtype=float)
        output[name] = float(scores.mean())
        if name != "confidence":
            output[f"{name}_vs_confidence"] = paired_interval(
                confidence,
                scores,
                draws=args.bootstrap,
                seed=args.seed + index,
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai2d-random", type=Path)
    parser.add_argument("--ai2d-confidence", type=Path)
    parser.add_argument("--ai2d-dprm-confidence", type=Path)
    parser.add_argument("--ai2d-dprm-random", type=Path)
    parser.add_argument("--ai2d-expected", type=int, default=500)
    parser.add_argument("--ai2d-doc-min", type=int)
    parser.add_argument("--ai2d-doc-max", type=int)
    parser.add_argument("--rwqa-confidence", type=Path)
    parser.add_argument("--rwqa-dprm", type=Path)
    parser.add_argument("--rwqa-doc-min", type=int, default=256)
    parser.add_argument("--rwqa-doc-max", type=int, default=765)
    parser.add_argument("--chartqa-confidence", type=Path)
    parser.add_argument("--chartqa-dprm", type=Path)
    parser.add_argument("--chartqa-expected", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument(
        "--base-summary",
        type=Path,
        help="optional existing multimodal JSON whose non-LLaDA keys are preserved",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload: dict[str, Any] = (
        json.loads(args.base_summary.read_text(encoding="utf-8"))
        if args.base_summary
        else {}
    )
    existing_llada = payload.get("llada_v", {})
    payload["llada_v"] = existing_llada if isinstance(existing_llada, dict) else {}
    payload["llada_v"]["protocol"] = {
        "bootstrap_draws": args.bootstrap,
        "bootstrap_seed": args.seed,
        "pairing_key": "doc_id",
        "selection": "controller fixed before each declared evaluation interval",
    }
    ai2d = optional_ai2d_orders(args)
    if ai2d:
        payload["llada_v"]["ai2d"] = ai2d

    if bool(args.rwqa_confidence) != bool(args.rwqa_dprm):
        raise ValueError("RealWorldQA requires both confidence and DPRM files")
    if args.rwqa_confidence:
        summary, doc_ids, baseline, method, rows = summarize_pair(
            args.rwqa_confidence,
            args.rwqa_dprm,
            scorer=target_score,
            doc_min=args.rwqa_doc_min,
            doc_max=args.rwqa_doc_max,
            draws=args.bootstrap,
            seed=args.seed + 10,
        )
        formats = np.asarray([classify_answer_format(question(rows[i])) for i in doc_ids])
        summary["by_prompt_format"] = {}
        for offset, label in enumerate(("choice", "numeric", "open")):
            mask = formats == label
            if not np.any(mask):
                continue
            summary["by_prompt_format"][label] = {
                "n": int(mask.sum()),
                "confidence": float(baseline[mask].mean()),
                "dprm_confidence": float(method[mask].mean()),
                "paired_delta": paired_interval(
                    baseline[mask],
                    method[mask],
                    draws=args.bootstrap,
                    seed=args.seed + 11 + offset,
                ),
                "wins": int(np.sum((baseline[mask] == 0.0) & (method[mask] == 1.0))),
                "losses": int(np.sum((baseline[mask] == 1.0) & (method[mask] == 0.0))),
            }
        payload["llada_v"]["realworldqa"] = summary

    if bool(args.chartqa_confidence) != bool(args.chartqa_dprm):
        raise ValueError("ChartQA requires both confidence and DPRM files")
    if args.chartqa_confidence:
        summary, doc_ids, baseline, method, rows = summarize_pair(
            args.chartqa_confidence,
            args.chartqa_dprm,
            scorer=chartqa_score,
            doc_min=None,
            doc_max=None,
            draws=args.bootstrap,
            seed=args.seed + 20,
        )
        if args.chartqa_expected is not None and len(doc_ids) != args.chartqa_expected:
            raise ValueError(
                f"expected {args.chartqa_expected} paired ChartQA documents, "
                f"found {len(doc_ids)}"
            )
        formats = np.asarray([classify_answer_format(question(rows[i])) for i in doc_ids])
        numeric = formats == "numeric"
        if np.any(numeric):
            summary["numeric"] = {
                "n": int(numeric.sum()),
                "confidence": float(baseline[numeric].mean()),
                "dprm_confidence": float(method[numeric].mean()),
                "paired_delta": paired_interval(
                    baseline[numeric],
                    method[numeric],
                    draws=args.bootstrap,
                    seed=args.seed + 21,
                ),
            }
        payload["llada_v"]["chartqa_frozen_transfer"] = summary

    if not payload["llada_v"]:
        raise ValueError("provide at least one complete task pair")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
