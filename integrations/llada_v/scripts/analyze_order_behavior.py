#!/usr/bin/env python3
"""Summarize token-level reveal behavior for the strict RealWorldQA audit."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer


NUMBER_WORDS = {
    "no",
    "none",
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
}


def load_trace(path: Path, doc_min: int, doc_max: int) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            doc_id = int(row["doc_id"])
            if doc_min <= doc_id < doc_max:
                grouped.setdefault(doc_id, []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["step"]))
    return grouped


def decode_step(tokenizer: Any, row: dict[str, Any]) -> str:
    return tokenizer.decode([int(value) for value in row.get("selected_token_ids", [])])


def token_kind(text: str) -> str:
    lowered = text.strip().lower()
    if "eot" in lowered or lowered in {"<eos>", "</s>"}:
        return "eot"
    words = set(re.findall(r"[a-z]+", lowered))
    if re.search(r"\d", lowered) or bool(words & NUMBER_WORDS):
        return "number"
    if re.search(r"[a-z]", lowered):
        return "lexical"
    return "punctuation"


def summarize_doc(tokenizer: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    positions = [int(row["selected_positions"][0]) for row in rows]
    origin = min(positions)
    permutation = tuple(position - origin for position in positions)
    decoded = [decode_step(tokenizer, row) for row in rows]
    kinds = [token_kind(text) for text in decoded]
    number_steps = [index for index, kind in enumerate(kinds) if kind == "number"]
    lexical_steps = [index for index, kind in enumerate(kinds) if kind == "lexical"]
    first_number = number_steps[0] if number_steps else None
    first_lexical = lexical_steps[0] if lexical_steps else None
    return {
        "permutation": permutation,
        "decoded_by_step": decoded,
        "kind_by_step": kinds,
        "first_number_step": first_number,
        "first_lexical_step": first_lexical,
        "lexical_before_number": (
            first_number is not None and first_lexical is not None and first_lexical < first_number
        ),
    }


def bootstrap_mean_delta(values: np.ndarray, seed: int, draws: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=float)
    for draw in range(draws):
        indices = rng.integers(0, len(values), size=len(values))
        estimates[draw] = values[indices].mean()
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
    }


def top_permutations(records: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    counts = Counter(record[key]["permutation"] for record in records)
    return [
        {"permutation": list(permutation), "count": count, "rate": count / len(records)}
        for permutation, count in counts.most_common(6)
    ]


def first_divergence(
    tokenizer: Any,
    baseline_rows: list[dict[str, Any]],
    dprm_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Read the first action disagreement while both policies share the same state."""
    for step, (baseline_row, dprm_row) in enumerate(zip(baseline_rows, dprm_rows)):
        baseline_position = int(baseline_row["selected_positions"][0])
        dprm_position = int(dprm_row["selected_positions"][0])
        if baseline_position == dprm_position:
            continue

        candidates = {
            int(candidate["position"]): candidate
            for candidate in dprm_row.get("dprm_candidates", [])
        }
        if baseline_position not in candidates or dprm_position not in candidates:
            raise RuntimeError(
                f"missing first-divergence candidates at step {step}: "
                f"confidence={baseline_position}, DPRM={dprm_position}"
            )
        confidence_candidate = candidates[baseline_position]
        dprm_candidate = candidates[dprm_position]
        return {
            "step": step,
            "confidence_position": baseline_position,
            "dprm_position": dprm_position,
            "confidence_token": decode_step(tokenizer, baseline_row),
            "dprm_token": decode_step(tokenizer, dprm_row),
            "confidence_action": confidence_candidate,
            "dprm_action": dprm_candidate,
            "local_confidence_delta": float(dprm_candidate["confidence"])
            - float(confidence_candidate["confidence"]),
            "reward_value_delta": float(dprm_candidate["dprm_value"])
            - float(confidence_candidate["dprm_value"]),
            "dprm_score_delta": float(dprm_candidate["dprm_score"])
            - float(confidence_candidate["dprm_score"]),
        }
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-json", type=Path, required=True)
    parser.add_argument("--baseline-trace", type=Path, required=True)
    parser.add_argument("--dprm-trace", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--doc-min", type=int, default=256)
    parser.add_argument("--doc-max", type=int, default=765)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    cases = {
        int(case["doc_id"]): case
        for case in json.loads(args.case_json.read_text())["all_cases"]
        if case["category"] == "numeric"
    }
    baseline_trace = load_trace(args.baseline_trace, args.doc_min, args.doc_max)
    dprm_trace = load_trace(args.dprm_trace, args.doc_min, args.doc_max)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=True
    )

    records = []
    for doc_id in sorted(cases):
        if doc_id not in baseline_trace or doc_id not in dprm_trace:
            raise RuntimeError(f"missing paired trace for document {doc_id}")
        case = cases[doc_id]
        baseline = summarize_doc(tokenizer, baseline_trace[doc_id])
        dprm = summarize_doc(tokenizer, dprm_trace[doc_id])
        records.append(
            {
                "doc_id": doc_id,
                "target": case["target"],
                "baseline_response": case["baseline_response"],
                "dprm_response": case["dprm_response"],
                "baseline_correct": bool(case["baseline_correct"]),
                "dprm_correct": bool(case["dprm_correct"]),
                "outcome": case["outcome"],
                "order_changed": baseline["permutation"] != dprm["permutation"],
                "baseline": baseline,
                "dprm": dprm,
                "first_divergence": first_divergence(
                    tokenizer, baseline_trace[doc_id], dprm_trace[doc_id]
                ),
            }
        )

    changed = [record for record in records if record["order_changed"]]
    paired_number_steps = [
        record
        for record in records
        if record["baseline"]["first_number_step"] is not None
        and record["dprm"]["first_number_step"] is not None
    ]
    number_step_deltas = np.asarray(
        [
            record["dprm"]["first_number_step"] - record["baseline"]["first_number_step"]
            for record in paired_number_steps
        ],
        dtype=float,
    )
    lexical_context_deltas = np.asarray(
        [
            float(record["dprm"]["lexical_before_number"])
            - float(record["baseline"]["lexical_before_number"])
            for record in records
        ],
        dtype=float,
    )
    winning_divergences = [
        record["first_divergence"]
        for record in records
        if record["outcome"] == "win" and record["first_divergence"] is not None
    ]
    payload = {
        "protocol": {
            "doc_interval": [args.doc_min, args.doc_max],
            "category": "prompt-only numeric/count",
            "token_classes": "EOT, numeric digit/number-word, lexical, punctuation",
            "bootstrap_draws": args.bootstrap,
        },
        "documents": len(records),
        "order_changed_documents": len(changed),
        "order_changed_rate": len(changed) / len(records),
        "changed_order_wins": sum(record["outcome"] == "win" for record in changed),
        "changed_order_losses": sum(record["outcome"] == "loss" for record in changed),
        "changed_order_same": sum(record["outcome"] == "same" for record in changed),
        "dprm_only_wins": sum(record["outcome"] == "win" for record in records),
        "lexical_before_number_in_dprm_only_wins": {
            "baseline": sum(
                record["baseline"]["lexical_before_number"]
                for record in records
                if record["outcome"] == "win"
            ),
            "dprm": sum(
                record["dprm"]["lexical_before_number"]
                for record in records
                if record["outcome"] == "win"
            ),
        },
        "paired_numeric_step_documents": len(paired_number_steps),
        "first_number_step_delta": bootstrap_mean_delta(
            number_step_deltas, args.seed, args.bootstrap
        ),
        "lexical_before_number_rate": {
            "baseline": sum(record["baseline"]["lexical_before_number"] for record in records)
            / len(records),
            "dprm": sum(record["dprm"]["lexical_before_number"] for record in records)
            / len(records),
        },
        "lexical_before_number_delta": bootstrap_mean_delta(
            lexical_context_deltas, args.seed + 1, args.bootstrap
        ),
        "top_baseline_permutations": top_permutations(records, "baseline"),
        "top_dprm_permutations": top_permutations(records, "dprm"),
        "dprm_only_win_first_divergence": {
            "documents": len(winning_divergences),
            "mean_local_confidence_delta": float(
                np.mean([row["local_confidence_delta"] for row in winning_divergences])
            ),
            "mean_reward_value_delta": float(
                np.mean([row["reward_value_delta"] for row in winning_divergences])
            ),
            "mean_dprm_score_delta": float(
                np.mean([row["dprm_score_delta"] for row in winning_divergences])
            ),
        },
        "discordant_cases": [
            record for record in records if record["outcome"] in {"win", "loss"}
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")

    delta = payload["first_number_step_delta"]
    context_delta = payload["lexical_before_number_delta"]
    lines = [
        "# RealWorldQA numeric/count reveal behavior",
        "",
        f"- Documents: {payload['documents']}",
        f"- Changed reveal permutation: {payload['order_changed_documents']}/"
        f"{payload['documents']} ({payload['order_changed_rate']:.3f})",
        f"- Outcomes within changed permutations: {payload['changed_order_wins']} wins, "
        f"{payload['changed_order_losses']} losses, {payload['changed_order_same']} ties",
        f"- First numeric-token step delta (DPRM - confidence): {delta['mean']:+.3f} "
        f"[{delta['ci_low']:+.3f}, {delta['ci_high']:+.3f}] over "
        f"{payload['paired_numeric_step_documents']} paired documents",
        f"- Lexical context before the first numeric token: "
        f"{payload['lexical_before_number_rate']['baseline']:.3f} confidence vs "
        f"{payload['lexical_before_number_rate']['dprm']:.3f} DPRM; paired delta "
        f"{context_delta['mean']:+.3f} [{context_delta['ci_low']:+.3f}, "
        f"{context_delta['ci_high']:+.3f}]",
        "",
        "| doc | outcome | target | confidence output | DPRM output | confidence reveal tokens | DPRM reveal tokens |",
        "|---:|---|---|---|---|---|---|",
    ]
    for record in payload["discordant_cases"]:
        lines.append(
            f"| {record['doc_id']} | {record['outcome']} | {record['target']} | "
            f"{record['baseline_response']} | {record['dprm_response']} | "
            f"{record['baseline']['decoded_by_step']} | {record['dprm']['decoded_by_step']} |"
        )
    divergence = payload["dprm_only_win_first_divergence"]
    lines.extend(
        [
            "",
            "## First action divergence in DPRM-only wins",
            "",
            "The two policies have identical committed states immediately before this row.",
            "",
            f"- Mean local-confidence delta (DPRM action - confidence action): "
            f"{divergence['mean_local_confidence_delta']:+.3f}",
            f"- Mean bucket-value delta: {divergence['mean_reward_value_delta']:+.3f}",
            f"- Mean final-score delta: {divergence['mean_dprm_score_delta']:+.3f}",
            "",
            "| doc | step | confidence token | DPRM token | local confidence delta | bucket value delta | final score delta |",
            "|---:|---:|---|---|---:|---:|---:|",
        ]
    )
    for record in payload["discordant_cases"]:
        if record["outcome"] != "win" or record["first_divergence"] is None:
            continue
        row = record["first_divergence"]
        lines.append(
            f"| {record['doc_id']} | {row['step']} | {row['confidence_token']} | "
            f"{row['dprm_token']} | {row['local_confidence_delta']:+.3f} | "
            f"{row['reward_value_delta']:+.3f} | {row['dprm_score_delta']:+.3f} |"
        )
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text("\n".join(lines) + "\n")
    print(json.dumps({key: payload[key] for key in payload if key != "discordant_cases"}, indent=2))


if __name__ == "__main__":
    main()
