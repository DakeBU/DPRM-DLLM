#!/usr/bin/env python3
"""Recompute the paper's Prism metrics from released question-level records."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def last_boxed(text: str) -> str | None:
    index = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if index < 0:
        return None
    if "\\boxed " in text[index : index + 8] and "{" not in text[index : index + 8]:
        return "\\boxed " + text[index:].split("\\boxed ")[-1].split("$")[0].strip()
    depth = 0
    for right in range(index, len(text)):
        if text[right] == "{":
            depth += 1
        elif text[right] == "}":
            depth -= 1
            if depth == 0:
                return text[index : right + 1]
    return None


def remove_box(text: str | None) -> str | None:
    if not text:
        return None
    if text.startswith("\\boxed "):
        return text[len("\\boxed ") :]
    for prefix in ("\\boxed{", "\\fbox{"):
        if text.startswith(prefix) and text.endswith("}"):
            return text[len(prefix) : -1]
    return text


def strip_answer(value: Any) -> str:
    text = str(value or "").strip()
    while re.search(r"(\d),(\d{3})", text):
        text = re.sub(r"(\d),(\d{3})", r"\1\2", text)
    text = text.replace("\n", "").replace("\\!", "")
    text = text.replace("tfrac", "frac").replace("dfrac", "frac")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("^{\\circ}", "").replace("^\\circ", "")
    text = text.replace("\\$", "").replace("\\%", "").replace("\%", "")
    if "=" in text and len(text.split("=")[0]) <= 5:
        text = text.split("=", 1)[1].strip()
    return text.replace(" ", "").rstrip(".")


def normalize(value: Any) -> float | str:
    text = strip_answer(value)
    try:
        parts = text.split("/")
        if len(parts) == 2:
            return float(parts[0]) / float(parts[1])
        return float(text)
    except (TypeError, ValueError, ZeroDivisionError):
        return text


def extract_answer(text: str) -> str:
    text = (text or "").replace("<|role_end|>", "").replace("</s>", "").strip()
    boxed = remove_box(last_boxed(text))
    if boxed:
        return strip_answer(boxed)
    tag = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if tag:
        return strip_answer(tag.group(1))
    tail = text[-200:]
    marker = "the answer is"
    if marker in tail.lower():
        after = tail[tail.lower().rfind(marker) + len(marker) :].strip()
        return strip_answer(re.split(r"[.\n]", after)[0].replace(":", "").replace("$", ""))
    numbers = re.findall(r"(?<!\d)-?\d+\.?\d*(?!\d)", text[-50:])
    return strip_answer(numbers[-1]) if numbers else ""


def equivalent(prediction: Any, target: Any) -> bool:
    left, right = normalize(prediction), normalize(target)
    if isinstance(left, float) and isinstance(right, float):
        return math.isclose(left, right, rel_tol=1e-4)
    return str(left) == str(right)


def score_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    target_text = str(record.get("target", ""))
    target = strip_answer(target_text.split("####")[-1] if "####" in target_text else target_text)
    trajectories = record.get("all_trajectories", [])
    if not trajectories:
        trajectories = [
            {"resp": item[0] if isinstance(item, list) else item, "score": 0.0, "rank": rank}
            for rank, item in enumerate(record.get("resps", []), start=1)
        ]
    parsed = []
    rank_correct = [False, False, False, False]
    for trajectory in trajectories:
        answer = extract_answer(str(trajectory.get("resp", "")))
        if not answer:
            continue
        rank = int(trajectory.get("rank", 0))
        correct = equivalent(answer, target)
        if 1 <= rank <= 4:
            rank_correct[rank - 1] = correct
        parsed.append(
            {
                "answer": answer,
                "value": normalize(answer),
                "score": float(trajectory.get("score", 0.0)),
            }
        )

    voted_correct = False
    if parsed:
        parsed.sort(key=lambda item: item["score"], reverse=True)
        voters = parsed[: max(1, int(len(parsed) * 0.6))]
        votes: dict[float | str, dict[str, Any]] = defaultdict(
            lambda: {"weight": 0.0, "max_score": -float("inf"), "answer": ""}
        )
        for item in voters:
            cell = votes[item["value"]]
            cell["weight"] += math.exp(item["score"])
            if item["score"] > cell["max_score"]:
                cell["max_score"] = item["score"]
                cell["answer"] = item["answer"]
        winner = max(votes.values(), key=lambda item: (item["weight"], item["max_score"]))
        voted_correct = equivalent(winner["answer"], target)

    return {
        "key": str(record.get("doc", {}).get("question", index)),
        "voted_correct": voted_correct,
        "rank1_correct": rank_correct[0],
        "any4_correct": any(rank_correct),
        "rank_correct": rank_correct,
        "nfe": int(record.get("nfe", 0)),
        "svf_calls": int(record.get("svf_calls", 0)),
    }


def load(path: Path) -> list[dict[str, Any]]:
    return [
        score_record(json.loads(line), index)
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines())
        if line.strip()
    ]


def means(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "voted_accuracy": float(np.mean([row["voted_correct"] for row in rows])),
        "rank1_accuracy": float(np.mean([row["rank1_correct"] for row in rows])),
        "any4_accuracy": float(np.mean([row["any4_correct"] for row in rows])),
        "rank_accuracy": [
            float(np.mean([row["rank_correct"][rank] for row in rows])) for rank in range(4)
        ],
        "mean_nfe": float(np.mean([row["nfe"] for row in rows])),
        "mean_svf_calls": float(np.mean([row["svf_calls"] for row in rows])),
    }


def paired_intervals(
    baseline: list[dict[str, Any]], method: list[dict[str, Any]], draws: int, seed: int
) -> dict[str, Any]:
    base = {row["key"]: row for row in baseline}
    treatment = {row["key"]: row for row in method}
    if set(base) != set(treatment):
        raise ValueError("baseline and method question sets differ")
    keys = sorted(base)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(keys), size=(draws, len(keys)))
    output = {}
    for field in ("voted_correct", "rank1_correct", "any4_correct", "nfe"):
        delta = np.asarray(
            [float(treatment[key][field]) - float(base[key][field]) for key in keys]
        )
        sampled = delta[indices].mean(axis=1)
        output[field] = {
            "method_minus_confidence": float(delta.mean()),
            "ci95": [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))],
            "wins": int((delta > 0).sum()),
            "ties": int((delta == 0).sum()),
            "losses": int((delta < 0).sum()),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confidence", type=Path, required=True)
    parser.add_argument("--dprm", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=956)
    args = parser.parse_args()

    confidence, dprm = load(args.confidence), load(args.dprm)
    if len(confidence) != 1319 or len(dprm) != 1319:
        raise ValueError("the paper protocol requires 1,319 GSM8K questions per method")
    result = {
        "confidence": means(confidence),
        "dprm": means(dprm),
        "paired_bootstrap": paired_intervals(confidence, dprm, args.draws, args.seed),
        "bootstrap_draws": args.draws,
        "bootstrap_seed": args.seed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("confidence", confidence), ("dprm", dprm)):
        with (args.output_dir / f"{name}_per_question.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
