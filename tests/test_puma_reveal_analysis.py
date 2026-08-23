from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "integrations" / "puma" / "scripts" / "analyze_reveal_order.py"


def write_trace(path: Path, *, correct: bool, steps: list[dict]) -> None:
    row = {
        "index": 7,
        "correct": correct,
        "prompt": "test",
        "gold_answer": "3",
        "code": "result = 3",
        "trace_steps": steps,
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_puma_content_diagnostics_and_case_export(tmp_path: Path) -> None:
    confidence = tmp_path / "confidence.jsonl"
    dprm = tmp_path / "dprm.jsonl"
    output = tmp_path / "summary.json"
    cases = tmp_path / "cases.jsonl"
    write_trace(
        confidence,
        correct=False,
        steps=[
            {"step": 0, "selected_positions": [2, 3], "selected_token_texts": ["a", "b"]},
            {"step": 1, "selected_positions": [4, 5], "selected_token_texts": ["1", "c"]},
            {"step": 2, "selected_positions": [8], "selected_token_texts": ["<|endoftext|>"]},
        ],
    )
    write_trace(
        dprm,
        correct=True,
        steps=[
            {"step": 0, "selected_positions": [2, 12], "selected_token_texts": ["a", "b"]},
            {"step": 1, "selected_positions": [1, 9], "selected_token_texts": ["c", "d"]},
            {"step": 2, "selected_positions": [4, 8], "selected_token_texts": ["1", "e"]},
        ],
    )

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--confidence",
            str(confidence),
            "--dprm",
            str(dprm),
            "--output",
            str(output),
            "--case-output",
            str(cases),
            "--bootstrap-iters",
            "100",
        ],
        check=True,
    )
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["accuracy"]["mean_delta"] == 1.0
    assert summary["accuracy"]["dprm_only_wins"] == 1
    assert summary["order_metrics"]["same_step_nonlocal_rate"]["mean_delta"] > 0
    assert summary["order_metrics"]["backfill_step_rate"]["mean_delta"] > 0
    assert len(cases.read_text(encoding="utf-8").splitlines()) == 1
