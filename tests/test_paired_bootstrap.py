from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "paired_bootstrap.py"


def write_jsonl(path: Path, values: list[int]) -> None:
    path.write_text(
        "".join(
            json.dumps({"example": index, "metrics": {"correct": value}}) + "\n"
            for index, value in enumerate(values)
        ),
        encoding="utf-8",
    )


def test_paired_bootstrap_from_nested_jsonl_fields(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    method = tmp_path / "method.jsonl"
    output = tmp_path / "summary.json"
    write_jsonl(baseline, [0, 0, 1, 1])
    write_jsonl(method, [1, 0, 1, 1])

    result = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--baseline", str(baseline), "--method", str(method),
            "--key", "example", "--value", "metrics.correct",
            "--scale", "100", "--output", str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["n"] == 4
    assert payload["baseline_mean"] == 50.0
    assert payload["method_mean"] == 75.0
    assert payload["benefit_delta"] == 25.0
    assert payload["wins"] == 1
    assert payload["ties"] == 3


def test_paired_bootstrap_rejects_unmatched_keys(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    method = tmp_path / "method.jsonl"
    output = tmp_path / "summary.json"
    write_jsonl(baseline, [0, 1])
    method.write_text('{"example": 8, "correct": 1}\n', encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--baseline", str(baseline), "--method", str(method),
            "--key", "example", "--value", "correct",
            "--baseline-value", "metrics.correct", "--output", str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "different keys" in result.stderr
