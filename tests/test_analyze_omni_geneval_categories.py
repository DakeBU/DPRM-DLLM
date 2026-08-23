from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO_ROOT
    / "integrations/omni_diffusion/matched/scripts/analyze_omni_geneval_categories.py"
)


def test_reports_paired_category_and_overall_effects(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "one", "tag": "single_object"}),
                json.dumps({"prompt": "two", "tag": "two_object"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    records = tmp_path / "records.json"
    records.write_text(
        json.dumps(
            {
                "progressive_confidence": [
                    {"prompt_id": "p0", "prompt": "one", "clip_cosine": 0.2},
                    {"prompt_id": "p1", "prompt": "two", "clip_cosine": 0.3},
                ],
                "dprm_confidence_warmup": [
                    {"prompt_id": "p0", "prompt": "one", "clip_cosine": 0.2},
                    {"prompt_id": "p1", "prompt": "two", "clip_cosine": 0.4},
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "summary.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--records",
            str(records),
            "--metadata",
            str(metadata),
            "--output",
            str(output),
            "--metrics",
            "clip_cosine",
            "--bootstrap-samples",
            "100",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    rows = {(row["group_type"], row["group"]): row for row in payload["rows"]}
    assert rows[("all", "all")]["mean_delta"] == pytest.approx(0.05)
    assert rows[("tag", "single_object")]["mean_delta"] == 0.0
    assert rows[("tag", "two_object")]["mean_delta"] == pytest.approx(0.1)
