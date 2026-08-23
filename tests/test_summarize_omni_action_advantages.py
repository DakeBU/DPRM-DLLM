from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "integrations/omni_diffusion/matched/scripts/summarize_omni_action_advantages.py"
)


def test_summarizes_stage_rank_groups(tmp_path: Path) -> None:
    records = tmp_path / "advantages.json"
    rows = []
    for step, quantile, deltas in [
        (64, 0.9, [(0.1, 0.2), (0.3, -0.1)]),
        (96, 0.95, [(-0.2, 0.1), (0.4, 0.3)]),
    ]:
        for primary, secondary in deltas:
            rows.append(
                {
                    "step": step,
                    "requested_quantile": quantile,
                    "clip_advantage": primary,
                    "clip_b32_advantage": secondary,
                    "confidence_gap_from_default": -0.05,
                }
            )
    records.write_text(json.dumps({"design": "test", "branches": rows}))
    output = tmp_path / "summary.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--records",
            str(records),
            "--output",
            str(output),
            "--resamples",
            "100",
        ],
        check=True,
    )
    result = json.loads(output.read_text())
    assert len(result["groups"]) == 2
    assert result["groups"][0]["step"] == 64
    assert result["groups"][0]["clip_cosine"]["mean_delta"] == 0.2
    assert result["groups"][1]["both_metrics_improved"] == 1
