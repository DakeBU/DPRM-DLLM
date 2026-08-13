from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RENDERER = REPO_ROOT / "integrations" / "omni_diffusion" / "matched" / "scripts" / "render_omni_matched_latex.py"


def paired_payload() -> dict:
    return {
        "comparisons_by_metric": {
            "clip_cosine": [
                {
                    "baseline": "progressive_confidence",
                    "method": "dprm_confidence_warmup",
                    "matched_prompts": 96,
                    "baseline_mean": 0.25,
                    "method_mean": 0.26,
                },
                {
                    "baseline": "random",
                    "method": "progressive_confidence",
                    "matched_prompts": 96,
                    "baseline_mean": 0.22,
                    "method_mean": 0.25,
                },
            ],
            "clip_b32_cosine": [
                {
                    "baseline": "progressive_confidence",
                    "method": "dprm_confidence_warmup",
                    "matched_prompts": 96,
                    "baseline_mean": 0.24,
                    "method_mean": 0.248,
                },
                {
                    "baseline": "random",
                    "method": "progressive_confidence",
                    "matched_prompts": 96,
                    "baseline_mean": 0.21,
                    "method_mean": 0.24,
                },
            ],
        }
    }


def invoke_renderer(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    promotion = tmp_path / "promotion"
    promotion.mkdir(exist_ok=True)
    paired = tmp_path / "paired.json"
    paired.write_text(json.dumps(paired_payload()), encoding="utf-8")
    report = promotion / "promotion_report.json"
    if report.exists():
        payload = json.loads(report.read_text(encoding="utf-8"))
        payload["evidence_sha256"] = {
            "paired": hashlib.sha256(paired.read_bytes()).hexdigest()
        }
        report.write_text(json.dumps(payload), encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            str(RENDERER),
            "--promotion-dir",
            str(promotion),
            "--paired",
            str(paired),
            "--rows-output",
            str(tmp_path / "rows.tex"),
            "--aggregate-output",
            str(tmp_path / "aggregate.tex"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_renderer_refuses_result_without_promotion_marker(tmp_path: Path) -> None:
    promotion = tmp_path / "promotion"
    promotion.mkdir()
    (promotion / "promotion_report.json").write_text(
        json.dumps({"passed": True}), encoding="utf-8"
    )
    result = invoke_renderer(tmp_path)
    assert result.returncode != 0
    assert "refusing to render unpromoted Omni results" in result.stderr
    assert not (tmp_path / "rows.tex").exists()


def test_renderer_emits_table_macros_after_promotion(tmp_path: Path) -> None:
    promotion = tmp_path / "promotion"
    promotion.mkdir()
    (promotion / "promotion_report.json").write_text(
        json.dumps({"passed": True}), encoding="utf-8"
    )
    (promotion / "MANUSCRIPT_PROMOTION_READY").write_text("\n", encoding="utf-8")
    result = invoke_renderer(tmp_path)
    assert result.returncode == 0, result.stderr
    rows = (tmp_path / "rows.tex").read_text(encoding="utf-8")
    aggregate = (tmp_path / "aggregate.tex").read_text(encoding="utf-8")
    assert r"\newcommand{\omnimatchedrows}" in rows
    assert r"DPRM & \bestcell{0.26000} & \bestcell{0.24800} & 1 \\" in rows
    assert r"\newcommand{\multimodalaggregaterow}" in aggregate
