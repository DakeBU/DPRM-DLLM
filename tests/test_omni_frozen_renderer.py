import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "integrations/omni_diffusion/matched/scripts/render_omni_frozen_latex.py"


def row(metric, baseline, method, baseline_mean, method_mean):
    return {
        "metric": metric,
        "baseline": baseline,
        "method": method,
        "matched_prompts": 489,
        "baseline_mean": baseline_mean,
        "method_mean": method_mean,
        "mean_delta": method_mean - baseline_mean,
    }


def test_frozen_renderer_requires_and_renders_promoted_results(tmp_path):
    clip_l = row("clip_cosine", "progressive_confidence", "dprm_confidence_warmup", 0.25, 0.26)
    clip_b = row("clip_b32_cosine", "progressive_confidence", "dprm_confidence_warmup", 0.31, 0.32)
    random_l = row("clip_cosine", "random", "progressive_confidence", 0.20, 0.25)
    random_b = row("clip_b32_cosine", "random", "progressive_confidence", 0.27, 0.31)
    paired = {"comparisons_by_metric": {"clip_cosine": [clip_l, random_l], "clip_b32_cosine": [clip_b, random_b]}}
    promotion = {"passed": True, "primary": clip_l, "secondary": clip_b}
    paired_path = tmp_path / "paired.json"
    promotion_path = tmp_path / "promotion.json"
    rows_path = tmp_path / "rows.tex"
    aggregate_path = tmp_path / "aggregate.tex"
    multimodal_path = tmp_path / "multimodal.json"
    paired_path.write_text(json.dumps(paired), encoding="utf-8")
    promotion_path.write_text(json.dumps(promotion), encoding="utf-8")
    multimodal_path.write_text(
        json.dumps(
            {
                "llada_v": {
                    "ai2d": {"n": 500, "confidence": 0.658, "dprm_confidence": 0.692},
                    "realworldqa": {"n": 509, "confidence": 0.4735, "dprm_confidence": 0.4892},
                    "chartqa_frozen_transfer": {"n": 500, "confidence": 0.696, "dprm_confidence": 0.710},
                }
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [sys.executable, str(SCRIPT), "--numeric-promotion", str(promotion_path), "--paired", str(paired_path), "--multimodal-summary", str(multimodal_path), "--rows-output", str(rows_path), "--aggregate-output", str(aggregate_path)],
        check=True,
    )
    assert "Omni default & 0.25000 & 0.31000 & 1" in rows_path.read_text()
    assert r"DPRM & \bestcell{0.26000} & \bestcell{0.32000} & 1" in rows_path.read_text()
    assert r"\newcommand{\multimodalaggregaterow}" in aggregate_path.read_text()


def test_frozen_renderer_ignores_pending_lladav_tasks(tmp_path):
    clip_l = row("clip_cosine", "progressive_confidence", "dprm_confidence_warmup", 0.25, 0.26)
    clip_b = row("clip_b32_cosine", "progressive_confidence", "dprm_confidence_warmup", 0.31, 0.32)
    random_l = row("clip_cosine", "random", "progressive_confidence", 0.20, 0.25)
    random_b = row("clip_b32_cosine", "random", "progressive_confidence", 0.27, 0.31)
    paired = {"comparisons_by_metric": {"clip_cosine": [clip_l, random_l], "clip_b32_cosine": [clip_b, random_b]}}
    promotion = {"passed": True, "primary": clip_l, "secondary": clip_b}
    paired_path = tmp_path / "paired.json"
    promotion_path = tmp_path / "promotion.json"
    rows_path = tmp_path / "rows.tex"
    aggregate_path = tmp_path / "aggregate.tex"
    multimodal_path = tmp_path / "multimodal.json"
    paired_path.write_text(json.dumps(paired), encoding="utf-8")
    promotion_path.write_text(json.dumps(promotion), encoding="utf-8")
    multimodal_path.write_text(
        json.dumps(
            {
                "llada_v": {
                    "ai2d": {"status": "formal_pending_clean_rerun"},
                    "realworldqa": {
                        "n": 509,
                        "confidence": 0.4735,
                        "dprm_confidence": 0.4892,
                    },
                    "chartqa_frozen_transfer": {
                        "status": "formal_pending_exact_controller_rerun"
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--numeric-promotion",
            str(promotion_path),
            "--paired",
            str(paired_path),
            "--multimodal-summary",
            str(multimodal_path),
            "--rows-output",
            str(rows_path),
            "--aggregate-output",
            str(aggregate_path),
        ],
        check=True,
    )
    aggregate = aggregate_path.read_text(encoding="utf-8")
    assert "0.96472" in aggregate
    assert r"\bestcell{1.00000}" in aggregate
