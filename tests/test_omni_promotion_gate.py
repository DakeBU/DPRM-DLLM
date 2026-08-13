from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER = REPO_ROOT / "integrations" / "omni_diffusion" / "matched" / "scripts" / "check_omni_matched_promotion.py"


def run_gate(tmp_path: Path, *, ci_low: float) -> dict:
    paired = tmp_path / "paired.json"
    divergence = tmp_path / "divergence.json"
    controller = tmp_path / "controller.json"
    manifest = tmp_path / "run_manifest.json"
    training_audit = tmp_path / "training_contract_audit.json"
    output = tmp_path / "promotion" / "promotion_report.json"
    paired.write_text(
        json.dumps(
            {
                "comparisons_by_metric": {
                    "clip_cosine": [
                        {
                            "baseline": "progressive_confidence",
                            "method": "dprm_confidence_warmup",
                            "matched_prompts": 96,
                            "ci95_low": ci_low,
                            "mean_delta": 0.01,
                        }
                    ],
                    "clip_b32_cosine": [
                        {
                            "baseline": "progressive_confidence",
                            "method": "dprm_confidence_warmup",
                            "matched_prompts": 96,
                            "ci95_low": -0.01,
                            "mean_delta": 0.005,
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    divergence.write_text(
        json.dumps(
            {
                "comparisons": [
                    {
                        "reference": "progressive_confidence",
                        "method": "dprm_confidence_warmup",
                        "moved_position_fraction": 0.1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    controller.write_text(
        json.dumps(
            {
                "metadata": {
                    "deployment_contract": {
                        "paths_per_prompt": 1,
                        "terminal_reward_calls_at_test": 0,
                        "complete_image_selection": False,
                        "fixed_t2i_scaffold": True,
                        "ordered_visual_positions": 256,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    training_audit.write_text(json.dumps({"passed": True}), encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "paths_per_prompt": 1,
                "prompt_offset": 2300,
                "main_figure_prompt_id": "prompt_2300",
                "supplement_figure_prompt_ids": [
                    "prompt_2301",
                    "prompt_2302",
                    "prompt_2303",
                ],
                "test_time_terminal_rollouts": 0,
                "complete_image_selection": False,
                "outcome_ranked_visual_selection": False,
                "aesthetic_scoring": False,
                "evaluation_metrics": ["CLIP-L/14", "CLIP-B/32"],
                "fixed_t2i_scaffold": True,
                "ordered_action_space": "256 visual-code positions; four T2I format tokens fixed",
                "prompt_count": 96,
                "training_contract_audit": str(training_audit),
                "training_contract_audit_sha256": __import__("hashlib")
                .sha256(training_audit.read_bytes())
                .hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--paired",
            str(paired),
            "--divergence",
            str(divergence),
            "--controller",
            str(controller),
            "--run-manifest",
            str(manifest),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def test_promotion_gate_replaces_stale_marker(tmp_path: Path) -> None:
    promotion = tmp_path / "promotion"
    promotion.mkdir()
    failed = promotion / "RESULT_COMPLETE_NOT_PROMOTED"
    failed.write_text("stale\n", encoding="utf-8")
    report = run_gate(tmp_path, ci_low=0.001)
    assert report["passed"] is True
    assert (promotion / "MANUSCRIPT_PROMOTION_READY").is_file()
    assert not failed.exists()
    assert set(report["evidence_sha256"]) == {
        "paired",
        "divergence",
        "controller",
        "run_manifest",
        "training_contract_audit",
    }


def test_failed_gate_cannot_leave_ready_marker(tmp_path: Path) -> None:
    promotion = tmp_path / "promotion"
    promotion.mkdir()
    ready = promotion / "MANUSCRIPT_PROMOTION_READY"
    ready.write_text("stale\n", encoding="utf-8")
    report = run_gate(tmp_path, ci_low=-0.001)
    assert report["passed"] is False
    assert (promotion / "RESULT_COMPLETE_NOT_PROMOTED").is_file()
    assert not ready.exists()
