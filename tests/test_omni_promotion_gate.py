from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER = REPO_ROOT / "integrations" / "omni_diffusion" / "matched" / "scripts" / "check_omni_matched_promotion.py"


def run_gate(
    tmp_path: Path,
    *,
    ci_low: float,
    expected_prompts: int = 96,
    external_prompts: bool = False,
) -> dict:
    paired = tmp_path / "paired.json"
    divergence = tmp_path / "divergence.json"
    controller = tmp_path / "controller.json"
    manifest = tmp_path / "run_manifest.json"
    training_audit = tmp_path / "training_contract_audit.json"
    visual_manifest = tmp_path / "visual_prompts.json"
    visual_validation = tmp_path / "visual_prompt_validation.json"
    output = tmp_path / "promotion" / "promotion_report.json"
    paired.write_text(
        json.dumps(
            {
                "comparisons_by_metric": {
                    "clip_cosine": [
                        {
                            "baseline": "progressive_confidence",
                            "method": "dprm_confidence_warmup",
                            "matched_prompts": expected_prompts,
                            "ci95_low": ci_low,
                            "mean_delta": 0.01,
                        }
                    ],
                    "clip_b32_cosine": [
                        {
                            "baseline": "progressive_confidence",
                            "method": "dprm_confidence_warmup",
                            "matched_prompts": expected_prompts,
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
                        "direct_override_fraction": 0.01,
                        "has_direct_override": 0.5,
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
    visual_manifest.write_text(json.dumps({"prompts": [2300, 2301, 2302, 2303]}), encoding="utf-8")
    visual_validation.write_text(
        json.dumps(
            {
                "passed": True,
                "prompt_count": expected_prompts,
                "unique_prompt_count": expected_prompts,
                "unique_prompt_id_count": expected_prompts,
            }
        ),
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            {
                "paths_per_prompt": 1,
                "prompt_offset": 2300,
                "prompt_jsonl": str(visual_manifest) if external_prompts else "",
                "fixed_visual_prompt_ids": [] if external_prompts else [
                    "prompt_2300", "prompt_2301", "prompt_2302", "prompt_2303"
                ],
                "test_time_terminal_rollouts": 0,
                "complete_image_selection": False,
                "outcome_ranked_visual_selection": False,
                "aesthetic_scoring": False,
                "evaluation_metrics": ["CLIP-L/14", "CLIP-B/32"],
                "fixed_t2i_scaffold": True,
                "ordered_action_space": "256 visual-code positions; four T2I format tokens fixed",
                "prompt_count": expected_prompts,
                "training_contract_audit": str(training_audit),
                "training_contract_audit_sha256": __import__("hashlib")
                .sha256(training_audit.read_bytes())
                .hexdigest(),
                "visual_prompt_preregistration": str(visual_manifest),
                "visual_prompt_preregistration_sha256": hashlib.sha256(
                    visual_manifest.read_bytes()
                ).hexdigest(),
                "visual_prompt_validation": str(visual_validation),
                "visual_prompt_validation_sha256": hashlib.sha256(
                    visual_validation.read_bytes()
                ).hexdigest(),
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
            "--expected-prompts",
            str(expected_prompts),
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
        "visual_prompt_preregistration",
        "visual_prompt_validation",
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


def test_promotion_gate_accepts_complete_512_prompt_confirmation(
    tmp_path: Path,
) -> None:
    report = run_gate(tmp_path, ci_low=0.001, expected_prompts=512)
    assert report["passed"] is True
    assert report["clip_l14"]["matched_prompts"] == 512


def test_promotion_gate_accepts_frozen_external_prompt_split(tmp_path: Path) -> None:
    report = run_gate(
        tmp_path,
        ci_low=0.001,
        expected_prompts=512,
        external_prompts=True,
    )
    assert report["passed"] is True
    assert report["checks"]["visual_or_external_prompts_are_preregistered"] is True
