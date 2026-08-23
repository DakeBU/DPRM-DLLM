import json
from pathlib import Path
import subprocess
import sys


SCRIPT = (
    Path(__file__).parents[1]
    / "integrations/omni_diffusion/matched/scripts/select_omni_online_action_value.py"
)


def row(tmp_path: Path, prompt: str, seed: int, l14: float, b32: float, *, raw: float, default: float) -> dict:
    image = tmp_path / f"{prompt}_{seed}_{l14}.png"
    image.write_bytes(b"image")
    return {
        "prompt": prompt,
        "seed": seed,
        "image_path": str(image),
        "clip_cosine": l14,
        "clip_b32_cosine": b32,
        "counterfactual_override": {
            "applied": True,
            "visual_index": 16,
            "confidence": 0.5,
            "raw_order_score": raw,
            "rank_quantile": 0.9,
            "shared_provisional_visual_token_ids": [11, 12, 13, seed],
            "default_candidate_index": 3,
            "default_sequence_position": 32,
            "default_visual_index": 32,
            "default_confidence": 0.9,
            "default_raw_order_score": default,
            "confidence_gap_from_default": -0.2,
        },
    }


def test_selects_reward_tilted_action_and_reports_uniform_control(tmp_path: Path) -> None:
    confidence = [
        row(tmp_path, "a", 1, 0.10, 0.20, raw=-0.1, default=-0.1),
        row(tmp_path, "b", 2, 0.20, 0.30, raw=-0.1, default=-0.1),
    ]
    q90 = [
        row(tmp_path, "a", 1, 0.16, 0.24, raw=-0.2, default=-0.1),
        row(tmp_path, "b", 2, 0.18, 0.28, raw=-0.2, default=-0.1),
    ]
    q95 = [
        row(tmp_path, "a", 1, 0.12, 0.21, raw=-0.15, default=-0.1),
        row(tmp_path, "b", 2, 0.21, 0.31, raw=-0.15, default=-0.1),
    ]
    records = tmp_path / "records.json"
    records.write_text(
        json.dumps({"confidence": confidence, "step96_q0.90": q90, "step96_q0.95": q95}),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--records",
            str(records),
            "--output-dir",
            str(output),
            "--branch-methods",
            "step96_q0.90",
            "step96_q0.95",
            "--fixed-guidance",
            "1",
            "--reward-scale",
            "0.03",
            "--bootstrap",
            "100",
        ],
        check=True,
    )
    result = json.loads((output / "online_action_value_summary.json").read_text())
    assert result["candidate_rollouts_per_prompt"] == 3
    assert result["selected_evaluation"]["metrics"]["clip_cosine"]["dprm_mean"] == 0.185
    assert result["selected_evaluation"]["metrics"]["clip_cosine"]["uniform_mean"] == 0.16166666666666668
    assert result["selected_evaluation"]["metrics"]["clip_cosine"]["reward_only_mean"] == 0.185
    assert len(result["records"][0]["shared_action_state_sha256"]) == 64
    assert len(list((output / "selected_images").glob("*.png"))) == 2


def test_rejects_forced_branches_from_different_states(tmp_path: Path) -> None:
    confidence = [row(tmp_path, "a", 1, 0.10, 0.20, raw=-0.1, default=-0.1)]
    q90 = [row(tmp_path, "a", 1, 0.16, 0.24, raw=-0.2, default=-0.1)]
    q95 = [row(tmp_path, "a", 1, 0.12, 0.21, raw=-0.15, default=-0.1)]
    q95[0]["counterfactual_override"]["shared_provisional_visual_token_ids"] = [99]
    records = tmp_path / "records.json"
    records.write_text(
        json.dumps({"confidence": confidence, "step96_q0.90": q90, "step96_q0.95": q95}),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--records",
            str(records),
            "--output-dir",
            str(tmp_path / "output"),
            "--branch-methods",
            "step96_q0.90",
            "step96_q0.95",
            "--fixed-guidance",
            "1",
            "--bootstrap",
            "10",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "do not share the same action state" in completed.stderr
