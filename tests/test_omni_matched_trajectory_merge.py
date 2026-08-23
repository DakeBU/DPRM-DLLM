from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MERGER = REPO_ROOT / "integrations" / "omni_diffusion" / "matched" / "scripts" / "merge_omni_matched_trajectories.py"


def trajectory_row(prompt: str, policy: str, step: int) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Generate an image based on the provided text description.\n"
                    + prompt
                ),
            },
            {
                "role": "assistant",
                "content": "<|begin_of_image|><|image_7|><|end_of_image|>",
            },
        ],
        "dprm_source_index": 17,
        "dprm_trajectory_policy": policy,
        "dprm_trajectory_step": step,
        "dprm_next_action_step": step + 1,
        "dprm_revealed_visual_indices": list(range(step + 1)),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def merger_command(
    random_path: Path,
    confidence_path: Path,
    dprm_path: Path,
    output_dir: Path,
    forbidden: Path,
) -> list[str]:
    return [
        sys.executable,
        str(MERGER),
        "--random-shards",
        str(random_path),
        "--confidence-shards",
        str(confidence_path),
        "--dprm-shards",
        str(dprm_path),
        "--output-dir",
        str(output_dir),
        "--forbidden-prompts",
        str(forbidden),
    ]


def test_matched_trajectory_merge_records_pairing_hashes(tmp_path: Path) -> None:
    prompt = "three children playing with toys on a beach"
    random_path = tmp_path / "random.jsonl"
    confidence_path = tmp_path / "confidence.jsonl"
    dprm_path = tmp_path / "dprm.jsonl"
    rows_by_policy = {
        random_path: "random",
        confidence_path: "progressive_confidence",
        dprm_path: "dprm_confidence_warmup",
    }
    for path, policy in rows_by_policy.items():
        write_jsonl(path, [trajectory_row(prompt, policy, step) for step in (31, 63)])
    forbidden = tmp_path / "forbidden.txt"
    forbidden.write_text("unrelated prompt\n", encoding="utf-8")
    output = tmp_path / "merged"

    result = subprocess.run(
        merger_command(random_path, confidence_path, dprm_path, output, forbidden),
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    expected_key_hash = hashlib.sha256(b"17:31\n17:63").hexdigest()
    assert result.returncode == 0
    assert manifest["rows_per_policy"] == 2
    assert manifest["unique_prompts"] == 1
    assert manifest["paired_state_key_count"] == 2
    assert manifest["post_action_checkpoints"] == [31, 63]
    assert manifest["training_next_action_steps"] == [32, 64]
    assert manifest["paired_state_key_sha256"] == expected_key_hash
    assert manifest["policy_pairing_verified"] is True
    assert manifest["clean_target_pairing_verified"] is True
    assert manifest["canvas_divergence"]["dprm_vs_confidence"] == {
        "paired_canvases": 2,
        "different_canvases": 0,
        "different_fraction": 0.0,
        "mean_substituted_positions": 0.0,
        "by_post_action_step": {
            "31": {
                "paired_canvases": 1,
                "different_canvases": 0,
                "different_fraction": 0.0,
                "mean_substituted_positions": 0.0,
                "max_substituted_positions": 0,
            },
            "63": {
                "paired_canvases": 1,
                "different_canvases": 0,
                "different_fraction": 0.0,
                "mean_substituted_positions": 0.0,
                "max_substituted_positions": 0,
            },
        },
    }
    assert len(manifest["paired_clean_target_sha256"]) == 64
    assert manifest["forbidden_prompt_overlap"] == 0


def test_matched_trajectory_merge_rejects_policy_mismatch(tmp_path: Path) -> None:
    prompt = "three children playing with toys on a beach"
    random_path = tmp_path / "random.jsonl"
    confidence_path = tmp_path / "confidence.jsonl"
    dprm_path = tmp_path / "dprm.jsonl"
    write_jsonl(random_path, [trajectory_row(prompt, "random", 31)])
    write_jsonl(confidence_path, [trajectory_row(prompt, "random", 31)])
    write_jsonl(dprm_path, [trajectory_row(prompt, "dprm_confidence_warmup", 31)])
    forbidden = tmp_path / "forbidden.txt"
    forbidden.write_text("unrelated prompt\n", encoding="utf-8")

    result = subprocess.run(
        merger_command(
            random_path, confidence_path, dprm_path, tmp_path / "merged", forbidden
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "trajectory-policy mismatch" in result.stderr


def test_matched_trajectory_merge_rejects_prompt_leakage(tmp_path: Path) -> None:
    prompt = "three children playing with toys on a beach"
    random_path = tmp_path / "random.jsonl"
    confidence_path = tmp_path / "confidence.jsonl"
    dprm_path = tmp_path / "dprm.jsonl"
    write_jsonl(random_path, [trajectory_row(prompt, "random", 31)])
    write_jsonl(
        confidence_path, [trajectory_row(prompt, "progressive_confidence", 31)]
    )
    write_jsonl(
        dprm_path, [trajectory_row(prompt, "dprm_confidence_warmup", 31)]
    )
    forbidden = tmp_path / "forbidden.txt"
    forbidden.write_text(prompt + "\n", encoding="utf-8")

    result = subprocess.run(
        merger_command(
            random_path, confidence_path, dprm_path, tmp_path / "merged", forbidden
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "training/evaluation prompt leakage" in result.stderr


def test_matched_trajectory_merge_rejects_clean_target_mismatch(tmp_path: Path) -> None:
    prompt = "three children playing with toys on a beach"
    random_path = tmp_path / "random.jsonl"
    confidence_path = tmp_path / "confidence.jsonl"
    dprm_path = tmp_path / "dprm.jsonl"
    random_row = trajectory_row(prompt, "random", 31)
    confidence_row = trajectory_row(prompt, "progressive_confidence", 31)
    dprm_row = trajectory_row(prompt, "dprm_confidence_warmup", 31)
    dprm_row["messages"][1]["content"] = (
        "<|begin_of_image|><|image_999|><|end_of_image|>"
    )
    write_jsonl(random_path, [random_row])
    write_jsonl(confidence_path, [confidence_row])
    write_jsonl(dprm_path, [dprm_row])
    forbidden = tmp_path / "forbidden.txt"
    forbidden.write_text("unrelated prompt\n", encoding="utf-8")

    result = subprocess.run(
        merger_command(
            random_path, confidence_path, dprm_path, tmp_path / "merged", forbidden
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "do not share paired prompts, states, and clean targets" in result.stderr
