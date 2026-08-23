import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "integrations/omni_diffusion/matched/scripts/publish_omni_online_results.py"
)


def test_publishes_only_five_path_confirmation_with_random_control(tmp_path: Path) -> None:
    delta = {"mean_delta": 0.02, "ci95_low": 0.01, "ci95_high": 0.03}
    summary = {
        "format": "omni_online_rank_bucket_dprm_v1",
        "prompt_count": 2,
        "candidate_rollouts_per_prompt": 5,
        "candidate_methods": [
            "confidence",
            "step96_q0.70",
            "step96_q0.85",
            "step96_q0.90",
            "step96_q0.95",
        ],
        "reward_scale": 0.03,
        "selected_guidance": 4.0,
        "selection_metric": "clip_cosine",
        "independent_check_metric": "clip_b32_cosine",
        "records": [
            {"shared_action_state_verification": "full_canvas:4/4"},
            {"shared_action_state_verification": "full_canvas:4/4"},
        ],
        "selected_evaluation": {
            "selected_action_counts": {"confidence": 1, "step96_q0.90": 1},
            "override_fraction": 0.5,
            "metrics": {
                "clip_cosine": {
                        "confidence_mean": 0.20,
                        "uniform_mean": 0.19,
                        "reward_only_mean": 0.221,
                        "dprm_mean": 0.22,
                    "dprm_minus_confidence": delta,
                },
                "clip_b32_cosine": {
                        "confidence_mean": 0.30,
                        "uniform_mean": 0.29,
                        "reward_only_mean": 0.311,
                        "dprm_mean": 0.31,
                    "dprm_minus_confidence": delta,
                },
            },
        },
    }
    def scored_rows(l14: tuple[float, float], b32: tuple[float, float]) -> list[dict]:
        return [
            {"prompt": "a", "seed": 1, "clip_cosine": l14[0], "clip_b32_cosine": b32[0]},
            {"prompt": "b", "seed": 2, "clip_cosine": l14[1], "clip_b32_cosine": b32[1]},
        ]

    scored = {"random": scored_rows((0.1, 0.2), (0.2, 0.3))}
    scored["confidence"] = scored_rows((0.1, 0.3), (0.2, 0.4))
    for method in summary["candidate_methods"][1:]:
        scored[method] = scored_rows((0.1, 0.3), (0.2, 0.4))
    manifest = {
        "prompt_count": 2,
        "candidate_rollouts_per_prompt": 5,
        "fixed_guidance": 4.0,
        "random_control": True,
        "checkpoint": "checkpoint-1000",
        "prompt_file_sha256": "abc",
        "action_step": 96,
        "confidence_rank_quantiles": "0.70 0.85 0.90 0.95",
    }
    paths = {}
    for name, payload in (("summary", summary), ("scored", scored), ("manifest", manifest)):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(payload), encoding="utf-8")
    output_json = tmp_path / "release.json"
    output_tex = tmp_path / "rows.tex"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--summary",
            str(paths["summary"]),
            "--scored-records",
            str(paths["scored"]),
            "--run-manifest",
            str(paths["manifest"]),
            "--output-json",
            str(output_json),
            "--output-tex",
            str(output_tex),
            "--expected-prompts",
            "2",
        ],
        check=True,
    )
    release = json.loads(output_json.read_text())
    assert release["methods"]["dprm"]["paths"] == 5
    assert release["methods"]["random"]["clip_l14"] == pytest.approx(0.15)
    assert "\\bestcell{0.22000}" in output_tex.read_text()
