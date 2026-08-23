import json
import subprocess
import sys
from pathlib import Path

from dprm.omni_order import OmniStageRankCodeDPRM, OmniStageRankSpatialDPRM


def test_fits_action_conditioned_stage_rank_spatial_controller(tmp_path: Path):
    records = {
        "branches": [
            {
                "applied": True,
                "step": 96,
                "rank_bin": 6,
                "visual_index": 0,
                "advantage": 0.4,
            },
            {
                "applied": True,
                "step": 96,
                "rank_bin": 6,
                "visual_index": 7,
                "advantage": 0.2,
            },
            {
                "applied": True,
                "step": 128,
                "rank_bin": 5,
                "visual_index": 255,
                "advantage": -0.1,
            },
        ]
    }
    source = tmp_path / "records.json"
    output = tmp_path / "controller.json"
    source.write_text(json.dumps(records), encoding="utf-8")
    script = (
        Path(__file__).resolve().parents[1]
        / "integrations/omni_diffusion/matched/scripts/fit_omni_action_bucket_controller.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--records",
            str(source),
            "--output",
            str(output),
            "--active-steps",
            "96",
            "128",
            "--spatial-bins",
            "4",
            "--shrinkage",
            "0",
        ],
        check=True,
    )

    controller, metadata = OmniStageRankSpatialDPRM.load_artifact(output)
    assert controller.active_steps == (96, 128)
    assert controller.counts[0][6][0] == 2
    assert controller.reward_values[0][6][0] > 0
    assert controller.reward_values[1][5][3] < 0
    assert metadata["accepted_records"] == 3
    assert metadata["deployment_contract"]["terminal_reward_calls_at_test"] == 0


def test_rebins_continuous_rank_quantile(tmp_path: Path):
    source = tmp_path / "records.json"
    source.write_text(
        json.dumps(
            {
                "branches": [
                    {
                        "applied": True,
                        "step": 96,
                        "rank_bin": 7,
                        "rank_quantile": 0.98,
                        "visual_index": 20,
                        "advantage": 1.0,
                        "prompt": "p",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "controller.json"
    script = (
        Path(__file__).resolve().parents[1]
        / "integrations/omni_diffusion/matched/scripts/fit_omni_action_bucket_controller.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--records",
            str(source),
            "--output",
            str(output),
            "--active-steps",
            "96",
            "--rank-bins",
            "64",
            "--spatial-bins",
            "1",
        ],
        check=True,
    )
    payload = json.loads(output.read_text())
    assert payload["config"]["counts"][0][62][0] == 1
    assert payload["config"]["counts"][0][63][0] == 0


def test_fits_stage_rank_code_controller(tmp_path: Path):
    source = tmp_path / "records.json"
    source.write_text(
        json.dumps(
            {
                "branches": [
                    {
                        "step": 64,
                        "rank_quantile": 0.9,
                        "provisional_token_id": 168072 + 7000,
                        "advantage": 0.5,
                        "prompt": "p1",
                    },
                    {
                        "step": 64,
                        "rank_quantile": 0.9,
                        "provisional_token_id": 168072 + 7100,
                        "advantage": 0.3,
                        "prompt": "p2",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "code_controller.json"
    script = (
        Path(__file__).resolve().parents[1]
        / "integrations/omni_diffusion/matched/scripts/fit_omni_action_code_controller.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--records",
            str(source),
            "--output",
            str(output),
            "--active-steps",
            "64",
            "--min-count",
            "2",
            "--shrinkage",
            "0",
        ],
        check=True,
    )
    controller, metadata = OmniStageRankCodeDPRM.load_artifact(output)
    assert controller.counts[0][57][3] == 2
    assert controller.reward_values[0][57][3] > 0
    assert metadata["positive_ready_buckets"] == 1
