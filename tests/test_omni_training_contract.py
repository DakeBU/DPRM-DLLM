from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDITOR = REPO_ROOT / "integrations" / "omni_diffusion" / "matched" / "scripts" / "audit_omni_training_contract.py"
OMNI_OVERLAY = REPO_ROOT / "integrations" / "omni_diffusion" / "matched" / "overlay"
ORDERS = ("random_matched", "confidence_matched", "dprm_matched")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_fixture(tmp_path: Path) -> tuple[Path, Path]:
    train_root = tmp_path / "training"
    controller = tmp_path / "controller.json"
    inference_hook = tmp_path / "omni_t2i_smoke.py"
    controller.write_text('{"format":"omni_bucket_table_dprm_v1"}\n')
    inference_hook.write_text("# fixed formal inference hook\n")
    for index, order in enumerate(ORDERS):
        branch = train_root / order
        checkpoint = branch / "checkpoint-1000"
        checkpoint.mkdir(parents=True)
        (checkpoint / "trainer_state.json").write_text('{"global_step": 1000}\n')
        data_json = tmp_path / f"{order}.jsonl"
        data_json.write_text(json.dumps({"policy": order, "index": index}) + "\n")
        data_config = tmp_path / f"{order}.yaml"
        data_config.write_text(f"dataset: {order}\n")
        manifest = {
            "order": order,
            "shared_initial_checkpoint": "/models/shared",
            "shared_checkpoint_index_sha256": "initial-hash",
            "data_config": str(data_config),
            "data_config_sha256": sha256(data_config),
            "data_json": str(data_json),
            "data_json_sha256": sha256(data_json),
            "controller": str(controller),
            "controller_sha256": sha256(controller),
            "trainer_sha256": sha256(
                OMNI_OVERLAY / "tools" / "trainer_v4_51_3.py"
            ),
            "dataset_sha256": sha256(
                OMNI_OVERLAY
                / "omni_diffusion"
                / "data"
                / "dataset_qwen2.py"
            ),
            "order_code_sha256": sha256(REPO_ROOT / "src" / "dprm" / "omni_order.py"),
            "inference_hook": str(inference_hook),
            "inference_hook_sha256": sha256(inference_hook),
            "precomputed_trajectory_mode": "hybrid",
            "current_model_policy_refresh": True,
            "seed": 956,
            "distributed_world_size": 4,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "max_steps": 1000,
            "learning_rate": "1e-5",
            "warmup_ratio": "0.03",
            "trainable_last_n_layers": 2,
            "reveal_budget": 1,
        }
        (branch / "branch_manifest.json").write_text(json.dumps(manifest) + "\n")
    return train_root, controller


def invoke(tmp_path: Path, train_root: Path, controller: Path):
    return subprocess.run(
        [
            sys.executable,
            str(AUDITOR),
            "--train-root",
            str(train_root),
            "--controller",
            str(controller),
            "--step",
            "1000",
            "--omni-root",
            str(OMNI_OVERLAY),
            "--release-root",
            str(REPO_ROOT),
            "--output",
            str(tmp_path / "audit.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_training_contract_accepts_three_matched_branches(tmp_path: Path) -> None:
    train_root, controller = make_fixture(tmp_path)
    result = invoke(tmp_path, train_root, controller)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "audit.json").read_text())
    assert report["passed"] is True
    assert set(report["policy_data_json_sha256"]) == set(ORDERS)
    assert len(set(report["policy_data_json_sha256"].values())) == 3


def test_training_contract_rejects_trainer_mismatch(tmp_path: Path) -> None:
    train_root, controller = make_fixture(tmp_path)
    path = train_root / "dprm_matched" / "branch_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["trainer_sha256"] = "different-trainer"
    path.write_text(json.dumps(manifest) + "\n")
    result = invoke(tmp_path, train_root, controller)
    assert result.returncode != 0
    assert "training-contract mismatch for trainer_sha256" in result.stderr


def test_training_contract_rejects_mutated_trajectory(tmp_path: Path) -> None:
    train_root, controller = make_fixture(tmp_path)
    manifest = json.loads(
        (train_root / "confidence_matched" / "branch_manifest.json").read_text()
    )
    Path(manifest["data_json"]).write_text("mutated\n")
    result = invoke(tmp_path, train_root, controller)
    assert result.returncode != 0
    assert "trajectory JSONL hash mismatch" in result.stderr


def test_training_contract_rejects_mutated_inference_hook(tmp_path: Path) -> None:
    train_root, controller = make_fixture(tmp_path)
    manifest = json.loads(
        (train_root / "dprm_matched" / "branch_manifest.json").read_text()
    )
    Path(manifest["inference_hook"]).write_text("# changed after training\n")
    result = invoke(tmp_path, train_root, controller)
    assert result.returncode != 0
    assert "inference hook changed after training" in result.stderr


def test_training_contract_rejects_incomplete_checkpoint(tmp_path: Path) -> None:
    train_root, controller = make_fixture(tmp_path)
    state = train_root / "random_matched" / "checkpoint-1000" / "trainer_state.json"
    state.write_text('{"global_step": 999}\n')
    result = invoke(tmp_path, train_root, controller)
    assert result.returncode != 0
    assert "checkpoint step mismatch" in result.stderr


def test_training_contract_rejects_mutated_data_config(tmp_path: Path) -> None:
    train_root, controller = make_fixture(tmp_path)
    manifest = json.loads(
        (train_root / "dprm_matched" / "branch_manifest.json").read_text()
    )
    Path(manifest["data_config"]).write_text("mutated: true\n")
    result = invoke(tmp_path, train_root, controller)
    assert result.returncode != 0
    assert "trajectory config hash mismatch" in result.stderr
