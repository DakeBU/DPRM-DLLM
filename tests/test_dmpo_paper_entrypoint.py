from __future__ import annotations

import os
import subprocess
import sys
import importlib.util
from pathlib import Path

import torch
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "integrations" / "dmpo" / "overlay" / "DMPO" / "run_paper_dmpo.sh"
HOST_LAUNCHER = ROOT / "integrations" / "dmpo" / "overlay" / "DMPO" / "run_dmpo.sh"
COMPAT_ENTRYPOINT = ROOT / "integrations" / "dmpo" / "overlay" / "DMPO" / "dmpo_train_compat.py"
COMPAT_MODULE = ROOT / "integrations" / "dmpo" / "overlay" / "transformers_compat.py"
FAST_SAMPLER = ROOT / "integrations" / "dmpo" / "overlay" / "fast_samplers" / "fast_dllm" / "generate.py"
MATCHED_ARGS = ROOT / "integrations" / "dmpo" / "scripts" / "verify_matched_training_args.py"


def dry_run(task: str, policy: str) -> dict[str, str]:
    env = os.environ.copy()
    env["DPRM_DRY_RUN"] = "1"
    completed = subprocess.run(
        ["bash", str(SCRIPT), task, policy],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return dict(line.split("=", 1) for line in completed.stdout.splitlines())


def test_math_dprm_paper_configuration() -> None:
    config = dry_run("math", "dprm_confidence")
    assert config["LEARNING_RATE"] == "3e-6"
    assert config["GRADIENT_ACCUMULATION_STEPS"] == "4"
    assert config["NUM_GENERATIONS"] == "8"
    assert config["LOSS_PROGRESSIVE_ORDER_POLICY"] == "dprm_soft_bon"
    assert config["LOSS_PROGRESSIVE_DPRM_WARMUP_POLICY"] == "confidence"
    assert config["SAMPLER_REMASKING"] == "dprm_soft_bon"


def test_countdown_random_warmup_paper_configuration() -> None:
    config = dry_run("countdown", "dprm_random")
    assert config["LEARNING_RATE"] == "1e-6"
    assert config["GRADIENT_ACCUMULATION_STEPS"] == "2"
    assert config["LOSS_PROGRESSIVE_DPRM_WARMUP_POLICY"] == "random"
    assert config["LOSS_PROGRESSIVE_THRESHOLD"] == "0.9"


def test_random_masking_uses_host_low_confidence_decoder() -> None:
    config = dry_run("gsm8k", "random")
    assert config["LOSS_MASK_SAMPLER"] == "random"
    assert config["SAMPLER_REMASKING"] == "low_confidence"


def test_host_launcher_uses_versioned_transformers_compatibility_entrypoint() -> None:
    launcher = HOST_LAUNCHER.read_text(encoding="utf-8")
    compatibility = COMPAT_ENTRYPOINT.read_text(encoding="utf-8")
    compatibility_module = COMPAT_MODULE.read_text(encoding="utf-8")
    assert '"${LAUNCHER[@]}" dmpo_train_compat.py' in launcher
    assert 'install_llada_tp_plan_guard()' in compatibility
    assert 'model._tp_plan = {}' in compatibility_module
    assert 'runpy.run_path' in compatibility


def test_eval_entrypoints_install_transformers_guard() -> None:
    eval_root = ROOT / "integrations" / "dmpo" / "overlay" / "eval"
    for name in (
        "eval.py",
        "eval_passk_gsm8k_single.py",
        "eval_passk_math_single.py",
        "eval_passk_countdown_single.py",
    ):
        source = (eval_root / name).read_text(encoding="utf-8")
        assert "from transformers_compat import install_llada_tp_plan_guard" in source
        assert "install_llada_tp_plan_guard()" in source


def test_fast_sampler_executes_dprm_remasking_branch() -> None:
    overlay = ROOT / "integrations" / "dmpo" / "overlay"
    sys.path.insert(0, str(overlay))
    try:
        spec = importlib.util.spec_from_file_location("released_fast_dllm", FAST_SAMPLER)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        from dprm_guidance import OnlineDPRMEstimator

        estimator = OnlineDPRMEstimator(
            num_phases=1,
            num_bins=2,
            reward_temperature=1.0,
            dprm_lambda=4.0,
            ready_count=1,
            mode="analytic",
        )
        estimator.counts[:] = 1
        estimator.exp_reward_sum[:] = torch.tensor([[4.0, 1.0]])
        logits = torch.log(torch.tensor([[[0.4, 0.3, 0.3], [0.8, 0.1, 0.1]]]))
        x = torch.zeros((1, 2), dtype=torch.long)
        selected_tokens, transfer = module.get_transfer_index(
            logits,
            temperature=0.0,
            remasking="dprm_soft_bon",
            mask_index=torch.ones((1, 2), dtype=torch.bool),
            x=x,
            num_transfer_tokens=torch.tensor([1]),
            dprm_estimator=estimator,
            dprm_phase=torch.tensor([0]),
            dprm_global_step=5000,
            dprm_force_full=True,
        )
        assert selected_tokens.shape == x.shape
        assert transfer.tolist() == [[True, False]]
    finally:
        sys.path.remove(str(overlay))


def test_matched_training_args_accept_only_order_policy_difference(tmp_path: Path) -> None:
    shared = {
        "seed": 42,
        "max_steps": 5000,
        "learning_rate": 1e-6,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "generation_batch_size": 4,
        "num_generations": 8,
        "num_iterations": 8,
        "num_replicates": 2,
        "compute_ref_log_prob_elbo_size": 2,
        "alpha": 0.04,
        "sampler_steps": 128,
        "temperature": 0.2,
        "loss_mask_sampler": "progressive",
        "loss_progressive_k": 8,
        "loss_progressive_phase_init": "random",
        "loss_progressive_threshold": 0.9,
    }
    confidence = tmp_path / "confidence"
    dprm = tmp_path / "dprm"
    confidence.mkdir()
    dprm.mkdir()
    torch.save(
        SimpleNamespace(**shared, loss_progressive_order_policy="confidence"),
        confidence / "training_args.bin",
    )
    torch.save(
        SimpleNamespace(**shared, loss_progressive_order_policy="dprm_soft_bon"),
        dprm / "training_args.bin",
    )
    output = tmp_path / "matched.json"
    subprocess.run(
        [
            sys.executable,
            str(MATCHED_ARGS),
            "--source-root",
            str(tmp_path),
            "--confidence",
            str(confidence),
            "--dprm",
            str(dprm),
            "--output",
            str(output),
        ],
        check=True,
    )
    assert __import__("json").loads(output.read_text())["matched"] is True
