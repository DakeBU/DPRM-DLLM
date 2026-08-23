from pathlib import Path

import torch

from dprm.controller import DPRMConfig, OnlineDPRMController


ROOT = Path(__file__).resolve().parents[1]


def test_sdpo_reported_phase_count_is_the_executable_default():
    launcher = (
        ROOT / "integrations/sdpo/overlay/scripts/run_sdpo_dna_variant.sh"
    ).read_text(encoding="utf-8")
    evaluator = (
        ROOT / "integrations/sdpo/overlay/scripts/run_sdpo_dna_eval_compare.sh"
    ).read_text(encoding="utf-8")
    train = (ROOT / "integrations/sdpo/overlay/finetune_sdpo.py").read_text(
        encoding="utf-8"
    )
    eval_bootstrap = (
        ROOT / "integrations/sdpo/overlay/eval_dna_bootstrap.py"
    ).read_text(encoding="utf-8")

    assert "DPRM_PHASE_BINS=${DPRM_PHASE_BINS:-1}" in launcher
    assert "DPRM_PHASE_BINS=${DPRM_PHASE_BINS:-1}" in evaluator
    assert '--dprm_phase_bins "${DPRM_PHASE_BINS}"' in evaluator
    assert "--dprm_phase_bins', type=int, default=1" in train
    assert '"--dprm_phase_bins", type=int, default=1' in eval_bootstrap


def test_puma_paper_configs_enable_the_trained_sampled_shortlist():
    config_root = ROOT / "integrations/puma/overlay/yaml_files"
    for name in (
        "tinygsm_puma_dprm.yaml",
        "tinygsm_puma_dprm_random.yaml",
        "tinygsm_block_puma_dprm.yaml",
        "sudoku_puma_dprm.yaml",
    ):
        text = (config_root / name).read_text(encoding="utf-8")
        assert "sampled_shortlist: true" in text


def test_puma_paper_config_fixes_training_and_decode_settings():
    text = (
        ROOT / "integrations/puma/overlay/yaml_files/tinygsm_puma_dprm.yaml"
    ).read_text(encoding="utf-8")
    for value in (
        "num_bins: 16",
        "reward_beta: 1.0",
        "warmup_steps: 2000",
        "switch_steps: 60000",
        "ready_count: 128",
        "candidate_multiplier: 4",
        "min_candidates: 8",
        "max_candidates: 64",
        "temperature: 0.0",
        "unmasking_num: [2, 3]",
    ):
        assert value in text


def test_puma_evaluator_requires_and_passes_checkpoint_local_state():
    text = (
        ROOT / "integrations/puma/overlay/eval_dprm_checkpoint.py"
    ).read_text(encoding="utf-8")
    for value in (
        'checkpoint.get("dprm_order_state")',
        "DPRM evaluation requires checkpoint['dprm_order_state']",
        'dprm_state=dprm_state if args.confidence == "dprm_soft_bon" else None',
        '"dprm_state_loaded": args.confidence == "dprm_soft_bon"',
        '"dprm_table_shape": table_shape',
        '"example_id": start + offset',
    ):
        assert value in text


def test_puma_readme_evaluates_the_two_trained_endpoints():
    text = (ROOT / "integrations/puma/README.md").read_text(encoding="utf-8")
    for value in (
        "PUMA_CONFIDENCE_CHECKPOINT=ckpts/puma_confidence/ema_step=2000000.pt",
        "PUMA_DPRM_CHECKPOINT=ckpts/puma_dprm/ema_step=2000000.pt",
        "--cfg yaml_files/tinygsm_puma.yaml",
        '--ckpt "$PUMA_CONFIDENCE_CHECKPOINT"',
        "--cfg yaml_files/tinygsm_puma_dprm.yaml",
        '--ckpt "$PUMA_DPRM_CHECKPOINT"',
    ):
        assert value in text


def test_checkpoint_state_restores_effective_puma_bucket_dimensions():
    controller = OnlineDPRMController(
        DPRMConfig(num_phases=16, confidence_bins=16, aux_bins=1)
    )
    controller.load_state_dict(
        {
            "counts": torch.ones(39, 16, 1),
            "exp_reward_sums": torch.ones(39, 16, 1),
        }
    )
    assert controller.counts.shape == (39, 16, 1)
    assert controller.cfg.num_phases == 39
    assert controller.cfg.confidence_bins == 16


def test_dmpo_paper_launcher_fixes_the_reported_training_controller():
    text = (
        ROOT / "integrations/dmpo/overlay/DMPO/run_paper_dmpo.sh"
    ).read_text(encoding="utf-8")
    for value in (
        "LOSS_PROGRESSIVE_K=8",
        "LOSS_PROGRESSIVE_DPRM_BINS=16",
        "LOSS_PROGRESSIVE_DPRM_REWARD_TEMPERATURE=1.0",
        "LOSS_PROGRESSIVE_DPRM_WARMUP_STEPS=500",
        "LOSS_PROGRESSIVE_DPRM_SWITCH_STEPS=2000",
        "LOSS_PROGRESSIVE_DPRM_READY_COUNT=128",
        "LOSS_PROGRESSIVE_DPRM_MAX_CANDIDATES=32",
        "MAX_STEPS=5000",
        "SAMPLER_STEPS=128",
        "TEMPERATURE=0.2",
    ):
        assert value in text


def test_prism_paper_launcher_fixes_search_and_order_settings():
    text = (
        ROOT
        / "integrations/prism/overlay/LLaDA2mini/LLaDA2mini_Prism/scripts/run_gsm8k.sh"
    ).read_text(encoding="utf-8")
    for value in (
        "hts_N=16",
        "final_K=4",
        "temperature=0.7",
        "dprm_num_bins=16",
        "dprm_phase_buckets=8",
        "dprm_warmup_pct=0.2",
        "dprm_switch_pct=0.7",
        "dprm_ready_count=64",
    ):
        assert value in text


def test_dplm_reported_scalarizations_override_the_base_schedule():
    root = ROOT / "integrations/dplm/overlay/configs/experiment/dplm2"
    for name, reward in (
        ("dprm_joint_ws_dplm_650m.yaml", "aar_structure_weighted_sum"),
        ("dprm_joint_tcheby_dplm_650m.yaml", "aar_structure_tchebycheff"),
    ):
        text = (root / name).read_text(encoding="utf-8")
        for value in (
            "max_steps: 5000",
            f"reward: {reward}",
            "reward_aa_weight: 0.5",
            "reward_structure_weight: 0.5",
            "warmup_steps: 500",
            "switch_steps: 2000",
            "ready_count: 128",
        ):
            assert value in text


def test_dcm_terminal_protocol_fixes_disjoint_calibration_and_evaluation():
    text = (
        ROOT / "integrations/dcm/scripts/run_terminal_calibration.sh"
    ).read_text(encoding="utf-8")
    for value in (
        "--max-train-cells 256",
        "--branch-steps 0,8,16,24",
        "--num-phases 4",
        "--confidence-bins 16",
        "--ready-count 64",
        "--split train --cell-offset 256 --max-cells 96",
        "--split val --bootstrap 5000",
    ):
        assert value in text


def test_genmol_preference_launcher_fixes_scalarization_and_endpoints():
    text = (
        ROOT / "integrations/genmol/scripts/run_preference_sweep.sh"
    ).read_text(encoding="utf-8")
    for value in (
        "MAX_STEPS=${MAX_STEPS:-5000}",
        "SCALARIZATION=${SCALARIZATION:-smooth_tchebycheff}",
        'weights=("[0.95,0.05]" "[0.55,0.45]" "[0.05,0.95]")',
        "DPRM_TCHEBYCHEFF_TEMPERATURE=0.05",
        "DPRM_TCHEBYCHEFF_AUGMENTATION=0.05",
    ):
        assert value in text


def test_omni_pipeline_separates_fit_development_and_confirmation():
    fit = (
        ROOT / "integrations/omni_diffusion/matched/fit_paper_controller.sh"
    ).read_text(encoding="utf-8")
    develop = (
        ROOT
        / "integrations/omni_diffusion/matched/develop_public_base_fallback_controller.sh"
    ).read_text(encoding="utf-8")
    endpoint = (
        ROOT
        / "integrations/omni_diffusion/matched/scripts/select_and_confirm_matched_endpoints.sh"
    ).read_text(encoding="utf-8")
    evaluate = (
        ROOT
        / "integrations/omni_diffusion/matched/scripts/evaluate_omni_frozen_controller.sh"
    ).read_text(encoding="utf-8")
    training = (
        ROOT
        / "integrations/omni_diffusion/matched/scripts/train_matched_branches.sh"
    ).read_text(encoding="utf-8")
    matched_evaluate = (
        ROOT
        / "integrations/omni_diffusion/matched/scripts/evaluate_matched_branches.sh"
    ).read_text(encoding="utf-8")
    for value in (
        "--development-count 64",
        "sed -n '2001,2048p'",
        "--primary-reward-weight 0.25 --secondary-reward-weight 0.75",
        "--num-phases 1",
        "--confidence-bins 8",
    ):
        assert value in fit
    for value in (
        '"bh_lowconf_g075|${table_bh}|0.075|0.020|low"',
        '"bh_lowconf_g300|${table_bh}|0.300|0.050|low"',
        '"eq_mid_g300|${table_eq}|0.300|0.050|mid"',
        '"lh_mid_g300|${table_lh}|0.300|0.050|mid"',
        "--min-prompt-override-fraction 0.05 --require-positive-primary-ci",
    ):
        assert value in develop
    for value in (
        'STEP_A="${DPRM_OMNI_ENDPOINT_A:-500}"',
        'STEP_B="${DPRM_OMNI_ENDPOINT_B:-1000}"',
        'DEV_COUNT="${DPRM_OMNI_DEV_COUNT:-128}"',
        'CONFIRM_COUNT="${DPRM_OMNI_CONFIRM_COUNT:-512}"',
        'DPRM_OMNI_EVAL_ROLE=confirmation',
        'DPRM_OMNI_FIXED_VISUAL_PROMPT_IDS=""',
    ):
        assert value in endpoint
    for value in (
        'COUNT="${DPRM_OMNI_EVAL_COUNT:-512}"',
        '"test_time_terminal_rollouts": 0',
        '"complete_image_selection": false',
    ):
        assert value in evaluate
    assert 'DPRM_OMNI_DPRM_SCORER="${MATCHED_SCORER}"' in training
    assert 'HYBRID_ROLLIN="${DPRM_OMNI_HYBRID_ROLLIN:-1}"' in training
    assert "export DPRM_OMNI_PRECOMPUTED_TRAJECTORY=1" in training
    assert 'if [[ "${RESUME_FROM_CHECKPOINT}" == "auto" ]]' in training
    assert 'EFFECTIVE_RESUME="${LATEST_COMPLETE_CKPT}"' in training
    assert 'PROMPT_JSONL="${DPRM_OMNI_PROMPT_JSONL:-}"' in matched_evaluate
    assert '"prompt_jsonl_sha256"' in matched_evaluate


def test_lladav_realworldqa_protocol_fixes_all_three_intervals():
    text = (
        ROOT / "integrations/llada_v/run_realworldqa_protocol.sh"
    ).read_text(encoding="utf-8")
    for value in (
        "--max-docs-per-task 128",
        "--num-phases 1 --confidence-bins 8",
        "--aux-mode format_eot_position",
        "--doc-min 128 --doc-max 256",
        "--rwqa-doc-min 256 --rwqa-doc-max 765",
        "--bootstrap 5000 --seed 20260811",
    ):
        assert value in text


def test_lladav_ai2d_protocol_is_disjoint_and_confirmation_is_paired():
    text = (
        ROOT / "integrations/llada_v/run_ai2d_protocol.sh"
    ).read_text(encoding="utf-8")
    for value in (
        "FIT_DOCS=${FIT_DOCS:-128}",
        "DEV_MIN=${DEV_MIN:-128}",
        "DEV_MAX=${DEV_MAX:-256}",
        "TEST_MIN=${TEST_MIN:-256}",
        "TEST_MAX=${TEST_MAX:-500}",
        "for phases in 1 4",
        "for guidance in 1 2 4 8",
        '--doc-min "${TEST_MIN}" --doc-max "${TEST_MAX}"',
    ):
        assert value in text


def test_lladav_public_runner_installs_overlay_and_uses_lmms_directory():
    apply_text = (
        ROOT / "integrations/llada_v/apply_overlay.sh"
    ).read_text(encoding="utf-8")
    runner_text = (
        ROOT / "integrations/llada_v/run_lmms_eval.sh"
    ).read_text(encoding="utf-8")

    for value in (
        'EXPECTED_COMMIT="f8b02ce04b09f4f271fe55a3652059a73bbc7a32"',
        'dprm_generation.py:train/llava/dprm_generation.py',
        'host/fast_dllm_hook.py:train/llava/hooks/fast_dllm_hook.py',
        'host/modeling_llada.py:train/llava/model/language_model/modeling_llada.py',
        'host/llava_onevision_llada.py:eval/lmms-eval/lmms_eval/models/llava_onevision_llada.py',
        'tasks/ai2d_lite.yaml:eval/lmms-eval/lmms_eval/tasks/ai2d/ai2d_lite.yaml',
        'tasks/realworldqa.yaml:eval/lmms-eval/lmms_eval/tasks/realworldqa/realworldqa.yaml',
        'tasks/chartqa_lite.yaml:eval/lmms-eval/lmms_eval/tasks/chartqa/chartqa_lite.yaml',
    ):
        assert value in apply_text

    for value in (
        'bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/apply_overlay.sh"',
        'cd "${LLADA_V_LMMS_ROOT}/eval/lmms-eval"',
        'export PYTHONPATH="${LLADA_V_LMMS_ROOT}/train:${PYTHONPATH:-}"',
        'GEN_STEPS="${GEN_STEPS}" "${PYTHON}" -',
        '"${ACCELERATE}" launch --num_processes=1 -m lmms_eval',
    ):
        assert value in runner_text

    ai2d_text = (
        ROOT / "integrations/llada_v/overlay/tasks/ai2d_lite.yaml"
    ).read_text(encoding="utf-8")
    assert "dataset_path: lmms-lab/LMMs-Eval-Lite" in ai2d_text
    assert "token: false" in ai2d_text
