from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = (
    REPO_ROOT
    / "integrations"
    / "omni_diffusion"
    / "matched"
    / "scripts"
)
ORCHESTRATOR = SCRIPT_DIR / "orchestrate_matched_pipeline.sh"
FIT_CONTROLLER = SCRIPT_DIR.parent / "fit_paper_controller.sh"
FALLBACK_CONTROLLER = (
    SCRIPT_DIR.parent / "develop_public_base_fallback_controller.sh"
)
ACTION_CONTROLLER_EVALUATOR = (
    SCRIPT_DIR.parent / "evaluate_action_conditioned_controllers.sh"
)


def test_orchestrator_local_entrypoints_exist() -> None:
    source = ORCHESTRATOR.read_text(encoding="utf-8")
    referenced = set(
        re.findall(r'\$\{SCRIPT_DIR\}/([A-Za-z0-9_.-]+\.(?:py|sh))', source)
    )
    assert referenced
    missing = sorted(name for name in referenced if not (SCRIPT_DIR / name).is_file())
    assert not missing, f"missing Omni pipeline entrypoints: {missing}"


def test_release_runner_uses_matched_training_and_external_confirmation() -> None:
    runner = (SCRIPT_DIR.parent / "run_pipeline.sh").read_text(encoding="utf-8")
    required = {
        'bash "${MATCHED_ROOT}/fit_paper_controller.sh"',
        'bash "${MATCHED_ROOT}/develop_public_base_fallback_controller.sh"',
        'scripts/freeze_partiprompts_split.py',
        'DPRM_OMNI_TRAIN_ORDERS="confidence_matched dprm_matched"',
        "DPRM_OMNI_HYBRID_ROLLIN=1",
        "DPRM_OMNI_MAX_STEPS=500",
        "DPRM_OMNI_DEVELOPMENT_CONTROLLER_VALIDATED=1",
        'bash "${MATCHED_ROOT}/scripts/orchestrate_matched_pipeline.sh"',
        'exec bash "${MATCHED_ROOT}/scripts/select_and_confirm_matched_endpoints.sh"',
    }
    missing = sorted(fragment for fragment in required if fragment not in runner)
    assert not missing, f"incomplete public Omni protocol: {missing}"
    assert runner.index('bash "${MATCHED_ROOT}/fit_paper_controller.sh"') < runner.index(
        'bash "${MATCHED_ROOT}/develop_public_base_fallback_controller.sh"'
    )
    assert 'if [[ ! -s "${OMNI_DEVELOPMENT_ROOT}/formal_controller.json" ]]' not in runner


def test_release_evaluation_parallelizes_orders_before_shared_statistics() -> None:
    evaluator = (SCRIPT_DIR / "evaluate_matched_branches.sh").read_text(
        encoding="utf-8"
    )
    assert 'EVAL_GPUS_TEXT="${DPRM_OMNI_EVAL_GPUS:-${DPRM_OMNI_EVAL_GPU:-0}}"' in evaluator
    assert 'assigned_gpus+=("${EVAL_GPUS[$gpu_idx]}")' in evaluator
    assert 'shard_idx=$((local_idx % ${#shard_jobs[@]}))' in evaluator
    assert 'generation_pids+=("$!")' in evaluator
    assert 'for pid in "${generation_pids[@]}"' in evaluator
    assert evaluator.index('for pid in "${generation_pids[@]}"') < evaluator.index(
        'summarize_omni_eval.py'
    )


def test_paper_controller_fit_is_frozen_and_prompt_disjoint() -> None:
    source = FIT_CONTROLLER.read_text(encoding="utf-8")
    required = {
        "--development-count 64 --salt dprm-omni-geneval-v1",
        "sed -n '2001,2048p'",
        "--secondary-clip-model openai/clip-vit-base-patch32",
        "--primary-reward-weight 0.25 --secondary-reward-weight 0.75",
        "--reward-normalization paired_prompt_advantage",
        "--num-phases 1 --phase-source step --confidence-bins 8",
        "--confidence-binning development_quantile --aux-bins 16",
        'bash "${MATCHED_ROOT}/develop_low_confidence_controller.sh"',
    }
    missing = sorted(fragment for fragment in required if fragment not in source)
    assert not missing, f"paper Omni controller fit drifted: {missing}"


def test_low_confidence_controller_configuration() -> None:
    source = (SCRIPT_DIR.parent / "develop_low_confidence_controller.sh").read_text(
        encoding="utf-8"
    )
    required = {
        "--guidance-scale 0.075",
        "--ready-count 4",
        "--reward-action-steps 96 112 128 144 160",
        "--max-base-score-gap 0.02",
        "--max-reward-confidence-bin 0",
        "--min-prompt-override-fraction 0.05",
        "--require-positive-primary-ci",
    }
    missing = sorted(fragment for fragment in required if fragment not in source)
    assert not missing, f"paper Omni controller configuration drifted: {missing}"


def test_matched_trajectory_schedule_covers_controller_actions() -> None:
    source = ORCHESTRATOR.read_text(encoding="utf-8")
    assert 'config.get("reward_action_steps", config.get("active_steps", []))' in source
    assert "{step - 1 for step in reward_steps}" in source
    assert '--checkpoints "${POST_ACTION_CHECKPOINTS[@]}"' in source


def test_bucket_controller_owns_its_deployment_steps() -> None:
    source = (SCRIPT_DIR / "omni_t2i_smoke.py").read_text(encoding="utf-8")
    assert "if scorer.reward_action_steps" in source
    assert "for step in scorer.reward_action_steps" in source


def test_public_base_fallback_is_disjoint_and_single_path() -> None:
    source = FALLBACK_CONTROLLER.read_text(encoding="utf-8")
    required = {
        "OMNI_DEVELOPMENT_PROMPT_FILE",
        '"test_time_terminal_rollouts": 0',
        '"complete_image_selection": false',
        "--selection-prompt-file",
        "--primary-reward-weight",
        "--secondary-reward-weight",
    }
    missing = sorted(fragment for fragment in required if fragment not in source)
    assert not missing, f"Omni fallback protocol drifted: {missing}"
    assert "bh_lowconf_g075" in source
    assert "bh_lowconf_g150" in source
    assert "bh_lowconf_g300" in source
    for label in (
        "bh_mid_g150",
        "bh_mid_g300",
        "eq_mid_g150",
        "eq_mid_g300",
        "lh_mid_g150",
        "lh_mid_g300",
    ):
        assert label in source
    assert "terminal_utility_weights" in source
    assert "--require-positive-primary-ci" in source
    assert "Omni development grid incomplete" in source
    assert 'result = output / f\'omni_t2i_{row["order_policy"]}.json\'' in source
    assert "OMNI_TABLE_PROMPT_COUNT:-48" in source
    assert "OMNI_TABLE_NUM_PHASES:-1" in source
    assert "OMNI_CONTROLLER_TABLE_ONLY:-0" in source


def test_action_controller_evaluator_ignores_non_controller_json() -> None:
    source = ACTION_CONTROLLER_EVALUATOR.read_text(encoding="utf-8")
    assert "artifact_format=\"$(jq -r '.format // empty'" in source
    for artifact_format in (
        "omni_bucket_table_dprm_v1",
        "omni_rank_bucket_dprm_v1",
        "omni_stage_rank_spatial_dprm_v1",
        "omni_stage_rank_code_dprm_v1",
    ):
        assert artifact_format in source
    assert "*) continue ;;" in source
    assert "OMNI_ACTION_REQUIRE_POSITIVE_PRIMARY_CI:-1" in source
    assert "selection_args+=(--require-positive-primary-ci)" in source


def test_step96_refinement_selects_then_validates() -> None:
    source = (
        SCRIPT_DIR.parent / "refine_step96_bucket_controller.sh"
    ).read_text(encoding="utf-8")
    assert "--reward-action-steps 96" in source
    assert "OMNI_ACTION_REQUIRE_POSITIVE_PRIMARY_CI=0" in source
    assert '"reserved_validation_split": "development prompts 64--127"' in source


def test_omni_readme_matches_the_online_action_value_protocol() -> None:
    source = (SCRIPT_DIR.parent.parent / "README.md").read_text(encoding="utf-8")
    assert "At visual step 96" in source
    assert "`0.70`, `0.85`," in source
    assert "`0.90`, and `0.95`" in source
    assert "128 development prompts" in source
    assert "512 disjoint confirmation prompts" in source
    assert "OMNI_ONLINE_FIXED_GUIDANCE=8" in source
    assert "run_online_action_value_controller.sh" in source


def test_matched_evaluator_supports_fixed_nonselective_qualitative_protocol() -> None:
    evaluator = (SCRIPT_DIR / "evaluate_matched_branches.sh").read_text(
        encoding="utf-8"
    )
    prompts = REPO_ROOT / "reproducibility" / "omni_qualitative_prompts.jsonl"
    rows = [__import__("json").loads(line) for line in prompts.read_text().splitlines()]
    assert [row["prompt_id"] for row in rows] == [
        "beach_three_children",
        "boy_flute_kittens",
    ]
    assert 'DPRM_OMNI_SKIP_PROMOTION:-0' in evaluator
    assert 'QUALITATIVE_PROTOCOL_COMPLETE' in evaluator


def test_endpoint_protocol_selects_before_confirmation() -> None:
    source = (SCRIPT_DIR / "select_and_confirm_matched_endpoints.sh").read_text(
        encoding="utf-8"
    )
    assert "DPRM_OMNI_RESUME_FROM_CHECKPOINT=auto" in source
    assert "DPRM_OMNI_SAVE_TOTAL_LIMIT=2" in source
    assert "select_omni_training_endpoint.py" in source
    assert source.index("select_omni_training_endpoint.py") < source.index(
        "# The confirmation prompt file is first consumed after the selection artifact"
    )
    assert "DPRM_OMNI_EVAL_ROLE=confirmation" in source
    assert '"confirmation_data_read_before_selection": false' in source
    assert "omni_qualitative_prompts.jsonl" in source
    assert source.index("final_selection.json") < source.index(
        "DPRM_OMNI_EVAL_ROLE=qualitative"
    )
