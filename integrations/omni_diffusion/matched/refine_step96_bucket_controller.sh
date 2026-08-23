#!/usr/bin/env bash
set -euo pipefail

MATCHED_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${MATCHED_ROOT}/scripts"
: "${OMNI_SOURCE_TABLE:?set the current-checkpoint DPRM bucket table}"
: "${OMNI_STEP96_CONTROLLER_DIR:?set the frozen-controller output directory}"
: "${OMNI_ACTION_EVAL_ROOT:?set the development evaluation output directory}"
: "${VIRTUAL_ENV:?set the experiment environment}"

PYTHON="${PYTHON:-${VIRTUAL_ENV}/bin/python}"
mkdir -p "${OMNI_STEP96_CONTROLLER_DIR}"

# The first development half localized the transferable intervention to step 96.
# This locked grid varies only the two deployment scalars already present in the
# bucketized DPRM controller. The second development half is reserved for the
# selected controller and is not used by this script.
specs=(
  "step96_g030_gap005|0.30|0.05"
  "step96_g060_gap005|0.60|0.05"
  "step96_g060_gap010|0.60|0.10"
  "step96_g100_gap010|1.00|0.10"
)
for spec in "${specs[@]}"; do
  IFS='|' read -r label guidance gap <<< "${spec}"
  "${PYTHON}" "${SCRIPT_DIR}/freeze_omni_bucket_controller.py" \
    --source-table "${OMNI_SOURCE_TABLE}" \
    --output "${OMNI_STEP96_CONTROLLER_DIR}/${label}.json" \
    --guidance-scale "${guidance}" \
    --ready-count 4 \
    --policy-warmup-steps 0 \
    --reward-action-steps 96 \
    --max-base-score-gap "${gap}"
done

cat > "${OMNI_STEP96_CONTROLLER_DIR}/grid_manifest.json" <<JSON
{
  "design": "step-96 refinement of the current-checkpoint bucketized DPRM controller",
  "source_table": "${OMNI_SOURCE_TABLE}",
  "reward_action_steps": [96],
  "guidance_scales": [0.30, 0.60, 1.00],
  "max_base_score_gaps": [0.05, 0.10],
  "terminal_reward_calls_at_test": 0,
  "complete_image_selection": false,
  "selection_split": "development prompts 0--63",
  "reserved_validation_split": "development prompts 64--127"
}
JSON

export OMNI_ACTION_CONTROLLER_DIR="${OMNI_STEP96_CONTROLLER_DIR}"
export OMNI_ACTION_EVAL_DESIGN="step-96 refinement of current-checkpoint bucketized DPRM"
export OMNI_ACTION_REQUIRE_POSITIVE_PRIMARY_CI=0
bash "${MATCHED_ROOT}/evaluate_action_conditioned_controllers.sh"
