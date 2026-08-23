#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
: "${OMNI_ROOT:?set OMNI_ROOT}"
: "${VIRTUAL_ENV:?activate the experiment environment}"
: "${DPRM_OMNI_MODEL_PATH:?set DPRM_OMNI_MODEL_PATH}"
: "${DPRM_OMNI_IMAGE_TOKENIZER:?set DPRM_OMNI_IMAGE_TOKENIZER}"
: "${DPRM_OMNI_MATCHED_ROOT:?set DPRM_OMNI_MATCHED_ROOT}"
: "${DPRM_OMNI_GATE_CONTROLLER:?set DPRM_OMNI_GATE_CONTROLLER}"
: "${DPRM_OMNI_PROMPT_DEV_JSONL:?set the frozen development prompt JSONL}"
: "${DPRM_OMNI_PROMPT_CONFIRM_JSONL:?set the untouched confirmation prompt JSONL}"

PYTHON="${PYTHON:-${VIRTUAL_ENV}/bin/python}"
RUN_ROOT="${DPRM_OMNI_MATCHED_ROOT}"
TRAIN_ROOT="${DPRM_OMNI_TRAIN_OUT:-${RUN_ROOT}/matched_training_v2}"
TRAJ_ROOT="${RUN_ROOT}/matched_trajectories_v2"
CONTROLLER="${DPRM_OMNI_GATE_CONTROLLER}"
ENDPOINT_ROOT="${DPRM_OMNI_ENDPOINT_ROOT:-${RUN_ROOT}/endpoint_protocol}"
STEP_A="${DPRM_OMNI_ENDPOINT_A:-500}"
STEP_B="${DPRM_OMNI_ENDPOINT_B:-1000}"
DEV_COUNT="${DPRM_OMNI_DEV_COUNT:-128}"
CONFIRM_COUNT="${DPRM_OMNI_CONFIRM_COUNT:-512}"
GPUS="${DPRM_OMNI_GPUS:-0,1,2,3,4,5}"
NPROC="${DPRM_OMNI_NPROC:-6}"

DEV_A="${DPRM_OMNI_DEV_A_OUT:-${RUN_ROOT}/partiprompts_dev_step${STEP_A}}"
DEV_B="${DPRM_OMNI_DEV_B_OUT:-${RUN_ROOT}/partiprompts_dev_step${STEP_B}}"
SELECTION="${ENDPOINT_ROOT}/selection.json"
CONFIRM_OUT="${ENDPOINT_ROOT}/partiprompts_confirmation"

mkdir -p "${ENDPOINT_ROOT}"
exec 8>"${ENDPOINT_ROOT}/endpoint_protocol.lock"
if ! flock -n 8; then
  echo "another Omni endpoint protocol owns ${ENDPOINT_ROOT}" >&2
  exit 0
fi
source "${VIRTUAL_ENV}/bin/activate"
export PYTHONPATH="${RELEASE_ROOT}/src:${OMNI_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

for path in \
  "${CONTROLLER}" \
  "${DPRM_OMNI_PROMPT_DEV_JSONL}" \
  "${DPRM_OMNI_PROMPT_CONFIRM_JSONL}" \
  "${TRAJ_ROOT}/merged/random_matched.yaml" \
  "${TRAJ_ROOT}/merged/confidence_matched.yaml" \
  "${TRAJ_ROOT}/merged/dprm_matched.yaml" \
  "${DEV_A}/EVAL_COMPLETE" \
  "${DEV_A}/summary/paired/paired_clip_summary.json" \
  "${DEV_A}/summary/order_divergence/paired_order_divergence.json"; do
  [[ -s "${path}" ]] || { echo "missing endpoint-protocol input: ${path}" >&2; exit 2; }
done

for order in confidence_matched dprm_matched; do
  [[ -s "${TRAIN_ROOT}/${order}/checkpoint-${STEP_A}/trainer_state.json" ]] || {
    echo "missing ${order} checkpoint-${STEP_A}" >&2
    exit 2
  }
done

cat > "${ENDPOINT_ROOT}/protocol_manifest.json" <<JSON
{
  "protocol": "development-only endpoint selection followed by one untouched confirmation evaluation",
  "candidate_steps": [${STEP_A}, ${STEP_B}],
  "development_prompt_count": ${DEV_COUNT},
  "confirmation_prompt_count": ${CONFIRM_COUNT},
  "development_prompt_sha256": "$(sha256sum "${DPRM_OMNI_PROMPT_DEV_JSONL}" | awk '{print $1}')",
  "confirmation_prompt_sha256": "$(sha256sum "${DPRM_OMNI_PROMPT_CONFIRM_JSONL}" | awk '{print $1}')",
  "controller_sha256": "$(sha256sum "${CONTROLLER}" | awk '{print $1}')",
  "shared_initial_checkpoint": "${DPRM_OMNI_MODEL_PATH}",
  "training_orders": ["confidence_matched", "dprm_matched"],
  "training_hyperparameters": {
    "learning_rate": 1e-5,
    "warmup_ratio": 0.03,
    "trainable_last_n_layers": 2,
    "seed": 956,
    "hybrid_current_model_rollin": true
  },
  "selection_rule": "positive mean delta on both CLIP encoders, measurable order divergence, then largest equal-weight mean delta",
  "confirmation_data_read_before_selection": false
}
JSON

if [[ ! -s "${TRAIN_ROOT}/confidence_matched/checkpoint-${STEP_B}/trainer_state.json" \
   || ! -s "${TRAIN_ROOT}/dprm_matched/checkpoint-${STEP_B}/trainer_state.json" ]]; then
  OMNI_ROOT="${OMNI_ROOT}" \
  DPRM_OMNI_MODEL_PATH="${DPRM_OMNI_MODEL_PATH}" \
  DPRM_OMNI_OUT_BASE="${TRAIN_ROOT}" \
  DPRM_OMNI_ORDERS="confidence_matched dprm_matched" \
  DPRM_OMNI_DATA_CONFIG="${TRAJ_ROOT}/merged/random_matched.yaml" \
  DPRM_OMNI_CONFIDENCE_DATA_CONFIG="${TRAJ_ROOT}/merged/confidence_matched.yaml" \
  DPRM_OMNI_DPRM_DATA_CONFIG="${TRAJ_ROOT}/merged/dprm_matched.yaml" \
  DPRM_OMNI_DPRM_SCORER="${CONTROLLER}" \
  DPRM_OMNI_HYBRID_ROLLIN=1 \
  DPRM_OMNI_GPUS="${GPUS}" \
  DPRM_OMNI_NPROC="${NPROC}" \
  DPRM_OMNI_MAX_STEPS="${STEP_B}" \
  DPRM_OMNI_SAVE_STEPS="${STEP_A}" \
  DPRM_OMNI_SAVE_TOTAL_LIMIT=2 \
  DPRM_OMNI_RESUME_FROM_CHECKPOINT=auto \
  DPRM_OMNI_LEARNING_RATE=1e-5 \
  DPRM_OMNI_WARMUP_RATIO=0.03 \
  DPRM_OMNI_TRAINABLE_LAST_N_LAYERS=2 \
  WANDB_MODE="${WANDB_MODE:-offline}" \
  bash "${SCRIPT_DIR}/train_matched_branches.sh" \
    >> "${ENDPOINT_ROOT}/resume_to_${STEP_B}.log" 2>&1
fi
date -Is > "${ENDPOINT_ROOT}/TRAINING_ENDPOINTS_COMPLETE"

if [[ ! -s "${DEV_B}/EVAL_COMPLETE" ]]; then
  DPRM_OMNI_TRAIN_OUT="${TRAIN_ROOT}" \
  DPRM_OMNI_GATE_CONTROLLER="${CONTROLLER}" \
  DPRM_OMNI_IMAGE_TOKENIZER="${DPRM_OMNI_IMAGE_TOKENIZER}" \
  DPRM_OMNI_PROMPT_JSONL="${DPRM_OMNI_PROMPT_DEV_JSONL}" \
  DPRM_OMNI_EVAL_OUT="${DEV_B}" \
  DPRM_OMNI_EVAL_COUNT="${DEV_COUNT}" \
  DPRM_OMNI_EVAL_GPUS="${GPUS}" \
  DPRM_OMNI_EVAL_STEP="${STEP_B}" \
  DPRM_OMNI_EVAL_ROLE=development \
  DPRM_OMNI_FIXED_VISUAL_PROMPT_IDS="" \
  bash "${SCRIPT_DIR}/evaluate_matched_branches.sh" \
    >> "${DEV_B}.log" 2>&1
fi

"${PYTHON}" "${SCRIPT_DIR}/select_omni_training_endpoint.py" \
  --candidate "${STEP_A}" \
    "${DEV_A}/summary/paired/paired_clip_summary.json" \
    "${DEV_A}/summary/order_divergence/paired_order_divergence.json" \
  --candidate "${STEP_B}" \
    "${DEV_B}/summary/paired/paired_clip_summary.json" \
    "${DEV_B}/summary/order_divergence/paired_order_divergence.json" \
  --expected-prompts "${DEV_COUNT}" --output "${SELECTION}"

SELECTED_STEP="$(jq -r '.selected_step // empty' "${SELECTION}")"
if [[ -z "${SELECTED_STEP}" ]]; then
  date -Is > "${ENDPOINT_ROOT}/DEVELOPMENT_SELECTION_FAILED"
  exit 2
fi

# The confirmation prompt file is first consumed after the selection artifact
# has been written. No endpoint or controller choice follows this evaluation.
if [[ ! -s "${CONFIRM_OUT}/EVAL_COMPLETE" ]]; then
  DPRM_OMNI_TRAIN_OUT="${TRAIN_ROOT}" \
  DPRM_OMNI_GATE_CONTROLLER="${CONTROLLER}" \
  DPRM_OMNI_IMAGE_TOKENIZER="${DPRM_OMNI_IMAGE_TOKENIZER}" \
  DPRM_OMNI_PROMPT_JSONL="${DPRM_OMNI_PROMPT_CONFIRM_JSONL}" \
  DPRM_OMNI_EVAL_OUT="${CONFIRM_OUT}" \
  DPRM_OMNI_EVAL_COUNT="${CONFIRM_COUNT}" \
  DPRM_OMNI_EVAL_GPUS="${GPUS}" \
  DPRM_OMNI_EVAL_STEP="${SELECTED_STEP}" \
  DPRM_OMNI_EVAL_ROLE=confirmation \
  DPRM_OMNI_FIXED_VISUAL_PROMPT_IDS="" \
  bash "${SCRIPT_DIR}/evaluate_matched_branches.sh" \
    >> "${CONFIRM_OUT}.log" 2>&1
fi

jq --argjson selected_step "${SELECTED_STEP}" \
  '. + {selected_step: $selected_step, confirmation_data_read: true}' \
  "${SELECTION}" > "${ENDPOINT_ROOT}/final_selection.json"

QUALITATIVE_PROMPTS="${DPRM_OMNI_QUALITATIVE_PROMPTS:-${RELEASE_ROOT}/reproducibility/omni_qualitative_prompts.jsonl}"
QUALITATIVE_OUT="${ENDPOINT_ROOT}/qualitative_examples"
if [[ ! -s "${QUALITATIVE_OUT}/EVAL_COMPLETE" ]]; then
  DPRM_OMNI_TRAIN_OUT="${TRAIN_ROOT}" \
  DPRM_OMNI_GATE_CONTROLLER="${CONTROLLER}" \
  DPRM_OMNI_IMAGE_TOKENIZER="${DPRM_OMNI_IMAGE_TOKENIZER}" \
  DPRM_OMNI_PROMPT_JSONL="${QUALITATIVE_PROMPTS}" \
  DPRM_OMNI_EVAL_OUT="${QUALITATIVE_OUT}" \
  DPRM_OMNI_EVAL_COUNT=2 DPRM_OMNI_EVAL_OFFSET=2500 \
  DPRM_OMNI_EVAL_GPUS="${GPUS}" DPRM_OMNI_EVAL_STEP="${SELECTED_STEP}" \
  DPRM_OMNI_EVAL_ROLE=qualitative \
  DPRM_OMNI_FIXED_VISUAL_PROMPT_IDS="2500 2501" \
  DPRM_OMNI_SKIP_PROMOTION=1 \
  bash "${SCRIPT_DIR}/evaluate_matched_branches.sh" \
    >> "${QUALITATIVE_OUT}.log" 2>&1
fi
date -Is > "${ENDPOINT_ROOT}/ENDPOINT_PROTOCOL_COMPLETE"
