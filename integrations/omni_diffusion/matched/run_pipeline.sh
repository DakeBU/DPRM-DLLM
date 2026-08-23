#!/usr/bin/env bash
set -euo pipefail

MATCHED_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${MATCHED_ROOT}/../../.." && pwd)"
EXPECTED_COMMIT="c4f4625f84197a72d556ea00f10e5b2775524252"
PARTI_SHA256="fab29e41bb512a169b56acab4cf2a41dcb675e285df2efcde6640c7dd3c440eb"
PARTI_URL="https://raw.githubusercontent.com/google-research/parti/5a657978134374ce28973948331b319adef164bd/PartiPrompts.tsv"

: "${OMNI_ROOT:?set the Omni-Diffusion checkout}"
: "${OMNI_MODEL_PATH:?set the public Omni checkpoint}"
: "${OMNI_IMAGE_TOKENIZER_PATH:?set the MagViT tokenizer path}"
: "${OMNI_DATA_JSON:?set the tokenized JourneyDB JSONL}"
: "${OMNI_GENEVAL_PROMPT_SOURCE:?set the 553-line unique GenEval prompt file}"
: "${OMNI_PARTIPROMPTS_SOURCE:?set the pinned PartiPrompts.tsv file}"
: "${OMNI_DEVELOPMENT_ROOT:?set the controller-fit output directory}"
: "${OMNI_RUN_ROOT:?set the matched-training and evaluation output directory}"
: "${VIRTUAL_ENV:?activate the experiment environment}"

PYTHON="${PYTHON:-${VIRTUAL_ENV}/bin/python}"
SPLIT_ROOT="${OMNI_PARTIPROMPTS_SPLIT_ROOT:-${OMNI_RUN_ROOT}/partiprompts_split}"
actual_commit="$(git -C "${OMNI_ROOT}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
  echo "Omni-Diffusion commit mismatch: ${actual_commit} != ${EXPECTED_COMMIT}" >&2
  exit 2
fi
for path in "${OMNI_MODEL_PATH}" "${OMNI_IMAGE_TOKENIZER_PATH}" \
  "${OMNI_DATA_JSON}" "${OMNI_GENEVAL_PROMPT_SOURCE}" \
  "${OMNI_PARTIPROMPTS_SOURCE}"; do
  [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 2; }
done

cp -a "${MATCHED_ROOT}/overlay/." "${OMNI_ROOT}/"
source "${VIRTUAL_ENV}/bin/activate"
export PYTHONPATH="${RELEASE_ROOT}/src:${OMNI_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export OMNI_DEVELOPMENT_GPUS="${OMNI_DEVELOPMENT_GPUS:-0 1}"

"${PYTHON}" "${MATCHED_ROOT}/scripts/freeze_partiprompts_split.py" \
  --source "${OMNI_PARTIPROMPTS_SOURCE}" --output-root "${SPLIT_ROOT}" \
  --source-url "${PARTI_URL}" --expected-sha256 "${PARTI_SHA256}" \
  --development-count 128 --confirmation-count 512

fit_status=0
bash "${MATCHED_ROOT}/fit_paper_controller.sh" || fit_status=$?
for order in random progressive_confidence; do
  [[ $(find "${OMNI_DEVELOPMENT_ROOT}/table_rollouts/${order}" \
    -type f -name COMPLETE 2>/dev/null | wc -l) -eq 48 ]] || {
    echo "controller fitting failed before producing 48 ${order} rollouts" >&2
    exit "${fit_status:-2}"
  }
done
export OMNI_DEVELOPMENT_PROMPT_FILE="${OMNI_DEVELOPMENT_ROOT}/split/development_prompts.txt"
bash "${MATCHED_ROOT}/develop_public_base_fallback_controller.sh"
for path in "${OMNI_DEVELOPMENT_ROOT}/formal_controller.json" \
  "${OMNI_DEVELOPMENT_ROOT}/DEVELOPMENT_COMPLETE"; do
  [[ -s "${path}" ]] || { echo "missing fitted controller artifact: ${path}" >&2; exit 2; }
done

export DPRM_OMNI_MATCHED_ROOT="${OMNI_RUN_ROOT}"
export DPRM_OMNI_GATE_CONTROLLER="${OMNI_DEVELOPMENT_ROOT}/formal_controller.json"
export DPRM_OMNI_MODEL_PATH="${OMNI_MODEL_PATH}"
export DPRM_OMNI_IMAGE_TOKENIZER="${OMNI_IMAGE_TOKENIZER_PATH}"
export DPRM_OMNI_DATA_JSON="${OMNI_DATA_JSON}"
export DPRM_OMNI_DEVELOPMENT_CONTROLLER_VALIDATED=1
export DPRM_OMNI_TRAJECTORY_COUNT="${DPRM_OMNI_TRAJECTORY_COUNT:-256}"
export DPRM_OMNI_TRAIN_ORDERS="confidence_matched dprm_matched"
export DPRM_OMNI_HYBRID_ROLLIN=1
export DPRM_OMNI_MAX_STEPS=500
export DPRM_OMNI_SAVE_STEPS=500
export DPRM_OMNI_SAVE_TOTAL_LIMIT=2
export DPRM_OMNI_TRAINABLE_LAST_N_LAYERS=2
export DPRM_OMNI_MATCHED_EVAL_OUT="${OMNI_RUN_ROOT}/partiprompts_dev_step500"
export DPRM_OMNI_PROMPT_JSONL="${SPLIT_ROOT}/development.jsonl"
export DPRM_OMNI_EVAL_COUNT=128
export DPRM_OMNI_EVAL_STEP=500
export DPRM_OMNI_EVAL_ROLE=development
export DPRM_OMNI_FIXED_VISUAL_PROMPT_IDS=""
export WANDB_MODE="${WANDB_MODE:-offline}"

bash "${MATCHED_ROOT}/scripts/orchestrate_matched_pipeline.sh"

export DPRM_OMNI_PROMPT_DEV_JSONL="${SPLIT_ROOT}/development.jsonl"
export DPRM_OMNI_PROMPT_CONFIRM_JSONL="${SPLIT_ROOT}/confirmation.jsonl"
export DPRM_OMNI_DEV_A_OUT="${OMNI_RUN_ROOT}/partiprompts_dev_step500"
export DPRM_OMNI_TRAIN_OUT="${OMNI_RUN_ROOT}/matched_training_v2"
exec bash "${MATCHED_ROOT}/scripts/select_and_confirm_matched_endpoints.sh"
