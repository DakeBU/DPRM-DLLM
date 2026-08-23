#!/usr/bin/env bash
set -euo pipefail

MATCHED_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${MATCHED_ROOT}/../../.." && pwd)"
SCRIPT_DIR="${MATCHED_ROOT}/scripts"
: "${OMNI_ROOT:?set the Omni-Diffusion checkout}"
: "${OMNI_MODEL_PATH:?set the frozen public Omni checkpoint}"
: "${OMNI_IMAGE_TOKENIZER_PATH:?set the MagViT tokenizer path}"
: "${OMNI_DATA_JSON:?set the tokenized JourneyDB JSONL}"
: "${OMNI_GENEVAL_PROMPT_SOURCE:?set the 553-line unique GenEval prompt file}"
: "${OMNI_DEVELOPMENT_ROOT:?set the controller-fit output directory}"
: "${VIRTUAL_ENV:?activate the experiment environment}"

PYTHON="${PYTHON:-${VIRTUAL_ENV}/bin/python}"
GPUS_TEXT="${OMNI_DEVELOPMENT_GPUS:-0 1}"
read -r -a GPUS <<< "${GPUS_TEXT//,/ }"
source "${VIRTUAL_ENV}/bin/activate"
export PYTHONPATH="${RELEASE_ROOT}/src:${OMNI_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
mkdir -p "${OMNI_DEVELOPMENT_ROOT}"/{table_rollouts,table,split,logs}

run_jobs() {
  local jobs="$1" log_prefix="$2" pids=() failed=0
  for gpu in "${GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${SCRIPT_DIR}/run_omni_t2i_manifest.py" \
      --smoke-script "${SCRIPT_DIR}/omni_t2i_smoke.py" \
      --model-path "${OMNI_MODEL_PATH}" \
      --image-tokenizer-path "${OMNI_IMAGE_TOKENIZER_PATH}" \
      --jobs "${jobs}" > "${OMNI_DEVELOPMENT_ROOT}/logs/${log_prefix}_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  (( failed == 0 ))
}

"${PYTHON}" "${SCRIPT_DIR}/build_omni_prompt_split.py" \
  --source "${OMNI_GENEVAL_PROMPT_SOURCE}" \
  --development-output "${OMNI_DEVELOPMENT_ROOT}/split/development_prompts.txt" \
  --confirmation-output "${OMNI_DEVELOPMENT_ROOT}/split/confirmation_prompts.txt" \
  --manifest-output "${OMNI_DEVELOPMENT_ROOT}/split/manifest.json" \
  --development-count 64 --salt dprm-omni-geneval-v1

mapfile -t TABLE_PROMPTS < <(
  jq -r '.messages[0].content | split("\n") | .[1:] | join("\n")' "${OMNI_DATA_JSON}" \
    | awk 'NF && !seen[$0]++' | sed -n '2001,2048p'
)
[[ ${#TABLE_PROMPTS[@]} -eq 48 ]] || {
  echo "expected 48 unique JourneyDB prompts" >&2
  exit 2
}

jobs="${OMNI_DEVELOPMENT_ROOT}/table_jobs.jsonl"
: > "${jobs}"
for order in random progressive_confidence; do
  for local_idx in "${!TABLE_PROMPTS[@]}"; do
    idx=$((2000 + local_idx))
    dir="${OMNI_DEVELOPMENT_ROOT}/table_rollouts/${order}/prompt_$(printf '%04d' "${idx}")"
    extra='["--fixed-t2i-scaffold","--trace-order-stats","auto","--trace-num-phases","1","--trace-confidence-bins","8","--trace-aux-bins","16","--trace-provisional-phases"]'
    jq -cn --arg output_dir "${dir}" --arg prompt "${TABLE_PROMPTS[$local_idx]}" \
      --arg order "${order}" --argjson seed "$((20272000 + idx))" \
      --argjson extra "${extra}" \
      '{output_dir:$output_dir,prompt:$prompt,order_policy:$order,seed:$seed,steps:260,max_tokens:260,extra_args:$extra}' \
      >> "${jobs}"
  done
done
run_jobs "${jobs}" table

table="${OMNI_DEVELOPMENT_ROOT}/table/omni_dual_l025_b075.json"
CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${PYTHON}" "${SCRIPT_DIR}/build_omni_dprm_table.py" \
  --rollout-root "${OMNI_DEVELOPMENT_ROOT}/table_rollouts" \
  --orders random progressive_confidence --out "${table}" \
  --clip-model openai/clip-vit-large-patch14 \
  --secondary-clip-model openai/clip-vit-base-patch32 \
  --primary-reward-weight 0.25 --secondary-reward-weight 0.75 \
  --device cuda:0 --deduplicate-prompt-text \
  --reward-normalization paired_prompt_advantage \
  --num-phases 1 --phase-source step --confidence-bins 8 \
  --confidence-binning development_quantile --aux-bins 16 \
  --reward-temperature 1.0 --ready-count 4 --warmup-steps 0 \
  --switch-steps 64 --require-fixed-visual-canvas

export OMNI_DUAL_TABLE_B_HEAVY="${table}"
export OMNI_DEVELOPMENT_PROMPT_FILE="${OMNI_DEVELOPMENT_ROOT}/split/development_prompts.txt"
bash "${MATCHED_ROOT}/develop_low_confidence_controller.sh"

