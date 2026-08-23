#!/usr/bin/env bash
set -euo pipefail

MATCHED_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${MATCHED_ROOT}/../../.." && pwd)"
SCRIPT_DIR="${MATCHED_ROOT}/scripts"
: "${OMNI_ROOT:?set the Omni-Diffusion checkout}"
: "${OMNI_MODEL_PATH:?set the frozen Omni checkpoint used for table fitting}"
: "${OMNI_IMAGE_TOKENIZER_PATH:?set the MagViT tokenizer path}"
: "${OMNI_DATA_JSON:?set the tokenized JourneyDB JSONL}"
: "${OMNI_DEVELOPMENT_ROOT:?set the output directory}"
: "${OMNI_DEVELOPMENT_PROMPT_FILE:?set the frozen development prompt file}"
: "${VIRTUAL_ENV:?set the experiment environment}"

PYTHON="${PYTHON:-${VIRTUAL_ENV}/bin/python}"
TABLE_PROMPT_OFFSET="${OMNI_TABLE_PROMPT_OFFSET:-2000}"
TABLE_PROMPT_COUNT="${OMNI_TABLE_PROMPT_COUNT:-48}"
GPUS_TEXT="${OMNI_DEVELOPMENT_GPUS:-0 1}"
read -r -a GPUS <<< "${GPUS_TEXT//,/ }"
source "${VIRTUAL_ENV}/bin/activate"
export PYTHONPATH="${RELEASE_ROOT}/src:${OMNI_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
mkdir -p "${OMNI_DEVELOPMENT_ROOT}"/{table_rollouts,logs}

mapfile -t TABLE_PROMPTS < <(
  jq -r '.messages[0].content | split("\n") | .[1:] | join("\n")' "${OMNI_DATA_JSON}" \
    | awk 'NF && !seen[$0]++' \
    | sed -n "$((TABLE_PROMPT_OFFSET + 1)),$((TABLE_PROMPT_OFFSET + TABLE_PROMPT_COUNT))p"
)
[[ ${#TABLE_PROMPTS[@]} -eq ${TABLE_PROMPT_COUNT} ]] || {
  echo "expected ${TABLE_PROMPT_COUNT} unique JourneyDB table prompts" >&2
  exit 2
}

jobs="${OMNI_DEVELOPMENT_ROOT}/step_local_table_jobs.jsonl"
: > "${jobs}"
for order in random progressive_confidence; do
  for local_idx in "${!TABLE_PROMPTS[@]}"; do
    idx=$((TABLE_PROMPT_OFFSET + local_idx))
    dir="${OMNI_DEVELOPMENT_ROOT}/table_rollouts/${order}/prompt_$(printf '%04d' "${idx}")"
    extra='["--fixed-t2i-scaffold","--trace-order-stats","auto","--trace-num-phases","1","--trace-confidence-bins","8","--trace-aux-bins","16","--trace-provisional-phases"]'
    jq -cn --arg output_dir "${dir}" --arg prompt "${TABLE_PROMPTS[$local_idx]}" \
      --arg order "${order}" --argjson seed "$((20272000 + idx))" \
      --argjson extra "${extra}" \
      '{output_dir:$output_dir,prompt:$prompt,order_policy:$order,seed:$seed,steps:260,max_tokens:260,extra_args:$extra}' \
      >> "${jobs}"
  done
done

pids=()
for gpu in "${GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${SCRIPT_DIR}/run_omni_t2i_manifest.py" \
    --smoke-script "${SCRIPT_DIR}/omni_t2i_smoke.py" \
    --model-path "${OMNI_MODEL_PATH}" \
    --image-tokenizer-path "${OMNI_IMAGE_TOKENIZER_PATH}" \
    --jobs "${jobs}" > "${OMNI_DEVELOPMENT_ROOT}/logs/table_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
(( failed == 0 )) || { echo "step-local table rollout worker failed" >&2; exit 2; }

for order in random progressive_confidence; do
  [[ $(find "${OMNI_DEVELOPMENT_ROOT}/table_rollouts/${order}" -type f -name COMPLETE | wc -l) -eq ${TABLE_PROMPT_COUNT} ]] || {
    echo "incomplete step-local ${order} table rollouts" >&2
    exit 2
  }
done

export OMNI_CONTROLLER_DEVELOPMENT_DESIGN="step-local bucket recalibration"
bash "${MATCHED_ROOT}/develop_public_base_fallback_controller.sh"
