#!/usr/bin/env bash
set -euo pipefail

MATCHED_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${MATCHED_ROOT}/../../.." && pwd)"
SCRIPT_DIR="${MATCHED_ROOT}/scripts"
: "${OMNI_ROOT:?set the Omni-Diffusion checkout}"
: "${OMNI_MODEL_PATH:?set the frozen Omni checkpoint}"
: "${OMNI_IMAGE_TOKENIZER_PATH:?set the MagViT tokenizer path}"
: "${OMNI_DATA_JSON:?set the tokenized JourneyDB JSONL}"
: "${OMNI_ACTION_FIT_ROOT:?set the action-fit output directory}"
: "${VIRTUAL_ENV:?set the experiment environment}"

PYTHON="${PYTHON:-${VIRTUAL_ENV}/bin/python}"
GPUS_TEXT="${OMNI_ACTION_FIT_GPUS:-0 1}"
read -r -a GPUS <<< "${GPUS_TEXT//,/ }"
PROMPT_OFFSET="${OMNI_ACTION_FIT_OFFSET:-2000}"
PROMPT_COUNT="${OMNI_ACTION_FIT_COUNT:-48}"
STEPS_TEXT="${OMNI_ACTION_STEPS:-${OMNI_ACTION_STEP:-96}}"
read -r -a STEPS <<< "${STEPS_TEXT}"
QUANTILES_TEXT="${OMNI_ACTION_QUANTILES:-0.70 0.85}"
read -r -a QUANTILES <<< "${QUANTILES_TEXT}"
RANK_BINS="${OMNI_ACTION_RANK_BINS:-8}"
SPATIAL_BINS="${OMNI_ACTION_SPATIAL_BINS:-4}"

source "${VIRTUAL_ENV}/bin/activate"
export PYTHONPATH="${RELEASE_ROOT}/src:${OMNI_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
mkdir -p "${OMNI_ACTION_FIT_ROOT}"/{baseline,branches,controllers,logs,records}

mapfile -t PROMPTS < <(
  jq -r '.messages[0].content | split("\n") | .[1:] | join("\n")' "${OMNI_DATA_JSON}" \
    | awk 'NF && !seen[$0]++' \
    | sed -n "$((PROMPT_OFFSET + 1)),$((PROMPT_OFFSET + PROMPT_COUNT))p"
)
[[ ${#PROMPTS[@]} -eq ${PROMPT_COUNT} ]] || {
  echo "expected ${PROMPT_COUNT} unique action-fit prompts" >&2
  exit 2
}

jobs="${OMNI_ACTION_FIT_ROOT}/action_jobs.jsonl"
: > "${jobs}"
for local_idx in "${!PROMPTS[@]}"; do
  idx=$((PROMPT_OFFSET + local_idx))
  prompt="${PROMPTS[$local_idx]}"
  base_dir="${OMNI_ACTION_FIT_ROOT}/baseline/prompt_$(printf '%04d' "${idx}")"
  jq -cn --arg output_dir "${base_dir}" --arg prompt "${prompt}" \
    --argjson seed "$((20276000 + idx))" \
    '{output_dir:$output_dir,prompt:$prompt,order_policy:"progressive_confidence",seed:$seed,steps:260,max_tokens:260,extra_args:["--fixed-t2i-scaffold"]}' \
    >> "${jobs}"
  for step in "${STEPS[@]}"; do
    for quantile in "${QUANTILES[@]}"; do
      branch_dir="${OMNI_ACTION_FIT_ROOT}/branches/step${step}_q${quantile}/prompt_$(printf '%04d' "${idx}")"
      extra="$(jq -cn --arg step "${step}" --arg quantile "${quantile}" \
        --arg rank_bins "${RANK_BINS}" \
        '["--fixed-t2i-scaffold","--force-order-step",$step,"--force-confidence-quantile",$quantile,"--force-confidence-bins","8","--force-rank-bins",$rank_bins,"--force-aux-bins","16","--require-forced-action"]')"
      jq -cn --arg output_dir "${branch_dir}" --arg prompt "${prompt}" \
        --argjson seed "$((20276000 + idx))" --argjson extra "${extra}" \
        '{output_dir:$output_dir,prompt:$prompt,order_policy:"progressive_confidence",seed:$seed,steps:260,max_tokens:260,extra_args:$extra}' \
        >> "${jobs}"
    done
  done
done

cat > "${OMNI_ACTION_FIT_ROOT}/run_manifest.json" <<JSON
{
  "design": "offline action-conditioned DPRM fit",
  "checkpoint": "${OMNI_MODEL_PATH}",
  "prompt_offset": ${PROMPT_OFFSET},
  "prompt_count": ${PROMPT_COUNT},
  "action_steps": "${STEPS_TEXT}",
  "confidence_rank_quantiles": "${QUANTILES_TEXT}",
  "rank_bins": ${RANK_BINS},
  "spatial_bins": ${SPATIAL_BINS},
  "terminal_reward_calls_at_test": 0
}
JSON

pids=()
for gpu in "${GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${SCRIPT_DIR}/run_omni_t2i_manifest.py" \
    --smoke-script "${SCRIPT_DIR}/omni_t2i_smoke.py" \
    --model-path "${OMNI_MODEL_PATH}" \
    --image-tokenizer-path "${OMNI_IMAGE_TOKENIZER_PATH}" \
    --jobs "${jobs}" > "${OMNI_ACTION_FIT_ROOT}/logs/action_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
(( failed == 0 )) || { echo "action-fit rollout worker failed" >&2; exit 2; }

"${PYTHON}" "${SCRIPT_DIR}/collect_omni_action_records.py" \
  --root "${OMNI_ACTION_FIT_ROOT}" \
  --output "${OMNI_ACTION_FIT_ROOT}/records/unscored.json"
CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${PYTHON}" "${SCRIPT_DIR}/score_omni_records_clip.py" \
  --records "${OMNI_ACTION_FIT_ROOT}/records/unscored.json" \
  --output "${OMNI_ACTION_FIT_ROOT}/records/clip_l14.json" \
  --model openai/clip-vit-large-patch14 --metric-name clip_cosine --device cuda:0
CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${PYTHON}" "${SCRIPT_DIR}/score_omni_records_clip.py" \
  --records "${OMNI_ACTION_FIT_ROOT}/records/clip_l14.json" \
  --output "${OMNI_ACTION_FIT_ROOT}/records/two_encoder.json" \
  --model openai/clip-vit-base-patch32 --metric-name clip_b32_cosine --device cuda:0
"${PYTHON}" "${SCRIPT_DIR}/build_omni_action_advantages.py" \
  --records "${OMNI_ACTION_FIT_ROOT}/records/two_encoder.json" \
  --output "${OMNI_ACTION_FIT_ROOT}/records/action_advantages.json"
"${PYTHON}" "${SCRIPT_DIR}/summarize_omni_action_advantages.py" \
  --records "${OMNI_ACTION_FIT_ROOT}/records/action_advantages.json" \
  --output "${OMNI_ACTION_FIT_ROOT}/records/action_summary.json"

for guidance in 0.25 0.50 1.00; do
  tag="g$(printf '%03d' "$(awk -v value="${guidance}" 'BEGIN {print 100*value}')")"
  "${PYTHON}" "${SCRIPT_DIR}/fit_omni_action_bucket_controller.py" \
    --records "${OMNI_ACTION_FIT_ROOT}/records/action_advantages.json" \
    --output "${OMNI_ACTION_FIT_ROOT}/controllers/rank_spatial_${tag}.json" \
    --active-steps "${STEPS[@]}" --rank-bins "${RANK_BINS}" --spatial-bins "${SPATIAL_BINS}" \
    --beta 1.0 --guidance-scale "${guidance}" --min-count 1 --shrinkage 4
done
date -Is > "${OMNI_ACTION_FIT_ROOT}/FIT_COMPLETE"
