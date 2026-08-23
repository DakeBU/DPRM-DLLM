#!/usr/bin/env bash
set -euo pipefail

MATCHED_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${MATCHED_ROOT}/../../.." && pwd)"
SCRIPT_DIR="${MATCHED_ROOT}/scripts"
: "${OMNI_ROOT:?set OMNI_ROOT}"
: "${OMNI_MODEL_PATH:?set the shared Omni checkpoint}"
: "${OMNI_IMAGE_TOKENIZER_PATH:?set the MagViT tokenizer path}"
: "${OMNI_DATA_JSON:?set the tokenized JourneyDB JSONL}"
: "${OMNI_DEVELOPMENT_ROOT:?set the development output directory}"
: "${VIRTUAL_ENV:?activate the experiment environment}"
PYTHON="${PYTHON:-${VIRTUAL_ENV}/bin/python}"
GPUS_TEXT="${OMNI_DEVELOPMENT_GPUS:-0}"
read -r -a GPUS <<< "${GPUS_TEXT}"

source "${VIRTUAL_ENV}/bin/activate"
export PYTHONPATH="${RELEASE_ROOT}/src:${OMNI_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
mkdir -p "${OMNI_DEVELOPMENT_ROOT}"/{table_rollouts,controllers,gate,logs}

run_jobs() {
  local jobs="$1" log_prefix="$2"
  local pids=()
  for gpu in "${GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${SCRIPT_DIR}/run_omni_t2i_manifest.py" \
      --smoke-script "${SCRIPT_DIR}/omni_t2i_smoke.py" \
      --model-path "${OMNI_MODEL_PATH}" \
      --image-tokenizer-path "${OMNI_IMAGE_TOKENIZER_PATH}" \
      --jobs "${jobs}" >> "${OMNI_DEVELOPMENT_ROOT}/logs/${log_prefix}_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "${pid}"; done
}

mapfile -t TABLE_PROMPTS < <(
  jq -r '.messages[0].content | split("\n") | .[1:] | join("\n")' "${OMNI_DATA_JSON}" \
    | awk 'NF && !seen[$0]++' | sed -n '2001,2048p'
)
[[ ${#TABLE_PROMPTS[@]} -eq 48 ]] || exit 2
table_jobs="${OMNI_DEVELOPMENT_ROOT}/table_jobs.jsonl"
: > "${table_jobs}"
for order in random progressive_confidence; do
  for local_idx in "${!TABLE_PROMPTS[@]}"; do
    idx=$((2000 + local_idx))
    dir="${OMNI_DEVELOPMENT_ROOT}/table_rollouts/${order}/prompt_$(printf '%04d' "${idx}")"
    extra='["--fixed-t2i-scaffold","--trace-order-stats","auto","--trace-num-phases","1","--trace-confidence-bins","8","--trace-aux-bins","16"]'
    jq -cn --arg output_dir "${dir}" --arg prompt "${TABLE_PROMPTS[$local_idx]}" \
      --arg order "${order}" --argjson seed "$((20272000 + idx))" --argjson extra "${extra}" \
      '{output_dir:$output_dir,prompt:$prompt,order_policy:$order,seed:$seed,steps:260,max_tokens:260,extra_args:$extra}' \
      >> "${table_jobs}"
  done
done
run_jobs "${table_jobs}" table

table="${OMNI_DEVELOPMENT_ROOT}/development_table.json"
CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${PYTHON}" "${SCRIPT_DIR}/build_omni_dprm_table.py" \
  --rollout-root "${OMNI_DEVELOPMENT_ROOT}/table_rollouts" \
  --orders random progressive_confidence --out "${table}" \
  --clip-model openai/clip-vit-large-patch14 --device cuda:0 \
  --deduplicate-prompt-text --reward-normalization paired_prompt_advantage \
  --num-phases 1 --phase-source step --confidence-bins 8 \
  --confidence-binning development_quantile --aux-bins 16 \
  --reward-temperature 1.0 --ready-count 4 --warmup-steps 0 --switch-steps 64 \
  --require-fixed-visual-canvas

specs=(
  "s1_g075_m050|0.075|0.050|96"
  "s3_g050_m050|0.050|0.050|96 128 160"
  "s3_g075_m020|0.075|0.020|96 128 160"
  "s3_g075_m050|0.075|0.050|96 128 160"
  "s5_g050_m050|0.050|0.050|64 96 128 160 192"
  "s5_g075_m020|0.075|0.020|64 96 128 160 192"
  "s5_g075_m050|0.075|0.050|64 96 128 160 192"
  "s5_g100_m050|0.100|0.050|64 96 128 160 192"
  "s1_g150_m100|0.150|0.100|96"
  "s1b_g150_m100|0.150|0.100|128"
  "s3_g100_m100|0.100|0.100|96 128 160"
  "s3_g150_m050|0.150|0.050|96 128 160"
  "s3_g150_m100|0.150|0.100|96 128 160"
  "s3_g250_m100|0.250|0.100|96 128 160"
  "s5_g150_m100|0.150|0.100|64 96 128 160 192"
  "s5_g250_m100|0.250|0.100|64 96 128 160 192"
)
labels=(confidence)
controllers=("")
candidate_map="${OMNI_DEVELOPMENT_ROOT}/candidate_map.tsv"
: > "${candidate_map}"
for spec in "${specs[@]}"; do
  IFS='|' read -r label guidance gap steps_text <<< "${spec}"
  read -r -a steps <<< "${steps_text}"
  controller="${OMNI_DEVELOPMENT_ROOT}/controllers/${label}.json"
  "${PYTHON}" "${SCRIPT_DIR}/freeze_omni_bucket_controller.py" \
    --source-table "${table}" --output "${controller}" \
    --guidance-scale "${guidance}" --ready-count 4 --policy-warmup-steps 0 \
    --reward-action-steps "${steps[@]}" --max-base-score-gap "${gap}"
  labels+=("${label}"); controllers+=("${controller}")
  printf '%s\t%s\n' "${label}" "${controller}" >> "${candidate_map}"
done

mapfile -t GATE_PROMPTS < <(
  jq -r '.messages[0].content | split("\n") | .[1:] | join("\n")' "${OMNI_DATA_JSON}" \
    | awk 'NF && !seen[$0]++' | sed -n '2101,2132p'
)
[[ ${#GATE_PROMPTS[@]} -eq 32 ]] || exit 2
gate_jobs="${OMNI_DEVELOPMENT_ROOT}/gate_jobs.jsonl"
: > "${gate_jobs}"
for method_idx in "${!labels[@]}"; do
  for local_idx in "${!GATE_PROMPTS[@]}"; do
    idx=$((2100 + local_idx)); label="${labels[$method_idx]}"; order=progressive_confidence
    extra='["--fixed-t2i-scaffold","--trace-order-stats","auto","--trace-num-phases","1","--trace-confidence-bins","8","--trace-aux-bins","16"]'
    if [[ -n "${controllers[$method_idx]}" ]]; then
      order=dprm_confidence_warmup
      extra="$(jq -cn --arg scorer "${controllers[$method_idx]}" '["--fixed-t2i-scaffold","--trace-order-stats","auto","--trace-num-phases","1","--trace-confidence-bins","8","--trace-aux-bins","16","--dprm-order-scorer",$scorer,"--dprm-warmup-steps","0"]')"
    fi
    dir="${OMNI_DEVELOPMENT_ROOT}/gate/${label}/prompt_$(printf '%04d' "${idx}")"
    jq -cn --arg output_dir "${dir}" --arg prompt "${GATE_PROMPTS[$local_idx]}" \
      --arg order "${order}" --argjson seed "$((20273000 + idx))" --argjson extra "${extra}" \
      '{output_dir:$output_dir,prompt:$prompt,order_policy:$order,seed:$seed,steps:260,max_tokens:260,extra_args:$extra}' \
      >> "${gate_jobs}"
  done
done
run_jobs "${gate_jobs}" gate

summary="${OMNI_DEVELOPMENT_ROOT}/gate/sweep_summary.json"
CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${PYTHON}" \
  "${SCRIPT_DIR}/summarize_omni_controller_sweep.py" \
  --root "${OMNI_DEVELOPMENT_ROOT}/gate" --labels "${labels[@]}" \
  --output "${summary}" --clip-model openai/clip-vit-large-patch14 --device cuda:0
"${PYTHON}" "${SCRIPT_DIR}/select_omni_stagewise_controller.py" \
  --root "${OMNI_DEVELOPMENT_ROOT}/gate" --summary "${summary}" \
  --candidate-map "${candidate_map}" \
  --selected-output "${OMNI_DEVELOPMENT_ROOT}/selected_controller.json" \
  --decision-output "${OMNI_DEVELOPMENT_ROOT}/selection.json"
"${PYTHON}" "${SCRIPT_DIR}/prepare_omni_formal_controller.py" \
  --input "${OMNI_DEVELOPMENT_ROOT}/selected_controller.json" \
  --selection-decision "${OMNI_DEVELOPMENT_ROOT}/selection.json" \
  --table-prompt-range 2000 2047 \
  --selection-prompt-range 2100 2131 \
  --output "${OMNI_DEVELOPMENT_ROOT}/formal_controller.json"
date -Is > "${OMNI_DEVELOPMENT_ROOT}/DEVELOPMENT_COMPLETE"
