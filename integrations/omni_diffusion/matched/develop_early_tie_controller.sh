#!/usr/bin/env bash
set -euo pipefail

MATCHED_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${MATCHED_ROOT}/../../.." && pwd)"
SCRIPT_DIR="${MATCHED_ROOT}/scripts"
: "${OMNI_ROOT:?set OMNI_ROOT}"
: "${OMNI_MODEL_PATH:?set the shared Omni checkpoint}"
: "${OMNI_IMAGE_TOKENIZER_PATH:?set the MagViT tokenizer path}"
: "${OMNI_DATA_JSON:?set the tokenized JourneyDB JSONL}"
: "${OMNI_SOURCE_TABLE:?set the development-frozen bucket table}"
: "${OMNI_DEVELOPMENT_ROOT:?set the development output directory}"
: "${VIRTUAL_ENV:?activate the experiment environment}"
PYTHON="${PYTHON:-${VIRTUAL_ENV}/bin/python}"
OFFSET="${OMNI_DEVELOPMENT_OFFSET:-3000}"
COUNT="${OMNI_DEVELOPMENT_COUNT:-64}"
PROMPT_FILE="${OMNI_DEVELOPMENT_PROMPT_FILE:-}"
GPUS_TEXT="${OMNI_DEVELOPMENT_GPUS:-1 2 3 6}"
read -r -a GPUS <<< "${GPUS_TEXT//,/ }"

source "${VIRTUAL_ENV}/bin/activate"
export PYTHONPATH="${RELEASE_ROOT}/src:${OMNI_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
mkdir -p "${OMNI_DEVELOPMENT_ROOT}"/{controllers,gate,logs,summary}

specs=(
  "even4_g050_m020|0.050|0.020|64 96 128 160"
  "even4_g075_m020|0.075|0.020|64 96 128 160"
  "odd4_g050_m020|0.050|0.020|80 112 144 176"
  "odd4_g075_m020|0.075|0.020|80 112 144 176"
  "early5_g050_m020|0.050|0.020|64 80 96 112 128"
  "early5_g075_m020|0.075|0.020|64 80 96 112 128"
  "middle5_g050_m020|0.050|0.020|96 112 128 144 160"
  "middle5_g075_m020|0.075|0.020|96 112 128 144 160"
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
    --source-table "${OMNI_SOURCE_TABLE}" --output "${controller}" \
    --guidance-scale "${guidance}" --ready-count 4 --policy-warmup-steps 0 \
    --reward-action-steps "${steps[@]}" --max-base-score-gap "${gap}"
  labels+=("${label}")
  controllers+=("${controller}")
  printf '%s\t%s\n' "${label}" "${controller}" >> "${candidate_map}"
done

if [[ -n "${PROMPT_FILE}" ]]; then
  mapfile -t PROMPTS < <(awk 'NF && !seen[$0]++' "${PROMPT_FILE}" | sed -n '1,'"${COUNT}"'p')
else
  mapfile -t PROMPTS < <(
    jq -r '.messages[0].content | split("\n") | .[1:] | join("\n")' "${OMNI_DATA_JSON}" \
      | awk 'NF && !seen[$0]++' | sed -n "$((OFFSET + 1)),$((OFFSET + COUNT))p"
  )
fi
[[ ${#PROMPTS[@]} -eq ${COUNT} ]] || {
  echo "expected ${COUNT} unique prompts, found ${#PROMPTS[@]}" >&2
  exit 2
}
PROMPT_FILE_JSON=null
PROMPT_FILE_SHA_JSON=null
SELECTION_PROMPT_ARGS=(--selection-prompt-range "${OFFSET}" "$((OFFSET + COUNT - 1))")
if [[ -n "${PROMPT_FILE}" ]]; then
  PROMPT_FILE_JSON="$(printf '%s' "${PROMPT_FILE}" | jq -Rsa .)"
  PROMPT_FILE_SHA_JSON="\"$(sha256sum "${PROMPT_FILE}" | awk '{print $1}')\""
  SELECTION_PROMPT_ARGS=(--selection-prompt-file "${PROMPT_FILE}")
fi

jobs="${OMNI_DEVELOPMENT_ROOT}/gate_jobs.jsonl"
: > "${jobs}"
for method_idx in "${!labels[@]}"; do
  for local_idx in "${!PROMPTS[@]}"; do
    idx=$((OFFSET + local_idx))
    label="${labels[$method_idx]}"
    order=progressive_confidence
    extra='["--fixed-t2i-scaffold","--trace-order-stats","auto","--trace-num-phases","1","--trace-confidence-bins","8","--trace-aux-bins","16","--trace-provisional-phases"]'
    if [[ -n "${controllers[$method_idx]}" ]]; then
      order=dprm_confidence_warmup
      extra="$(jq -cn --argjson current "${extra}" --arg scorer "${controllers[$method_idx]}" \
        '$current + ["--dprm-order-scorer",$scorer,"--dprm-warmup-steps","0"]')"
    fi
    dir="${OMNI_DEVELOPMENT_ROOT}/gate/${label}/prompt_$(printf '%04d' "${idx}")"
    jq -cn --arg output_dir "${dir}" --arg prompt "${PROMPTS[$local_idx]}" \
      --arg order "${order}" --argjson seed "$((20274000 + idx))" \
      --argjson extra "${extra}" \
      '{output_dir:$output_dir,prompt:$prompt,order_policy:$order,seed:$seed,steps:260,max_tokens:260,extra_args:$extra}' \
      >> "${jobs}"
  done
done

cat > "${OMNI_DEVELOPMENT_ROOT}/run_manifest.json" <<JSON
{
  "design": "early-action confidence-tie controller development",
  "claim_eligible": false,
  "prompt_offset": ${OFFSET},
  "prompt_count": ${COUNT},
  "prompt_file": ${PROMPT_FILE_JSON},
  "prompt_file_sha256": ${PROMPT_FILE_SHA_JSON},
  "source_table": "${OMNI_SOURCE_TABLE}",
  "source_table_sha256": "$(sha256sum "${OMNI_SOURCE_TABLE}" | awk '{print $1}')",
  "shared_checkpoint": "${OMNI_MODEL_PATH}",
  "selection_metrics": ["CLIP-L/14", "CLIP-B/32"],
  "test_time_terminal_rollouts": 0,
  "complete_image_selection": false
}
JSON

pids=()
for gpu in "${GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${SCRIPT_DIR}/run_omni_t2i_manifest.py" \
    --smoke-script "${SCRIPT_DIR}/omni_t2i_smoke.py" \
    --model-path "${OMNI_MODEL_PATH}" \
    --image-tokenizer-path "${OMNI_IMAGE_TOKENIZER_PATH}" \
    --jobs "${jobs}" > "${OMNI_DEVELOPMENT_ROOT}/logs/gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
(( failed == 0 )) || exit 2

CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${PYTHON}" "${SCRIPT_DIR}/summarize_omni_eval.py" \
  --eval-root "${OMNI_DEVELOPMENT_ROOT}/gate" --orders "${labels[@]}" \
  --out-dir "${OMNI_DEVELOPMENT_ROOT}/summary" \
  --clip-model openai/clip-vit-large-patch14 --device cuda:0 --no-aesthetic --strict-clip
CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${PYTHON}" "${SCRIPT_DIR}/score_omni_records_clip.py" \
  --records "${OMNI_DEVELOPMENT_ROOT}/summary/records.json" \
  --output "${OMNI_DEVELOPMENT_ROOT}/summary/records_two_encoder.json" \
  --model "${DPRM_OMNI_CLIP_B32_PATH:-openai/clip-vit-base-patch32}" \
  --metric-name clip_b32_cosine --device cuda:0
comparisons=()
for label in "${labels[@]:1}"; do comparisons+=("confidence:${label}"); done
"${PYTHON}" "${SCRIPT_DIR}/analyze_omni_paired_results.py" \
  --records "${OMNI_DEVELOPMENT_ROOT}/summary/records_two_encoder.json" \
  --output-dir "${OMNI_DEVELOPMENT_ROOT}/summary/paired" \
  --comparisons "${comparisons[@]}"
"${PYTHON}" "${SCRIPT_DIR}/select_omni_dual_clip_controller.py" \
  --root "${OMNI_DEVELOPMENT_ROOT}/gate" \
  --paired-summary "${OMNI_DEVELOPMENT_ROOT}/summary/paired/paired_clip_summary.json" \
  --candidate-map "${candidate_map}" \
  --selected-output "${OMNI_DEVELOPMENT_ROOT}/selected_controller.json" \
  --decision-output "${OMNI_DEVELOPMENT_ROOT}/selection.json"
"${PYTHON}" "${SCRIPT_DIR}/prepare_omni_formal_controller.py" \
  --input "${OMNI_DEVELOPMENT_ROOT}/selected_controller.json" \
  --selection-decision "${OMNI_DEVELOPMENT_ROOT}/selection.json" \
  --table-prompt-range 2000 2047 \
  "${SELECTION_PROMPT_ARGS[@]}" \
  --controller-only-host \
  --output "${OMNI_DEVELOPMENT_ROOT}/formal_controller.json"
date -Is > "${OMNI_DEVELOPMENT_ROOT}/DEVELOPMENT_COMPLETE"
