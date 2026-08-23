#!/usr/bin/env bash
set -euo pipefail

MATCHED_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${MATCHED_ROOT}/../../.." && pwd)"
SCRIPT_DIR="${MATCHED_ROOT}/scripts"
: "${OMNI_ROOT:?set OMNI_ROOT}"
: "${OMNI_MODEL_PATH:?set the shared Omni checkpoint}"
: "${OMNI_IMAGE_TOKENIZER_PATH:?set the MagViT tokenizer path}"
: "${OMNI_DATA_JSON:?set the tokenized JourneyDB JSONL}"
: "${OMNI_DUAL_TABLE_B_HEAVY:?set the CLIP-B-heavy dual-CLIP table}"
: "${OMNI_DEVELOPMENT_ROOT:?set the development output directory}"
: "${OMNI_DEVELOPMENT_PROMPT_FILE:?set the frozen development prompt file}"
: "${VIRTUAL_ENV:?activate the experiment environment}"
PYTHON="${PYTHON:-${VIRTUAL_ENV}/bin/python}"
OFFSET="${OMNI_DEVELOPMENT_OFFSET:-3000}"
COUNT="${OMNI_DEVELOPMENT_COUNT:-64}"
GPUS_TEXT="${OMNI_DEVELOPMENT_GPUS:-0 1}"
read -r -a GPUS <<< "${GPUS_TEXT//,/ }"

source "${VIRTUAL_ENV}/bin/activate"
export PYTHONPATH="${RELEASE_ROOT}/src:${OMNI_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "${OMNI_DEVELOPMENT_ROOT}"/{controllers,gate,logs,summary}

label=bh_lowconf_g075
controller="${OMNI_DEVELOPMENT_ROOT}/controllers/${label}.json"
"${PYTHON}" "${SCRIPT_DIR}/freeze_omni_bucket_controller.py" \
  --source-table "${OMNI_DUAL_TABLE_B_HEAVY}" --output "${controller}" \
  --guidance-scale 0.075 --ready-count 4 --policy-warmup-steps 0 \
  --reward-action-steps 96 112 128 144 160 --max-base-score-gap 0.02 \
  --max-reward-confidence-bin 0
printf '%s\t%s\n' "${label}" "${controller}" > "${OMNI_DEVELOPMENT_ROOT}/candidate_map.tsv"

mapfile -t PROMPTS < <(
  awk 'NF && !seen[$0]++' "${OMNI_DEVELOPMENT_PROMPT_FILE}" | head -n "${COUNT}"
)
[[ ${#PROMPTS[@]} -eq ${COUNT} ]] || {
  echo "expected ${COUNT} unique prompts, found ${#PROMPTS[@]}" >&2
  exit 2
}

jobs="${OMNI_DEVELOPMENT_ROOT}/gate_jobs.jsonl"
: > "${jobs}"
for method in confidence "${label}"; do
  for local_idx in "${!PROMPTS[@]}"; do
    idx=$((OFFSET + local_idx))
    order=progressive_confidence
    extra='["--fixed-t2i-scaffold","--trace-order-stats","auto","--trace-num-phases","1","--trace-confidence-bins","8","--trace-aux-bins","16","--trace-provisional-phases"]'
    if [[ "${method}" != confidence ]]; then
      order=dprm_confidence_warmup
      extra="$(jq -cn --argjson current "${extra}" --arg scorer "${controller}" \
        '$current + ["--dprm-order-scorer",$scorer,"--dprm-warmup-steps","0"]')"
    fi
    dir="${OMNI_DEVELOPMENT_ROOT}/gate/${method}/prompt_$(printf '%04d' "${idx}")"
    jq -cn --arg output_dir "${dir}" --arg prompt "${PROMPTS[$local_idx]}" \
      --arg order "${order}" --argjson seed "$((20274000 + idx))" \
      --argjson extra "${extra}" \
      '{output_dir:$output_dir,prompt:$prompt,order_policy:$order,seed:$seed,steps:260,max_tokens:260,extra_args:$extra}' \
      >> "${jobs}"
  done
done

read -r prompt_file_sha256 _ < <(sha256sum "${OMNI_DEVELOPMENT_PROMPT_FILE}")
read -r source_table_sha256 _ < <(sha256sum "${OMNI_DUAL_TABLE_B_HEAVY}")
cat > "${OMNI_DEVELOPMENT_ROOT}/run_manifest.json" <<JSON
{
  "design": "low-confidence-bin Omni controller development",
  "claim_eligible": false,
  "prompt_count": ${COUNT},
  "prompt_file": "${OMNI_DEVELOPMENT_PROMPT_FILE}",
  "prompt_file_sha256": "${prompt_file_sha256}",
  "source_table_sha256": "${source_table_sha256}",
  "development_rationale": "reward tilt is restricted to the lowest frozen confidence bin",
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
  --eval-root "${OMNI_DEVELOPMENT_ROOT}/gate" --orders confidence "${label}" \
  --out-dir "${OMNI_DEVELOPMENT_ROOT}/summary" \
  --clip-model openai/clip-vit-large-patch14 --device cuda:0 --no-aesthetic --strict-clip
CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${PYTHON}" "${SCRIPT_DIR}/score_omni_records_clip.py" \
  --records "${OMNI_DEVELOPMENT_ROOT}/summary/records.json" \
  --output "${OMNI_DEVELOPMENT_ROOT}/summary/records_two_encoder.json" \
  --model "${DPRM_OMNI_CLIP_B32_PATH:-openai/clip-vit-base-patch32}" \
  --metric-name clip_b32_cosine --device cuda:0
"${PYTHON}" "${SCRIPT_DIR}/analyze_omni_paired_results.py" \
  --records "${OMNI_DEVELOPMENT_ROOT}/summary/records_two_encoder.json" \
  --output-dir "${OMNI_DEVELOPMENT_ROOT}/summary/paired" \
  --comparisons "confidence:${label}"
"${PYTHON}" "${SCRIPT_DIR}/select_omni_dual_clip_controller.py" \
  --root "${OMNI_DEVELOPMENT_ROOT}/gate" \
  --paired-summary "${OMNI_DEVELOPMENT_ROOT}/summary/paired/paired_clip_summary.json" \
  --candidate-map "${OMNI_DEVELOPMENT_ROOT}/candidate_map.tsv" \
  --selected-output "${OMNI_DEVELOPMENT_ROOT}/selected_controller.json" \
  --decision-output "${OMNI_DEVELOPMENT_ROOT}/selection.json" \
  --min-prompt-override-fraction 0.05 --require-positive-primary-ci
"${PYTHON}" "${SCRIPT_DIR}/prepare_omni_formal_controller.py" \
  --input "${OMNI_DEVELOPMENT_ROOT}/selected_controller.json" \
  --selection-decision "${OMNI_DEVELOPMENT_ROOT}/selection.json" \
  --source-table-root "$(dirname "${OMNI_DUAL_TABLE_B_HEAVY}")" \
  --table-prompt-range 2000 2047 \
  --selection-prompt-file "${OMNI_DEVELOPMENT_PROMPT_FILE}" \
  --controller-only-host --output "${OMNI_DEVELOPMENT_ROOT}/formal_controller.json"
date -Is > "${OMNI_DEVELOPMENT_ROOT}/DEVELOPMENT_COMPLETE"
