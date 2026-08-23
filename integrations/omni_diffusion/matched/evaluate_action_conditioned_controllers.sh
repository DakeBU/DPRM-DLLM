#!/usr/bin/env bash
set -euo pipefail

MATCHED_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${MATCHED_ROOT}/../../.." && pwd)"
SCRIPT_DIR="${MATCHED_ROOT}/scripts"
: "${OMNI_ROOT:?set the Omni-Diffusion checkout}"
: "${OMNI_MODEL_PATH:?set the frozen Omni checkpoint}"
: "${OMNI_IMAGE_TOKENIZER_PATH:?set the MagViT tokenizer path}"
: "${OMNI_ACTION_CONTROLLER_DIR:?set the directory containing fitted controllers}"
: "${OMNI_ACTION_EVAL_ROOT:?set the development evaluation output directory}"
: "${OMNI_ACTION_EVAL_PROMPT_FILE:?set the frozen development prompt file}"
: "${VIRTUAL_ENV:?set the experiment environment}"

PYTHON="${PYTHON:-${VIRTUAL_ENV}/bin/python}"
GPUS_TEXT="${OMNI_ACTION_EVAL_GPUS:-0 1}"
read -r -a GPUS <<< "${GPUS_TEXT//,/ }"
COUNT="${OMNI_ACTION_EVAL_COUNT:-64}"
PROMPT_OFFSET="${OMNI_ACTION_EVAL_PROMPT_OFFSET:-0}"
INDEX_OFFSET="${OMNI_ACTION_EVAL_INDEX_OFFSET:-3500}"
SEED_OFFSET="${OMNI_ACTION_EVAL_SEED_OFFSET:-20274000}"

source "${VIRTUAL_ENV}/bin/activate"
export PYTHONPATH="${RELEASE_ROOT}/src:${OMNI_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
mkdir -p "${OMNI_ACTION_EVAL_ROOT}"/{gate,logs,summary}

PROMPT_START=$((PROMPT_OFFSET + 1))
PROMPT_END=$((PROMPT_OFFSET + COUNT))
prompt_stream() {
  if [[ "${OMNI_ACTION_EVAL_PROMPT_FILE}" == *.jsonl ]]; then
    "${PYTHON}" - "${OMNI_ACTION_EVAL_PROMPT_FILE}" <<'PY'
import json
import sys
from pathlib import Path

with Path(sys.argv[1]).open(encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        prompt = row.get("prompt", row.get("text"))
        if prompt:
            print(str(prompt).strip())
PY
  else
    awk 'NF' "${OMNI_ACTION_EVAL_PROMPT_FILE}"
  fi
}
mapfile -t PROMPTS < <(
  prompt_stream | awk 'NF && !seen[$0]++' | sed -n "${PROMPT_START},${PROMPT_END}p"
)
[[ ${#PROMPTS[@]} -eq ${COUNT} ]] || {
  echo "expected ${COUNT} unique prompts, found ${#PROMPTS[@]}" >&2
  exit 2
}

labels=(confidence)
controllers=("")
candidate_map="${OMNI_ACTION_EVAL_ROOT}/candidate_map.tsv"
: > "${candidate_map}"
while IFS= read -r controller; do
  artifact_format="$(jq -r '.format // empty' "${controller}")"
  case "${artifact_format}" in
    omni_bucket_table_dprm_v1|omni_rank_bucket_dprm_v1|omni_stage_rank_spatial_dprm_v1|omni_stage_rank_code_dprm_v1) ;;
    *) continue ;;
  esac
  label="$(basename "${controller}" .json)"
  labels+=("${label}")
  controllers+=("${controller}")
  printf '%s\t%s\n' "${label}" "${controller}" >> "${candidate_map}"
done < <(find -L "${OMNI_ACTION_CONTROLLER_DIR}" -maxdepth 1 -type f -name '*.json' | sort)
[[ ${#labels[@]} -gt 1 ]] || {
  echo "no controller JSON files under ${OMNI_ACTION_CONTROLLER_DIR}" >&2
  exit 2
}

jobs="${OMNI_ACTION_EVAL_ROOT}/gate_jobs.jsonl"
: > "${jobs}"
for method_idx in "${!labels[@]}"; do
  for local_idx in "${!PROMPTS[@]}"; do
    idx=$((INDEX_OFFSET + local_idx))
    label="${labels[$method_idx]}"
    order=progressive_confidence
    extra='["--fixed-t2i-scaffold","--trace-order-stats","auto","--trace-num-phases","8","--trace-confidence-bins","8","--trace-aux-bins","16"]'
    if [[ -n "${controllers[$method_idx]}" ]]; then
      order=dprm_confidence_warmup
      extra="$(jq -cn --argjson current "${extra}" --arg scorer "${controllers[$method_idx]}" \
        '$current + ["--dprm-order-scorer",$scorer,"--dprm-warmup-steps","0"]')"
    fi
    output_dir="${OMNI_ACTION_EVAL_ROOT}/gate/${label}/prompt_$(printf '%04d' "${idx}")"
    jq -cn --arg output_dir "${output_dir}" --arg prompt "${PROMPTS[$local_idx]}" \
      --arg order "${order}" --argjson seed "$((SEED_OFFSET + idx))" \
      --argjson extra "${extra}" \
      '{output_dir:$output_dir,prompt:$prompt,order_policy:$order,seed:$seed,steps:260,max_tokens:260,extra_args:$extra}' \
      >> "${jobs}"
  done
done

cat > "${OMNI_ACTION_EVAL_ROOT}/run_manifest.json" <<JSON
{
  "design": "${OMNI_ACTION_EVAL_DESIGN:-single-path development evaluation of offline action-conditioned DPRM}",
  "claim_eligible": false,
  "checkpoint": "${OMNI_MODEL_PATH}",
  "prompt_file": "${OMNI_ACTION_EVAL_PROMPT_FILE}",
  "prompt_file_sha256": "$(sha256sum "${OMNI_ACTION_EVAL_PROMPT_FILE}" | awk '{print $1}')",
  "prompt_count": ${COUNT},
  "prompt_offset": ${PROMPT_OFFSET},
  "index_offset": ${INDEX_OFFSET},
  "seed_offset": ${SEED_OFFSET},
  "selection_requires_positive_primary_ci": ${OMNI_ACTION_REQUIRE_POSITIVE_PRIMARY_CI:-1},
  "terminal_reward_calls_at_test": 0,
  "complete_image_selection": false
}
JSON

pids=()
for gpu in "${GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${SCRIPT_DIR}/run_omni_t2i_manifest.py" \
    --smoke-script "${SCRIPT_DIR}/omni_t2i_smoke.py" \
    --model-path "${OMNI_MODEL_PATH}" \
    --image-tokenizer-path "${OMNI_IMAGE_TOKENIZER_PATH}" \
    --jobs "${jobs}" > "${OMNI_ACTION_EVAL_ROOT}/logs/gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
(( failed == 0 )) || { echo "controller evaluation worker failed" >&2; exit 2; }

CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${PYTHON}" "${SCRIPT_DIR}/summarize_omni_eval.py" \
  --eval-root "${OMNI_ACTION_EVAL_ROOT}/gate" --orders "${labels[@]}" \
  --out-dir "${OMNI_ACTION_EVAL_ROOT}/summary" \
  --clip-model openai/clip-vit-large-patch14 --device cuda:0 --no-aesthetic --strict-clip
CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${PYTHON}" "${SCRIPT_DIR}/score_omni_records_clip.py" \
  --records "${OMNI_ACTION_EVAL_ROOT}/summary/records.json" \
  --output "${OMNI_ACTION_EVAL_ROOT}/summary/records_two_encoder.json" \
  --model "${DPRM_OMNI_CLIP_B32_PATH:-openai/clip-vit-base-patch32}" \
  --metric-name clip_b32_cosine --device cuda:0
comparisons=()
for label in "${labels[@]:1}"; do comparisons+=("confidence:${label}"); done
"${PYTHON}" "${SCRIPT_DIR}/analyze_omni_paired_results.py" \
  --records "${OMNI_ACTION_EVAL_ROOT}/summary/records_two_encoder.json" \
  --output-dir "${OMNI_ACTION_EVAL_ROOT}/summary/paired" \
  --comparisons "${comparisons[@]}"
selection_args=(
  --root "${OMNI_ACTION_EVAL_ROOT}/gate"
  --paired-summary "${OMNI_ACTION_EVAL_ROOT}/summary/paired/paired_clip_summary.json"
  --candidate-map "${candidate_map}"
  --selected-output "${OMNI_ACTION_EVAL_ROOT}/selected_controller.json"
  --decision-output "${OMNI_ACTION_EVAL_ROOT}/selection.json"
  --min-prompt-override-fraction 0.05
)
if [[ "${OMNI_ACTION_REQUIRE_POSITIVE_PRIMARY_CI:-1}" == 1 ]]; then
  selection_args+=(--require-positive-primary-ci)
fi
"${PYTHON}" "${SCRIPT_DIR}/select_omni_dual_clip_controller.py" \
  "${selection_args[@]}"
date -Is > "${OMNI_ACTION_EVAL_ROOT}/EVALUATION_COMPLETE"
