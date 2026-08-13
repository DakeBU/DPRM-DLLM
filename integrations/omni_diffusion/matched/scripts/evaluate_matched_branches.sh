#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OMNI_ROOT="${OMNI_ROOT:?set OMNI_ROOT}"
ENV="${VIRTUAL_ENV:?activate the experiment environment before launching}"
PYTHON="${PYTHON:-${ENV}/bin/python}"
TRAIN_ROOT="${DPRM_OMNI_TRAIN_OUT:?set DPRM_OMNI_TRAIN_OUT}"
CONTROLLER="${DPRM_OMNI_GATE_CONTROLLER:?set DPRM_OMNI_GATE_CONTROLLER}"
DATA="${DPRM_OMNI_DATA_JSON:-${OMNI_ROOT}/datasets/jsonl/JourneyDB/BLIP3o_JourneyDB_T2I_tokenized.jsonl}"
IMAGE_TOKENIZER="${DPRM_OMNI_IMAGE_TOKENIZER:?set DPRM_OMNI_IMAGE_TOKENIZER}"
OUT="${DPRM_OMNI_EVAL_OUT:?set DPRM_OMNI_EVAL_OUT}"
STEP="${DPRM_OMNI_EVAL_STEP:-500}"
OFFSET="${DPRM_OMNI_EVAL_OFFSET:-2300}"
COUNT="${DPRM_OMNI_EVAL_COUNT:-96}"
GPU="${DPRM_OMNI_EVAL_GPU:-0}"

declare -A CKPT=(
  [progressive_confidence]="${TRAIN_ROOT}/confidence_matched/checkpoint-${STEP}"
  [dprm_confidence_warmup]="${TRAIN_ROOT}/dprm_matched/checkpoint-${STEP}"
)
RANDOM_CKPT="${DPRM_OMNI_RANDOM_CKPT:-}"
if [[ "${DPRM_OMNI_INCLUDE_TRAINED_RANDOM:-0}" == "1" ]]; then
  RANDOM_CKPT="${TRAIN_ROOT}/random_matched/checkpoint-${STEP}"
fi
if [[ -n "${RANDOM_CKPT}" ]]; then
  CKPT[random]="${RANDOM_CKPT}"
fi
for order in "${!CKPT[@]}"; do
  [[ -s "${CKPT[$order]}/trainer_state.json" ]] || {
    echo "missing trained checkpoint for ${order}: ${CKPT[$order]}" >&2
    exit 2
  }
done
mkdir -p "${OUT}"
source "${ENV}/bin/activate"
export PYTHONPATH="${OMNI_ROOT}:${RELEASE_ROOT}/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
"${PYTHON}" "${SCRIPT_DIR}/audit_omni_training_contract.py" \
  --train-root "${TRAIN_ROOT}" --controller "${CONTROLLER}" --step "${STEP}" \
  --omni-root "${OMNI_ROOT}" --release-root "${RELEASE_ROOT}" \
  --output "${OUT}/training_contract_audit.json"
mapfile -t PROMPTS < <(
  jq -r '.messages[0].content | split("\n") | .[1:] | join("\n")' "${DATA}" \
    | awk 'NF && !seen[$0]++' | sed -n "$((OFFSET + 1)),$((OFFSET + COUNT))p"
)
[[ ${#PROMPTS[@]} -eq ${COUNT} ]] || exit 2
cat > "${OUT}/run_manifest.json" <<JSON
{
  "design": "train-test-matched single-path independent evaluation",
  "training_root": "${TRAIN_ROOT}",
  "controller": "${CONTROLLER}",
  "prompt_offset": ${OFFSET},
  "prompt_count": ${COUNT},
  "main_figure_prompt_id": "prompt_$(printf '%04d' "${OFFSET}")",
  "supplement_figure_prompt_ids": [
    "prompt_$(printf '%04d' "$((OFFSET + 1))")",
    "prompt_$(printf '%04d' "$((OFFSET + 2))")",
    "prompt_$(printf '%04d' "$((OFFSET + 3))")"
  ],
  "paths_per_prompt": 1,
  "ordered_action_space": "256 visual-code positions; four T2I format tokens fixed",
  "fixed_t2i_scaffold": true,
  "test_time_terminal_rollouts": 0,
  "complete_image_selection": false,
  "outcome_ranked_visual_selection": false,
  "aesthetic_scoring": false,
  "evaluation_metrics": ["CLIP-L/14", "CLIP-B/32"],
  "training_contract_audit": "${OUT}/training_contract_audit.json",
  "training_contract_audit_sha256": "$(sha256sum "${OUT}/training_contract_audit.json" | awk '{print $1}')",
  "orders": [$(printf '"%s",' "${!CKPT[@]}" | sed 's/,$//')]
}
JSON

orders=(progressive_confidence dprm_confidence_warmup)
[[ -n "${RANDOM_CKPT}" ]] && orders=(random "${orders[@]}")
for order in "${orders[@]}"; do
  jobs="${OUT}/jobs_${order}.jsonl"
  : > "${jobs}"
  for local_idx in "${!PROMPTS[@]}"; do
    idx=$((OFFSET + local_idx)); dir="${OUT}/${order}/prompt_$(printf '%04d' "${idx}")"
    mkdir -p "${dir}"
    extra='["--fixed-t2i-scaffold","--trace-order-stats","auto","--trace-num-phases","1","--trace-confidence-bins","8","--trace-aux-bins","16","--trace-provisional-phases"]'
    if [[ "${order}" == "dprm_confidence_warmup" ]]; then
      extra="$(jq -cn --arg scorer "${CONTROLLER}" '["--fixed-t2i-scaffold","--trace-order-stats","auto","--trace-num-phases","1","--trace-confidence-bins","8","--trace-aux-bins","16","--trace-provisional-phases","--dprm-order-scorer",$scorer,"--dprm-warmup-steps","0"]')"
    fi
    # The first four prompt ids are fixed before scoring and retain decoded
    # intermediate canvases for the manuscript mechanism audit.
    if (( local_idx < 4 )); then
      extra="$(jq -cn --argjson current "${extra}" '$current + ["--save-history-frames","--history-frame-stride","32","--history-frame-limit","9"]')"
    fi
    jq -cn --arg output_dir "${dir}" --arg prompt "${PROMPTS[$local_idx]}" \
      --arg order "${order}" --argjson seed "$((20270000 + idx))" \
      --argjson extra "${extra}" \
      '{output_dir:$output_dir,prompt:$prompt,order_policy:$order,seed:$seed,steps:260,max_tokens:260,extra_args:$extra}' \
      >> "${jobs}"
  done
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${SCRIPT_DIR}/run_omni_t2i_manifest.py" \
    --smoke-script "${SCRIPT_DIR}/omni_t2i_smoke.py" \
    --model-path "${CKPT[$order]}" --image-tokenizer-path "${IMAGE_TOKENIZER}" \
    --jobs "${jobs}" >> "${OUT}/${order}.log" 2>&1
done
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${SCRIPT_DIR}/summarize_omni_eval.py" \
  --eval-root "${OUT}" --orders "${orders[@]}" \
  --out-dir "${OUT}/summary" --clip-model openai/clip-vit-large-patch14 --device cuda:0 \
  --no-aesthetic --strict-clip
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${SCRIPT_DIR}/score_omni_records_clip.py" \
  --records "${OUT}/summary/records.json" \
  --output "${OUT}/summary/records_two_encoder.json" \
  --model "${DPRM_OMNI_CLIP_B32_PATH:-openai/clip-vit-base-patch32}" \
  --metric-name clip_b32_cosine --device cuda:0
"${PYTHON}" "${SCRIPT_DIR}/analyze_omni_paired_results.py" \
  --records "${OUT}/summary/records_two_encoder.json" \
  --output-dir "${OUT}/summary/paired"
"${PYTHON}" "${SCRIPT_DIR}/analyze_omni_paired_order_divergence.py" \
  --formal-root "${OUT}" --output-dir "${OUT}/summary/order_divergence"
"${PYTHON}" "${SCRIPT_DIR}/package_omni_formal_visual_audit.py" \
  --records "${OUT}/summary/records_two_encoder.json" \
  --summary "${OUT}/summary/summary.json" \
  --out-dir "${OUT}/human_visual_audit" --orders "${orders[@]}" --num-examples 12
"${PYTHON}" "${SCRIPT_DIR}/package_omni_matched_intermediates.py" \
  --records "${OUT}/summary/records_two_encoder.json" \
  --output-dir "${OUT}/fixed_intermediate_canvases" --fixed-count 4
"${PYTHON}" "${SCRIPT_DIR}/check_omni_matched_promotion.py" \
  --paired "${OUT}/summary/paired/paired_clip_summary.json" \
  --divergence "${OUT}/summary/order_divergence/paired_order_divergence.json" \
  --controller "${CONTROLLER}" --run-manifest "${OUT}/run_manifest.json" \
  --output "${OUT}/promotion/promotion_report.json" \
  --expected-prompts "${COUNT}"
date -Is > "${OUT}/EVAL_COMPLETE"
