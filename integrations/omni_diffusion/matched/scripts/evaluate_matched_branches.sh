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
PROMPT_JSONL="${DPRM_OMNI_PROMPT_JSONL:-}"
IMAGE_TOKENIZER="${DPRM_OMNI_IMAGE_TOKENIZER:?set DPRM_OMNI_IMAGE_TOKENIZER}"
OUT="${DPRM_OMNI_EVAL_OUT:?set DPRM_OMNI_EVAL_OUT}"
STEP="${DPRM_OMNI_EVAL_STEP:-1000}"
OFFSET="${DPRM_OMNI_EVAL_OFFSET:-2500}"
COUNT="${DPRM_OMNI_EVAL_COUNT:-96}"
EVAL_ROLE="${DPRM_OMNI_EVAL_ROLE:-confirmation}"
EVAL_GPUS_TEXT="${DPRM_OMNI_EVAL_GPUS:-${DPRM_OMNI_EVAL_GPU:-0}}"
read -r -a EVAL_GPUS <<< "${EVAL_GPUS_TEXT//,/ }"
[[ ${#EVAL_GPUS[@]} -gt 0 ]] || { echo "no Omni evaluation GPU supplied" >&2; exit 2; }
METRIC_GPU="${EVAL_GPUS[0]}"
if [[ -n "${PROMPT_JSONL}" ]]; then
  DEFAULT_FIXED_VISUAL_PROMPT_IDS=""
else
  DEFAULT_FIXED_VISUAL_PROMPT_IDS="2501 2521 2524 2554"
fi
FIXED_VISUAL_PROMPT_IDS_TEXT="${DPRM_OMNI_FIXED_VISUAL_PROMPT_IDS-${DEFAULT_FIXED_VISUAL_PROMPT_IDS}}"
read -r -a FIXED_VISUAL_PROMPT_IDS <<< "${FIXED_VISUAL_PROMPT_IDS_TEXT}"
VISUAL_PROMPT_MANIFEST="${DPRM_OMNI_VISUAL_PROMPT_MANIFEST:-${RELEASE_ROOT}/reproducibility/omni_visual_prompts.json}"

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
  --orders confidence_matched dprm_matched \
  --output "${OUT}/training_contract_audit.json"
if [[ -n "${PROMPT_JSONL}" ]]; then
  [[ -s "${PROMPT_JSONL}" ]] || { echo "missing frozen prompt JSONL: ${PROMPT_JSONL}" >&2; exit 2; }
  mapfile -t PROMPTS < <(jq -r '.prompt' "${PROMPT_JSONL}" | sed -n "1,${COUNT}p")
else
  mapfile -t PROMPTS < <(
    jq -r '.messages[0].content | split("\n") | .[1:] | join("\n")' "${DATA}" \
      | awk 'NF && !seen[$0]++' | sed -n "$((OFFSET + 1)),$((OFFSET + COUNT))p"
  )
fi
[[ ${#PROMPTS[@]} -eq ${COUNT} ]] || exit 2
if [[ -n "${PROMPT_JSONL}" ]]; then
  "${PYTHON}" - "${PROMPT_JSONL}" "${COUNT}" "${OUT}/prompt_validation.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

source, expected_count, output = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
selected = rows[:expected_count]
prompts = [row.get("prompt", "").strip() for row in selected]
prompt_ids = [row.get("prompt_id", "").strip() for row in selected]
if len(selected) != expected_count or not all(prompts) or not all(prompt_ids):
    raise SystemExit("frozen Omni prompt JSONL has missing rows, prompts, or prompt ids")
if len(set(prompts)) != expected_count or len(set(prompt_ids)) != expected_count:
    raise SystemExit("frozen Omni prompt JSONL contains duplicate prompts or prompt ids")
payload = {
    "prompt_count": expected_count,
    "unique_prompt_count": len(set(prompts)),
    "unique_prompt_id_count": len(set(prompt_ids)),
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
}
output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
  VISUAL_PROMPT_MANIFEST="${PROMPT_JSONL}"
  VISUAL_PROMPT_VALIDATION="${OUT}/prompt_validation.json"
else
  "${PYTHON}" "${SCRIPT_DIR}/validate_omni_visual_prompts.py" \
    --data "${DATA}" --manifest "${VISUAL_PROMPT_MANIFEST}" \
    --eval-offset "${OFFSET}" --eval-count "${COUNT}" \
    --expected-prompt-ids "${FIXED_VISUAL_PROMPT_IDS[@]}" \
    --output "${OUT}/visual_prompt_validation.json"
  VISUAL_PROMPT_VALIDATION="${OUT}/visual_prompt_validation.json"
fi
FIXED_VISUAL_IDS_JSON="$(
  printf '%s\n' "${FIXED_VISUAL_PROMPT_IDS[@]}" \
    | awk 'NF {printf "prompt_%04d\n", $1}' \
    | jq -R . | jq -s .
)"
cat > "${OUT}/run_manifest.json" <<JSON
{
  "design": "train-test-matched single-path independent evaluation",
  "training_root": "${TRAIN_ROOT}",
  "controller": "${CONTROLLER}",
  "prompt_jsonl": "${PROMPT_JSONL}",
  "prompt_jsonl_sha256": "$([[ -s "${PROMPT_JSONL}" ]] && sha256sum "${PROMPT_JSONL}" | awk '{print $1}' || printf '')",
  "prompt_offset": ${OFFSET},
  "prompt_count": ${COUNT},
  "evaluation_role": "${EVAL_ROLE}",
  "fixed_visual_prompt_ids": ${FIXED_VISUAL_IDS_JSON},
  "visual_prompt_preregistration": "${VISUAL_PROMPT_MANIFEST}",
  "visual_prompt_preregistration_sha256": "$(sha256sum "${VISUAL_PROMPT_MANIFEST}" | awk '{print $1}')",
  "visual_prompt_validation": "${VISUAL_PROMPT_VALIDATION}",
  "visual_prompt_validation_sha256": "$(sha256sum "${VISUAL_PROMPT_VALIDATION}" | awk '{print $1}')",
  "paths_per_prompt": 1,
  "ordered_action_space": "256 visual-code positions; four T2I format tokens fixed",
  "fixed_t2i_scaffold": true,
  "test_time_terminal_rollouts": 0,
  "complete_image_selection": false,
  "outcome_ranked_visual_selection": false,
  "aesthetic_scoring": false,
  "evaluation_metrics": ["CLIP-L/14", "CLIP-B/32"],
  "evaluation_gpus": [$(printf '"%s",' "${EVAL_GPUS[@]}" | sed 's/,$//')],
  "training_contract_audit": "${OUT}/training_contract_audit.json",
  "training_contract_audit_sha256": "$(sha256sum "${OUT}/training_contract_audit.json" | awk '{print $1}')",
  "orders": [$(printf '"%s",' "${!CKPT[@]}" | sed 's/,$//')]
}
JSON

orders=(progressive_confidence dprm_confidence_warmup)
[[ -n "${RANDOM_CKPT}" ]] && orders=(random "${orders[@]}")
if (( ${#EVAL_GPUS[@]} < ${#orders[@]} )); then
  echo "Omni evaluation needs at least one GPU per order" >&2
  exit 2
fi
generation_pids=()
for order_idx in "${!orders[@]}"; do
  order="${orders[$order_idx]}"
  assigned_gpus=()
  for gpu_idx in "${!EVAL_GPUS[@]}"; do
    if (( gpu_idx % ${#orders[@]} == order_idx )); then
      assigned_gpus+=("${EVAL_GPUS[$gpu_idx]}")
    fi
  done
  jobs="${OUT}/jobs_${order}.jsonl"
  : > "${jobs}"
  shard_jobs=()
  for shard_idx in "${!assigned_gpus[@]}"; do
    shard="${OUT}/jobs_${order}_shard_${shard_idx}.jsonl"
    : > "${shard}"
    shard_jobs+=("${shard}")
  done
  for local_idx in "${!PROMPTS[@]}"; do
    idx=$((OFFSET + local_idx)); dir="${OUT}/${order}/prompt_$(printf '%04d' "${idx}")"
    mkdir -p "${dir}"
    extra='["--fixed-t2i-scaffold","--trace-order-stats","auto","--trace-num-phases","1","--trace-confidence-bins","8","--trace-aux-bins","16","--trace-provisional-phases"]'
    if [[ "${order}" == "dprm_confidence_warmup" ]]; then
      extra="$(jq -cn --arg scorer "${CONTROLLER}" '["--fixed-t2i-scaffold","--trace-order-stats","auto","--trace-num-phases","1","--trace-confidence-bins","8","--trace-aux-bins","16","--trace-provisional-phases","--dprm-order-scorer",$scorer,"--dprm-warmup-steps","0"]')"
    fi
    save_history=0
    for fixed_id in "${FIXED_VISUAL_PROMPT_IDS[@]}"; do
      [[ "${idx}" == "${fixed_id}" ]] && save_history=1
    done
    if (( save_history == 1 )); then
      extra="$(jq -cn --argjson current "${extra}" '$current + ["--save-history-frames","--history-frame-stride","32","--history-frame-limit","9"]')"
    fi
    job_line="$(jq -cn --arg output_dir "${dir}" --arg prompt "${PROMPTS[$local_idx]}" \
      --arg order "${order}" --argjson seed "$((20270000 + idx))" \
      --argjson extra "${extra}" \
      '{output_dir:$output_dir,prompt:$prompt,order_policy:$order,seed:$seed,steps:260,max_tokens:260,extra_args:$extra}')"
    printf '%s\n' "${job_line}" >> "${jobs}"
    shard_idx=$((local_idx % ${#shard_jobs[@]}))
    printf '%s\n' "${job_line}" >> "${shard_jobs[$shard_idx]}"
  done
  for shard_idx in "${!shard_jobs[@]}"; do
    order_gpu="${assigned_gpus[$shard_idx]}"
    CUDA_VISIBLE_DEVICES="${order_gpu}" "${PYTHON}" "${SCRIPT_DIR}/run_omni_t2i_manifest.py" \
      --smoke-script "${SCRIPT_DIR}/omni_t2i_smoke.py" \
      --model-path "${CKPT[$order]}" --image-tokenizer-path "${IMAGE_TOKENIZER}" \
      --jobs "${shard_jobs[$shard_idx]}" \
      >> "${OUT}/${order}_shard_${shard_idx}.log" 2>&1 &
    generation_pids+=("$!")
  done
done
generation_failed=0
for pid in "${generation_pids[@]}"; do
  if ! wait "${pid}"; then
    generation_failed=1
  fi
done
(( generation_failed == 0 )) || exit 2
CUDA_VISIBLE_DEVICES="${METRIC_GPU}" "${PYTHON}" "${SCRIPT_DIR}/summarize_omni_eval.py" \
  --eval-root "${OUT}" --orders "${orders[@]}" \
  --out-dir "${OUT}/summary" --clip-model openai/clip-vit-large-patch14 --device cuda:0 \
  --no-aesthetic --strict-clip
CUDA_VISIBLE_DEVICES="${METRIC_GPU}" "${PYTHON}" "${SCRIPT_DIR}/score_omni_records_clip.py" \
  --records "${OUT}/summary/records.json" \
  --output "${OUT}/summary/records_two_encoder.json" \
  --model "${DPRM_OMNI_CLIP_B32_PATH:-openai/clip-vit-base-patch32}" \
  --metric-name clip_b32_cosine --device cuda:0
"${PYTHON}" "${SCRIPT_DIR}/analyze_omni_paired_results.py" \
  --records "${OUT}/summary/records_two_encoder.json" \
  --output-dir "${OUT}/summary/paired"
"${PYTHON}" "${SCRIPT_DIR}/analyze_omni_paired_order_divergence.py" \
  --formal-root "${OUT}" --output-dir "${OUT}/summary/order_divergence"
visual_args=()
if (( ${#FIXED_VISUAL_PROMPT_IDS[@]} > 0 )); then
  visual_args=(--fixed-prompt-ids $(printf 'prompt_%04d ' "${FIXED_VISUAL_PROMPT_IDS[@]}"))
fi
"${PYTHON}" "${SCRIPT_DIR}/package_omni_formal_visual_audit.py" \
  --records "${OUT}/summary/records_two_encoder.json" \
  --summary "${OUT}/summary/summary.json" \
  --out-dir "${OUT}/human_visual_audit" --orders "${orders[@]}" --num-examples 12 \
  "${visual_args[@]}"
date -Is > "${OUT}/human_visual_audit/VISUAL_REVIEW_PENDING"
if (( ${#FIXED_VISUAL_PROMPT_IDS[@]} > 0 )); then
  "${PYTHON}" "${SCRIPT_DIR}/package_omni_matched_intermediates.py" \
    --records "${OUT}/summary/records_two_encoder.json" \
    --output-dir "${OUT}/fixed_intermediate_canvases" \
    --prompt-ids $(printf 'prompt_%04d ' "${FIXED_VISUAL_PROMPT_IDS[@]}")
fi
if [[ "${DPRM_OMNI_SKIP_PROMOTION:-0}" != "1" ]]; then
  "${PYTHON}" "${SCRIPT_DIR}/check_omni_matched_promotion.py" \
    --paired "${OUT}/summary/paired/paired_clip_summary.json" \
    --divergence "${OUT}/summary/order_divergence/paired_order_divergence.json" \
    --controller "${CONTROLLER}" --run-manifest "${OUT}/run_manifest.json" \
    --output "${OUT}/promotion/promotion_report.json" \
    --expected-prompts "${COUNT}" --role "${EVAL_ROLE}"
else
  date -Is > "${OUT}/QUALITATIVE_PROTOCOL_COMPLETE"
fi
date -Is > "${OUT}/EVAL_COMPLETE"
