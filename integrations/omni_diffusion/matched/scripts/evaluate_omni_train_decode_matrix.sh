#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OMNI_ROOT="${OMNI_ROOT:?set OMNI_ROOT}"
ENV="${VIRTUAL_ENV:?activate the experiment environment before launching}"
PYTHON="${PYTHON:-${ENV}/bin/python}"
DATA="${DPRM_OMNI_DATA_JSON:?set DPRM_OMNI_DATA_JSON}"
IMAGE_TOKENIZER="${DPRM_OMNI_IMAGE_TOKENIZER:?set DPRM_OMNI_IMAGE_TOKENIZER}"
CONTROLLER="${DPRM_OMNI_GATE_CONTROLLER:?set DPRM_OMNI_GATE_CONTROLLER}"
CONFIDENCE_CKPT="${DPRM_OMNI_CONFIDENCE_CHECKPOINT:?set DPRM_OMNI_CONFIDENCE_CHECKPOINT}"
DPRM_CKPT="${DPRM_OMNI_DPRM_CHECKPOINT:?set DPRM_OMNI_DPRM_CHECKPOINT}"
OUT="${DPRM_OMNI_MATRIX_OUT:?set DPRM_OMNI_MATRIX_OUT}"
OFFSET="${DPRM_OMNI_MATRIX_OFFSET:-1000}"
COUNT="${DPRM_OMNI_MATRIX_COUNT:-128}"
GPUS_TEXT="${DPRM_OMNI_MATRIX_GPUS:-1,2,3,6}"
read -r -a GPUS <<< "${GPUS_TEXT//,/ }"
[[ ${#GPUS[@]} -ge 4 ]] || { echo "the four-cell matrix requires four GPUs" >&2; exit 2; }

for path in "${DATA}" "${IMAGE_TOKENIZER}" "${CONTROLLER}" \
  "${CONFIDENCE_CKPT}" "${DPRM_CKPT}"; do
  [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 2; }
done

source "${ENV}/bin/activate"
export PYTHONPATH="${OMNI_ROOT}:${RELEASE_ROOT}/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${OUT}"

mapfile -t PROMPTS < <(
  jq -r '.messages[0].content | split("\n") | .[1:] | join("\n")' "${DATA}" \
    | awk 'NF && !seen[$0]++' | sed -n "$((OFFSET + 1)),$((OFFSET + COUNT))p"
)
[[ ${#PROMPTS[@]} -eq ${COUNT} ]] || {
  echo "expected ${COUNT} unique prompts, found ${#PROMPTS[@]}" >&2
  exit 2
}

cells=(
  confidence_train__confidence_decode
  confidence_train__dprm_decode
  dprm_train__confidence_decode
  dprm_train__dprm_decode
)
models=("${CONFIDENCE_CKPT}" "${CONFIDENCE_CKPT}" "${DPRM_CKPT}" "${DPRM_CKPT}")
policies=(progressive_confidence dprm_confidence_warmup progressive_confidence dprm_confidence_warmup)

cat > "${OUT}/run_manifest.json" <<JSON
{
  "design": "post-hoc train-by-decode mechanism matrix",
  "claim_eligible": false,
  "prompt_offset": ${OFFSET},
  "prompt_count": ${COUNT},
  "paths_per_cell_per_prompt": 1,
  "confidence_checkpoint": "${CONFIDENCE_CKPT}",
  "dprm_checkpoint": "${DPRM_CKPT}",
  "controller": "${CONTROLLER}",
  "controller_sha256": "$(sha256sum "${CONTROLLER}" | awk '{print $1}')",
  "test_time_terminal_rollouts": 0,
  "complete_image_selection": false,
  "cells": [$(printf '"%s",' "${cells[@]}" | sed 's/,$//')]
}
JSON

pids=()
for i in "${!cells[@]}"; do
  cell="${cells[$i]}"
  model="${models[$i]}"
  policy="${policies[$i]}"
  jobs="${OUT}/jobs_${cell}.jsonl"
  : > "${jobs}"
  for local_idx in "${!PROMPTS[@]}"; do
    idx=$((OFFSET + local_idx))
    dir="${OUT}/${cell}/prompt_$(printf '%04d' "${idx}")"
    extra='["--fixed-t2i-scaffold","--trace-order-stats","auto","--trace-num-phases","1","--trace-confidence-bins","8","--trace-aux-bins","16","--trace-provisional-phases"]'
    if [[ "${policy}" == "dprm_confidence_warmup" ]]; then
      extra="$(jq -cn --argjson current "${extra}" --arg scorer "${CONTROLLER}" \
        '$current + ["--dprm-order-scorer",$scorer,"--dprm-warmup-steps","0"]')"
    fi
    jq -cn --arg output_dir "${dir}" --arg prompt "${PROMPTS[$local_idx]}" \
      --arg order "${policy}" --argjson seed "$((20270000 + idx))" \
      --argjson extra "${extra}" \
      '{output_dir:$output_dir,prompt:$prompt,order_policy:$order,seed:$seed,steps:260,max_tokens:260,extra_args:$extra}' \
      >> "${jobs}"
  done
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "${PYTHON}" "${SCRIPT_DIR}/run_omni_t2i_manifest.py" \
    --smoke-script "${SCRIPT_DIR}/omni_t2i_smoke.py" \
    --model-path "${model}" --image-tokenizer-path "${IMAGE_TOKENIZER}" \
    --jobs "${jobs}" > "${OUT}/${cell}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
(( failed == 0 )) || exit 2

CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${PYTHON}" "${SCRIPT_DIR}/summarize_omni_eval.py" \
  --eval-root "${OUT}" --orders "${cells[@]}" --out-dir "${OUT}/summary" \
  --clip-model openai/clip-vit-large-patch14 --device cuda:0 --no-aesthetic --strict-clip
CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${PYTHON}" "${SCRIPT_DIR}/score_omni_records_clip.py" \
  --records "${OUT}/summary/records.json" \
  --output "${OUT}/summary/records_two_encoder.json" \
  --model "${DPRM_OMNI_CLIP_B32_PATH:-openai/clip-vit-base-patch32}" \
  --metric-name clip_b32_cosine --device cuda:0
"${PYTHON}" "${SCRIPT_DIR}/analyze_omni_paired_results.py" \
  --records "${OUT}/summary/records_two_encoder.json" \
  --output-dir "${OUT}/summary/paired" \
  --comparisons \
    confidence_train__confidence_decode:confidence_train__dprm_decode \
    dprm_train__confidence_decode:dprm_train__dprm_decode \
    confidence_train__confidence_decode:dprm_train__confidence_decode \
    confidence_train__dprm_decode:dprm_train__dprm_decode
date -Is > "${OUT}/MATRIX_COMPLETE"
