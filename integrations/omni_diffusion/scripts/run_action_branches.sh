#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OMNI_ROOT="${OMNI_ROOT:?set OMNI_ROOT to the Omni-Diffusion checkout}"
MODEL_PATH="${OMNI_MODEL_PATH:?set OMNI_MODEL_PATH to the evaluated checkpoint}"
IMAGE_TOKENIZER_PATH="${OMNI_IMAGE_TOKENIZER_PATH:?set OMNI_IMAGE_TOKENIZER_PATH}"
DATA_JSON="${OMNI_DATA_JSON:-${OMNI_ROOT}/datasets/jsonl/JourneyDB/BLIP3o_JourneyDB_T2I_tokenized.jsonl}"
OUT_ROOT="${OMNI_ACTION_ROOT:-${OMNI_ROOT}/outputs/omni_action_branches}"
PROMPT_OFFSET="${OMNI_PROMPT_OFFSET:-2000}"
PROMPT_COUNT="${OMNI_PROMPT_COUNT:-96}"
SEED_BASE="${OMNI_SEED_BASE:-20268000}"
STEP="${OMNI_ACTION_STEP:-96}"
read -r -a QUANTILES <<< "${OMNI_ACTION_QUANTILES:-0.15 0.3 0.7 0.85}"
read -r -a GPUS <<< "${GPU_IDS:-0 1 2 3 4 5 6 7}"

mkdir -p "${OUT_ROOT}/logs"
export PYTHONPATH="${OMNI_ROOT}:${PYTHONPATH:-}"
mapfile -t PROMPTS < <(
  jq -r '.messages[0].content | split("\n") | .[1:] | join("\n")' "${DATA_JSON}" |
    awk 'NF && !seen[$0]++' |
    sed -n "$((PROMPT_OFFSET + 1)),$((PROMPT_OFFSET + PROMPT_COUNT))p"
)
if (( ${#PROMPTS[@]} != PROMPT_COUNT )); then
  echo "expected ${PROMPT_COUNT} prompts, found ${#PROMPTS[@]}" >&2
  exit 2
fi

run_one() {
  local kind="$1" prompt_idx="$2" prompt="$3" gpu="$4" quantile="${5:-}"
  local out_dir extra=()
  if [[ "${kind}" == baseline ]]; then
    out_dir="${OUT_ROOT}/baseline/prompt_$(printf '%04d' "${prompt_idx}")"
  else
    out_dir="${OUT_ROOT}/branches/step${STEP}_q${quantile}/prompt_$(printf '%04d' "${prompt_idx}")"
    extra=(--force-order-step "${STEP}" --force-confidence-quantile "${quantile}"
      --force-confidence-bins 8 --force-rank-bins 8 --force-aux-bins 16
      --require-forced-action)
  fi
  mkdir -p "${out_dir}"
  [[ -s "${out_dir}/omni_t2i_progressive_confidence.json" ]] && return 0
  CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT_DIR}/generate_four_orders.py" \
    --model-path "${MODEL_PATH}" \
    --image-tokenizer-path "${IMAGE_TOKENIZER_PATH}" \
    --output-dir "${out_dir}" \
    --prompt "${prompt}" \
    --order-policy progressive_confidence \
    --steps 260 --max-tokens 260 \
    --seed "$((SEED_BASE + prompt_idx))" \
    "${extra[@]}" > "${out_dir}/run.log" 2>&1
}

pids=()
job=0
wait_batch() {
  local failed=0 pid
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  pids=()
  (( failed == 0 ))
}
for local_idx in "${!PROMPTS[@]}"; do
  prompt_idx=$((PROMPT_OFFSET + local_idx))
  prompt="${PROMPTS[$local_idx]}"
  gpu="${GPUS[$((job % ${#GPUS[@]}))]}"; job=$((job + 1))
  run_one baseline "${prompt_idx}" "${prompt}" "${gpu}" & pids+=("$!")
  (( ${#pids[@]} < ${#GPUS[@]} )) || wait_batch
  for quantile in "${QUANTILES[@]}"; do
    gpu="${GPUS[$((job % ${#GPUS[@]}))]}"; job=$((job + 1))
    run_one branch "${prompt_idx}" "${prompt}" "${gpu}" "${quantile}" & pids+=("$!")
    (( ${#pids[@]} < ${#GPUS[@]} )) || wait_batch
  done
done
(( ${#pids[@]} == 0 )) || wait_batch
printf 'complete\n' > "${OUT_ROOT}/ROLLOUTS_COMPLETE"
echo "action branches: ${OUT_ROOT}"
