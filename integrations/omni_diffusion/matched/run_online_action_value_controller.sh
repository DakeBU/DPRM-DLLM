#!/usr/bin/env bash
set -euo pipefail

MATCHED_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${MATCHED_ROOT}/../../.." && pwd)"
SCRIPT_DIR="${MATCHED_ROOT}/scripts"
: "${OMNI_ROOT:?set the Omni-Diffusion checkout}"
: "${OMNI_MODEL_PATH:?set the frozen Omni checkpoint}"
: "${OMNI_IMAGE_TOKENIZER_PATH:?set the MagViT tokenizer path}"
: "${OMNI_ONLINE_PROMPT_FILE:?set a frozen JSONL prompt split}"
: "${OMNI_ONLINE_ROOT:?set the output directory}"
: "${VIRTUAL_ENV:?set the experiment environment}"

PYTHON="${PYTHON:-${VIRTUAL_ENV}/bin/python}"
GPUS_TEXT="${OMNI_ONLINE_GPUS:-1 2 3 4 5 6 7}"
read -r -a GPUS <<< "${GPUS_TEXT//,/ }"
COUNT="${OMNI_ONLINE_COUNT:-128}"
PROMPT_OFFSET="${OMNI_ONLINE_PROMPT_OFFSET:-0}"
INDEX_OFFSET="${OMNI_ONLINE_INDEX_OFFSET:-0}"
SEED_OFFSET="${OMNI_ONLINE_SEED_OFFSET:-20279000}"
ACTION_STEP="${OMNI_ONLINE_ACTION_STEP:-96}"
QUANTILES_TEXT="${OMNI_ONLINE_QUANTILES:-0.70 0.85 0.90 0.95}"
read -r -a QUANTILES <<< "${QUANTILES_TEXT}"

source "${VIRTUAL_ENV}/bin/activate"
export PYTHONPATH="${RELEASE_ROOT}/src:${OMNI_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
mkdir -p "${OMNI_ONLINE_ROOT}"/{baseline,branches,logs,random,records,selection}

mapfile -t PROMPTS < <(
  "${PYTHON}" - "${OMNI_ONLINE_PROMPT_FILE}" "${PROMPT_OFFSET}" "${COUNT}" <<'PY'
import json
import sys
from pathlib import Path

path, offset, count = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
for row in rows[offset : offset + count]:
    prompt = row.get("prompt", row.get("text"))
    if prompt:
        print(str(prompt).strip())
PY
)
[[ ${#PROMPTS[@]} -eq ${COUNT} ]] || {
  echo "expected ${COUNT} prompts, found ${#PROMPTS[@]}" >&2
  exit 2
}

jobs="${OMNI_ONLINE_ROOT}/jobs.jsonl"
: > "${jobs}"
common_extra='["--fixed-t2i-scaffold"]'
if [[ "${OMNI_ONLINE_SAVE_HISTORY:-0}" == 1 ]]; then
  common_extra="$(jq -cn --argjson current "${common_extra}" \
    '$current + ["--save-history-frames","--history-frame-stride","32","--history-frame-limit","12","--trace-order-stats","auto","--trace-provisional-phases","--trace-num-phases","8","--trace-confidence-bins","8","--trace-aux-bins","16"]')"
fi
for local_idx in "${!PROMPTS[@]}"; do
  idx=$((INDEX_OFFSET + local_idx))
  prompt="${PROMPTS[$local_idx]}"
  seed=$((SEED_OFFSET + idx))
  base_dir="${OMNI_ONLINE_ROOT}/baseline/prompt_$(printf '%04d' "${idx}")"
  jq -cn --arg output_dir "${base_dir}" --arg prompt "${prompt}" --argjson seed "${seed}" --argjson extra "${common_extra}" \
    '{output_dir:$output_dir,prompt:$prompt,order_policy:"progressive_confidence",seed:$seed,steps:260,max_tokens:260,extra_args:$extra}' \
    >> "${jobs}"
  if [[ "${OMNI_ONLINE_INCLUDE_RANDOM:-0}" == 1 ]]; then
    random_dir="${OMNI_ONLINE_ROOT}/random/prompt_$(printf '%04d' "${idx}")"
    jq -cn --arg output_dir "${random_dir}" --arg prompt "${prompt}" --argjson seed "${seed}" --argjson extra "${common_extra}" \
      '{output_dir:$output_dir,prompt:$prompt,order_policy:"random",seed:$seed,steps:260,max_tokens:260,extra_args:$extra}' \
      >> "${jobs}"
  fi
  for quantile in "${QUANTILES[@]}"; do
    branch_dir="${OMNI_ONLINE_ROOT}/branches/step${ACTION_STEP}_q${quantile}/prompt_$(printf '%04d' "${idx}")"
    extra="$(jq -cn --argjson current "${common_extra}" --arg step "${ACTION_STEP}" --arg quantile "${quantile}" \
      '$current + ["--force-order-step",$step,"--force-confidence-quantile",$quantile,"--force-confidence-bins","8","--force-rank-bins","64","--force-aux-bins","16","--require-forced-action"]')"
    jq -cn --arg output_dir "${branch_dir}" --arg prompt "${prompt}" --argjson seed "${seed}" --argjson extra "${extra}" \
      '{output_dir:$output_dir,prompt:$prompt,order_policy:"progressive_confidence",seed:$seed,steps:260,max_tokens:260,extra_args:$extra}' \
      >> "${jobs}"
  done
done

cat > "${OMNI_ONLINE_ROOT}/run_manifest.json" <<JSON
{
  "design": "online action-conditioned rank-bucket DPRM",
  "checkpoint": "${OMNI_MODEL_PATH}",
  "prompt_file": "${OMNI_ONLINE_PROMPT_FILE}",
  "prompt_file_sha256": "$(sha256sum "${OMNI_ONLINE_PROMPT_FILE}" | awk '{print $1}')",
  "prompt_offset": ${PROMPT_OFFSET},
  "prompt_count": ${COUNT},
  "action_step": ${ACTION_STEP},
  "confidence_rank_quantiles": "${QUANTILES_TEXT}",
  "candidate_rollouts_per_prompt": $((1 + ${#QUANTILES[@]})),
  "fixed_guidance": ${OMNI_ONLINE_FIXED_GUIDANCE:-null},
  "random_control": $([[ "${OMNI_ONLINE_INCLUDE_RANDOM:-0}" == 1 ]] && echo true || echo false),
  "selection_reward": "CLIP-L/14 cosine",
  "independent_check": "CLIP-B/32 cosine",
  "complete_candidate_path_selection": true,
  "human_selection": false
}
JSON

pids=()
for gpu in "${GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${SCRIPT_DIR}/run_omni_t2i_manifest.py" \
    --smoke-script "${SCRIPT_DIR}/omni_t2i_smoke.py" \
    --model-path "${OMNI_MODEL_PATH}" \
    --image-tokenizer-path "${OMNI_IMAGE_TOKENIZER_PATH}" \
    --jobs "${jobs}" > "${OMNI_ONLINE_ROOT}/logs/gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
(( failed == 0 )) || { echo "online action-value rollout worker failed" >&2; exit 2; }

"${PYTHON}" "${SCRIPT_DIR}/collect_omni_action_records.py" \
  --root "${OMNI_ONLINE_ROOT}" --output "${OMNI_ONLINE_ROOT}/records/unscored.json"
CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${PYTHON}" "${SCRIPT_DIR}/score_omni_records_clip.py" \
  --records "${OMNI_ONLINE_ROOT}/records/unscored.json" \
  --output "${OMNI_ONLINE_ROOT}/records/clip_l14.json" \
  --model openai/clip-vit-large-patch14 --metric-name clip_cosine --device cuda:0
CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${PYTHON}" "${SCRIPT_DIR}/score_omni_records_clip.py" \
  --records "${OMNI_ONLINE_ROOT}/records/clip_l14.json" \
  --output "${OMNI_ONLINE_ROOT}/records/two_encoder.json" \
  --model openai/clip-vit-base-patch32 --metric-name clip_b32_cosine --device cuda:0

branch_methods=()
for quantile in "${QUANTILES[@]}"; do branch_methods+=("step${ACTION_STEP}_q${quantile}"); done
selection_args=(
  --records "${OMNI_ONLINE_ROOT}/records/two_encoder.json"
  --output-dir "${OMNI_ONLINE_ROOT}/selection"
  --branch-methods "${branch_methods[@]}"
  --reward-scale "${OMNI_ONLINE_REWARD_SCALE:-0.03}"
)
if [[ -n "${OMNI_ONLINE_FIXED_GUIDANCE:-}" ]]; then
  selection_args+=(--fixed-guidance "${OMNI_ONLINE_FIXED_GUIDANCE}")
else
  read -r -a guidance_grid <<< "${OMNI_ONLINE_GUIDANCE_GRID:-0.25 0.5 1.0 2.0 4.0 8.0}"
  selection_args+=(--guidance-grid "${guidance_grid[@]}")
fi
"${PYTHON}" "${SCRIPT_DIR}/select_omni_online_action_value.py" "${selection_args[@]}"
date -Is > "${OMNI_ONLINE_ROOT}/COMPLETE"
