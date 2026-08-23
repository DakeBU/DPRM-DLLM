#!/usr/bin/env bash
set -euo pipefail

MATCHED_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${MATCHED_ROOT}/../../.." && pwd)"
SCRIPT_DIR="${MATCHED_ROOT}/scripts"
: "${OMNI_ROOT:?set OMNI_ROOT}"
: "${OMNI_MODEL_PATH:?set the frozen public Omni checkpoint}"
: "${OMNI_IMAGE_TOKENIZER_PATH:?set the MagViT tokenizer path}"
: "${OMNI_DEVELOPMENT_ROOT:?set the controller-fit output directory}"
: "${OMNI_DEVELOPMENT_PROMPT_FILE:?set the frozen GenEval development prompt file}"
: "${VIRTUAL_ENV:?activate the experiment environment}"

PYTHON="${PYTHON:-${VIRTUAL_ENV}/bin/python}"
COUNT="${OMNI_DEVELOPMENT_COUNT:-64}"
OFFSET="${OMNI_DEVELOPMENT_OFFSET:-3000}"
TABLE_PROMPT_OFFSET="${OMNI_TABLE_PROMPT_OFFSET:-2000}"
TABLE_PROMPT_COUNT="${OMNI_TABLE_PROMPT_COUNT:-48}"
TABLE_NUM_PHASES="${OMNI_TABLE_NUM_PHASES:-1}"
GPUS_TEXT="${OMNI_DEVELOPMENT_GPUS:-0 1}"
read -r -a GPUS <<< "${GPUS_TEXT//,/ }"

source "${VIRTUAL_ENV}/bin/activate"
export PYTHONPATH="${RELEASE_ROOT}/src:${OMNI_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
mkdir -p "${OMNI_DEVELOPMENT_ROOT}"/{controllers,gate,logs,summary,table}

rollouts="${OMNI_DEVELOPMENT_ROOT}/table_rollouts"
for order in random progressive_confidence; do
  [[ $(find "${rollouts}/${order}" -type f -name COMPLETE | wc -l) -eq ${TABLE_PROMPT_COUNT} ]] || {
    echo "expected ${TABLE_PROMPT_COUNT} complete ${order} table rollouts" >&2
    exit 2
  }
done

build_table() {
  local primary_weight=$1 secondary_weight=$2 output=$3
  if [[ ! -s "${output}" ]]; then
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${PYTHON}" "${SCRIPT_DIR}/build_omni_dprm_table.py" \
      --rollout-root "${rollouts}" --orders random progressive_confidence \
      --out "${output}" --clip-model openai/clip-vit-large-patch14 \
      --secondary-clip-model openai/clip-vit-base-patch32 \
      --primary-reward-weight "${primary_weight}" \
      --secondary-reward-weight "${secondary_weight}" \
      --device cuda:0 --deduplicate-prompt-text \
      --reward-normalization paired_prompt_advantage \
      --num-phases "${TABLE_NUM_PHASES}" --phase-source step --confidence-bins 8 \
      --confidence-binning development_quantile --aux-bins 16 \
      --reward-temperature 1.0 --ready-count 4 --warmup-steps 0 \
      --switch-steps 64 --require-fixed-visual-canvas
  fi
}

table_bh="${OMNI_DEVELOPMENT_ROOT}/table/omni_dual_l025_b075.json"
table_eq="${OMNI_DEVELOPMENT_ROOT}/table/omni_dual_l050_b050.json"
table_lh="${OMNI_DEVELOPMENT_ROOT}/table/omni_dual_l075_b025.json"
build_table 0.25 0.75 "${table_bh}"
build_table 0.50 0.50 "${table_eq}"
build_table 0.75 0.25 "${table_lh}"

if [[ "${OMNI_CONTROLLER_TABLE_ONLY:-0}" == 1 ]]; then
  cat > "${OMNI_DEVELOPMENT_ROOT}/table_fit_manifest.json" <<JSON
{
  "design": "current-checkpoint bucketized DPRM table fit",
  "model_path": "${OMNI_MODEL_PATH}",
  "table_prompt_range": [${TABLE_PROMPT_OFFSET}, $((TABLE_PROMPT_OFFSET + TABLE_PROMPT_COUNT - 1))],
  "table_prompt_count": ${TABLE_PROMPT_COUNT},
  "num_phases": ${TABLE_NUM_PHASES},
  "confidence_bins": 8,
  "spatial_bins": 16,
  "reward_normalization": "paired_prompt_advantage",
  "terminal_reward_calls_at_test": 0
}
JSON
  date -Is > "${OMNI_DEVELOPMENT_ROOT}/TABLE_FIT_COMPLETE"
  exit 0
fi

# Fixed before reading development outcomes. The grid compares the conservative
# low-confidence action window with the middle-stage pattern seen in the
# independent exact-action diagnostic.
specs=(
  "bh_lowconf_g075|${table_bh}|0.075|0.020|low"
  "bh_lowconf_g150|${table_bh}|0.150|0.050|low"
  "bh_lowconf_g300|${table_bh}|0.300|0.050|low"
  "bh_mid_g150|${table_bh}|0.150|0.050|mid"
  "bh_mid_g300|${table_bh}|0.300|0.050|mid"
  "eq_mid_g150|${table_eq}|0.150|0.050|mid"
  "eq_mid_g300|${table_eq}|0.300|0.050|mid"
  "lh_mid_g150|${table_lh}|0.150|0.050|mid"
  "lh_mid_g300|${table_lh}|0.300|0.050|mid"
)
labels=(confidence)
controllers=("")
candidate_map="${OMNI_DEVELOPMENT_ROOT}/candidate_map.tsv"
: > "${candidate_map}"
for spec in "${specs[@]}"; do
  IFS='|' read -r label table guidance gap family <<< "${spec}"
  controller="${OMNI_DEVELOPMENT_ROOT}/controllers/${label}.json"
  args=(
    --source-table "${table}" --output "${controller}"
    --guidance-scale "${guidance}" --ready-count 4 --policy-warmup-steps 0
    --max-base-score-gap "${gap}"
  )
  if [[ "${family}" == low ]]; then
    args+=(--reward-action-steps 96 112 128 144 160 --max-reward-confidence-bin 0)
  else
    args+=(--reward-action-steps 96 128 160)
  fi
  "${PYTHON}" "${SCRIPT_DIR}/freeze_omni_bucket_controller.py" "${args[@]}"
  labels+=("${label}")
  controllers+=("${controller}")
  printf '%s\t%s\n' "${label}" "${controller}" >> "${candidate_map}"
done

mapfile -t PROMPTS < <(
  "${PYTHON}" - "${OMNI_DEVELOPMENT_PROMPT_FILE}" "${COUNT}" <<'PY'
import json
import sys
from pathlib import Path


def prompt_from_row(row):
    if isinstance(row, str):
        return row.strip()
    if not isinstance(row, dict):
        return ""
    prompt = row.get("prompt", row.get("text"))
    if prompt is None and row.get("messages"):
        content = str(row["messages"][0].get("content", ""))
        lines = content.splitlines()
        prompt = "\n".join(lines[1:]).strip() if len(lines) > 1 else content
    return str(prompt or "").strip()


path = Path(sys.argv[1])
count = int(sys.argv[2])
seen = set()
for raw in path.read_text(encoding="utf-8").splitlines():
    raw = raw.strip()
    if not raw:
        continue
    try:
        row = json.loads(raw)
    except json.JSONDecodeError:
        row = raw
    prompt = prompt_from_row(row)
    if not prompt or prompt in seen:
        continue
    seen.add(prompt)
    print(prompt)
    if len(seen) == count:
        break
PY
)
[[ ${#PROMPTS[@]} -eq ${COUNT} ]] || {
  echo "expected ${COUNT} unique development prompts" >&2
  exit 2
}

jobs="${OMNI_DEVELOPMENT_ROOT}/fallback_gate_jobs.jsonl"
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

prompt_sha=$(sha256sum "${OMNI_DEVELOPMENT_PROMPT_FILE}" | awk '{print $1}')
cat > "${OMNI_DEVELOPMENT_ROOT}/fallback_run_manifest.json" <<JSON
{
  "design": "${OMNI_CONTROLLER_DEVELOPMENT_DESIGN:-frozen-table dual-encoder controller development}",
  "model_path": "${OMNI_MODEL_PATH}",
  "claim_eligible": false,
  "prompt_count": ${COUNT},
  "prompt_file": "${OMNI_DEVELOPMENT_PROMPT_FILE}",
  "prompt_file_sha256": "${prompt_sha}",
  "candidate_families": ["lowest-confidence five-action middle window", "three-action middle stage"],
  "guidance": [0.075, 0.15, 0.30],
  "native_score_margin": [0.02, 0.05],
  "terminal_utility_weights": [[0.25,0.75],[0.5,0.5],[0.75,0.25]],
  "table_prompt_range": [${TABLE_PROMPT_OFFSET}, $((TABLE_PROMPT_OFFSET + TABLE_PROMPT_COUNT - 1))],
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
    --jobs "${jobs}" > "${OMNI_DEVELOPMENT_ROOT}/logs/fallback_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
"${PYTHON}" - "${jobs}" "${failed}" <<-'PY'
import json
import sys
from pathlib import Path

jobs_path = Path(sys.argv[1])
worker_failure = bool(int(sys.argv[2]))
rows = [
    json.loads(line)
    for line in jobs_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
missing = []
for row in rows:
    output = Path(row["output_dir"])
    result = output / f'omni_t2i_{row["order_policy"]}.json'
    if (
        not result.is_file()
        or result.stat().st_size == 0
        or not (output / "COMPLETE").is_file()
    ):
        missing.append(str(output))
if missing:
    raise SystemExit(
        f"Omni development grid incomplete: {len(missing)}/{len(rows)} jobs missing; "
        f"first={missing[0]}"
    )
print(
    json.dumps(
        {
            "jobs": len(rows),
            "complete": len(rows),
            "worker_failure_observed": worker_failure,
        }
    )
)
PY

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
  --decision-output "${OMNI_DEVELOPMENT_ROOT}/selection.json" \
  --min-prompt-override-fraction 0.05 --require-positive-primary-ci
"${PYTHON}" "${SCRIPT_DIR}/prepare_omni_formal_controller.py" \
  --input "${OMNI_DEVELOPMENT_ROOT}/selected_controller.json" \
  --selection-decision "${OMNI_DEVELOPMENT_ROOT}/selection.json" \
  --source-table-root "${OMNI_DEVELOPMENT_ROOT}/table" \
  --table-prompt-range "${TABLE_PROMPT_OFFSET}" "$((TABLE_PROMPT_OFFSET + TABLE_PROMPT_COUNT - 1))" \
  --selection-prompt-file "${OMNI_DEVELOPMENT_PROMPT_FILE}" \
  --controller-only-host --output "${OMNI_DEVELOPMENT_ROOT}/formal_controller.json"
date -Is > "${OMNI_DEVELOPMENT_ROOT}/DEVELOPMENT_COMPLETE"
