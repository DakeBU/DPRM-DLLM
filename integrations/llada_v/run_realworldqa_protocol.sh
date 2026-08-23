#!/usr/bin/env bash
set -euo pipefail

RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${LLADA_V_OUTPUT_ROOT:?set the RealWorldQA protocol output directory}"
: "${LLADA_V_MODEL_PATH:?set the frozen GSAI-ML/LLaDA-V checkpoint}"
: "${LLADA_V_LMMS_ROOT:?set the lmms-eval checkout used by LLaDA-V}"

PYTHON=${PYTHON:-python}
GPU=${LLADA_V_GPU:-0}
BASELINE="${LLADA_V_OUTPUT_ROOT}/baseline"
TABLE_ROOT="${LLADA_V_OUTPUT_ROOT}/tables"
mkdir -p "${TABLE_ROOT}" "${LLADA_V_OUTPUT_ROOT}/development"

run_eval() {
  local root=$1 order=$2 limit=$3 table=${4:-} guidance=${5:-4}
  LLADA_V_OUTPUT_ROOT="${root}" LLADA_V_TASK=realworldqa \
  LLADA_V_ORDER="${order}" LLADA_V_LIMIT="${limit}" LLADA_V_GPU="${GPU}" \
  DPRM_LLADAV_TABLE="${table}" DPRM_LLADAV_GUIDANCE="${guidance}" \
    bash "${RELEASE_ROOT}/integrations/llada_v/run_lmms_eval.sh"
}

# One confidence run supplies the fitting interval, the development baseline,
# and the held-out baseline. Targets after document 127 are never used to fit.
run_eval "${BASELINE}" progressive_confidence 765
baseline_samples=$(find "${BASELINE}" -name '*_samples_realworldqa.jsonl' -print -quit)
[[ -s "${baseline_samples}" ]] || { echo "missing baseline samples" >&2; exit 2; }

for position_bins in 2 4; do
  table="${TABLE_ROOT}/p1_b8_pos${position_bins}.json"
  "${PYTHON}" "${RELEASE_ROOT}/integrations/llada_v/scripts/build_dprm_table.py" \
    --eval-root "${BASELINE}" --out "${table}" --orders progressive_confidence \
    --tasks realworldqa --reward-mode target --reward-normalization centered \
    --num-phases 1 --confidence-bins 8 --aux-mode format_eot_position \
    --position-bins "${position_bins}" --source-num-phases 8 \
    --source-confidence-bins 16 --source-aux-bins 16 --max-docs-per-task 128 \
    --reward-temperature 1 --guidance-scale 1 --ready-count 4 \
    --warmup-steps 0 --switch-steps 4
done

selection_args=()
for position_bins in 2 4; do
  table="${TABLE_ROOT}/p1_b8_pos${position_bins}.json"
  for guidance in 1 4 8; do
    label="p${position_bins}_g${guidance}"
    root="${LLADA_V_OUTPUT_ROOT}/development/${label}"
    run_eval "${root}" dprm_confidence_warmup 256 "${table}" "${guidance}"
    selection_args+=(--candidate "${label}=${root}")
  done
done

"${PYTHON}" "${RELEASE_ROOT}/integrations/llada_v/scripts/select_controller.py" \
  --baseline-root "${BASELINE}" "${selection_args[@]}" --task realworldqa \
  --doc-min 128 --doc-max 256 --output "${LLADA_V_OUTPUT_ROOT}/selection.json"

selected=$("${PYTHON}" - "${LLADA_V_OUTPUT_ROOT}/selection.json" <<'PY'
import json, sys
selected = json.load(open(sys.argv[1], encoding="utf-8")).get("selected")
if not selected:
    raise SystemExit("no active positive-development controller")
print(selected)
PY
)
position_bins=${selected%%_*}; position_bins=${position_bins#p}
guidance=${selected##*_g}
table="${TABLE_ROOT}/p1_b8_pos${position_bins}.json"
run_eval "${LLADA_V_OUTPUT_ROOT}/confirmation" dprm_confidence_warmup 765 \
  "${table}" "${guidance}"

dprm_samples=$(find "${LLADA_V_OUTPUT_ROOT}/confirmation" \
  -name '*_samples_realworldqa.jsonl' -print -quit)
[[ -s "${dprm_samples}" ]] || { echo "missing DPRM samples" >&2; exit 2; }
"${PYTHON}" "${RELEASE_ROOT}/integrations/llada_v/scripts/summarize_multimodal_results.py" \
  --rwqa-confidence "${baseline_samples}" --rwqa-dprm "${dprm_samples}" \
  --rwqa-doc-min 256 --rwqa-doc-max 765 --bootstrap 5000 --seed 20260811 \
  --output "${LLADA_V_OUTPUT_ROOT}/summary.json"
date -Is > "${LLADA_V_OUTPUT_ROOT}/EVALUATION_COMPLETE"
