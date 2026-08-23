#!/usr/bin/env bash
set -euo pipefail

RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${LLADA_V_LMMS_ROOT:?set the pinned LLaDA-V checkout}"
: "${LLADA_V_MODEL_PATH:?set the frozen GSAI-ML/LLaDA-V checkpoint}"
: "${LLADA_V_OUTPUT_ROOT:?set the AI2D protocol output directory}"

OUT=${LLADA_V_OUTPUT_ROOT}
GPU=${LLADA_V_GPU:-0}
PYTHON=${PYTHON:-python}
ACCELERATE=${ACCELERATE:-accelerate}
FIT_DOCS=${FIT_DOCS:-128}
DEV_MIN=${DEV_MIN:-128}
DEV_MAX=${DEV_MAX:-256}
TEST_MIN=${TEST_MIN:-256}
TEST_MAX=${TEST_MAX:-500}

mkdir -p "${OUT}"
cat > "${OUT}/PROTOCOL.txt" <<EOF
Frozen-host AI2D protocol declared before evaluation.
Fit documents: [0, ${FIT_DOCS})
Development documents: [${DEV_MIN}, ${DEV_MAX})
Confirmation documents: [${TEST_MIN}, ${TEST_MAX})
Candidate phase counts: {1,4}; confidence bins: 8; auxiliary state: EOT x four relative-position cells.
Candidate guidance: {1,2,4,8}; readiness: 4; force-full decode gate.
Selection: highest development target-normalized accuracy among active positive-delta controllers.
Confirmation: one confidence path and one frozen DPRM path per document; 5,000 paired bootstrap draws.
EOF

run_eval() {
  local output=$1 order=$2 limit=$3 table=${4:-} guidance=${5:-4}
  LLADA_V_OUTPUT_ROOT="${output}" LLADA_V_TASK=ai2d_lite \
  LLADA_V_ORDER="${order}" LLADA_V_LIMIT="${limit}" LLADA_V_GPU="${GPU}" \
  DPRM_LLADAV_TABLE="${table}" DPRM_LLADAV_GUIDANCE="${guidance}" \
  DPRM_LLADAV_READY_COUNT=4 DPRM_LLADAV_WARMUP_STEPS=0 \
  DPRM_LLADAV_SWITCH_STEPS=4 DPRM_LLADAV_FORCE_FULL=1 DPRM_LLADAV_TRACE=1 \
  PYTHON="${PYTHON}" ACCELERATE="${ACCELERATE}" \
    bash "${RELEASE_ROOT}/integrations/llada_v/run_lmms_eval.sh"
}

BASELINE=${OUT}/baseline
run_eval "${BASELINE}" progressive_confidence "${TEST_MAX}"

TABLE_ROOT=${OUT}/tables
mkdir -p "${TABLE_ROOT}"
for phases in 1 4; do
  table=${TABLE_ROOT}/p${phases}_b8_eot_pos4.json
  "${PYTHON}" "${RELEASE_ROOT}/integrations/llada_v/scripts/build_dprm_table.py" \
    --eval-root "${BASELINE}" --out "${table}" \
    --orders progressive_confidence --tasks ai2d_lite \
    --reward-mode target --reward-normalization centered \
    --num-phases "${phases}" --confidence-bins 8 --aux-mode eot_position \
    --position-bins 4 --source-num-phases 8 --source-confidence-bins 16 \
    --source-aux-bins 16 --max-docs-per-task "${FIT_DOCS}" \
    --reward-temperature 1 --guidance-scale 1 --ready-count 4 \
    --warmup-steps 0 --switch-steps 4
done

selection_args=()
for phases in 1 4; do
  table=${TABLE_ROOT}/p${phases}_b8_eot_pos4.json
  for guidance in 1 2 4 8; do
    label=p${phases}_g${guidance}
    candidate=${OUT}/development/${label}
    run_eval "${candidate}" dprm_confidence_warmup "${DEV_MAX}" "${table}" "${guidance}"
    selection_args+=(--candidate "${label}=${candidate}")
  done
done

"${PYTHON}" "${RELEASE_ROOT}/integrations/llada_v/scripts/select_controller.py" \
  --baseline-root "${BASELINE}" "${selection_args[@]}" --task ai2d_lite \
  --doc-min "${DEV_MIN}" --doc-max "${DEV_MAX}" \
  --output "${OUT}/development_selection.json"

selected=$("${PYTHON}" - "${OUT}/development_selection.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8")).get("selected")
if not value:
    raise SystemExit("no active positive-development AI2D controller")
print(value)
PY
)
phases=${selected%%_*}; phases=${phases#p}
guidance=${selected##*_g}
table=${TABLE_ROOT}/p${phases}_b8_eot_pos4.json
cat > "${OUT}/frozen_controller.json" <<EOF
{
  "label": "${selected}",
  "table": "${table}",
  "guidance": ${guidance},
  "ready_count": 4,
  "warmup_steps": 0,
  "switch_steps": 4,
  "force_full": true,
  "fit_interval": [0, ${FIT_DOCS}],
  "development_interval": [${DEV_MIN}, ${DEV_MAX}],
  "confirmation_interval": [${TEST_MIN}, ${TEST_MAX}]
}
EOF

FORMAL=${OUT}/confirmation
run_eval "${FORMAL}" dprm_confidence_warmup "${TEST_MAX}" "${table}" "${guidance}"
"${PYTHON}" "${RELEASE_ROOT}/integrations/llada_v/scripts/select_controller.py" \
  --baseline-root "${BASELINE}" --candidate "${selected}=${FORMAL}" \
  --task ai2d_lite --doc-min "${TEST_MIN}" --doc-max "${TEST_MAX}" \
  --no-require-positive-delta --output "${OUT}/confirmation_audit.json"
date -Is > "${OUT}/EVALUATION_COMPLETE"
