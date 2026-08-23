#!/usr/bin/env bash
set -euo pipefail

: "${LLADA_V_LMMS_ROOT:?set the lmms-eval checkout used by LLaDA-V}"
: "${LLADA_V_MODEL_PATH:?set the frozen GSAI-ML/LLaDA-V checkpoint}"
: "${LLADA_V_OUTPUT_ROOT:?set the evaluation output directory}"

bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/apply_overlay.sh"

TASK=${LLADA_V_TASK:-realworldqa}
ORDER=${LLADA_V_ORDER:-progressive_confidence}
GPU=${LLADA_V_GPU:-0}
LIMIT=${LLADA_V_LIMIT:-}
TABLE=${DPRM_LLADAV_TABLE:-}
GUIDANCE=${DPRM_LLADAV_GUIDANCE:-4}
READY=${DPRM_LLADAV_READY_COUNT:-4}
WARMUP=${DPRM_LLADAV_WARMUP_STEPS:-0}
SWITCH=${DPRM_LLADAV_SWITCH_STEPS:-4}
FORCE_FULL=${DPRM_LLADAV_FORCE_FULL:-1}
TRACE=${DPRM_LLADAV_TRACE:-1}
PYTHON=${PYTHON:-python}
ACCELERATE=${ACCELERATE:-accelerate}

case "${ORDER}" in
  random|progressive_confidence|dprm_confidence_warmup|dprm_random_warmup) ;;
  *) echo "unsupported order: ${ORDER}" >&2; exit 2 ;;
esac
if [[ "${ORDER}" == dprm_* && ! -s "${TABLE}" ]]; then
  echo "${ORDER} requires DPRM_LLADAV_TABLE" >&2
  exit 2
fi

case "${TASK}" in
  ai2d_lite|realworldqa)
    GEN_LENGTH=4; GEN_STEPS=4 ;;
  chartqa_lite|chartqa)
    GEN_LENGTH=16; GEN_STEPS=16 ;;
  *)
    echo "unsupported paper task: ${TASK}" >&2
    exit 2 ;;
esac

OUT="${LLADA_V_OUTPUT_ROOT}/${ORDER}/${TASK}"
TRACE_PATH="${OUT}/order_trace.jsonl"
mkdir -p "${OUT}"
if [[ -e "${OUT}/.done" ]]; then
  echo "already complete: ${OUT}"
  exit 0
fi

GEN_KWARGS=$(
  ORDER="${ORDER}" TABLE="${TABLE}" GUIDANCE="${GUIDANCE}" READY="${READY}" \
  WARMUP="${WARMUP}" SWITCH="${SWITCH}" FORCE_FULL="${FORCE_FULL}" \
  TRACE="${TRACE}" TRACE_PATH="${TRACE_PATH}" GEN_LENGTH="${GEN_LENGTH}" \
  GEN_STEPS="${GEN_STEPS}" "${PYTHON}" - <<'PY'
import json
import os

payload = {
    "temperature": 0,
    "cfg": 0,
    "remasking": os.environ["ORDER"],
    "gen_length": int(os.environ["GEN_LENGTH"]),
    "block_length": int(os.environ["GEN_LENGTH"]),
    "gen_steps": int(os.environ["GEN_STEPS"]),
    "stopping_criteria": ["\n"],
    "think_mode": "no_think",
}
if os.environ.get("TABLE"):
    payload.update(
        dprm_table=os.environ["TABLE"],
        dprm_guidance_scale=float(os.environ["GUIDANCE"]),
        dprm_ready_count=int(os.environ["READY"]),
        dprm_warmup_steps=int(os.environ["WARMUP"]),
        dprm_switch_steps=int(os.environ["SWITCH"]),
        dprm_force_full=os.environ["FORCE_FULL"] == "1",
    )
if os.environ["TRACE"] == "1":
    payload.update(
        trace_order_stats=os.environ["TRACE_PATH"],
        trace_num_phases=8,
        trace_confidence_bins=16,
        trace_aux_bins=16,
    )
print(json.dumps(payload, separators=(",", ":")))
PY
)

cat > "${OUT}/run_manifest.json" <<JSON
{
  "task": "${TASK}",
  "order": "${ORDER}",
  "model": "${LLADA_V_MODEL_PATH}",
  "table": "${TABLE}",
  "guidance": ${GUIDANCE},
  "ready_count": ${READY},
  "warmup_steps": ${WARMUP},
  "switch_steps": ${SWITCH},
  "force_full": ${FORCE_FULL},
  "limit": "${LIMIT}"
}
JSON

(
  cd "${LLADA_V_LMMS_ROOT}/eval/lmms-eval"
  export PYTHONPATH="${LLADA_V_LMMS_ROOT}/train:${PYTHONPATH:-}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${ACCELERATE}" launch --num_processes=1 -m lmms_eval \
    --model llava_onevision_llada \
    --model_args "pretrained=${LLADA_V_MODEL_PATH},conv_template=llava_llada,model_name=llava_llada" \
    --tasks "${TASK}" --batch_size 1 --log_samples \
    --log_samples_suffix "${TASK}_${ORDER}" --output_path "${OUT}" \
    --gen_kwargs "${GEN_KWARGS}" ${LIMIT:+--limit "${LIMIT}"}
) > "${OUT}/run.log" 2>&1

if grep -q "Error during evaluation" "${OUT}/run.log"; then
  echo "lmms-eval reported an error: ${OUT}/run.log" >&2
  exit 1
fi
touch "${OUT}/.done"
