#!/usr/bin/env bash
set -euo pipefail

: "${LLADA_V_LMMS_ROOT:?set the pinned LLaDA-V checkout}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OVERLAY="${ROOT}/integrations/llada_v/overlay"
EXPECTED_COMMIT="f8b02ce04b09f4f271fe55a3652059a73bbc7a32"

actual_commit="$(git -C "${LLADA_V_LMMS_ROOT}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
  echo "LLaDA-V commit mismatch: ${actual_commit} != ${EXPECTED_COMMIT}" >&2
  exit 2
fi

[[ -d "${LLADA_V_LMMS_ROOT}/train/llava" ]] || {
  echo "missing LLaDA-V train tree: ${LLADA_V_LMMS_ROOT}/train/llava" >&2
  exit 2
}
[[ -d "${LLADA_V_LMMS_ROOT}/eval/lmms-eval/lmms_eval" ]] || {
  echo "missing lmms-eval tree: ${LLADA_V_LMMS_ROOT}/eval/lmms-eval/lmms_eval" >&2
  exit 2
}

declare -a copies=(
  "dprm_generation.py:train/llava/dprm_generation.py"
  "host/fast_dllm_hook.py:train/llava/hooks/fast_dllm_hook.py"
  "host/modeling_llada.py:train/llava/model/language_model/modeling_llada.py"
  "host/llava_onevision_llada.py:eval/lmms-eval/lmms_eval/models/llava_onevision_llada.py"
  "tasks/ai2d_lite.yaml:eval/lmms-eval/lmms_eval/tasks/ai2d/ai2d_lite.yaml"
  "tasks/realworldqa.yaml:eval/lmms-eval/lmms_eval/tasks/realworldqa/realworldqa.yaml"
  "tasks/chartqa_lite.yaml:eval/lmms-eval/lmms_eval/tasks/chartqa/chartqa_lite.yaml"
)
for mapping in "${copies[@]}"; do
  source_path="${OVERLAY}/${mapping%%:*}"
  target_path="${LLADA_V_LMMS_ROOT}/${mapping#*:}"
  [[ -f "${source_path}" ]] || { echo "missing overlay: ${source_path}" >&2; exit 2; }
  [[ -f "${target_path}" || "${target_path}" == */dprm_generation.py ]] || {
    echo "missing LLaDA-V target: ${target_path}" >&2
    exit 2
  }
  mkdir -p "$(dirname "${target_path}")"
  cp "${source_path}" "${target_path}"
done
