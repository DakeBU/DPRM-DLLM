#!/usr/bin/env bash
set -euo pipefail

MATCHED_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${MATCHED_ROOT}/../../.." && pwd)"
EXPECTED_COMMIT="c4f4625f84197a72d556ea00f10e5b2775524252"

: "${OMNI_ROOT:?set OMNI_ROOT}"
: "${OMNI_MODEL_PATH:?set OMNI_MODEL_PATH}"
: "${OMNI_IMAGE_TOKENIZER_PATH:?set OMNI_IMAGE_TOKENIZER_PATH}"
: "${OMNI_DATA_JSON:?set OMNI_DATA_JSON}"
: "${OMNI_CONTROLLER:?set OMNI_CONTROLLER to the frozen development controller}"
: "${OMNI_RUN_ROOT:?set OMNI_RUN_ROOT}"
: "${VIRTUAL_ENV:?activate the experiment environment}"

actual_commit="$(git -C "${OMNI_ROOT}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
  echo "Omni-Diffusion commit mismatch: ${actual_commit} != ${EXPECTED_COMMIT}" >&2
  exit 2
fi
for path in "${OMNI_MODEL_PATH}" "${OMNI_IMAGE_TOKENIZER_PATH}" \
  "${OMNI_DATA_JSON}" "${OMNI_CONTROLLER}"; do
  [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 2; }
done

# The overlay is a snapshot of the host files used by the matched experiment.
# Applying it is deterministic and leaves the upstream revision visible in git.
cp -a "${MATCHED_ROOT}/overlay/." "${OMNI_ROOT}/"

export PYTHON="${PYTHON:-${VIRTUAL_ENV}/bin/python}"
export PYTHONPATH="${RELEASE_ROOT}/src:${OMNI_ROOT}:${PYTHONPATH:-}"
export DPRM_OMNI_MATCHED_ROOT="${OMNI_RUN_ROOT}"
export DPRM_OMNI_MODEL_PATH="${OMNI_MODEL_PATH}"
export DPRM_OMNI_IMAGE_TOKENIZER="${OMNI_IMAGE_TOKENIZER_PATH}"
export DPRM_OMNI_DATA_JSON="${OMNI_DATA_JSON}"
export DPRM_OMNI_GATE_CONTROLLER="${OMNI_CONTROLLER}"
export DPRM_OMNI_SKIP_PRETRAIN_GATE=1
export DPRM_OMNI_TRAIN_ORDERS="random_matched confidence_matched dprm_matched"
export DPRM_OMNI_HYBRID_ROLLIN=1
export DPRM_OMNI_INCLUDE_TRAINED_RANDOM=1
export DPRM_OMNI_TRAJECTORY_COUNT="${OMNI_TRAJECTORY_COUNT:-256}"
export DPRM_OMNI_TRAIN_SOURCE_UNIQUE_OFFSET="${OMNI_TRAIN_SOURCE_UNIQUE_OFFSET:-2400}"
export DPRM_OMNI_MAX_STEPS="${OMNI_MAX_STEPS:-1000}"
export DPRM_OMNI_SAVE_STEPS="${OMNI_SAVE_STEPS:-500}"
export DPRM_OMNI_EVAL_STEP="${OMNI_EVAL_STEP:-1000}"
export DPRM_OMNI_EVAL_OFFSET="${OMNI_EVAL_OFFSET:-2300}"
export DPRM_OMNI_EVAL_COUNT="${OMNI_EVAL_COUNT:-96}"

exec bash "${MATCHED_ROOT}/scripts/orchestrate_matched_pipeline.sh"
