#!/usr/bin/env bash
set -euo pipefail

: "${OMNI_DEVELOPMENT_ROOT:?set the completed development root}"
: "${DPRM_OMNI_EVAL_OUT:?set the confirmation output root}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTROLLER="${OMNI_DEVELOPMENT_ROOT}/formal_controller.json"
SELECTION="${OMNI_DEVELOPMENT_ROOT}/selection.json"

while [[ ! -s "${OMNI_DEVELOPMENT_ROOT}/DEVELOPMENT_COMPLETE" ]]; do
  if [[ -s "${SELECTION}" ]] \
    && ! jq -e '.passed == true' "${SELECTION}" >/dev/null; then
    echo "development gate did not select a controller" >&2
    exit 3
  fi
  sleep 30
done

[[ -s "${CONTROLLER}" ]] || {
  echo "development completed without a frozen controller" >&2
  exit 3
}

export DPRM_OMNI_CONTROLLER="${CONTROLLER}"
bash "${SCRIPT_DIR}/scripts/evaluate_omni_frozen_controller.sh"
