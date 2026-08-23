#!/usr/bin/env bash
set -euo pipefail

MATCHED_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${MATCHED_ROOT}/scripts"
: "${OMNI_ACTION_ADVANTAGES:?set the action advantage JSON}"
: "${OMNI_ACTION_CONTROLLER_DIR:?set the controller output directory}"
: "${VIRTUAL_ENV:?set the experiment environment}"
PYTHON="${PYTHON:-${VIRTUAL_ENV}/bin/python}"
ACTIVE_STEPS_TEXT="${OMNI_ACTION_ACTIVE_STEPS:-96}"
read -r -a ACTIVE_STEPS <<< "${ACTIVE_STEPS_TEXT}"
RANK_BINS="${OMNI_ACTION_RANK_BINS:-8}"
SPATIAL_BINS_TEXT="${OMNI_ACTION_GRID_SPATIAL_BINS:-1 4}"
read -r -a SPATIAL_BINS_GRID <<< "${SPATIAL_BINS_TEXT}"
GUIDANCE_TEXT="${OMNI_ACTION_GRID_GUIDANCE:-0.50 1.00 2.00 4.00}"
read -r -a GUIDANCE_GRID <<< "${GUIDANCE_TEXT}"
BETA_TEXT="${OMNI_ACTION_GRID_BETA:-1.00}"
read -r -a BETA_GRID <<< "${BETA_TEXT}"
SHRINKAGE="${OMNI_ACTION_GRID_SHRINKAGE:-4}"
MIN_COUNT="${OMNI_ACTION_GRID_MIN_COUNT:-1}"

source "${VIRTUAL_ENV}/bin/activate"
mkdir -p "${OMNI_ACTION_CONTROLLER_DIR}"

# Rank-only tables pool evidence across the canvas. Rank-spatial tables retain
# a 2x2 spatial partition when the development rollouts support that detail.
for beta in "${BETA_GRID[@]}"; do
  beta_tag="b$(printf '%03d' "$(awk -v value="${beta}" 'BEGIN {print 100*value}')")"
  for spatial_bins in "${SPATIAL_BINS_GRID[@]}"; do
    for guidance in "${GUIDANCE_GRID[@]}"; do
      guidance_tag="g$(printf '%03d' "$(awk -v value="${guidance}" 'BEGIN {print 100*value}')")"
      output="${OMNI_ACTION_CONTROLLER_DIR}/${beta_tag}_rank${spatial_bins}_${guidance_tag}.json"
      "${PYTHON}" "${SCRIPT_DIR}/fit_omni_action_bucket_controller.py" \
        --records "${OMNI_ACTION_ADVANTAGES}" --output "${output}" \
        --active-steps "${ACTIVE_STEPS[@]}" --rank-bins "${RANK_BINS}" --spatial-bins "${spatial_bins}" \
        --beta "${beta}" --guidance-scale "${guidance}" \
        --min-count "${MIN_COUNT}" --shrinkage "${SHRINKAGE}"
    done
  done
done
date -Is > "${OMNI_ACTION_CONTROLLER_DIR}/GRID_COMPLETE"
