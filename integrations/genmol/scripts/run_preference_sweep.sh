#!/usr/bin/env bash
set -euo pipefail

GENMOL_ROOT=${GENMOL_ROOT:?set GENMOL_ROOT to the upstream GenMol checkout}
DPRM_ROOT=${DPRM_ROOT:-}
ENV_ACTIVATE=${ENV_ACTIVATE:-}
RUNNER=${RUNNER:-${GENMOL_ROOT}/run_ordering_train.sh}
OUTPUT_ROOT=${OUTPUT_ROOT:-${GENMOL_ROOT}/outputs/dprm_genmol_preference_sweep}
MAX_STEPS=${MAX_STEPS:-5000}
GPU_LIST=${GPU_LIST:-0,1,2}
SCALARIZATION=${SCALARIZATION:-smooth_tchebycheff}

case "${SCALARIZATION}" in
  weighted_sum) REWARD_MODE=molecular_weighted_sum ;;
  smooth_tchebycheff) REWARD_MODE=molecular_tchebycheff ;;
  *) echo "SCALARIZATION must be weighted_sum or smooth_tchebycheff" >&2; exit 2 ;;
esac

IFS=, read -r -a GPUS <<< "${GPU_LIST}"
labels=(qed balanced sa)
weights=("[0.95,0.05]" "[0.55,0.45]" "[0.05,0.95]")
if (( ${#GPUS[@]} < ${#labels[@]} )); then
  echo "GPU_LIST must provide at least ${#labels[@]} entries" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
if [[ -n "${DPRM_ROOT}" ]]; then
  export PYTHONPATH="${DPRM_ROOT}/src:${GENMOL_ROOT}/src:${PYTHONPATH:-}"
fi
pids=()
for index in "${!labels[@]}"; do
  label=${labels[$index]}
  (
    if [[ -n "${ENV_ACTIVATE}" ]]; then source "${ENV_ACTIVATE}"; fi
    DPRM_REWARD_MODE=${REWARD_MODE} \
    DPRM_OBJECTIVE_WEIGHTS=${weights[$index]} \
    DPRM_AUX_MODE=molecular_token_class \
    DPRM_TCHEBYCHEFF_TEMPERATURE=0.05 \
    DPRM_TCHEBYCHEFF_AUGMENTATION=0.05 \
    WANDB_MODE=offline \
      bash "${RUNNER}" "${GPUS[$index]}" "dprm-genmol-${SCALARIZATION}-${label}" \
        dprm_random "${OUTPUT_ROOT}/${label}" "${MAX_STEPS}"
  ) >"${OUTPUT_ROOT}/${label}.launcher.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "${pid}"; done
touch "${OUTPUT_ROOT}/.training_complete"
