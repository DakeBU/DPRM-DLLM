#!/usr/bin/env bash
set -euo pipefail

DCM_ROOT=${DCM_ROOT:?set DCM_ROOT to the upstream DCM checkout}
DPRM_ROOT=${DPRM_ROOT:-}
ENV_ACTIVATE=${ENV_ACTIVATE:-}
OUTPUT_ROOT=${OUTPUT_ROOT:-${DCM_ROOT}/experiments/dprm_dcm_preference_sweep}
CONFIG=${CONFIG:-configs/rnaseq_dprm_multiobjective_dentate.yaml}
EPOCHS=${EPOCHS:-50}
GPU_LIST=${GPU_LIST:-0,1,2,3}

IFS=, read -r -a GPUS <<< "${GPU_LIST}"
labels=(recovery mae balanced zero)
weights=("0.90 0.075 0.025" "0.05 0.90 0.05" "0.45 0.35 0.20" "0.025 0.075 0.90")
if (( ${#GPUS[@]} < ${#labels[@]} )); then
  echo "GPU_LIST must provide at least ${#labels[@]} entries" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
if [[ -n "${DPRM_ROOT}" ]]; then
  export PYTHONPATH="${DPRM_ROOT}/src:${DCM_ROOT}/src:${PYTHONPATH:-}"
fi
pids=()
for index in "${!labels[@]}"; do
  label=${labels[$index]}
  output=${OUTPUT_ROOT}/${label}
  read -r w0 w1 w2 <<< "${weights[$index]}"
  (
    cd "${DCM_ROOT}"
    if [[ -n "${ENV_ACTIVATE}" ]]; then source "${ENV_ACTIVATE}"; fi
    CUDA_VISIBLE_DEVICES=${GPUS[$index]} python3 scripts/train_rnaseq.py \
      --config "${CONFIG}" --checkpoint_dir "${output}" --resume "" \
      --num_epochs "${EPOCHS}" --order_policy dprm_confidence \
      --dprm_reward_mode reconstruction_tchebycheff \
      --dprm_objective_weights "${w0}" "${w1}" "${w2}" \
      --dprm_gene_aux_mode predicted_zero \
      --wandb_mode offline
  ) >"${OUTPUT_ROOT}/${label}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "${pid}"; done
touch "${OUTPUT_ROOT}/.training_complete"
