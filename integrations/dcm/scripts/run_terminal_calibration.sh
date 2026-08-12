#!/usr/bin/env bash
set -euo pipefail

DCM_ROOT=${DCM_ROOT:?set DCM_ROOT to the upstream DCM checkout}
DPRM_ROOT=${DPRM_ROOT:?set DPRM_ROOT to the DPRM-DLLM checkout}
ENV_ACTIVATE=${ENV_ACTIVATE:-}
GPU=${GPU:-0}
RUN_REPLICATION=${RUN_REPLICATION:-1}
REPLICATION_SEEDS=${REPLICATION_SEEDS:-20260815,20260816}
OUTPUT_ROOT=${OUTPUT_ROOT:-${DCM_ROOT}/experiments/dprm_terminal_calibration}
EVAL_ROOT=${EVAL_ROOT:-${DCM_ROOT}/eval_outputs/dprm_terminal_calibration}
CHECKPOINT=${CHECKPOINT:-${DCM_ROOT}/experiments/dcm_single_cell_real_progressive_orderfix/final.pt}

if [[ -n "${ENV_ACTIVATE}" ]]; then source "${ENV_ACTIVATE}"; fi
export PYTHONPATH="${DPRM_ROOT}/src:${DCM_ROOT}/src:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_ROOT}" "${EVAL_ROOT}"
cd "${DCM_ROOT}"

if [[ ! -f "${OUTPUT_ROOT}/.complete" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" python3 scripts/calibrate_dcm_terminal_order.py \
    --checkpoint "${CHECKPOINT}" --output-dir "${OUTPUT_ROOT}" \
    --max-train-cells 256 --batch-size 8 --branch-steps 0,8,16,24 \
    --num-phases 4 --confidence-bins 16 --ready-count 64
fi

series=(
  "Confidence=configs/rnaseq_progressive_dentate.yaml=${CHECKPOINT}=confidence"
  "DPRM-recovery=configs/rnaseq_dprm_multiobjective_dentate.yaml=${OUTPUT_ROOT}/recovery/calibrated.pt=dprm_confidence"
  "DPRM-MAE=configs/rnaseq_dprm_multiobjective_dentate.yaml=${OUTPUT_ROOT}/mae/calibrated.pt=dprm_confidence"
  "DPRM-balanced=configs/rnaseq_dprm_multiobjective_dentate.yaml=${OUTPUT_ROOT}/balanced/calibrated.pt=dprm_confidence"
  "DPRM-zero=configs/rnaseq_dprm_multiobjective_dentate.yaml=${OUTPUT_ROOT}/zero/calibrated.pt=dprm_confidence"
)

for guidance in 0.5 1.0 2.0 4.0; do
  target="${EVAL_ROOT}/development_g${guidance}"
  if [[ -f "${target}/.complete" ]]; then continue; fi
  command=(python3 scripts/eval_dcm_ordering_bootstrap.py
    --output-dir "${target}" --split train --cell-offset 256 --max-cells 96
    --bootstrap 1000 --seed 20260812 --num-steps 32 --num-samples 2
    --guidance-scale "${guidance}")
  for item in "${series[@]}"; do command+=(--series "${item}"); done
  CUDA_VISIBLE_DEVICES="${GPU}" "${command[@]}"
  touch "${target}/.complete"
done

python3 "${DPRM_ROOT}/integrations/dcm/scripts/select_terminal_guidance.py" "${EVAL_ROOT}"
guidance=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["guidance"])' "${EVAL_ROOT}/development_selection.json")

formal="${EVAL_ROOT}/formal"
if [[ ! -f "${formal}/.complete" ]]; then
  command=(python3 scripts/eval_dcm_ordering_bootstrap.py
    --output-dir "${formal}" --split val --bootstrap 5000 --seed 20260812
    --num-steps 32 --num-samples 4 --guidance-scale "${guidance}")
  for item in "${series[@]}"; do command+=(--series "${item}"); done
  CUDA_VISIBLE_DEVICES="${GPU}" "${command[@]}"
  touch "${formal}/.complete"
fi

if [[ "${RUN_REPLICATION}" == "1" ]]; then
  IFS=, read -r -a replication_seeds <<< "${REPLICATION_SEEDS}"
  for seed in "${replication_seeds[@]}"; do
    replication="${EVAL_ROOT}/formal_replication_${seed}"
    if [[ -f "${replication}/.complete" ]]; then continue; fi
    command=(python3 scripts/eval_dcm_ordering_bootstrap.py
      --output-dir "${replication}" --split val --bootstrap 5000 --seed "${seed}"
      --num-steps 32 --num-samples 4 --guidance-scale "${guidance}")
    for item in "${series[@]}"; do command+=(--series "${item}"); done
    CUDA_VISIBLE_DEVICES="${GPU}" "${command[@]}"
    touch "${replication}/.complete"
  done
fi
