#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-<DPRM_DPLM_REPO>}"
ENV_ROOT="${ENV_ROOT:-<PYTHON_ENV>}"
GPU="${GPU:-7}"
RUN_ROOT="${RUN_ROOT:-<DPRM_DPLM_RUN_ROOT>/dprm_dplm2bit_650m_gpu${GPU}_fair}"
SHORT_TMP_ROOT="${SHORT_TMP_ROOT:-/tmp/dprm2bit_g${GPU}}"
RUN_NAME="${RUN_NAME:-dprm_dplm2bit_650m_gpu${GPU}_fair}"
TRAIN_FORCE_RESTART="${TRAIN_FORCE_RESTART:-false}"
RESUME_CKPT="${RESUME_CKPT:-${RUN_ROOT}/checkpoints/last.ckpt}"
WANDB_ID="${WANDB_ID:-}"

mkdir -p "${RUN_ROOT}"

source "${ENV_ROOT}/bin/activate"

export PROJECT_ROOT
export CUDA_VISIBLE_DEVICES="${GPU}"
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DIR="${RUN_ROOT}/wandb"
export WANDB_CACHE_DIR="${RUN_ROOT}/wandb-cache"
export TMPDIR="${SHORT_TMP_ROOT}"

mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${TMPDIR}"

cd "${PROJECT_ROOT}"

if [[ -z "${WANDB_ID}" && -L "${WANDB_DIR}/latest-run" ]]; then
  latest_run="$(basename "$(readlink -f "${WANDB_DIR}/latest-run")")"
  WANDB_ID="${latest_run##*-}"
fi

cmd=(
  python
  train.py
  experiment=dplm2/dprm_dplm_650m
  logger=wandb
  project=anonymous-dprm-review
  "name=${RUN_NAME}"
  trainer=ddp_bf16
  trainer.devices=1
  ++trainer.strategy=auto
  ++trainer.num_nodes=1
  trainer.gradient_clip_val=0.5
  "train.force_restart=${TRAIN_FORCE_RESTART}"
  +test=false
  "++paths.log_dir=${RUN_ROOT}"
  "paths.data_dir=${PROJECT_ROOT}/data-bin"
)

if [[ "${TRAIN_FORCE_RESTART}" != "true" ]]; then
  cmd+=("train.ckpt_path=${RESUME_CKPT}")
fi

if [[ -n "${WANDB_ID}" ]]; then
  cmd+=("logger.wandb.id=${WANDB_ID}")
fi

"${cmd[@]}"
