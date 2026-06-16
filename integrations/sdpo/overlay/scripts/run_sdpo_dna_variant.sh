#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-<SDPO_HOST_REPO>}
ENV=${ENV:-<PYTHON_ENV>/bin/activate}
GPU=${GPU:?set GPU}
RUN_NAME=${RUN_NAME:?set RUN_NAME}
ORDER_POLICY=${ORDER_POLICY:-baseline}

BASE_PATH=${BASE_PATH:-data_and_model/}
NUM_EPOCHS=${NUM_EPOCHS:-2}
K=${K:-2000}
LR=${LR:-1e-5}
SDPO_BETA=${SDPO_BETA:-0.5}
DPRM_BETA=${DPRM_BETA:-1.0}
DPRM_WARMUP_STEPS=${DPRM_WARMUP_STEPS:-100}
DPRM_SWITCH_STEPS=${DPRM_SWITCH_STEPS:-400}
DPRM_READY_COUNT=${DPRM_READY_COUNT:-64}
DPRM_SHORTLIST_SIZE=${DPRM_SHORTLIST_SIZE:-64}
EVAL_BATCHES=${EVAL_BATCHES:-10}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-64}
BOOTSTRAP=${BOOTSTRAP:-1000}
SEED=${SEED:-0}

if [[ "$ROOT" == "<SDPO_HOST_REPO>" || "$ENV" == "<PYTHON_ENV>/bin/activate" ]]; then
  echo "Set ROOT to the SDPO host repo and ENV to a Python environment before running." >&2
  exit 1
fi

cd "$ROOT"
source "$ENV"

while [ ! -f "${BASE_PATH}/.ready" ]; do
  echo "Waiting for ${BASE_PATH}/.ready"
  sleep 60
done

export CUDA_VISIBLE_DEVICES="$GPU"
export WANDB_PROJECT=${WANDB_PROJECT:-DPRM-DNA}
export WANDB_GROUP=${WANDB_GROUP:-sdpo-dna}
export WANDB_NAME="$RUN_NAME"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

OUT_DIR="${BASE_PATH}/dprm_sdpo_outputs/${RUN_NAME}"
mkdir -p "$OUT_DIR"

python finetune_sdpo.py \
  --seed "$SEED" \
  --base_path "$BASE_PATH" \
  --num_epochs "$NUM_EPOCHS" \
  --K "$K" \
  --lr "$LR" \
  --beta "$SDPO_BETA" \
  --save_path "${OUT_DIR}/model.pt" \
  --wandb True \
  --eval_every 999999 \
  --skip_final_inline_eval \
  --order_policy "$ORDER_POLICY" \
  --dprm_beta "$DPRM_BETA" \
  --dprm_warmup_steps "$DPRM_WARMUP_STEPS" \
  --dprm_switch_steps "$DPRM_SWITCH_STEPS" \
  --dprm_ready_count "$DPRM_READY_COUNT" \
  --dprm_shortlist_size "$DPRM_SHORTLIST_SIZE"

python eval_dna_bootstrap.py \
  --seed "$SEED" \
  --base_path "$BASE_PATH" \
  --model_path "dprm_sdpo_outputs/${RUN_NAME}/model.pt" \
  --num_sample_batches "$EVAL_BATCHES" \
  --num_samples_per_batch "$EVAL_BATCH_SIZE" \
  --bootstrap "$BOOTSTRAP" \
  --output "${OUT_DIR}/eval_bootstrap.json" \
  --order_policy "$ORDER_POLICY" \
  --dprm_beta "$DPRM_BETA" \
  --dprm_warmup_steps "$DPRM_WARMUP_STEPS" \
  --dprm_switch_steps "$DPRM_SWITCH_STEPS" \
  --dprm_ready_count "$DPRM_READY_COUNT" \
  --dprm_shortlist_size "$DPRM_SHORTLIST_SIZE"
