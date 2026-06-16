#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-<SDPO_HOST_REPO>}
ENV=${ENV:-<PYTHON_ENV>/bin/activate}
GPU=${GPU:-0}
BASE_PATH=${BASE_PATH:-data_and_model/}
EVAL_BATCHES=${EVAL_BATCHES:-10}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-64}
BOOTSTRAP=${BOOTSTRAP:-2000}
DPRM_BETA=${DPRM_BETA:-1.0}
DPRM_WARMUP_STEPS=${DPRM_WARMUP_STEPS:-100}
DPRM_SWITCH_STEPS=${DPRM_SWITCH_STEPS:-400}
DPRM_READY_COUNT=${DPRM_READY_COUNT:-64}
DPRM_SHORTLIST_SIZE=${DPRM_SHORTLIST_SIZE:-64}

if [[ "$ROOT" == "<SDPO_HOST_REPO>" || "$ENV" == "<PYTHON_ENV>/bin/activate" ]]; then
  echo "Set ROOT to the SDPO host repo and ENV to a Python environment before running." >&2
  exit 1
fi

cd "${ROOT}"
source "${ENV}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM=false

declare -a RUNS=(
  "sdpo-dna-baseline:baseline"
  "sdpo-dna-progressive:progressive"
  "sdpo-dna-dprm-confidence:dprm"
  "sdpo-dna-dprm-random:dprm_random"
)

for item in "${RUNS[@]}"; do
  IFS=: read -r run_name policy <<< "${item}"
  model_rel="dprm_sdpo_outputs/${run_name}/model.pt"
  output_path="${BASE_PATH}/dprm_sdpo_outputs/${run_name}/eval_bootstrap.json"
  if [ ! -f "${BASE_PATH}/${model_rel}" ]; then
    echo "[missing] ${BASE_PATH}/${model_rel}; skip ${run_name}" >&2
    continue
  fi
  echo "[eval] ${run_name} policy=${policy}"
  python eval_dna_bootstrap.py \
    --seed 0 \
    --base_path "${BASE_PATH}" \
    --model_path "${model_rel}" \
    --num_sample_batches "${EVAL_BATCHES}" \
    --num_samples_per_batch "${EVAL_BATCH_SIZE}" \
    --bootstrap "${BOOTSTRAP}" \
    --output "${output_path}" \
    --order_policy "${policy}" \
    --dprm_beta "${DPRM_BETA}" \
    --dprm_warmup_steps "${DPRM_WARMUP_STEPS}" \
    --dprm_switch_steps "${DPRM_SWITCH_STEPS}" \
    --dprm_ready_count "${DPRM_READY_COUNT}" \
    --dprm_shortlist_size "${DPRM_SHORTLIST_SIZE}"
done

python scripts/summarize_sdpo_dna_results.py \
  --output-root "${BASE_PATH}/dprm_sdpo_outputs" \
  --summary-dir "eval_outputs/sdpo_dna_ordering"
