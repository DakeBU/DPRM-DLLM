#!/bin/bash
# Note: If this file is called from loop.sh, then --job-name and --output below will be overridden

#SBATCH --job-name=dmpo
#SBATCH --partition=coe-gpu
#SBATCH --gres=gpu:A100:8
#SBATCH --time=02:00:00
# max 16 GPU hours, i.e., time <= 16h / num of GPUs
#SBATCH --mem-per-gpu=80G
# current machine: 8 x A100 80GB
# vanilla DMPO fits with num_replicates=4; progressive masking is heavier, so the defaults below reduce ELBO/replicate load
#SBATCH --cpus-per-task=8
#SBATCH --wait-all-nodes=1
#SBATCH --output=../outputs/%j.%x/.log

NUM_PROCESSES="${NUM_PROCESSES:-8}"

# If this file is called from loop.sh, then receive environment variables from it;
# Otherwise, generate randomly or based on the current timestamp
export WANDB_RUN_ID="${WANDB_RUN_ID:-"$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c8)"}"
echo "WandB Run ID: ${WANDB_RUN_ID}"

RUN_NAME="${RUN_NAME:-"$(date '+%m%d%H%M%S')"}"
echo "Run name: ${RUN_NAME}"



DATASET="${DATASET:-"gsm8k"}"
LOSS_MASK_SAMPLER="${LOSS_MASK_SAMPLER:-"random"}"
LOSS_PROGRESSIVE_K="${LOSS_PROGRESSIVE_K:-8}"
LOSS_PROGRESSIVE_PHASE_INIT="${LOSS_PROGRESSIVE_PHASE_INIT:-"random"}"
LOSS_PROGRESSIVE_THRESHOLD="${LOSS_PROGRESSIVE_THRESHOLD:-""}"
LOSS_PROGRESSIVE_ORDER_POLICY="${LOSS_PROGRESSIVE_ORDER_POLICY:-"confidence"}"
LOSS_PROGRESSIVE_DPRM_BINS="${LOSS_PROGRESSIVE_DPRM_BINS:-16}"
LOSS_PROGRESSIVE_DPRM_REWARD_TEMPERATURE="${LOSS_PROGRESSIVE_DPRM_REWARD_TEMPERATURE:-1.0}"
LOSS_PROGRESSIVE_DPRM_LAMBDA="${LOSS_PROGRESSIVE_DPRM_LAMBDA:-1.0}"
LOSS_PROGRESSIVE_DPRM_WARMUP_STEPS="${LOSS_PROGRESSIVE_DPRM_WARMUP_STEPS:-500}"
LOSS_PROGRESSIVE_DPRM_SWITCH_STEPS="${LOSS_PROGRESSIVE_DPRM_SWITCH_STEPS:-2000}"
LOSS_PROGRESSIVE_DPRM_READY_COUNT="${LOSS_PROGRESSIVE_DPRM_READY_COUNT:-128}"
LOSS_PROGRESSIVE_DPRM_MODE="${LOSS_PROGRESSIVE_DPRM_MODE:-sampled}"
LOSS_PROGRESSIVE_DPRM_CANDIDATE_MULTIPLIER="${LOSS_PROGRESSIVE_DPRM_CANDIDATE_MULTIPLIER:-4}"
LOSS_PROGRESSIVE_DPRM_MAX_CANDIDATES="${LOSS_PROGRESSIVE_DPRM_MAX_CANDIDATES:-32}"
LOSS_PROGRESSIVE_DPRM_MIN_CANDIDATES="${LOSS_PROGRESSIVE_DPRM_MIN_CANDIDATES:-8}"

if [ "${LOSS_MASK_SAMPLER}" = "progressive" ]; then
    PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
    GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-4}"
    NUM_REPLICATES="${NUM_REPLICATES:-2}"
    COMPUTE_REF_LOG_PROB_ELBO_SIZE="${COMPUTE_REF_LOG_PROB_ELBO_SIZE:-2}"
    LOSS_ANTITHETIC="${LOSS_ANTITHETIC:-false}"
else
    PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
    GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-8}"
    NUM_REPLICATES="${NUM_REPLICATES:-4}"
    COMPUTE_REF_LOG_PROB_ELBO_SIZE="${COMPUTE_REF_LOG_PROB_ELBO_SIZE:-4}"
    LOSS_ANTITHETIC="${LOSS_ANTITHETIC:-true}"
fi

GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
NUM_ITERATIONS="${NUM_ITERATIONS:-8}"
SAMPLER="${SAMPLER:-"pd_cache_prefix"}"
USE_FAST_SAMPLER="${USE_FAST_SAMPLER:-"fast_dllm"}"
SAMPLER_REMASKING="${SAMPLER_REMASKING:-""}"
SAMPLER_STEPS="${SAMPLER_STEPS:-128}"
TEMPERATURE="${TEMPERATURE:-0.2}"
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-"GSAI-ML/LLaDA-8B-Instruct"}"
SYNC_REF_MODEL="${SYNC_REF_MODEL:-true}"
REF_MODEL_SYNC_STEPS="${REF_MODEL_SYNC_STEPS:-64}"
ALPHA="${ALPHA:-0.04}"

if [ -z "${SAMPLER_REMASKING}" ]; then
    if [ "${LOSS_MASK_SAMPLER}" = "progressive" ] && [ "${LOSS_PROGRESSIVE_ORDER_POLICY}" = "dprm_soft_bon" ]; then
        SAMPLER_REMASKING="dprm_soft_bon"
    else
        SAMPLER_REMASKING="low_confidence"
    fi
fi

LOG_DIR="${LOG_DIR:-"../outputs/${DATASET}/${LOSS_MASK_SAMPLER}/${RUN_NAME}"}"
mkdir -p "${LOG_DIR}"
echo "Training logs and checkpoints will be saved to: ${LOG_DIR}"
echo "Loss mask sampler: ${LOSS_MASK_SAMPLER}"

MIN_PORT=10000; MAX_PORT=65535; RANDOM_PORT=$(( RANDOM % (MAX_PORT - MIN_PORT + 1) + MIN_PORT ))

# Check if there are previous checkpoints. If found, resume training from the last one
LAST_CHECKPOINT=$(ls -d "${LOG_DIR}/checkpoint-"* 2>/dev/null | sort -V | tail -n 1)
if [ -n "${LAST_CHECKPOINT}" ]; then
    echo "Found the last checkpoint: ${LAST_CHECKPOINT}, resuming training from this checkpoint."
    RESUME_FLAG="--resume_from_checkpoint ${LAST_CHECKPOINT}"
else
    echo "No previous checkpoints found, starting new training."
    RESUME_FLAG=""
fi

MASKING_ARGS=(
    --loss_mask_sampler "${LOSS_MASK_SAMPLER}"
    --loss_progressive_k "${LOSS_PROGRESSIVE_K}"
    --loss_progressive_phase_init "${LOSS_PROGRESSIVE_PHASE_INIT}"
    --loss_progressive_order_policy "${LOSS_PROGRESSIVE_ORDER_POLICY}"
    --loss_progressive_dprm_bins "${LOSS_PROGRESSIVE_DPRM_BINS}"
    --loss_progressive_dprm_reward_temperature "${LOSS_PROGRESSIVE_DPRM_REWARD_TEMPERATURE}"
    --loss_progressive_dprm_lambda "${LOSS_PROGRESSIVE_DPRM_LAMBDA}"
    --loss_progressive_dprm_warmup_steps "${LOSS_PROGRESSIVE_DPRM_WARMUP_STEPS}"
    --loss_progressive_dprm_switch_steps "${LOSS_PROGRESSIVE_DPRM_SWITCH_STEPS}"
    --loss_progressive_dprm_ready_count "${LOSS_PROGRESSIVE_DPRM_READY_COUNT}"
    --loss_progressive_dprm_mode "${LOSS_PROGRESSIVE_DPRM_MODE}"
    --loss_progressive_dprm_candidate_multiplier "${LOSS_PROGRESSIVE_DPRM_CANDIDATE_MULTIPLIER}"
    --loss_progressive_dprm_max_candidates "${LOSS_PROGRESSIVE_DPRM_MAX_CANDIDATES}"
    --loss_progressive_dprm_min_candidates "${LOSS_PROGRESSIVE_DPRM_MIN_CANDIDATES}"
)
if [ -n "${LOSS_PROGRESSIVE_THRESHOLD}" ]; then
    MASKING_ARGS+=(--loss_progressive_threshold "${LOSS_PROGRESSIVE_THRESHOLD}")
fi

srun accelerate launch \
    --config_file accelerate.yaml \
    --num_processes $NUM_PROCESSES \
    --main_process_port $RANDOM_PORT dmpo_train.py \
    --config dmpo_train_config.yaml \
    --dataset $DATASET \
    --run_name $RUN_NAME \
    --output_dir $LOG_DIR \
    ${RESUME_FLAG} \
    --advantage_centering true \
    --advantage_centering_unbias false \
    --advantage_centering_neg true \
    --compute_ref_log_prob_elbo true \
    --compute_ref_log_prob_elbo_size "${COMPUTE_REF_LOG_PROB_ELBO_SIZE}" \
    --centering_strength 1.0 \
    --alpha "${ALPHA}" \
    --num_generations "${NUM_GENERATIONS}" \
    --num_iterations "${NUM_ITERATIONS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --sync_ref_model "${SYNC_REF_MODEL}" \
    --ref_model_sync_steps "${REF_MODEL_SYNC_STEPS}" \
    --generation_batch_size "${GENERATION_BATCH_SIZE}" \
    --save_total_limit 10 \
    --loss_mask_non_eos false \
    --num_replicates "${NUM_REPLICATES}" \
    --loss_antithetic "${LOSS_ANTITHETIC}" \
    --loss "wdce" \
    --use_fast_sampler "${USE_FAST_SAMPLER}" \
    --sampler "${SAMPLER}" \
    --sampler_remasking "${SAMPLER_REMASKING}" \
    --sampler_steps "${SAMPLER_STEPS}" \
    --temperature "${TEMPERATURE}" \
    --pretrained_model_path "${PRETRAINED_MODEL_PATH}" \
    "${MASKING_ARGS[@]}"

# Note: append additional training arguments above, but avoid changing the config file
