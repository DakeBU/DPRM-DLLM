#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

ENV_ACTIVATE="${ENV_ACTIVATE:-<PYTHON_ENV>/bin/activate}"
source "$ENV_ACTIVATE"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
RANDOM_RUN_DIR="${RANDOM_RUN_DIR:-<DMPO_DPRM_REPO>/outputs/gsm8k/random/dmpo-random-0329_230101}"
PROGRESSIVE_RUN_DIR="${PROGRESSIVE_RUN_DIR:-<DMPO_DPRM_REPO>/outputs/gsm8k/progressive/dmpo-progressive-0329_225919}"
OUTPUT_DIR="${OUTPUT_DIR:-<DMPO_DPRM_REPO>/eval_outputs/passk_gsm8k_compare}"
GPU_IDS="${GPU_IDS:-1,2}"
VANILLA_GPU_ID="${VANILLA_GPU_ID:-}"
PROGRESSIVE_GPU_ID="${PROGRESSIVE_GPU_ID:-}"
BASE_GPU_ID="${BASE_GPU_ID:-}"
KS="${KS:-1 2 4 8 16 32}"
TEST_SIZE="${TEST_SIZE:-1319}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GEN_LENGTH="${GEN_LENGTH:-256}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-128}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
TEMPERATURE="${TEMPERATURE:-0.2}"
SAMPLER="${SAMPLER:-pd_cache_prefix}"
USE_FAST_SAMPLER="${USE_FAST_SAMPLER:-fast_dllm}"
BASE_REMASKING="${BASE_REMASKING:-low_confidence}"
VANILLA_REMASKING="${VANILLA_REMASKING:-low_confidence}"
PROGRESSIVE_REMASKING="${PROGRESSIVE_REMASKING:-low_confidence}"
BASE_DPRM_ESTIMATOR_PATH="${BASE_DPRM_ESTIMATOR_PATH:-}"
VANILLA_DPRM_ESTIMATOR_PATH="${VANILLA_DPRM_ESTIMATOR_PATH:-}"
PROGRESSIVE_DPRM_ESTIMATOR_PATH="${PROGRESSIVE_DPRM_ESTIMATOR_PATH:-}"
CHECKPOINT_MODE="${CHECKPOINT_MODE:-latest_common}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-}"
INCLUDE_BASE="${INCLUDE_BASE:-true}"
BASE_LABEL="${BASE_LABEL:-Base Model}"
VANILLA_LABEL="${VANILLA_LABEL:-DMPO}"
PROGRESSIVE_LABEL="${PROGRESSIVE_LABEL:-Progressive DMPO}"
SAVE_EVERY_BATCHES="${SAVE_EVERY_BATCHES:-1}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/raw"

if [[ -z "$VANILLA_GPU_ID" || -z "$PROGRESSIVE_GPU_ID" ]]; then
  GPU_CLEAN="${GPU_IDS// /}"
  IFS=',' read -r -a GPU_ARRAY <<< "$GPU_CLEAN"
  if [[ -z "$VANILLA_GPU_ID" ]]; then
    VANILLA_GPU_ID="${GPU_ARRAY[0]:-}"
  fi
  if [[ -z "$PROGRESSIVE_GPU_ID" ]]; then
    PROGRESSIVE_GPU_ID="${GPU_ARRAY[1]:-}"
  fi
  if [[ -z "$BASE_GPU_ID" ]]; then
    BASE_GPU_ID="${GPU_ARRAY[2]:-}"
  fi
fi

if [[ -z "$VANILLA_GPU_ID" || -z "$PROGRESSIVE_GPU_ID" ]]; then
  echo "Need two GPU ids. Set GPU_IDS=1,2 or set VANILLA_GPU_ID / PROGRESSIVE_GPU_ID." >&2
  exit 1
fi

latest_checkpoint() {
  local run_dir="$1"
  find "$run_dir" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1
}

checkpoint_at_step() {
  local run_dir="$1"
  local step="$2"
  local path="$run_dir/checkpoint-$step"
  if [[ ! -d "$path" ]]; then
    echo "Missing checkpoint: $path" >&2
    exit 1
  fi
  echo "$path"
}

latest_common_checkpoint() {
  local run_dir_a="$1"
  local run_dir_b="$2"
  local steps_a steps_b common_steps last_common
  steps_a="$(find "$run_dir_a" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sed 's/checkpoint-//' | sort -n)"
  steps_b="$(find "$run_dir_b" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sed 's/checkpoint-//' | sort -n)"
  common_steps="$(comm -12 <(printf '%s\n' "$steps_a") <(printf '%s\n' "$steps_b"))"
  last_common="$(printf '%s\n' "$common_steps" | tail -n 1)"
  if [[ -z "$last_common" ]]; then
    echo "No shared checkpoint step found between $run_dir_a and $run_dir_b" >&2
    exit 1
  fi
  echo "$last_common"
}

if [[ -n "$CHECKPOINT_STEP" ]]; then
  COMMON_STEP="$CHECKPOINT_STEP"
elif [[ "$CHECKPOINT_MODE" == "latest_common" ]]; then
  COMMON_STEP="$(latest_common_checkpoint "$RANDOM_RUN_DIR" "$PROGRESSIVE_RUN_DIR")"
else
  COMMON_STEP=""
fi

if [[ -n "$COMMON_STEP" ]]; then
  RANDOM_CKPT="$(checkpoint_at_step "$RANDOM_RUN_DIR" "$COMMON_STEP")"
  PROGRESSIVE_CKPT="$(checkpoint_at_step "$PROGRESSIVE_RUN_DIR" "$COMMON_STEP")"
else
  RANDOM_CKPT="$(latest_checkpoint "$RANDOM_RUN_DIR")"
  PROGRESSIVE_CKPT="$(latest_checkpoint "$PROGRESSIVE_RUN_DIR")"
fi

if [[ -z "$RANDOM_CKPT" || -z "$PROGRESSIVE_CKPT" ]]; then
  echo "Failed to resolve checkpoints." >&2
  exit 1
fi

VANILLA_STATE_DIR="${VANILLA_STATE_DIR:-$OUTPUT_DIR/raw/vanilla_step${COMMON_STEP:-latest}}"
PROGRESSIVE_STATE_DIR="${PROGRESSIVE_STATE_DIR:-$OUTPUT_DIR/raw/progressive_step${COMMON_STEP:-latest}}"
BASE_STATE_DIR="${BASE_STATE_DIR:-$OUTPUT_DIR/raw/base_step${COMMON_STEP:-latest}}"
mkdir -p "$VANILLA_STATE_DIR" "$PROGRESSIVE_STATE_DIR"
if [[ "$INCLUDE_BASE" == "true" ]]; then
  mkdir -p "$BASE_STATE_DIR"
fi

echo "Comparing checkpoints:"
echo "  ${VANILLA_LABEL}: $RANDOM_CKPT"
echo "  ${PROGRESSIVE_LABEL}: $PROGRESSIVE_CKPT"
if [[ -n "$COMMON_STEP" ]]; then
  echo "  Matched checkpoint step: $COMMON_STEP"
fi
echo "Running in parallel on GPUs:"
echo "  ${VANILLA_LABEL}: GPU $VANILLA_GPU_ID"
echo "  ${PROGRESSIVE_LABEL}: GPU $PROGRESSIVE_GPU_ID"
if [[ "$INCLUDE_BASE" == "true" ]]; then
  if [[ -n "$BASE_GPU_ID" && "$BASE_GPU_ID" != "$VANILLA_GPU_ID" && "$BASE_GPU_ID" != "$PROGRESSIVE_GPU_ID" ]]; then
    echo "  ${BASE_LABEL}: GPU $BASE_GPU_ID"
  else
    echo "  ${BASE_LABEL}: sequential on GPU $VANILLA_GPU_ID"
  fi
fi

(
  CUDA_VISIBLE_DEVICES="$VANILLA_GPU_ID" python "$PROJECT_ROOT/eval_passk_gsm8k_single.py" \
    --base_model_path "$BASE_MODEL_PATH" \
    --checkpoint "$RANDOM_CKPT" \
    --model_label "$VANILLA_LABEL" \
    --output_dir "$VANILLA_STATE_DIR" \
    --ks $KS \
    --test_size "$TEST_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --gen_length "$GEN_LENGTH" \
    --diffusion_steps "$DIFFUSION_STEPS" \
    --block_length "$BLOCK_LENGTH" \
    --temperature "$TEMPERATURE" \
    --sampler "$SAMPLER" \
    --use_fast_sampler "$USE_FAST_SAMPLER" \
    --remasking "$VANILLA_REMASKING" \
    --dprm_estimator_path "$VANILLA_DPRM_ESTIMATOR_PATH" \
    --save_every_batches "$SAVE_EVERY_BATCHES"
) 2>&1 | tee -a "$OUTPUT_DIR/vanilla_eval.log" &
VANILLA_PID=$!

(
  CUDA_VISIBLE_DEVICES="$PROGRESSIVE_GPU_ID" python "$PROJECT_ROOT/eval_passk_gsm8k_single.py" \
    --base_model_path "$BASE_MODEL_PATH" \
    --checkpoint "$PROGRESSIVE_CKPT" \
    --model_label "$PROGRESSIVE_LABEL" \
    --output_dir "$PROGRESSIVE_STATE_DIR" \
    --ks $KS \
    --test_size "$TEST_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --gen_length "$GEN_LENGTH" \
    --diffusion_steps "$DIFFUSION_STEPS" \
    --block_length "$BLOCK_LENGTH" \
    --temperature "$TEMPERATURE" \
    --sampler "$SAMPLER" \
    --use_fast_sampler "$USE_FAST_SAMPLER" \
    --remasking "$PROGRESSIVE_REMASKING" \
    --dprm_estimator_path "$PROGRESSIVE_DPRM_ESTIMATOR_PATH" \
    --save_every_batches "$SAVE_EVERY_BATCHES"
) 2>&1 | tee -a "$OUTPUT_DIR/progressive_eval.log" &
PROGRESSIVE_PID=$!

BASE_PID=""
BASE_STATUS=0
if [[ "$INCLUDE_BASE" == "true" && -n "$BASE_GPU_ID" && "$BASE_GPU_ID" != "$VANILLA_GPU_ID" && "$BASE_GPU_ID" != "$PROGRESSIVE_GPU_ID" ]]; then
(
  CUDA_VISIBLE_DEVICES="$BASE_GPU_ID" python "$PROJECT_ROOT/eval_passk_gsm8k_single.py" \
    --base_model_path "$BASE_MODEL_PATH" \
    --checkpoint "" \
    --model_label "$BASE_LABEL" \
    --output_dir "$BASE_STATE_DIR" \
    --ks $KS \
    --test_size "$TEST_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --gen_length "$GEN_LENGTH" \
    --diffusion_steps "$DIFFUSION_STEPS" \
    --block_length "$BLOCK_LENGTH" \
    --temperature "$TEMPERATURE" \
    --sampler "$SAMPLER" \
    --use_fast_sampler "$USE_FAST_SAMPLER" \
    --remasking "$BASE_REMASKING" \
    --dprm_estimator_path "$BASE_DPRM_ESTIMATOR_PATH" \
    --save_every_batches "$SAVE_EVERY_BATCHES"
) 2>&1 | tee -a "$OUTPUT_DIR/base_eval.log" &
BASE_PID=$!
fi

VANILLA_STATUS=0
PROGRESSIVE_STATUS=0
wait "$VANILLA_PID" || VANILLA_STATUS=$?
wait "$PROGRESSIVE_PID" || PROGRESSIVE_STATUS=$?
if [[ -n "$BASE_PID" ]]; then
  wait "$BASE_PID" || BASE_STATUS=$?
fi

if [[ "$VANILLA_STATUS" -ne 0 || "$PROGRESSIVE_STATUS" -ne 0 || "$BASE_STATUS" -ne 0 ]]; then
  echo "One or more model evaluations failed." >&2
  echo "  ${VANILLA_LABEL} status: $VANILLA_STATUS" >&2
  echo "  ${PROGRESSIVE_LABEL} status: $PROGRESSIVE_STATUS" >&2
  if [[ "$INCLUDE_BASE" == "true" ]]; then
    echo "  ${BASE_LABEL} status: $BASE_STATUS" >&2
  fi
  exit 1
fi

if [[ "$INCLUDE_BASE" == "true" && -z "$BASE_PID" ]]; then
(
  CUDA_VISIBLE_DEVICES="$VANILLA_GPU_ID" python "$PROJECT_ROOT/eval_passk_gsm8k_single.py" \
    --base_model_path "$BASE_MODEL_PATH" \
    --checkpoint "" \
    --model_label "$BASE_LABEL" \
    --output_dir "$BASE_STATE_DIR" \
    --ks $KS \
    --test_size "$TEST_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --gen_length "$GEN_LENGTH" \
    --diffusion_steps "$DIFFUSION_STEPS" \
    --block_length "$BLOCK_LENGTH" \
    --temperature "$TEMPERATURE" \
    --sampler "$SAMPLER" \
    --use_fast_sampler "$USE_FAST_SAMPLER" \
    --remasking "$BASE_REMASKING" \
    --dprm_estimator_path "$BASE_DPRM_ESTIMATOR_PATH" \
    --save_every_batches "$SAVE_EVERY_BATCHES"
) 2>&1 | tee -a "$OUTPUT_DIR/base_eval.log"
fi

COMBINE_ARGS=(
  --vanilla_state_dir "$VANILLA_STATE_DIR"
  --progressive_state_dir "$PROGRESSIVE_STATE_DIR"
  --output_dir "$OUTPUT_DIR"
  --base_label "$BASE_LABEL"
  --vanilla_label "$VANILLA_LABEL"
  --progressive_label "$PROGRESSIVE_LABEL"
)

if [[ "$INCLUDE_BASE" == "true" ]]; then
  COMBINE_ARGS+=(--base_state_dir "$BASE_STATE_DIR")
fi

python "$PROJECT_ROOT/combine_passk_gsm8k.py" "${COMBINE_ARGS[@]}"

echo "Saved outputs in $OUTPUT_DIR"
