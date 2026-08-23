#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 {gsm8k|math|countdown|all} {random|confidence|dprm_confidence|dprm_random}" >&2
}

TASK="${1:-}"
POLICY="${2:-}"
if [[ -z "$TASK" || -z "$POLICY" ]]; then
  usage
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$TASK" == "all" ]]; then
  for task in gsm8k math countdown; do
    "$0" "$task" "$POLICY"
  done
  exit 0
fi

case "$TASK" in
  gsm8k|countdown)
    LEARNING_RATE="1e-6"
    GRADIENT_ACCUMULATION_STEPS="2"
    ;;
  math)
    LEARNING_RATE="3e-6"
    GRADIENT_ACCUMULATION_STEPS="4"
    ;;
  *)
    usage
    exit 2
    ;;
esac

LOSS_MASK_SAMPLER="progressive"
LOSS_PROGRESSIVE_ORDER_POLICY="confidence"
LOSS_PROGRESSIVE_DPRM_WARMUP_POLICY="confidence"
SAMPLER_REMASKING="low_confidence"
case "$POLICY" in
  random)
    LOSS_MASK_SAMPLER="random"
    ;;
  confidence)
    ;;
  dprm_confidence)
    LOSS_PROGRESSIVE_ORDER_POLICY="dprm_soft_bon"
    SAMPLER_REMASKING="dprm_soft_bon"
    ;;
  dprm_random)
    LOSS_PROGRESSIVE_ORDER_POLICY="dprm_soft_bon"
    LOSS_PROGRESSIVE_DPRM_WARMUP_POLICY="random"
    SAMPLER_REMASKING="dprm_soft_bon"
    ;;
  *)
    usage
    exit 2
    ;;
esac

OUTPUT_ROOT="${DPRM_OUTPUT_ROOT:-../outputs/paper}"
RUN_NAME="${RUN_NAME:-dmpo-${TASK}-${POLICY}-paper-seed42}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/${TASK}/${POLICY}}"

export DATASET="$TASK"
export RUN_NAME LOG_DIR LEARNING_RATE GRADIENT_ACCUMULATION_STEPS
export LOSS_MASK_SAMPLER LOSS_PROGRESSIVE_ORDER_POLICY SAMPLER_REMASKING
export LOSS_PROGRESSIVE_DPRM_WARMUP_POLICY
export LOSS_PROGRESSIVE_K=8
export LOSS_PROGRESSIVE_THRESHOLD=0.9
export LOSS_PROGRESSIVE_DPRM_BINS=16
export LOSS_PROGRESSIVE_DPRM_REWARD_TEMPERATURE=1.0
export LOSS_PROGRESSIVE_DPRM_LAMBDA=1.0
export LOSS_PROGRESSIVE_DPRM_WARMUP_STEPS=500
export LOSS_PROGRESSIVE_DPRM_SWITCH_STEPS=2000
export LOSS_PROGRESSIVE_DPRM_READY_COUNT=128
export LOSS_PROGRESSIVE_DPRM_MODE=sampled
export LOSS_PROGRESSIVE_DPRM_CANDIDATE_MULTIPLIER=4
export LOSS_PROGRESSIVE_DPRM_MIN_CANDIDATES=8
export LOSS_PROGRESSIVE_DPRM_MAX_CANDIDATES=32
export MAX_STEPS=5000
export NUM_GENERATIONS=8
export NUM_ITERATIONS=8
export PER_DEVICE_TRAIN_BATCH_SIZE=4
export GENERATION_BATCH_SIZE=4
export NUM_REPLICATES=2
export COMPUTE_REF_LOG_PROB_ELBO_SIZE=2
export LOSS_ANTITHETIC=false
export SAMPLER_STEPS=128
export TEMPERATURE=0.2

if [[ "${DPRM_DRY_RUN:-0}" == "1" ]]; then
  env | sort | grep -E '^(DATASET|RUN_NAME|LOG_DIR|LEARNING_RATE|GRADIENT_ACCUMULATION_STEPS|LOSS_.*|SAMPLER_.*|MAX_STEPS|NUM_GENERATIONS|NUM_ITERATIONS|PER_DEVICE_TRAIN_BATCH_SIZE|GENERATION_BATCH_SIZE|NUM_REPLICATES|COMPUTE_REF_LOG_PROB_ELBO_SIZE|TEMPERATURE)='
  exit 0
fi

cd "$SCRIPT_DIR"
exec bash run_dmpo.sh
