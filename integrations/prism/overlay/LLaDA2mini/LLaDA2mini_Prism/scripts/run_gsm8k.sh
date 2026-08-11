#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PRISM_ROOT:?set PRISM_ROOT to the LLaDA2mini_Prism checkout}"
MODEL_PATH="${MODEL_PATH:-inclusionAI/LLaDA2.0-mini}"
BASE_OUTPUT_PATH="${BASE_OUTPUT_PATH:-${PROJECT_ROOT}/outputs/llada2_gsm8k}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"

cd "$PROJECT_ROOT"

LENGTH=256
STEPS=32
BLOCK=32
TASK="gsm8k"
TYPE="math"
ORDER_POLICY="${ORDER_POLICY:-confidence}"
NAME="${NAME:-win_0.1-0.6_s2_k4_${ORDER_POLICY}}"

mkdir -p "${BASE_OUTPUT_PATH}/${NAME}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" accelerate launch evaluation_script.py \
    --model LLaDA2 \
    --tasks ${TASK} \
    --batch_size 1 \
    --model_args "pretrained=${MODEL_PATH},assistant_prefix=<reasoning> " \
    --gen_kwargs "use_hts=True,hts_N=16,final_K=4,hts_survivor_k=2,hts_mode=True,hts_start_pct=0.1,hts_end_pct=0.6,pruning_interval=3,decay_factor=1.8,reward_mode=svf,task_type=${TYPE},steps=${STEPS},block_length=${BLOCK},gen_length=${LENGTH},temperature=0.7,order_policy=${ORDER_POLICY},dprm_num_bins=16,dprm_phase_buckets=8,dprm_reward_beta=1.0,dprm_lambda=1.0,dprm_warmup_pct=0.2,dprm_switch_pct=0.7,dprm_ready_count=64,dprm_candidate_multiplier=4,dprm_min_candidates=8,dprm_max_candidates=64,realtime_output=${BASE_OUTPUT_PATH}/${NAME}/res.jsonl" \
    --num_fewshot 0 \
    --output_path "${BASE_OUTPUT_PATH}/${NAME}"
