#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PRISM_ROOT:?set PRISM_ROOT to the Dream_Prism checkout}"
MODEL_PATH="${DREAM_MODEL_PATH:?set DREAM_MODEL_PATH to the Dream-v0-Instruct-7B checkpoint}"
BASE_OUTPUT_PATH="${BASE_OUTPUT_PATH:-${PROJECT_ROOT}/outputs/dream_gsm8k}"
GPU_IDS="${GPU_IDS:-0}"

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export HF_ALLOW_CODE_EVAL=1
export PYTHONPATH=.

TASK="gsm8k"      
LENGTH=256       
STEPS=256         
PORT="${PORT:-12334}"
ORDER_POLICY="${ORDER_POLICY:-confidence}"
NAME="${NAME:-win_0.1-0.6_s2_k4_${ORDER_POLICY}}"

mkdir -p "${BASE_OUTPUT_PATH}/${NAME}"

accelerate launch --main_process_port ${PORT} -m lm_eval\
    --model diffllm \
    --tasks ${TASK} \
    --batch_size 1 \
    --model_args "pretrained=${MODEL_PATH},trust_remote_code=True,dtype=bfloat16,max_new_tokens=${LENGTH},diffusion_steps=${STEPS}" \
    --gen_kwargs "use_hts=True,initial_N=16,final_K=4,hts_survivor_k=2,hts_mode=True,hts_start_pct=0.1,hts_end_pct=0.6,pruning_interval=3,decay_factor=1.8,reward_mode=svf,task_type=math,temperature=0.7,order_policy=${ORDER_POLICY},dprm_num_bins=16,dprm_phase_buckets=8,dprm_reward_beta=1.0,dprm_lambda=1.0,dprm_warmup_pct=0.2,dprm_switch_pct=0.7,dprm_ready_count=64,dprm_candidate_multiplier=4,dprm_min_candidates=8,dprm_max_candidates=64,realtime_output=${BASE_OUTPUT_PATH}/${NAME}/res.jsonl" \
    --num_fewshot 0 \
    --confirm_run_unsafe_code \
    --output_path "${BASE_OUTPUT_PATH}/${NAME}"
