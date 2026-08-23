#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OMNI_ROOT="${OMNI_ROOT:?set OMNI_ROOT}"
ENV="${VIRTUAL_ENV:?activate the experiment environment before launching}"
PYTHON="${PYTHON:-${ENV}/bin/python}"
MODEL_PATH="${DPRM_OMNI_MODEL_PATH:?set DPRM_OMNI_MODEL_PATH}"
IMAGE_TOKENIZER_PATH="${DPRM_OMNI_IMAGE_TOKENIZER:?set DPRM_OMNI_IMAGE_TOKENIZER}"
DATA_CONFIG="${DPRM_OMNI_DATA_CONFIG:-${OMNI_ROOT}/configs/dprm_journeydb_t2i_tokenized.yaml}"
CONFIDENCE_DATA_CONFIG="${DPRM_OMNI_CONFIDENCE_DATA_CONFIG:-${DATA_CONFIG}}"
DPRM_DATA_CONFIG="${DPRM_OMNI_DPRM_DATA_CONFIG:-${DATA_CONFIG}}"
DATA_JSON="${DPRM_OMNI_DATA_JSON:-${OMNI_ROOT}/datasets/jsonl/JourneyDB/BLIP3o_JourneyDB_T2I_tokenized.jsonl}"
OUT_BASE="${DPRM_OMNI_OUT_BASE:?set DPRM_OMNI_OUT_BASE}"
MAX_STEPS="${DPRM_OMNI_MAX_STEPS:-1000}"
SAVE_STEPS="${DPRM_OMNI_SAVE_STEPS:-750}"
SAVE_TOTAL_LIMIT="${DPRM_OMNI_SAVE_TOTAL_LIMIT:-1}"
WARMUP_RATIO="${DPRM_OMNI_WARMUP_RATIO:-0.03}"
LEARNING_RATE="${DPRM_OMNI_LEARNING_RATE:-1e-5}"
TRAINABLE_LAST_N_LAYERS="${DPRM_OMNI_TRAINABLE_LAST_N_LAYERS:--1}"
NUM_TRAIN_EPOCHS="${DPRM_OMNI_NUM_TRAIN_EPOCHS:-1}"
GPUS="${DPRM_OMNI_GPUS:-0,1,2,3,4,5,6,7}"
NPROC="${DPRM_OMNI_NPROC:-8}"
DATALOADER_NUM_WORKERS="${DPRM_OMNI_DATALOADER_NUM_WORKERS:-0}"
RESUME_FROM_CHECKPOINT="${DPRM_OMNI_RESUME_FROM_CHECKPOINT:-}"
GRADIENT_CHECKPOINTING="${DPRM_OMNI_GRADIENT_CHECKPOINTING:-False}"
DEEPSPEED_CONFIG="${DPRM_OMNI_DEEPSPEED_CONFIG:-${OMNI_ROOT}/scripts/deepspeed/ds_config_zero2.json}"

IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
if (( ${#GPU_IDS[@]} != NPROC )); then
  echo "DPRM_OMNI_NPROC=${NPROC} but DPRM_OMNI_GPUS contains ${#GPU_IDS[@]} devices: ${GPUS}" >&2
  exit 2
fi

read -r -a ORDERS <<< "${DPRM_OMNI_ORDERS:-random progressive_confidence}"
MATCHED_SCORER="${DPRM_OMNI_DPRM_SCORER:-}"
MATCHED_REVEAL_BUDGET="${DPRM_OMNI_MATCHED_REVEAL_BUDGET:-1}"
ONLINE_ROLLIN="${DPRM_OMNI_ONLINE_ROLLIN:-0}"
HYBRID_ROLLIN="${DPRM_OMNI_HYBRID_ROLLIN:-1}"

mkdir -p "${OUT_BASE}"
source "${ENV}/bin/activate"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export WANDB_PROJECT="${WANDB_PROJECT:-DPRM-multimodal-order}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONPATH="${RELEASE_ROOT}/src:${OMNI_ROOT}:${OMNI_ROOT}/third_party/GLM-4-Voice:${OMNI_ROOT}/third_party/GLM-4-Voice/third_party/Matcha-TTS:${PYTHONPATH:-}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${OMNI_ROOT}/.triton_cache}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
export DEEPSPEED_TIMEOUT="${DEEPSPEED_TIMEOUT:-120}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-0}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-0}"
export DPRM_DEEPSPEED_REDUCE_CHUNK_SIZE="${DPRM_DEEPSPEED_REDUCE_CHUNK_SIZE:-50000000}"
export DPRM_OMNI_REPEAT_TO_MAX_STEPS="${DPRM_OMNI_REPEAT_TO_MAX_STEPS:-1}"
export DPRM_SKIP_FINAL_SAVE="${DPRM_SKIP_FINAL_SAVE:-1}"
export DPRM_OMNI_DATA_REPEAT="${DPRM_OMNI_DATA_REPEAT:-4000}"
export DPRM_OMNI_SEQUENTIAL_SAMPLER="${DPRM_OMNI_SEQUENTIAL_SAMPLER:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export DPRM_OMNI_THEORY_LOG="${DPRM_OMNI_THEORY_LOG:-1}"
export DPRM_GRAD_VARIANCE_INTERVAL="${DPRM_GRAD_VARIANCE_INTERVAL:-100}"
export DPRM_GRAD_NORM_WINDOW="${DPRM_GRAD_NORM_WINDOW:-100}"
export DPRM_CONF_POSITION_WEIGHT="${DPRM_CONF_POSITION_WEIGHT:-0.90}"
export DPRM_RANDOM_POSITION_WEIGHT="${DPRM_RANDOM_POSITION_WEIGHT:-0.65}"

until [[ -s "${DATA_JSON}" ]]; do
  echo "waiting for Omni formal T2I json: ${DATA_JSON}"
  sleep 120
done

# Validate the complete run before starting any branch. This prevents a shared
# confidence continuation from consuming compute when its paired DPRM
# continuation is not backed by the selected development-frozen controller.
for ORDER in "${ORDERS[@]}"; do
  if [[ "${ORDER}" == "dprm_matched" && ! -s "${MATCHED_SCORER}" ]]; then
    echo "refusing dprm_matched: set DPRM_OMNI_DPRM_SCORER to a frozen development artifact" >&2
    exit 2
  fi
  if [[ "${ORDER}" == "dprm_matched" && "${DPRM_ALLOW_OMNI_EQUAL_WIDTH_BUCKETS:-0}" != "1" ]]; then
    "${PYTHON}" - "${MATCHED_SCORER}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
artifact_format = payload.get("format")
allowed_formats = {
    "omni_bucket_table_dprm_v1",
    "omni_stage_rank_code_dprm_v1",
    "omni_stage_rank_spatial_dprm_v1",
}
if artifact_format not in allowed_formats:
    raise SystemExit(f"unsupported formal Omni controller: {artifact_format!r}")
config = payload.get("config", {})
source_summary = payload.get("metadata", {}).get("source_summary", {})
if artifact_format == "omni_bucket_table_dprm_v1":
    if not config.get("confidence_bin_edges"):
        raise SystemExit("absolute-confidence tables require frozen quantile edges")
    if source_summary.get("prompt_text_deduplicated") is not True:
        raise SystemExit("absolute-confidence tables require unique source prompts")
else:
    if not config.get("active_steps"):
        raise SystemExit("rank-bucket tables require fixed active steps")
    if source_summary.get("action_conditioned_continuations") is not True:
        raise SystemExit("rank-bucket tables require action-conditioned continuations")
score_contract = payload.get("metadata", {}).get("score_contract", {})
expected_score = {
    "base_order_score": "negative_token_entropy",
    "position_selection_rule": "single_path_top1_adjusted_order_score",
}
for key, expected in expected_score.items():
    if score_contract.get(key) != expected:
        raise SystemExit(
            f"formal Omni DPRM score contract mismatch for {key}: "
            f"{score_contract.get(key)!r} != {expected!r}"
        )
expected_coordinate = (
    "exp_negative_token_entropy"
    if artifact_format == "omni_bucket_table_dprm_v1"
    else (
        "within_state_confidence_rank_and_provisional_code"
        if artifact_format == "omni_stage_rank_code_dprm_v1"
        else "within_state_confidence_rank"
    )
)
if score_contract.get("bucket_coordinate") != expected_coordinate:
    raise SystemExit("formal Omni DPRM bucket-coordinate contract mismatch")
deployment = payload.get("metadata", {}).get("deployment_contract", {})
if deployment.get("paths_per_prompt") != 1 or deployment.get("terminal_reward_calls_at_test") != 0:
    raise SystemExit("formal Omni DPRM must use one path and no terminal-reward calls at test time")
if deployment.get("complete_image_selection") is not False:
    raise SystemExit("formal Omni DPRM cannot select among completed images")
if deployment.get("fixed_t2i_scaffold") is not True:
    raise SystemExit("formal Omni DPRM requires the fixed T2I scaffold")
if deployment.get("ordered_visual_positions") != 256:
    raise SystemExit("formal Omni DPRM must order exactly 256 visual-code positions")
if source_summary.get("fixed_visual_canvas") is not True:
    raise SystemExit("formal Omni DPRM requires fixed-canvas development rollouts")
stagewise = payload.get("metadata", {}).get("stagewise_order_contract", {})
if not stagewise.get("reward_action_steps"):
    raise SystemExit("formal Omni DPRM requires fixed stagewise reward actions")
if artifact_format == "omni_bucket_table_dprm_v1" and stagewise.get("max_base_score_gap") is None:
    raise SystemExit("formal Omni DPRM requires confidence-ambiguity gating")
if stagewise.get("fallback") != "native confidence order":
    raise SystemExit("formal Omni DPRM must fall back to native confidence order")
PY
  fi
done

cd "${OMNI_ROOT}"
for ORDER in "${ORDERS[@]}"; do
  if [[ "${ORDER}" == "dprm_matched" && ! -s "${MATCHED_SCORER}" ]]; then
    echo "refusing dprm_matched: set DPRM_OMNI_DPRM_SCORER to a frozen development artifact" >&2
    exit 2
  fi
  if [[ "${ORDER}" == dprm* && "${ORDER}" != "dprm_matched" && "${DPRM_ALLOW_OMNI_POSITION_PROXY_DPRM:-0}" != "1" ]]; then
    echo "refusing ${ORDER}: current Omni training path only has a position/jitter proxy, not true bucketized DPRM" >&2
    echo "set DPRM_ALLOW_OMNI_POSITION_PROXY_DPRM=1 only for explicitly labeled diagnostic proxy runs" >&2
    exit 2
  fi
  RUN_NAME="omni_${ORDER}_formal_$(date +%Y%m%d_%H%M%S)"
  OUT_DIR="${OUT_BASE}/${ORDER}"
  mkdir -p "${OUT_DIR}"
  for CKPT in "${OUT_DIR}"/checkpoint-*; do
    [[ -d "${CKPT}" ]] || continue
    [[ -s "${CKPT}/trainer_state.json" ]] || {
      echo "incomplete checkpoint requires inspection: ${CKPT}" >&2
      exit 2
    }
  done
  DONE_CKPT="${OUT_DIR}/checkpoint-${MAX_STEPS}/trainer_state.json"
  if [[ -s "${DONE_CKPT}" ]]; then
    echo "skipping ${ORDER}: found completed checkpoint ${DONE_CKPT}"
    continue
  fi
  OVERWRITE_ARGS=()
  if ! find "${OUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -print -quit | grep -q .; then
    OVERWRITE_ARGS=(--overwrite_output_dir)
    echo "allowing fresh start for ${ORDER}: output_dir has no checkpoint"
  fi
  RESUME_ARGS=()
  if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    EFFECTIVE_RESUME="${RESUME_FROM_CHECKPOINT}"
    LATEST_COMPLETE_CKPT="$(
      find "${OUT_DIR}" -maxdepth 2 -type f -path '*/checkpoint-*/trainer_state.json' -size +0c \
        -printf '%h\n' 2>/dev/null \
        | awk -F'checkpoint-' -v target="${MAX_STEPS}" \
            '/checkpoint-[0-9]+$/ && ($2 + 0) <= (target + 0) {print $2 "\t" $0}' \
        | sort -n \
        | tail -1 \
        | cut -f2-
    )"
    if [[ "${RESUME_FROM_CHECKPOINT}" == "auto" ]]; then
      [[ -n "${LATEST_COMPLETE_CKPT}" ]] || {
        echo "cannot auto-resume ${ORDER}: no complete branch-local checkpoint" >&2
        exit 2
      }
      EFFECTIVE_RESUME="${LATEST_COMPLETE_CKPT}"
    elif [[ -n "${LATEST_COMPLETE_CKPT}" ]]; then
      EXPLICIT_STEP="${RESUME_FROM_CHECKPOINT##*checkpoint-}"
      LATEST_STEP="${LATEST_COMPLETE_CKPT##*checkpoint-}"
      if [[ "${EXPLICIT_STEP}" =~ ^[0-9]+$ && "${LATEST_STEP}" =~ ^[0-9]+$ ]] \
          && (( LATEST_STEP > EXPLICIT_STEP )); then
        EFFECTIVE_RESUME="${LATEST_COMPLETE_CKPT}"
      fi
    fi
    RESUME_ARGS=(--resume_from_checkpoint "${EFFECTIVE_RESUME}")
    OVERWRITE_ARGS=(--overwrite_output_dir)
    if [[ -s "${OUT_DIR}/branch_manifest.json" ]]; then
      SAVED_WORLD_SIZE="$(${PYTHON} - "${OUT_DIR}/branch_manifest.json" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["distributed_world_size"])
PY
)"
      if [[ "${SAVED_WORLD_SIZE}" != "${NPROC}" ]]; then
        echo "cannot resume ${ORDER}: checkpoint world size ${SAVED_WORLD_SIZE} != requested ${NPROC}" >&2
        exit 2
      fi
    fi
    echo "resuming ${ORDER} from latest complete checkpoint: ${EFFECTIVE_RESUME}"
  fi
  export DPRM_TRAIN_ORDER_POLICY="${ORDER}"
  export DPRM_OMNI_DPRM_SCORER="${MATCHED_SCORER}"
  export DPRM_OMNI_MATCHED_REVEAL_BUDGET="${MATCHED_REVEAL_BUDGET}"
  EFFECTIVE_DATA_CONFIG="${DATA_CONFIG}"
  if [[ "${ORDER}" == "random" || "${ORDER}" == "random_matched" ]]; then
    if [[ "${ONLINE_ROLLIN}" == "1" ]]; then
      unset DPRM_OMNI_PRECOMPUTED_TRAJECTORY || true
    elif [[ "${HYBRID_ROLLIN}" == "1" ]]; then
      export DPRM_OMNI_PRECOMPUTED_TRAJECTORY=hybrid
    else
      export DPRM_OMNI_PRECOMPUTED_TRAJECTORY=1
    fi
  elif [[ "${ORDER}" == "confidence_matched" ]]; then
    EFFECTIVE_DATA_CONFIG="${CONFIDENCE_DATA_CONFIG}"
    if [[ "${ONLINE_ROLLIN}" == "1" ]]; then
      unset DPRM_OMNI_PRECOMPUTED_TRAJECTORY || true
    elif [[ "${HYBRID_ROLLIN}" == "1" ]]; then
      export DPRM_OMNI_PRECOMPUTED_TRAJECTORY=hybrid
    else
      export DPRM_OMNI_PRECOMPUTED_TRAJECTORY=1
    fi
  elif [[ "${ORDER}" == "dprm_matched" ]]; then
    EFFECTIVE_DATA_CONFIG="${DPRM_DATA_CONFIG}"
    if [[ "${ONLINE_ROLLIN}" == "1" ]]; then
      unset DPRM_OMNI_PRECOMPUTED_TRAJECTORY || true
    elif [[ "${HYBRID_ROLLIN}" == "1" ]]; then
      export DPRM_OMNI_PRECOMPUTED_TRAJECTORY=hybrid
    else
      export DPRM_OMNI_PRECOMPUTED_TRAJECTORY=1
    fi
  else
    unset DPRM_OMNI_PRECOMPUTED_TRAJECTORY || true
  fi
  [[ -s "${EFFECTIVE_DATA_CONFIG}" ]] || {
    echo "missing data config for ${ORDER}: ${EFFECTIVE_DATA_CONFIG}" >&2
    exit 2
  }
  EFFECTIVE_DATA_JSON="$("${PYTHON}" - "${EFFECTIVE_DATA_CONFIG}" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
datasets = payload.get("dataset", {})
if len(datasets) != 1:
    raise SystemExit(f"expected one matched Omni dataset in {config_path}")
entry = next(iter(datasets.values()))
paths = entry.get("json_paths", [])
if len(paths) != 1:
    raise SystemExit(f"expected one matched Omni JSONL in {config_path}")
path = Path(paths[0])
if not path.is_absolute():
    path = (config_path.parent / path).resolve()
if not path.is_file():
    raise SystemExit(f"matched Omni JSONL does not exist: {path}")
print(path)
PY
)"
  TRAJECTORY_STAGE_CONTRACT="$("${PYTHON}" - "${EFFECTIVE_DATA_JSON}" "${MATCHED_SCORER}" <<'PY'
import json
import sys
from pathlib import Path

data_path, controller_path = map(Path, sys.argv[1:])
post_actions = set()
next_actions = set()
rows = 0
with data_path.open(encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        post = int(row["dprm_trajectory_step"])
        next_action = int(row["dprm_next_action_step"])
        revealed = len(row["dprm_revealed_visual_indices"])
        if next_action != post + 1 or revealed != next_action:
            raise SystemExit(
                f"invalid post-action training canvas in {data_path}: "
                f"post={post}, next={next_action}, revealed={revealed}"
            )
        post_actions.add(post)
        next_actions.add(next_action)
        rows += 1
if rows == 0:
    raise SystemExit(f"empty matched trajectory data: {data_path}")
controller = json.loads(controller_path.read_text(encoding="utf-8"))
reward_actions = controller.get("config", {}).get("reward_action_steps", [])
if not reward_actions:
    reward_actions = controller.get("config", {}).get("active_steps", [])
if not reward_actions:
    reward_actions = controller.get("metadata", {}).get("stagewise_order_contract", {}).get(
        "reward_action_steps", []
    )
missing = sorted(set(map(int, reward_actions)) - next_actions)
if missing:
    raise SystemExit(f"matched trajectories do not train reward actions {missing}")
print(json.dumps({
    "post_action_checkpoints": sorted(post_actions),
    "policy_input_visible_counts": sorted(next_actions),
    "training_next_action_steps": sorted(next_actions),
    "loss_state_visible_counts": sorted(value + 1 for value in next_actions),
    "hybrid_transition_then_loss": True,
    "controller_reward_action_steps": sorted(map(int, reward_actions)),
    "reward_action_coverage_verified": True,
}))
PY
)"
  cat > "${OUT_DIR}/branch_manifest.json" <<JSON
{
  "order": "${ORDER}",
  "shared_initial_checkpoint": "${MODEL_PATH}",
  "shared_checkpoint_index_sha256": "$(sha256sum "${MODEL_PATH}/model.safetensors.index.json" | awk '{print $1}')",
  "data_config": "${EFFECTIVE_DATA_CONFIG}",
  "data_config_sha256": "$(sha256sum "${EFFECTIVE_DATA_CONFIG}" | awk '{print $1}')",
  "data_json": "${EFFECTIVE_DATA_JSON}",
  "data_json_sha256": "$(sha256sum "${EFFECTIVE_DATA_JSON}" | awk '{print $1}')",
  "trajectory_stage_contract": ${TRAJECTORY_STAGE_CONTRACT},
  "controller": "${MATCHED_SCORER}",
  "controller_sha256": "$([[ -s "${MATCHED_SCORER}" ]] && sha256sum "${MATCHED_SCORER}" | awk '{print $1}' || printf '')",
  "trainer_sha256": "$(sha256sum "${OMNI_ROOT}/tools/trainer_v4_51_3.py" | awk '{print $1}')",
  "dataset_sha256": "$(sha256sum "${OMNI_ROOT}/omni_diffusion/data/dataset_qwen2.py" | awk '{print $1}')",
  "order_code_sha256": "$(sha256sum "${RELEASE_ROOT}/src/dprm/omni_order.py" | awk '{print $1}')",
  "inference_hook": "${SCRIPT_DIR}/omni_t2i_smoke.py",
  "inference_hook_sha256": "$(sha256sum "${SCRIPT_DIR}/omni_t2i_smoke.py" | awk '{print $1}')",
  "precomputed_trajectory_mode": "${DPRM_OMNI_PRECOMPUTED_TRAJECTORY:-0}",
  "current_model_policy_refresh": $([[ "${DPRM_OMNI_PRECOMPUTED_TRAJECTORY:-0}" == "hybrid" ]] && printf true || printf false),
  "seed": 956,
  "distributed_world_size": ${NPROC},
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 16,
  "max_steps": ${MAX_STEPS},
  "learning_rate": "${LEARNING_RATE}",
  "warmup_ratio": "${WARMUP_RATIO}",
  "trainable_last_n_layers": ${TRAINABLE_LAST_N_LAYERS},
  "reveal_budget": ${MATCHED_REVEAL_BUDGET}
}
JSON
  export DPRM_OMNI_THEORY_LOG_DIR="${OUT_DIR}/theory"
  echo "starting ${RUN_NAME} with DPRM_TRAIN_ORDER_POLICY=${DPRM_TRAIN_ORDER_POLICY}"
  echo "theory metrics: ${DPRM_OMNI_THEORY_LOG_DIR}/theory_metrics.jsonl"
  "${PYTHON}" -m torch.distributed.run \
    --nproc_per_node "${NPROC}" \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    --master_port "${DPRM_OMNI_MASTER_PORT:-45789}" \
    tools/finetune_dream_v4_51_3.py \
      --log_level info \
      --do_train \
      --config_name "${OMNI_ROOT}/omni_diffusion/models/dream/config_dream_resume.json" \
      --tokenizer_name "${MODEL_PATH}" \
      --model_name_or_path "${MODEL_PATH}" \
      --image_tokenizer_path "${IMAGE_TOKENIZER_PATH}" \
      --dataset_name "${EFFECTIVE_DATA_CONFIG}" \
      --bf16 True \
      --tf32 True \
      --torch_dtype bfloat16 \
      --output_dir "${OUT_DIR}" \
      "${OVERWRITE_ARGS[@]}" \
      "${RESUME_ARGS[@]}" \
      --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
      --max_steps "${MAX_STEPS}" \
      --per_device_train_batch_size 1 \
      --per_device_eval_batch_size 1 \
      --gradient_accumulation_steps 16 \
      --save_strategy steps \
      --save_steps "${SAVE_STEPS}" \
      --save_total_limit "${SAVE_TOTAL_LIMIT}" \
      --learning_rate "${LEARNING_RATE}" \
      --language_model_trainable_last_n_layers "${TRAINABLE_LAST_N_LAYERS}" \
      --max_grad_norm 1.0 \
      --weight_decay 0.0 \
      --adam_beta1 0.9 \
      --adam_beta2 0.95 \
      --adam_epsilon 1e-8 \
      --warmup_ratio "${WARMUP_RATIO}" \
      --lr_scheduler_type cosine \
      --logging_steps 1 \
      --report_to wandb \
      --model_max_length 3072 \
      --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
      --deepspeed "${DEEPSPEED_CONFIG}" \
      --trust_remote_code True \
      --ddp_timeout 7200 \
      --ddp_backend nccl \
      --attn_implementation sdpa \
      --seed 956 \
      --data_seed 956 \
      --reset_attention_mask \
      --reset_position_ids \
      --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
      --image_size 256 2>&1 | tee -a "${OUT_DIR}/train.log"
done
