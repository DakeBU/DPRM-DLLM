#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OMNI_ROOT="${OMNI_ROOT:?set OMNI_ROOT to the patched Omni-Diffusion checkout}"
ENV="${VIRTUAL_ENV:?activate the experiment environment before launching}"
PYTHON="${PYTHON:-${ENV}/bin/python}"
RUN_ROOT="${DPRM_OMNI_MATCHED_ROOT:?set DPRM_OMNI_MATCHED_ROOT}"
GATE="${DPRM_OMNI_GATE_OUT:-${RUN_ROOT}/gate16_offset2200_v2}"
CONTROLLER="${DPRM_OMNI_GATE_CONTROLLER:-${RUN_ROOT}/rank_bucket_step96_rank6.json}"
MODEL="${DPRM_OMNI_MODEL_PATH:?set DPRM_OMNI_MODEL_PATH}"
DATA="${DPRM_OMNI_DATA_JSON:?set DPRM_OMNI_DATA_JSON}"
TRAJ_ROOT="${RUN_ROOT}/matched_trajectories_v2"
TRAIN_OFFSET="${DPRM_OMNI_TRAJECTORY_OFFSET:-0}"
TRAIN_COUNT="${DPRM_OMNI_TRAJECTORY_COUNT:-256}"
SOURCE_UNIQUE_OFFSET="${DPRM_OMNI_TRAIN_SOURCE_UNIQUE_OFFSET:-2400}"
TRAJECTORY_GPUS="${DPRM_OMNI_TRAJECTORY_GPUS:-0,1,2,3,4,5,6,7}"
SHARD_SIZE="${DPRM_OMNI_TRAJECTORY_SHARD_SIZE:-16}"
MIN_FREE_MIB="${DPRM_OMNI_MIN_FREE_MIB:-28000}"
MAX_LAUNCH_UTIL="${DPRM_OMNI_MAX_LAUNCH_UTIL:-85}"
MAX_WORKERS_PER_GPU="${DPRM_OMNI_MAX_WORKERS_PER_GPU:-3}"
GPU_WORKER_CAPS="${DPRM_OMNI_GPU_WORKER_CAPS:-0:3,1:1,2:1,3:1,4:1,5:1,6:1,7:1}"
POLL_SECONDS="${DPRM_OMNI_POLL_SECONDS:-60}"
TRAIN_GPUS="${DPRM_OMNI_TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
TRAIN_MIN_FREE_MIB="${DPRM_OMNI_TRAIN_MIN_FREE_MIB:-50000}"
TRAIN_MIN_GPUS="${DPRM_OMNI_TRAIN_MIN_GPUS:-1}"
TRAIN_OUT="${DPRM_OMNI_TRAIN_OUT:-${RUN_ROOT}/matched_training_v2}"
TRAIN_ORDERS="${DPRM_OMNI_TRAIN_ORDERS:-confidence_matched dprm_matched}"

mkdir -p "${RUN_ROOT}"
exec 8>"${RUN_ROOT}/pipeline_scheduler.lock"
if ! flock -n 8; then
  echo "another matched Omni scheduler already owns ${RUN_ROOT}" >&2
  exit 0
fi
printf '%s\n' "$$" > "${RUN_ROOT}/pipeline_scheduler.pid"
cat > "${RUN_ROOT}/pipeline_manifest.json" <<JSON
{
  "model": "${MODEL}",
  "controller": "${CONTROLLER}",
  "trajectory_count": ${TRAIN_COUNT},
  "trajectory_checkpoints": [32, 64, 96, 128, 160, 192, 224],
  "train_orders": "${TRAIN_ORDERS}",
  "hybrid_rollin": ${DPRM_OMNI_HYBRID_ROLLIN:-0},
  "max_steps": ${DPRM_OMNI_MAX_STEPS:-500},
  "terminal_reward_calls_at_test": 0,
  "complete_image_selection": false
}
JSON

mkdir -p "${TRAJ_ROOT}/shards" "${TRAJ_ROOT}/logs"
source "${ENV}/bin/activate"
export PYTHONPATH="${OMNI_ROOT}:${RELEASE_ROOT}/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "${DPRM_OMNI_SKIP_PRETRAIN_GATE:-0}" == "1" ]]; then
  "${PYTHON}" - "${CONTROLLER}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("format") != "omni_bucket_table_dprm_v1":
    raise SystemExit("pretrain-gate bypass is limited to a frozen full-trajectory bucket controller")
PY
  date -Is > "${RUN_ROOT}/PRETRAIN_GATE_BYPASSED_FOR_MATCHED_TRAINING_TEST"
else
  while [[ ! -s "${GATE}/GATE_COMPLETE" ]]; do sleep "${POLL_SECONDS}"; done
  if ! "${PYTHON}" "${SCRIPT_DIR}/check_omni_gate.py" \
    --paired-summary "${GATE}/summary/paired/paired_clip_summary.json" \
    --output "${GATE}/gate_decision.json"; then
    date -Is > "${RUN_ROOT}/GATE_FAILED"
    exit 2
  fi
  date -Is > "${RUN_ROOT}/GATE_PASSED"
fi

# Development and gate prompts are excluded from the trajectory source. The
# ranges cover all Omni prompt sets used to select this controller.
jq -r '.messages[0].content | split("\n") | .[1:] | join("\n")' "${DATA}" \
  | awk 'NF && !seen[$0]++' | sed -n '1801,2396p' \
  > "${TRAJ_ROOT}/forbidden_prompts.txt"
"${PYTHON}" "${SCRIPT_DIR}/filter_omni_training_source.py" \
  --source "${DATA}" --forbidden-prompts "${TRAJ_ROOT}/forbidden_prompts.txt" \
  --output "${TRAJ_ROOT}/training_source.jsonl" --count "${TRAIN_COUNT}" \
  --unique-offset "${SOURCE_UNIQUE_OFFSET}" \
  > "${TRAJ_ROOT}/training_source_manifest.json"

run_shard() {
  local shard="$1" offset="$2" count="$3" gpu="$4"
  local stem="shard_$(printf '%04d' "${shard}")"
  local done="${TRAJ_ROOT}/shards/${stem}.complete"
  [[ -s "${done}" ]] && return 0
  rm -f "${TRAJ_ROOT}/shards/.${stem}"_*.tmp
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${SCRIPT_DIR}/build_omni_matched_trajectory_data.py" \
    --model-path "${MODEL}" --data-json "${TRAJ_ROOT}/training_source.jsonl" --controller "${CONTROLLER}" \
    --random-output "${TRAJ_ROOT}/shards/${stem}_random.jsonl" \
    --confidence-output "${TRAJ_ROOT}/shards/${stem}_confidence.jsonl" \
    --dprm-output "${TRAJ_ROOT}/shards/${stem}_dprm.jsonl" \
    --manifest-output "${TRAJ_ROOT}/shards/${stem}_manifest.json" \
    --offset "${offset}" --count "${count}" --checkpoints 32 64 96 128 160 192 224 \
    >> "${TRAJ_ROOT}/logs/${stem}.log" 2>&1
  date -Is > "${done}"
}

shard_builder_alive() {
  local manifest_path="$1" worker_pid argv
  while read -r worker_pid; do
    [[ -r "/proc/${worker_pid}/cmdline" ]] || continue
    argv="$(tr '\0' ' ' < "/proc/${worker_pid}/cmdline")"
    [[ "${argv}" == *"${manifest_path}"* ]] && return 0
  done < <(pgrep -f 'build_omni_matched_trajectory_data.py' || true)
  return 1
}

shards=$(((TRAIN_COUNT + SHARD_SIZE - 1) / SHARD_SIZE))
declare -A PID_GPU= PID_SHARD=
declare -a PIDS=()
while :; do
  # Recover a shard whose process exited before publishing its atomic output.
  # A live builder is identified by the shard-specific manifest path in argv.
  for ((shard=0; shard<shards; shard++)); do
    stem="shard_$(printf '%04d' "${shard}")"
    running="${TRAJ_ROOT}/shards/${stem}.running"
    [[ -s "${running}" && ! -s "${TRAJ_ROOT}/shards/${stem}.complete" ]] || continue
    if ! shard_builder_alive "${TRAJ_ROOT}/shards/${stem}_manifest.json"; then
      rm -f "${running}" "${TRAJ_ROOT}/shards/.${stem}"_*.tmp
    fi
  done

  complete=0
  for ((shard=0; shard<shards; shard++)); do
    [[ -s "${TRAJ_ROOT}/shards/shard_$(printf '%04d' "${shard}").complete" ]] && ((complete+=1))
  done
  (( complete == shards )) && break

  mapfile -t free_rows < <(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits)
  IFS=',' read -r -a allowed_trajectory_gpu_ids <<< "${TRAJECTORY_GPUS}"
  for row in "${free_rows[@]}"; do
    IFS=',' read -r gpu free util <<< "${row}"
    gpu="${gpu//[[:space:]]/}"
    free="${free//[[:space:]]/}"
    util="${util//[[:space:]]/}"
    [[ "${gpu}" =~ ^[0-9]+$ && "${free}" =~ ^[0-9]+$ && "${util}" =~ ^[0-9]+$ ]] || continue
    allowed=0
    for candidate_gpu in "${allowed_trajectory_gpu_ids[@]}"; do
      [[ "${gpu}" == "${candidate_gpu}" ]] && allowed=1
    done
    (( allowed == 1 )) || continue
    [[ "${free}" -ge "${MIN_FREE_MIB}" ]] || continue
    [[ "${util}" -le "${MAX_LAUNCH_UTIL}" ]] || continue
    workers_on_gpu=0
    while read -r worker_pid; do
      [[ -r "/proc/${worker_pid}/environ" ]] || continue
      worker_gpu="$(tr '\0' '\n' < "/proc/${worker_pid}/environ" \
        | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -n 1)"
      [[ "${worker_gpu}" == "${gpu}" ]] && ((workers_on_gpu+=1))
    done < <(pgrep -f 'python .*build_omni_matched_trajectory_data.py' || true)
    worker_cap="${MAX_WORKERS_PER_GPU}"
    IFS=',' read -r -a configured_worker_caps <<< "${GPU_WORKER_CAPS}"
    for configured_cap in "${configured_worker_caps[@]}"; do
      configured_gpu="${configured_cap%%:*}"
      configured_limit="${configured_cap#*:}"
      if [[ "${configured_gpu}" == "${gpu}" && "${configured_limit}" =~ ^[0-9]+$ ]]; then
        worker_cap="${configured_limit}"
        break
      fi
    done
    (( workers_on_gpu < worker_cap )) || continue
    next=""
    for ((shard=0; shard<shards; shard++)); do
      stem="shard_$(printf '%04d' "${shard}")"
      [[ -s "${TRAJ_ROOT}/shards/${stem}.complete" ]] && continue
      [[ -s "${TRAJ_ROOT}/shards/${stem}.running" ]] && continue
      next="${shard}"; break
    done
    [[ -n "${next}" ]] || continue
    offset=$((TRAIN_OFFSET + next * SHARD_SIZE))
    count="${SHARD_SIZE}"
    remaining=$((TRAIN_COUNT - next * SHARD_SIZE))
    (( remaining < count )) && count="${remaining}"
    stem="shard_$(printf '%04d' "${next}")"
    date -Is > "${TRAJ_ROOT}/shards/${stem}.running"
    (trap 'rm -f "'"${TRAJ_ROOT}/shards/${stem}.running"'"' EXIT; run_shard "${next}" "${offset}" "${count}" "${gpu}") &
    pid=$!; PIDS+=("${pid}"); PID_GPU[${pid}]="${gpu}"; PID_SHARD[${pid}]="${next}"
  done

  keep=()
  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      keep+=("${pid}")
    else
      wait "${pid}" || true
      unset 'PID_GPU['"${pid}"']' 'PID_SHARD['"${pid}"']'
    fi
  done
  PIDS=("${keep[@]}")
  sleep "${POLL_SECONDS}"
done

mapfile -t confidence_shards < <(find "${TRAJ_ROOT}/shards" -name 'shard_*_confidence.jsonl' | sort)
mapfile -t dprm_shards < <(find "${TRAJ_ROOT}/shards" -name 'shard_*_dprm.jsonl' | sort)
mapfile -t random_shards < <(find "${TRAJ_ROOT}/shards" -name 'shard_*_random.jsonl' | sort)
"${PYTHON}" "${SCRIPT_DIR}/merge_omni_matched_trajectories.py" \
  --random-shards "${random_shards[@]}" --confidence-shards "${confidence_shards[@]}" \
  --dprm-shards "${dprm_shards[@]}" \
  --output-dir "${TRAJ_ROOT}/merged" \
  --forbidden-prompts "${TRAJ_ROOT}/forbidden_prompts.txt"
date -Is > "${TRAJ_ROOT}/TRAJECTORIES_COMPLETE"

IFS=',' read -r -a allowed_train_gpu_ids <<< "${TRAIN_GPUS}"
while :; do
  train_gpu_ids=()
  declare -A free_by_gpu=()
  while IFS=',' read -r gpu free; do
    gpu="${gpu//[[:space:]]/}"
    free="${free//[[:space:]]/}"
    [[ "${gpu}" =~ ^[0-9]+$ && "${free}" =~ ^[0-9]+$ ]] || continue
    free_by_gpu[${gpu}]="${free}"
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
  for gpu in "${allowed_train_gpu_ids[@]}"; do
    if (( ${free_by_gpu[${gpu}]:-0} >= TRAIN_MIN_FREE_MIB )); then
      train_gpu_ids+=("${gpu}")
    fi
  done
  (( ${#train_gpu_ids[@]} >= TRAIN_MIN_GPUS )) && break
  sleep "${POLL_SECONDS}"
done
SELECTED_TRAIN_GPUS="$(IFS=,; echo "${train_gpu_ids[*]}")"

cat > "${RUN_ROOT}/training_execution_manifest.json" <<JSON
{
  "shared_initial_checkpoint": "${MODEL}",
  "shared_checkpoint_index_sha256": "$(sha256sum "${MODEL}/model.safetensors.index.json" | awk '{print $1}')",
  "controller": "${CONTROLLER}",
  "controller_sha256": "$(sha256sum "${CONTROLLER}" | awk '{print $1}')",
  "merged_data_manifest_sha256": "$(sha256sum "${TRAJ_ROOT}/merged/manifest.json" | awk '{print $1}')",
  "trainer_sha256": "$(sha256sum "${OMNI_ROOT}/tools/trainer_v4_51_3.py" | awk '{print $1}')",
  "dataset_sha256": "$(sha256sum "${OMNI_ROOT}/omni_diffusion/data/dataset_qwen2.py" | awk '{print $1}')",
  "order_code_sha256": "$(sha256sum "${RELEASE_ROOT}/src/dprm/omni_order.py" | awk '{print $1}')",
  "inference_hook_sha256": "$(sha256sum "${SCRIPT_DIR}/omni_t2i_smoke.py" | awk '{print $1}')",
  "branches": "${TRAIN_ORDERS}",
  "hybrid_current_model_policy_refresh": true,
  "max_steps": ${DPRM_OMNI_MAX_STEPS:-500},
  "selected_gpus": "${SELECTED_TRAIN_GPUS}"
}
JSON

DPRM_OMNI_MODEL_PATH="${MODEL}" \
DPRM_OMNI_OUT_BASE="${TRAIN_OUT}" \
DPRM_OMNI_ORDERS="${TRAIN_ORDERS}" \
DPRM_OMNI_DATA_CONFIG="${TRAJ_ROOT}/merged/random_matched.yaml" \
DPRM_OMNI_CONFIDENCE_DATA_CONFIG="${TRAJ_ROOT}/merged/confidence_matched.yaml" \
DPRM_OMNI_DPRM_DATA_CONFIG="${TRAJ_ROOT}/merged/dprm_matched.yaml" \
DPRM_OMNI_DPRM_SCORER="${CONTROLLER}" \
DPRM_OMNI_HYBRID_ROLLIN="${DPRM_OMNI_HYBRID_ROLLIN:-0}" \
DPRM_OMNI_GPUS="${SELECTED_TRAIN_GPUS}" \
DPRM_OMNI_NPROC="${#train_gpu_ids[@]}" \
DPRM_OMNI_MAX_STEPS="${DPRM_OMNI_MAX_STEPS:-500}" \
DPRM_OMNI_SAVE_STEPS="${DPRM_OMNI_SAVE_STEPS:-500}" \
DPRM_OMNI_TRAINABLE_LAST_N_LAYERS="${DPRM_OMNI_TRAINABLE_LAST_N_LAYERS:-2}" \
bash "${SCRIPT_DIR}/run_omni_formal_four_order_train.sh" \
  >> "${TRAIN_OUT}.log" 2>&1
date -Is > "${TRAIN_OUT}/TRAINING_COMPLETE"

EVAL_OUT="${DPRM_OMNI_MATCHED_EVAL_OUT:-${RUN_ROOT}/matched_eval_offset2300_v2}"
while :; do
  eval_gpu=""
  while IFS=',' read -r gpu free; do
    gpu="${gpu//[[:space:]]/}"
    free="${free//[[:space:]]/}"
    if [[ "${gpu}" =~ ^[0-9]+$ && "${free}" =~ ^[0-9]+$ ]] \
      && [[ "${free}" -ge "${MIN_FREE_MIB}" ]]; then
      eval_gpu="${gpu}"
      break
    fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
  [[ -n "${eval_gpu}" ]] && break
  sleep "${POLL_SECONDS}"
done
DPRM_OMNI_TRAIN_OUT="${TRAIN_OUT}" \
DPRM_OMNI_GATE_CONTROLLER="${CONTROLLER}" \
DPRM_OMNI_EVAL_OUT="${EVAL_OUT}" \
DPRM_OMNI_EVAL_GPU="${eval_gpu}" \
DPRM_OMNI_EVAL_STEP="${DPRM_OMNI_EVAL_STEP:-${DPRM_OMNI_MAX_STEPS:-500}}" \
DPRM_OMNI_INCLUDE_TRAINED_RANDOM="${DPRM_OMNI_INCLUDE_TRAINED_RANDOM:-0}" \
bash "${SCRIPT_DIR}/run_omni_matched_checkpoint_eval.sh" \
  >> "${EVAL_OUT}.log" 2>&1
date -Is > "${RUN_ROOT}/PIPELINE_COMPLETE"
rm -f "${RUN_ROOT}/pipeline_scheduler.pid"
