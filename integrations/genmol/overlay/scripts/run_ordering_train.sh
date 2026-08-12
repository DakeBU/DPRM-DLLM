#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 <gpu_id> <run_name> <order_policy> <output_dir> [max_steps]" >&2
  exit 2
fi

GPU_ID="$1"
RUN_NAME="$2"
ORDER_POLICY="$3"
OUTPUT_DIR="$4"
MAX_STEPS="${5:-5000}"

ROOT="${GENMOL_ROOT:?set GENMOL_ROOT to the upstream GenMol checkout}"
# Resolve before changing directories so repository-relative output paths keep
# the caller's meaning.
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"
cd "$ROOT"
if [[ -n "${ENV_ACTIVATE:-}" ]]; then
  source "$ENV_ACTIVATE"
fi

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export WANDB_PROJECT="${WANDB_PROJECT:-DLLM-DrugDesign}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_DIR="${WANDB_DIR:-${HOME}/.cache/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${HOME}/.cache/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${HOME}/.config/wandb}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-${HOME}/.cache/wandb-data}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-${HOME}/.cache/wandb-artifacts}"
export TMPDIR="${TMPDIR:-/tmp}"

mkdir -p "$OUTPUT_DIR" logs "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR" "$WANDB_ARTIFACT_DIR" "$TMPDIR"

export RUN_NAME ORDER_POLICY OUTPUT_DIR MAX_STEPS
export DPRM_REWARD_MODE="${DPRM_REWARD_MODE:-selected_confidence}"
export DPRM_OBJECTIVE_WEIGHTS="${DPRM_OBJECTIVE_WEIGHTS:-[0.55,0.45]}"
export DPRM_AUX_MODE="${DPRM_AUX_MODE:-molecular_token_class}"
export DPRM_TCHEBYCHEFF_TEMPERATURE="${DPRM_TCHEBYCHEFF_TEMPERATURE:-0.05}"
export DPRM_TCHEBYCHEFF_AUGMENTATION="${DPRM_TCHEBYCHEFF_AUGMENTATION:-0.05}"
export GENMOL_MANIFEST_PATH="$OUTPUT_DIR/run_manifest.json"
export GENMOL_STARTED_AT="$(date -Is)"
export GENMOL_GIT_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)"
export GENMOL_GIT_STATUS_SHORT="$(git -C "$ROOT" status --short 2>/dev/null || true)"
python3 - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "started_at": os.environ.get("GENMOL_STARTED_AT"),
    "project_root": os.environ.get("GENMOL_ROOT"),
    "git_commit": os.environ.get("GENMOL_GIT_COMMIT"),
    "git_status_short": os.environ.get("GENMOL_GIT_STATUS_SHORT"),
    "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "run_name": os.environ.get("RUN_NAME"),
    "order_policy": os.environ.get("ORDER_POLICY"),
    "output_dir": os.environ.get("OUTPUT_DIR"),
    "max_steps": os.environ.get("MAX_STEPS"),
    "dprm_reward_mode": os.environ.get("DPRM_REWARD_MODE"),
    "dprm_objective_weights": os.environ.get("DPRM_OBJECTIVE_WEIGHTS"),
    "dprm_aux_mode": os.environ.get("DPRM_AUX_MODE"),
    "dprm_tchebycheff_temperature": os.environ.get("DPRM_TCHEBYCHEFF_TEMPERATURE"),
    "dprm_tchebycheff_augmentation": os.environ.get("DPRM_TCHEBYCHEFF_AUGMENTATION"),
    "wandb_project": os.environ.get("WANDB_PROJECT"),
    "wandb_api_key_present": bool(os.environ.get("WANDB_API_KEY")),
    "command": [
        "python3", "scripts/train.py",
        "--config-name", "dprm_base",
        f"hydra.run.dir={os.environ.get('OUTPUT_DIR')}",
        f"callback.dirpath={os.environ.get('OUTPUT_DIR')}/checkpoints",
        "trainer.devices=1",
        f"trainer.max_steps={os.environ.get('MAX_STEPS')}",
        "loader.global_batch_size=128",
        "loader.num_workers=4",
        "training.use_bracket_safe=true",
        f"training.order_policy={os.environ.get('ORDER_POLICY')}",
        f"training.dprm_reward_mode={os.environ.get('DPRM_REWARD_MODE')}",
        f"training.dprm_objective_weights={os.environ.get('DPRM_OBJECTIVE_WEIGHTS')}",
        f"training.dprm_aux_mode={os.environ.get('DPRM_AUX_MODE')}",
        f"wandb.project={os.environ.get('WANDB_PROJECT')}",
        f"wandb.name={os.environ.get('RUN_NAME')}",
    ],
}
Path(os.environ["GENMOL_MANIFEST_PATH"]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

python3 scripts/train.py \
  --config-name dprm_base \
  hydra.run.dir="$OUTPUT_DIR" \
  callback.dirpath="$OUTPUT_DIR/checkpoints" \
  trainer.devices=1 \
  trainer.max_steps="$MAX_STEPS" \
  loader.global_batch_size=128 \
  loader.num_workers=4 \
  training.use_bracket_safe=true \
  training.order_policy="$ORDER_POLICY" \
  training.dprm_reward_mode="$DPRM_REWARD_MODE" \
  training.dprm_objective_weights="$DPRM_OBJECTIVE_WEIGHTS" \
  training.dprm_aux_mode="$DPRM_AUX_MODE" \
  training.dprm_tchebycheff_temperature="$DPRM_TCHEBYCHEFF_TEMPERATURE" \
  training.dprm_tchebycheff_augmentation="$DPRM_TCHEBYCHEFF_AUGMENTATION" \
  wandb.project="$WANDB_PROJECT" \
  wandb.name="$RUN_NAME" \
  2>&1 | tee -a "logs/${RUN_NAME}.log"
