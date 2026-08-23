#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OMNI_ROOT="${OMNI_ROOT:?set OMNI_ROOT}"
ENV="${VIRTUAL_ENV:?activate the experiment environment before launching}"
PYTHON="${PYTHON:-${ENV}/bin/python}"
DATA="${DPRM_OMNI_DATA_JSON:?set DPRM_OMNI_DATA_JSON}"
PROMPT_FILE="${DPRM_OMNI_PROMPT_FILE:-}"
MODEL="${DPRM_OMNI_SHARED_CHECKPOINT:?set DPRM_OMNI_SHARED_CHECKPOINT}"
IMAGE_TOKENIZER="${DPRM_OMNI_IMAGE_TOKENIZER:?set DPRM_OMNI_IMAGE_TOKENIZER}"
CONTROLLER="${DPRM_OMNI_CONTROLLER:?set DPRM_OMNI_CONTROLLER}"
OUT="${DPRM_OMNI_EVAL_OUT:?set DPRM_OMNI_EVAL_OUT}"
GENEVAL_METADATA="${DPRM_OMNI_GENEVAL_METADATA:-${OMNI_ROOT}/datasets/eval/GenEval/evaluation_metadata.jsonl}"
OFFSET="${DPRM_OMNI_EVAL_OFFSET:-3200}"
COUNT="${DPRM_OMNI_EVAL_COUNT:-512}"
GPUS_TEXT="${DPRM_OMNI_EVAL_GPUS:-1 2 3 6}"
read -r -a GPUS <<< "${GPUS_TEXT//,/ }"
FIXED_VISUAL_IDS_TEXT="${DPRM_OMNI_FIXED_VISUAL_PROMPT_IDS:-prompt_0187 prompt_0208 prompt_0315 prompt_0326}"
read -r -a FIXED_VISUAL_PROMPT_IDS <<< "${FIXED_VISUAL_IDS_TEXT//,/ }"

for path in "${DATA}" "${MODEL}" "${IMAGE_TOKENIZER}" "${CONTROLLER}" "${GENEVAL_METADATA}" ${PROMPT_FILE:+"${PROMPT_FILE}"}; do
  [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 2; }
done
source "${ENV}/bin/activate"
export PYTHONPATH="${OMNI_ROOT}:${RELEASE_ROOT}/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${OUT}"/{logs,summary}

"${PYTHON}" - "${CONTROLLER}" "${OFFSET}" "${COUNT}" "${PROMPT_FILE}" <<'PY'
import hashlib
import json
import sys

controller_path, offset, count, prompt_file = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
)
payload = json.load(open(controller_path, encoding="utf-8"))
metadata = payload.get("metadata", {})
deployment = metadata.get("deployment_contract", {})
match = metadata.get("train_test_order_match", {})
if deployment.get("paths_per_prompt") != 1:
    raise SystemExit("formal Omni evaluation requires one path per prompt")
if deployment.get("terminal_reward_calls_at_test") != 0:
    raise SystemExit("formal Omni evaluation forbids terminal-reward calls")
if deployment.get("complete_image_selection") is not False:
    raise SystemExit("formal Omni evaluation forbids completed-image selection")
if match.get("host_parameter_updates") != 0:
    raise SystemExit("controller-only evaluation requires a frozen host")
formal = set(range(offset, offset + count))
selection = metadata.get("development_selection", {})
if prompt_file:
    observed = hashlib.sha256(open(prompt_file, "rb").read()).hexdigest()
    if observed == selection.get("selection_prompt_file_sha256"):
        raise SystemExit("formal prompt file equals the development prompt file")
    development_hashes = set(selection.get("selection_prompt_text_sha256") or [])
    development_hashes.update(metadata.get("source_prompt_text_sha256") or [])
    formal_hashes = set()
    for line in open(prompt_file, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            prompt = line
        else:
            prompt = row.get("prompt", row.get("text"))
            if prompt is None and row.get("messages"):
                content = str(row["messages"][0].get("content", ""))
                lines = content.splitlines()
                prompt = "\n".join(lines[1:]).strip() if len(lines) > 1 else content
        prompt = str(prompt or "").strip()
        if prompt:
            formal_hashes.add(hashlib.sha256(prompt.encode("utf-8")).hexdigest())
    overlap = development_hashes & formal_hashes
    if overlap:
        raise SystemExit(
            f"formal prompt file overlaps development by {len(overlap)} prompt texts"
        )
for name in ("table_prompt_range", "selection_prompt_range"):
    bounds = selection.get(name)
    if bounds:
        used = set(range(int(bounds[0]), int(bounds[1]) + 1))
        if formal & used:
            raise SystemExit(f"formal prompt range overlaps {name}")
PY

if [[ -n "${PROMPT_FILE}" ]]; then
  mapfile -t PROMPTS < <(
    "${PYTHON}" - "${PROMPT_FILE}" "${COUNT}" <<'PY'
import json
import sys
from pathlib import Path

seen = set()
for raw in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    raw = raw.strip()
    if not raw:
        continue
    try:
        row = json.loads(raw)
    except json.JSONDecodeError:
        prompt = raw
    else:
        prompt = row.get("prompt", row.get("text"))
        if prompt is None and row.get("messages"):
            content = str(row["messages"][0].get("content", ""))
            lines = content.splitlines()
            prompt = "\n".join(lines[1:]).strip() if len(lines) > 1 else content
    prompt = str(prompt or "").strip()
    if not prompt or prompt in seen:
        continue
    seen.add(prompt)
    print(prompt)
    if len(seen) == int(sys.argv[2]):
        break
PY
  )
else
  mapfile -t PROMPTS < <(
    jq -r '.messages[0].content | split("\n") | .[1:] | join("\n")' "${DATA}" \
      | awk 'NF && !seen[$0]++' | sed -n "$((OFFSET + 1)),$((OFFSET + COUNT))p"
  )
fi
[[ ${#PROMPTS[@]} -eq ${COUNT} ]] || {
  echo "expected ${COUNT} unique prompts, found ${#PROMPTS[@]}" >&2
  exit 2
}
fixed_visual_jsonl="${OUT}/fixed_visual_prompts.jsonl"
: > "${fixed_visual_jsonl}"
for prompt_id in "${FIXED_VISUAL_PROMPT_IDS[@]}"; do
  [[ "${prompt_id}" =~ ^prompt_([0-9]+)$ ]] || {
    echo "invalid fixed visual prompt id: ${prompt_id}" >&2
    exit 2
  }
  local_idx=$((10#${BASH_REMATCH[1]} - OFFSET))
  (( local_idx >= 0 && local_idx < COUNT )) || {
    echo "fixed visual prompt is outside the formal split: ${prompt_id}" >&2
    exit 2
  }
  jq -cn --arg prompt_id "${prompt_id}" --arg prompt "${PROMPTS[$local_idx]}" \
    '{prompt_id:$prompt_id,prompt:$prompt}' >> "${fixed_visual_jsonl}"
done
jq -s . "${fixed_visual_jsonl}" > "${OUT}/fixed_visual_prompts.json"
rm -f "${fixed_visual_jsonl}"
FIXED_VISUAL_IDS_JSON="$(printf '%s\n' "${FIXED_VISUAL_PROMPT_IDS[@]}" | jq -R . | jq -s .)"
PROMPT_FILE_JSON=null
PROMPT_FILE_SHA_JSON=null
if [[ -n "${PROMPT_FILE}" ]]; then
  PROMPT_FILE_JSON="$(printf '%s' "${PROMPT_FILE}" | jq -Rsa .)"
  PROMPT_FILE_SHA_JSON="\"$(sha256sum "${PROMPT_FILE}" | awk '{print $1}')\""
fi

jobs="${OUT}/jobs.jsonl"
: > "${jobs}"
for label in random progressive_confidence dprm_confidence_warmup; do
  for local_idx in "${!PROMPTS[@]}"; do
    idx=$((OFFSET + local_idx))
    order="${label}"
    extra='["--fixed-t2i-scaffold","--trace-order-stats","auto","--trace-num-phases","1","--trace-confidence-bins","8","--trace-aux-bins","16","--trace-provisional-phases"]'
    if [[ "${label}" == dprm_confidence_warmup ]]; then
      extra="$(jq -cn --argjson current "${extra}" --arg scorer "${CONTROLLER}" \
        '$current + ["--dprm-order-scorer",$scorer,"--dprm-warmup-steps","0"]')"
    fi
    dir="${OUT}/${label}/prompt_$(printf '%04d' "${idx}")"
    jq -cn --arg output_dir "${dir}" --arg prompt "${PROMPTS[$local_idx]}" \
      --arg order "${order}" --argjson seed "$((20275000 + idx))" \
      --argjson extra "${extra}" \
      '{output_dir:$output_dir,prompt:$prompt,order_policy:$order,seed:$seed,steps:260,max_tokens:260,extra_args:$extra}' \
      >> "${jobs}"
  done
done

cat > "${OUT}/run_manifest.json" <<JSON
{
  "design": "frozen-host single-path Omni confidence-versus-DPRM confirmation",
  "claim_eligible": true,
  "prompt_offset": ${OFFSET},
  "prompt_count": ${COUNT},
  "prompt_file": ${PROMPT_FILE_JSON},
  "prompt_file_sha256": ${PROMPT_FILE_SHA_JSON},
  "paths_per_method_per_prompt": 1,
  "shared_checkpoint": "${MODEL}",
  "controller": "${CONTROLLER}",
  "controller_sha256": "$(sha256sum "${CONTROLLER}" | awk '{print $1}')",
  "test_time_terminal_rollouts": 0,
  "complete_image_selection": false,
  "fixed_visual_prompt_ids": ${FIXED_VISUAL_IDS_JSON},
  "fixed_visual_prompt_manifest": "${OUT}/fixed_visual_prompts.json",
  "outcome_ranked_visual_selection": false
}
JSON

pids=()
for gpu in "${GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${SCRIPT_DIR}/run_omni_t2i_manifest.py" \
    --smoke-script "${SCRIPT_DIR}/omni_t2i_smoke.py" \
    --model-path "${MODEL}" --image-tokenizer-path "${IMAGE_TOKENIZER}" \
    --jobs "${jobs}" > "${OUT}/logs/gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
(( failed == 0 )) || exit 2

CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${PYTHON}" "${SCRIPT_DIR}/summarize_omni_eval.py" \
  --eval-root "${OUT}" \
  --orders random progressive_confidence dprm_confidence_warmup \
  --out-dir "${OUT}/summary" --clip-model openai/clip-vit-large-patch14 \
  --device cuda:0 --no-aesthetic --strict-clip
CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${PYTHON}" "${SCRIPT_DIR}/score_omni_records_clip.py" \
  --records "${OUT}/summary/records.json" \
  --output "${OUT}/summary/records_two_encoder.json" \
  --model "${DPRM_OMNI_CLIP_B32_PATH:-openai/clip-vit-base-patch32}" \
  --metric-name clip_b32_cosine --device cuda:0
"${PYTHON}" "${SCRIPT_DIR}/analyze_omni_paired_results.py" \
  --records "${OUT}/summary/records_two_encoder.json" \
  --output-dir "${OUT}/summary/paired" \
  --comparisons random:progressive_confidence \
    progressive_confidence:dprm_confidence_warmup
"${PYTHON}" "${SCRIPT_DIR}/analyze_omni_geneval_categories.py" \
  --records "${OUT}/summary/records_two_encoder.json" \
  --metadata "${GENEVAL_METADATA}" \
  --output "${OUT}/summary/geneval_category_effects.json" \
  --tsv-output "${OUT}/summary/geneval_category_effects.tsv"
"${PYTHON}" "${SCRIPT_DIR}/package_omni_formal_visual_audit.py" \
  --records "${OUT}/summary/records_two_encoder.json" \
  --summary "${OUT}/summary/summary.json" \
  --out-dir "${OUT}/human_visual_audit" \
  --orders random progressive_confidence dprm_confidence_warmup \
  --num-examples 12 --fixed-prompt-ids "${FIXED_VISUAL_PROMPT_IDS[@]}"

"${PYTHON}" - "${OUT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
paired = json.loads(
    (root / "summary/paired/paired_clip_summary.json").read_text(encoding="utf-8")
)
by_metric = paired["comparisons_by_metric"]
primary = next(
    row for row in by_metric["clip_cosine"]
    if row["baseline"] == "progressive_confidence"
)
secondary = next(
    row for row in by_metric["clip_b32_cosine"]
    if row["baseline"] == "progressive_confidence"
)
overridden = []
for trace in sorted((root / "dprm_confidence_warmup").glob("prompt_*/*_order_trace.jsonl")):
    count = 0
    for line in trace.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        selected = row.get("selected_candidate_indices", [])
        default = row.get("confidence_default_candidate_index")
        if len(selected) == 1 and default is not None:
            count += int(int(selected[0]) != int(default))
    overridden.append(count > 0)
override_fraction = sum(overridden) / len(overridden)
passed = (
    primary["ci95_low"] > 0.0
    and secondary["mean_delta"] > 0.0
    and override_fraction >= 0.05
)
report = {
    "passed": passed,
    "primary": primary,
    "secondary": secondary,
    "prompt_fraction_with_direct_override": override_fraction,
    "requirements": {
        "primary_ci95_low_positive": True,
        "secondary_mean_delta_positive": True,
        "minimum_override_fraction": 0.05,
    },
}
(root / "numeric_promotion.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
if passed:
    (root / "NUMERIC_PROMOTION_PASSED").write_text("passed\n", encoding="utf-8")
print(json.dumps(report, indent=2))
PY
date -Is > "${OUT}/EVALUATION_COMPLETE"
