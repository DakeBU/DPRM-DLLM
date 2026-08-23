#!/usr/bin/env bash
set -euo pipefail

MATCHED_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${MATCHED_ROOT}/../../.." && pwd)"
SCRIPT_DIR="${MATCHED_ROOT}/scripts"

: "${OMNI_MODEL_PATH:?set the released step-1000 checkpoint directory}"
: "${OMNI_DEVELOPMENT_ROOT:?set the completed 128-prompt development directory}"
: "${OMNI_CONFIRMATION_ROOT:?set the completed 512-prompt confirmation directory}"
: "${OMNI_MECHANISM_BEACH_CONFIDENCE:?set the beach confidence trace directory}"
: "${OMNI_MECHANISM_BEACH_DPRM:?set the beach DPRM trace directory}"
: "${OMNI_MECHANISM_BOY_CONFIDENCE:?set the boy-and-kittens confidence trace directory}"
: "${OMNI_MECHANISM_BOY_DPRM:?set the boy-and-kittens DPRM trace directory}"
: "${OMNI_MECHANISM_RECORDS:?set the fixed mechanism action records JSON}"
: "${OMNI_RELEASE_ROOT:?set the Hugging Face staging directory}"

PYTHON="${PYTHON:-python}"
DEV_PROMPTS="${OMNI_DEVELOPMENT_PROMPTS:-${REPO_ROOT}/reproducibility/omni_partiprompts_development128.jsonl}"
CONF_PROMPTS="${OMNI_CONFIRMATION_PROMPTS:-${REPO_ROOT}/reproducibility/omni_partiprompts_confirmation512.jsonl}"
PUBLIC_RESULT="${OMNI_PUBLIC_RESULT:-${REPO_ROOT}/results/artifacts/omni_online_action_value_release.json}"
TARGET="${OMNI_RELEASE_ROOT}/omni_diffusion"

for path in \
  "${OMNI_MODEL_PATH}/model.safetensors.index.json" \
  "${OMNI_DEVELOPMENT_ROOT}/COMPLETE" \
  "${OMNI_DEVELOPMENT_ROOT}/records/two_encoder.json" \
  "${OMNI_DEVELOPMENT_ROOT}/selection/online_action_value_summary.json" \
  "${OMNI_DEVELOPMENT_ROOT}/run_manifest.json" \
  "${OMNI_CONFIRMATION_ROOT}/COMPLETE" \
  "${OMNI_CONFIRMATION_ROOT}/records/two_encoder.json" \
  "${OMNI_CONFIRMATION_ROOT}/selection/online_action_value_summary.json" \
  "${OMNI_CONFIRMATION_ROOT}/run_manifest.json" \
  "${OMNI_MECHANISM_RECORDS}" "${DEV_PROMPTS}" "${CONF_PROMPTS}" "${PUBLIC_RESULT}"; do
  [[ -s "${path}" ]] || { echo "missing release input: ${path}" >&2; exit 2; }
done

"${PYTHON}" - "${DEV_PROMPTS}" "${CONF_PROMPTS}" \
  "${OMNI_DEVELOPMENT_ROOT}/run_manifest.json" \
  "${OMNI_CONFIRMATION_ROOT}/run_manifest.json" "${PUBLIC_RESULT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

dev_path, conf_path, dev_manifest_path, conf_manifest_path, result_path = map(Path, sys.argv[1:])

def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

dev_rows, conf_rows = rows(dev_path), rows(conf_path)
if len(dev_rows) != 128 or len(conf_rows) != 512:
    raise SystemExit("Omni release requires 128 development and 512 confirmation prompts")
dev_prompts = {row.get("prompt", row.get("text")) for row in dev_rows}
conf_prompts = {row.get("prompt", row.get("text")) for row in conf_rows}
if dev_prompts & conf_prompts:
    raise SystemExit("Omni development and confirmation prompts overlap")

dev_manifest = json.loads(dev_manifest_path.read_text(encoding="utf-8"))
conf_manifest = json.loads(conf_manifest_path.read_text(encoding="utf-8"))
result = json.loads(result_path.read_text(encoding="utf-8"))
if dev_manifest.get("prompt_count") != 128 or dev_manifest.get("fixed_guidance") is not None:
    raise SystemExit("development manifest does not describe guidance selection")
if conf_manifest.get("prompt_count") != 512 or conf_manifest.get("fixed_guidance") != 8:
    raise SystemExit("confirmation manifest does not freeze guidance=8")
if conf_manifest.get("confidence_rank_quantiles") != "0.70 0.85 0.90 0.95":
    raise SystemExit("confirmation confidence-rank strata drifted")
if result.get("prompt_count") != 512 or result.get("guidance") != 8.0:
    raise SystemExit("public Omni result does not match the confirmation protocol")
if result["paired_deltas"]["clip_cosine"]["ci95_low"] <= 0:
    raise SystemExit("formal Omni CLIP-L/14 interval is not positive")
if result["paired_deltas"]["clip_b32_cosine"]["ci95_low"] <= 0:
    raise SystemExit("formal Omni CLIP-B/32 interval is not positive")

expected_hash = hashlib.sha256(conf_path.read_bytes()).hexdigest()
if result.get("prompt_file_sha256") != expected_hash:
    raise SystemExit("public Omni result uses a different confirmation split")
PY

rm -rf "${TARGET}"
mkdir -p "${TARGET}/checkpoint-1000" "${TARGET}/records" \
  "${TARGET}/reproducibility" "${TARGET}/mechanism_cases/beach" \
  "${TARGET}/mechanism_cases/boy_kittens"

# Retain inference files only. Optimizer and distributed-training state are not
# needed to reproduce the released controller evaluation.
for name in \
  added_tokens.json config.json configuration_dream.py generation_config.json \
  generation_utils.py merges.txt model.safetensors.index.json modeling_dream.py \
  modeling_sensevoice.py resampler_projector.py special_tokens_map.json \
  tokenization_dream.py tokenizer_config.json vocab.json; do
  [[ -f "${OMNI_MODEL_PATH}/${name}" ]] && cp -a "${OMNI_MODEL_PATH}/${name}" "${TARGET}/checkpoint-1000/"
done
for shard in "${OMNI_MODEL_PATH}"/model-*.safetensors; do
  [[ -f "${shard}" ]] || { echo "missing Omni model shards" >&2; exit 2; }
  cp -a "${shard}" "${TARGET}/checkpoint-1000/"
done

cp -a "${OMNI_DEVELOPMENT_ROOT}" "${TARGET}/records/development128"
cp -a "${OMNI_CONFIRMATION_ROOT}" "${TARGET}/records/confirmation512"
cp -a "${DEV_PROMPTS}" "${TARGET}/reproducibility/omni_partiprompts_development128.jsonl"
cp -a "${CONF_PROMPTS}" "${TARGET}/reproducibility/omni_partiprompts_confirmation512.jsonl"
cp -a "${PUBLIC_RESULT}" "${TARGET}/online_action_value_release.json"
cp -a "${OMNI_MECHANISM_BEACH_CONFIDENCE}" "${TARGET}/mechanism_cases/beach/confidence"
cp -a "${OMNI_MECHANISM_BEACH_DPRM}" "${TARGET}/mechanism_cases/beach/dprm"
cp -a "${OMNI_MECHANISM_BOY_CONFIDENCE}" "${TARGET}/mechanism_cases/boy_kittens/confidence"
cp -a "${OMNI_MECHANISM_BOY_DPRM}" "${TARGET}/mechanism_cases/boy_kittens/dprm"
cp -a "${OMNI_MECHANISM_RECORDS}" "${TARGET}/mechanism_cases/audit_records_b32.json"

"${PYTHON}" "${SCRIPT_DIR}/render_omni_mechanism_cases.py" \
  --confidence-dir "${TARGET}/mechanism_cases/beach/confidence" \
  --dprm-dir "${TARGET}/mechanism_cases/beach/dprm" \
  --formal-records "${TARGET}/mechanism_cases/audit_records_b32.json" \
  --prompt-id 20270085 --case-name Beach \
  --second-confidence-dir "${TARGET}/mechanism_cases/boy_kittens/confidence" \
  --second-dprm-dir "${TARGET}/mechanism_cases/boy_kittens/dprm" \
  --second-formal-records "${TARGET}/mechanism_cases/audit_records_b32.json" \
  --second-prompt-id 20270027 --second-case-name "Boy and kittens" \
  --output "${TARGET}/mechanism_cases/paper_figure.png"
cp -a "${REPO_ROOT}/reproducibility/omni_mechanism_cases.json" \
  "${TARGET}/mechanism_cases/manifest.json"

# Optional extended mechanism audit. The replay directory contains only the
# eight post-evaluation cases declared in the public manifest. The renderer
# checks that every replayed final image is byte-identical to the corresponding
# frozen confirmation image before producing the Supplementary figures.
if [[ -n "${OMNI_SUPPLEMENT_MECHANISM_ROOT:-}" ]]; then
  [[ -s "${OMNI_SUPPLEMENT_MECHANISM_ROOT}/replay_manifest.json" ]] || {
    echo "missing supplementary mechanism replay manifest" >&2
    exit 2
  }
  SUPPLEMENT_TARGET="${TARGET}/supplementary_mechanism_cases"
  mkdir -p "${SUPPLEMENT_TARGET}"
  for case_id in crowd_fireworks owl_family three_chairs donkey_lecture \
    robot_soccer armadillo_bagpipe avocado_armchair rat_gym; do
    cp -a "${OMNI_SUPPLEMENT_MECHANISM_ROOT}/${case_id}" "${SUPPLEMENT_TARGET}/${case_id}"
  done
  cp -a "${REPO_ROOT}/reproducibility/omni_supplementary_mechanism_cases.json" \
    "${SUPPLEMENT_TARGET}/case_manifest.json"
  "${PYTHON}" - \
    "${OMNI_SUPPLEMENT_MECHANISM_ROOT}/replay_manifest.json" \
    "${OMNI_CONFIRMATION_ROOT}" "${SUPPLEMENT_TARGET}/replay_manifest.json" <<'PY'
import json
import os
import sys
from pathlib import Path

source_path, confirmation_root, output_path = map(Path, sys.argv[1:])
payload = json.loads(source_path.read_text(encoding="utf-8"))
for case in payload["cases"]:
    case_id = case["id"]
    for key in ("source_confidence_image_path", "source_dprm_image_path"):
        relative = Path(case[key]).relative_to(confirmation_root)
        case[key] = str(Path("../records/confirmation512") / relative)
    case["confidence_dir"] = str(Path(case_id) / "confidence")
    case["dprm_dir"] = str(Path(case_id) / "dprm")
output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
  "${PYTHON}" "${SCRIPT_DIR}/render_omni_supplement_cases.py" \
    --replay-manifest "${SUPPLEMENT_TARGET}/replay_manifest.json" \
    --output-dir "${SUPPLEMENT_TARGET}/figures"
  tar -czf "${TARGET}/supplementary_mechanism_cases.tar.gz" \
    -C "${TARGET}" supplementary_mechanism_cases
fi

"${PYTHON}" - "${TARGET}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

target = Path(sys.argv[1])
items = [
    ("selected_dprm_checkpoint_index", "checkpoint-1000/model.safetensors.index.json"),
    ("selected_dprm_checkpoint_config", "checkpoint-1000/config.json"),
    ("development_records", "records/development128/records/two_encoder.json"),
    ("confirmation_records", "records/confirmation512/records/two_encoder.json"),
    ("formal_result", "online_action_value_release.json"),
    ("development_split", "reproducibility/omni_partiprompts_development128.jsonl"),
    ("confirmation_split", "reproducibility/omni_partiprompts_confirmation512.jsonl"),
    ("mechanism_manifest", "mechanism_cases/manifest.json"),
    ("mechanism_figure", "mechanism_cases/paper_figure.png"),
]
supplement = target / "supplementary_mechanism_cases"
if supplement.is_dir():
    items.extend([
        ("supplement_mechanism_manifest", "supplementary_mechanism_cases/case_manifest.json"),
        ("supplement_replay_manifest", "supplementary_mechanism_cases/replay_manifest.json"),
        ("supplement_replay_archive", "supplementary_mechanism_cases.tar.gz"),
    ])
    for path in sorted((supplement / "figures").glob("omni_mechanism_cases_*.png")):
        items.append((f"supplement_{path.stem}", str(path.relative_to(target))))
for path in sorted((target / "checkpoint-1000").glob("model-*.safetensors")):
    items.append((f"checkpoint_{path.stem.replace('-', '_')}", str(path.relative_to(target))))

artifacts = []
for artifact_id, relative in items:
    path = target / relative
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    artifacts.append({
        "id": artifact_id,
        "path": str(Path("omni_diffusion") / relative),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    })
(target / "manifest_fragment.json").write_text(
    json.dumps({"status": "complete", "artifacts": artifacts}, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(artifacts, indent=2))
PY
