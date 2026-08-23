# Release Checklist

Run the release from the repository root after all experiment packaging jobs
have completed.

## Verify code, artifacts, and paper

```bash
source .venv/bin/activate
export DPRM_ARTIFACT_ROOT="$HOME/DPRM-DLLM-artifacts"
export DPRM_PAPER_ROOT="$HOME/DPRM/Paper"

pytest -q
python scripts/verify_release.py
python scripts/verify_artifact_manifest.py \
  --artifact-root "$DPRM_ARTIFACT_ROOT" --require-complete
python scripts/audit_artifact_semantics.py \
  --artifact-root "$DPRM_ARTIFACT_ROOT"
python scripts/prepare_hf_release.py \
  --artifact-root "$DPRM_ARTIFACT_ROOT" --repo-root .
python scripts/verify_submission_ready.py \
  --paper-root "$DPRM_PAPER_ROOT"
git diff --check
```

`prepare_hf_release.py` refuses to write a model card when any host is pending,
or when an artifact byte size or SHA-256 digest differs from the frozen
manifest.

## Push GitHub without storing the PAT in the remote URL

```bash
read -rsp "GitHub PAT: " github_pat; echo
export github_pat
ASKPASS="$(mktemp)"
cat > "$ASKPASS" <<'SH'
#!/bin/sh
case "$1" in
  *Username*) printf '%s\n' x-access-token ;;
  *Password*) printf '%s\n' "$github_pat" ;;
esac
SH
chmod 700 "$ASKPASS"
GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
  git push https://github.com/DakeBU/DPRM-DLLM.git main
rm -f "$ASKPASS"
unset ASKPASS github_pat
```

Review and commit the release diff before running the push command. Do not put
the PAT in `origin`, a command argument, or a checked-in file.

## Upload the Hugging Face bundle

```bash
source .venv/bin/activate
python -m huggingface_hub.commands.huggingface_cli login
python -m huggingface_hub.commands.huggingface_cli repo create \
  DarkerBu/DPRM-DLLM --repo-type model --exist-ok
python -m huggingface_hub.commands.huggingface_cli upload-large-folder \
  DarkerBu/DPRM-DLLM "$DPRM_ARTIFACT_ROOT" \
  --repo-type model --num-workers 4
```

`upload-large-folder` records local progress and can be rerun after a network
interruption. Do not place the Hugging Face token in a command argument or a
checked-in file.

After upload, download the manifest and one small raw-record archive from the
Hub and rerun `verify_artifact_manifest.py` against the downloaded bundle.
