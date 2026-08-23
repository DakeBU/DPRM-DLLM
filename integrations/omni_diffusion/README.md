# DPRM-Omni-Diffusion

This integration changes which masked visual-code position Omni-Diffusion
commits next. The host predicts the token value and ranks positions by negative
token entropy. DPRM retains the checkpoint, visual tokenizer, provisional token
values, seed, and 260-step confidence continuation.

## Controller

At visual step 96, the controller starts from one shared canvas and evaluates
five declared actions:

- Omni's native lowest-entropy action;
- positions at current-prompt confidence-rank quantiles `0.70`, `0.85`,
  `0.90`, and `0.95`.

Each action receives one deterministic confidence continuation. CLIP-L/14 is
the terminal reward, and the deployed score is

```text
native_order_score + guidance * (CLIP-L(action) - CLIP-L(native)) / 0.03.
```

The selected path is the DPRM output. CLIP-B/32 is computed only after
selection as an independent semantic check. Uniform selection over the same
five paths and reward-only selection are compute-matched controls. This is the
`K=1` Monte Carlo action-value estimator and hard-tilt selection rule described
in the paper; it is not manual image selection.

## Paper Configuration

The host checkout is pinned to commit
`c4f4625f84197a72d556ea00f10e5b2775524252`. The frozen host checkpoint is the
step-1000 confidence-trained checkpoint in the release package. Rank strata,
action step, and reward scale are declared before evaluation. Guidance is
selected from `0.25 0.5 1 2 4 8` on 128 development prompts and fixed to `8`
before opening 512 disjoint confirmation prompts.

| Order | CLIP-L/14 | CLIP-B/32 | Paths |
|---|---:|---:|---:|
| Random | 0.17939 | 0.23381 | 1 |
| Omni default | 0.18661 | 0.23836 | 1 |
| Uniform action | 0.18566 | 0.23796 | 5 |
| DPRM | **0.21125** | **0.24854** | 5 |

DPRM minus confidence is `+0.02464 [0.02249, 0.02677]` on CLIP-L/14 and
`+0.01018 [0.00820, 0.01221]` on CLIP-B/32, using 5,000 paired bootstrap
resamples over prompt ids. The complete machine-readable report is
[`../../results/artifacts/omni_online_action_value_release.json`](../../results/artifacts/omni_online_action_value_release.json).
The four registered order commands are listed in
[`../../reproducibility/experiments.json`](../../reproducibility/experiments.json).

## Reproduction

Create an Omni-compatible environment, install this repository in editable
mode, and set:

```bash
export OMNI_ROOT="$HOME/checkouts/Omni-Diffusion"
export OMNI_MODEL_PATH="$HOME/models/dprm-omni-checkpoint-1000"
export OMNI_IMAGE_TOKENIZER_PATH="$HOME/models/magvitv2"
export VIRTUAL_ENV="$HOME/envs/omni"
export OMNI_ONLINE_GPUS="0 1 2 3 4 5 6 7"
```

Run development selection:

```bash
export OMNI_ONLINE_PROMPT_FILE="$PWD/reproducibility/omni_partiprompts_development128.jsonl"
export OMNI_ONLINE_ROOT="$HOME/outputs/omni_online_dev128"
export OMNI_ONLINE_COUNT=128
unset OMNI_ONLINE_FIXED_GUIDANCE
bash integrations/omni_diffusion/matched/run_online_action_value_controller.sh
```

Run the untouched confirmation with the selected guidance:

```bash
export OMNI_ONLINE_PROMPT_FILE="$PWD/reproducibility/omni_partiprompts_confirmation512.jsonl"
export OMNI_ONLINE_ROOT="$HOME/outputs/omni_online_confirmation512"
export OMNI_ONLINE_COUNT=512
export OMNI_ONLINE_FIXED_GUIDANCE=8
export OMNI_ONLINE_INCLUDE_RANDOM=1
bash integrations/omni_diffusion/matched/run_online_action_value_controller.sh

python integrations/omni_diffusion/matched/scripts/publish_omni_online_results.py \
  --summary "$OMNI_ONLINE_ROOT/selection/online_action_value_summary.json" \
  --scored-records "$OMNI_ONLINE_ROOT/records/two_encoder.json" \
  --run-manifest "$OMNI_ONLINE_ROOT/run_manifest.json" \
  --output results/artifacts/omni_online_action_value_release.json
```

The runner writes one job for every prompt/action pair, hashes the prompt split
and shared canvas, verifies that every forced branch changes exactly one
position action, scores all paths, and performs selection without human input.

## Supplementary Confirmation Cases

The Supplement contains eight additional cases from the frozen 512-prompt
confirmation. They are selected only after aggregate evaluation by the public
rule in
[`../../reproducibility/omni_supplementary_mechanism_cases.json`](../../reproducibility/omni_supplementary_mechanism_cases.json):
the DPRM action must differ from confidence, both CLIP encoders must improve,
and the prompt must expose an inspectable count, relation, or compositional
attribute. These cases do not select the controller or contribute to its mean.

The release bundle stores deterministic step-64/96/192 replays. Rebuild all
four Supplementary figures with:

```bash
python integrations/omni_diffusion/matched/scripts/render_omni_supplement_cases.py \
  --replay-manifest \
    "$DPRM_ARTIFACT_ROOT/omni_diffusion/supplementary_mechanism_cases/replay_manifest.json" \
  --output-dir "$HOME/outputs/omni_supplementary_figures"
```

Before rendering, the script checks SHA-256 digests of every replayed final
image against its source image in the frozen confirmation package. It aborts
if any pair differs.

## Mechanism Figure

The paper's beach and boy-with-kittens figure is a separate checkpoint-500
mechanism diagnostic. It uses the native action plus rank quantiles
`0.15 0.30 0.70 0.85`, seeds `20270085` and `20270027`, and terminal utility
`CLIP-L/14 + 0.01 * LAION aesthetic`. The selected visual indices are 229 and
192. This diagnostic is not included in the 512-prompt mean and is labeled
separately in the paper. The renderer is
`integrations/omni_diffusion/matched/scripts/render_omni_mechanism_cases.py`;
its manifest fixes prompt text, seed, candidate set, checkpoint identifier, and
input record paths.

After downloading the Hugging Face artifact to `$DPRM_ARTIFACT_ROOT`, rebuild
the figure with:

```bash
python integrations/omni_diffusion/matched/scripts/render_omni_mechanism_cases.py \
  --confidence-dir "$DPRM_ARTIFACT_ROOT/omni_diffusion/mechanism_cases/beach/confidence" \
  --dprm-dir "$DPRM_ARTIFACT_ROOT/omni_diffusion/mechanism_cases/beach/dprm" \
  --formal-records "$DPRM_ARTIFACT_ROOT/omni_diffusion/mechanism_cases/audit_records_b32.json" \
  --prompt-id 20270085 --case-name Beach \
  --second-confidence-dir "$DPRM_ARTIFACT_ROOT/omni_diffusion/mechanism_cases/boy_kittens/confidence" \
  --second-dprm-dir "$DPRM_ARTIFACT_ROOT/omni_diffusion/mechanism_cases/boy_kittens/dprm" \
  --second-formal-records "$DPRM_ARTIFACT_ROOT/omni_diffusion/mechanism_cases/audit_records_b32.json" \
  --second-prompt-id 20270027 --second-case-name "Boy and kittens" \
  --output integrations/omni_diffusion/omni_intermediate_canvas_case.png
```
