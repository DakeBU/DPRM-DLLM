# DPRM-Omni-Diffusion

Omni-Diffusion represents an image as masked visual codebook tokens and applies
a multimodal denoiser conditioned on text. The host sampler predicts token
values and commits visual positions over `260` steps. DPRM changes the committed
position; the denoiser, VQ tokenizer/decoder, prompt, seed, and continuation
decoder stay fixed.

## Matched Orders

- `random_matched`: uniformly sampled visual positions;
- `confidence_matched`: Omni's negative-token-entropy position order;
- `dprm_matched`: confidence proposal plus a frozen process-value table;
- `dprm_random_matched`: random-proposal DPRM control available in the registry.

The reported comparison trains random, confidence, and DPRM from one checkpoint.
Each branch receives teacher-forced canvases induced by its deployed order and
executes one current-model action with the same selector before the unchanged
denoising loss. CLIP-L/14 terminal utility builds the DPRM table on development
prompts; it is not a training loss. At evaluation every method generates one
image per prompt without a reward call or completed-image selection.

## Paper Protocol

The held-out audit uses `96` prompt-text-deduplicated prompts, one fixed seed per
prompt, `256` generated visual tokens, and the official `260`-step path.
CLIP-L/14 is the paired primary metric and CLIP-B/32 is a directional check.
The promotion gate also verifies order divergence, complete prompt pairing,
zero test-time reward calls, and exact training/deployment contracts.

## Reproduction

1. Check out Omni-Diffusion at commit `c4f4625f84197a72d556ea00f10e5b2775524252`.
2. Copy `matched/overlay/` into that checkout, preserving relative paths.
3. Install this package in the same environment and set the required paths:

   ```bash
   export OMNI_ROOT="$HOME/checkouts/Omni-Diffusion"
   export OMNI_MODEL_PATH="$HOME/models/omni-shared-checkpoint"
   export OMNI_IMAGE_TOKENIZER_PATH="$HOME/models/magvitv2"
   export OMNI_DATA_JSON="$HOME/data/tokenized_journeydb.jsonl"
   export OMNI_CONTROLLER="$HOME/artifacts/frozen_controller.json"
   export OMNI_RUN_ROOT="$HOME/outputs/omni-matched"
   ```

4. Run the matched pipeline. Development, trajectory, and evaluation prompt
   ranges must be disjoint; defaults and hashes are written to the run manifest.

   ```bash
   bash integrations/omni_diffusion/matched/run_pipeline.sh
   ```

The pipeline writes branch manifests, source and code hashes, checkpoint audits,
paired bootstrap statistics, the promotion decision, fixed-index images, and a
blinded visual-rating package. A failed promotion gate remains a reported
boundary result and is not rewritten as a positive row.

## Completed-Path Diagnostic

The files under `scripts/` outside `matched/` reproduce a completed-path action
search used to inspect one-step visual consequences. It evaluates multiple
terminal continuations and is therefore not the paper endpoint. Its values are
stored in `results/artifacts/omni_completed_path_diagnostic.csv`, not in the
canonical paper table.

For that diagnostic only, rebuild a shared-state canvas with:

```bash
python scripts/build_intermediate_canvas.py \
  --confidence-dir outputs/intermediate/confidence \
  --dprm-dir outputs/intermediate/dprm \
  --formal-records outputs/selection/matched_four_order_records_scored.json \
  --prompt-id "$PROMPT_ID" \
  --output outputs/intermediate/canvas_comparison.png
```

The two runs must use the same prompt, seed, checkpoint, tokenizer, and sampler.
The DPRM run changes one step-96 action with `--force-order-step 96` and the
quantile selected by the diagnostic DPRM-BoN record, then resumes confidence
decoding.

`generate_four_orders.py` rejects a DPRM label after warmup unless a real table
or action-value model is supplied. Diagnostic confidence fallback requires an
explicit flag and is excluded from formal results.
