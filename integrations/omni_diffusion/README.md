# DPRM-Omni-Diffusion

Omni-Diffusion represents an image as masked visual codebook tokens and applies
a multimodal denoiser conditioned on text. The host sampler predicts token
values and commits visual positions over `260` steps. DPRM changes the committed
position; the denoiser, VQ tokenizer/decoder, prompt, seed, and continuation
decoder stay fixed.

## Four Paper Orders

- `random`: uniformly sampled visual positions;
- `confidence`: highest-confidence visual positions;
- `DPRM-BoN-2`: at step `96`, evaluate two fixed confidence-quantile actions
  (`0.3`, `0.7`) plus the confidence fallback;
- `DPRM-BoN-4`: evaluate four actions (`0.15`, `0.3`, `0.7`, `0.85`) plus the
  confidence fallback.

Each action branch starts from the same partial visual canvas and completes with
confidence decoding. Hard terminal-utility selection implements the high-tilt
DPRM-BoN limit. The utility is CLIP-L/14 image-text cosine plus `0.01` times the
LAION aesthetic-predictor score. Uniform-3 and Uniform-5 sample from the same
branch records and are compute-matched controls.

## Paper Protocol

The held-out audit uses `96` prompts, one fixed seed per prompt, `256` generated
visual tokens, and the official `260`-step path. DPRM-BoN-2/4 require `3/5`
complete rollouts per prompt. CLIP-L/14 selects actions; CLIP-B/32 is an
independent post-selection evaluator. The compact statistics are in
[`../../results/artifacts/multimodal_summary.json`](../../results/artifacts/multimodal_summary.json).

## Reproduction

1. Copy `overlay/host/generation_utils.py` and `scripts/` into the Omni-Diffusion
   checkout. Set `OMNI_ROOT`, `OMNI_MODEL_PATH`, and
   `OMNI_IMAGE_TOKENIZER_PATH`. Set the output directory explicitly, for example
   `export OMNI_ACTION_ROOT="$OMNI_ROOT/outputs/omni_action_branches"`.
2. Run `scripts/run_action_branches.sh`. It selects 96 deduplicated JourneyDB
   prompts after offset 2000, assigns seed `20268000 + prompt_index`, and generates
   the confidence baseline plus quantiles `0.15`, `0.3`, `0.7`, and `0.85` from
   the same step-96 canvas. The prompt offset, count, seeds, quantiles, and GPUs
   have environment-variable overrides in the script.
3. Set `AESTHETIC_WEIGHTS` to the LAION aesthetic-predictor weights used by the
   evaluator and score the complete branches:

   ```bash
   python scripts/score_action_branches.py \
     --root "$OMNI_ACTION_ROOT" \
     --output "$OMNI_ACTION_ROOT/action_branches.json" \
     --aesthetic-weights "$AESTHETIC_WEIGHTS"
   ```

4. Run both DPRM shortlists and the compute-matched analysis:

   ```bash
   python scripts/select_dprm_bon.py \
     --records "$OMNI_ACTION_ROOT/action_branches.json" \
     --output-dir "$OMNI_ACTION_ROOT/selection"
   python scripts/analyze_compute_matched.py \
     --records "$OMNI_ACTION_ROOT/action_branches.json" \
     --output-dir "$OMNI_ACTION_ROOT/compute_matched"
   ```

5. Run random and confidence policies with the registry commands to reconstruct
   the four-order table and visual audit.

`generate_four_orders.py` rejects a DPRM label after warmup unless a real table
or action-value model is supplied. Diagnostic confidence fallback requires an
explicit flag and is excluded from formal results.
