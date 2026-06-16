# DPRM-Omni-Diffusion Patch Map

`DPRM-Omni-Diffusion` applies DPRM to visual-token ordering in text-to-image
generation. The Omni-Diffusion denoiser, tokenizer, MagViT/VQ image tokenizer,
prompt pipeline, and image sampler stay fixed. DPRM changes only which masked
visual tokens are revealed at each denoising step.

Upstream resources:

- Code: <https://github.com/VITA-MLLM/Omni-Diffusion>
- Model family: Omni-Diffusion multimodal discrete diffusion.

## Host Mapping

- `confidence`: probability or margin assigned to the proposed visual token.
- `candidate_mask`: currently masked visual-token positions eligible for reveal.
- `phase_ids`: denoising step bucket.
- `aux_bin_ids`: relative visual-token position bin, usually a raster-position
  bucket or coarse spatial bucket.
- `rewards`: image-level utility from the host evaluation, e.g. CLIP image-text
  cosine or another task-specific visual score.

## Ordering Variants

- `random`: reveal masked visual positions uniformly at random.
- `progressive_confidence`: reveal highest-confidence visual-token proposals.
- `dprm_confidence_warmup`: confidence warmup, then DPRM table guidance.
- `dprm_random_warmup`: random warmup, then DPRM table guidance.

For formal DPRM runs, a DPRM-labeled policy should consume a real table/hook
after warmup. Silent fallback to confidence makes the run indistinguishable from
the confidence baseline and should only be used for explicit debugging.

## Overlay Files

- `overlay/generation_order.py`: a small hook for Omni-style one-dimensional
  masked-token selection. It supports confidence/random/entropy/DPRM policies,
  table scoring, and trace logging.

Insert the hook where the host computes visual-token confidence and chooses
`number_transfer_tokens` masked positions to reveal. The sampled token values
still come from the Omni-Diffusion denoiser.

## Public Result Summary

The compact four-order text-to-image summary is in
`statistics_outputs/multimodal_order_results.csv`.

On the corrected 64-prompt official-step split, CLIP-L/14 mean image-text cosine
is `0.24915` for DPRM-confidence, `0.24744` for confidence-progressive, `0.22184`
for random, and `0.21456` for DPRM-random. This is a small positive
visual-token result for DPRM-confidence and a negative result for DPRM-random.

## Reproduction Sketch

1. Clone the upstream Omni-Diffusion host and install its model/tokenizer stack.
2. Add `overlay/generation_order.py` to the host import path.
3. Run `random` and `progressive_confidence` with order tracing enabled.
4. Score generated images with a fixed image-text metric and write a reward CSV.
5. Build a DPRM table with `examples/build_bucket_table_from_traces.py`.
6. Re-run the same prompt split and generation budget with DPRM order policies.

## Codex / Claude Guidance

Ask the assistant to preserve:

- the Omni-Diffusion model, tokenizer, image tokenizer, prompt split, and
  generation step budget;
- the original random/confidence order modes as explicit baselines;
- the same image scoring script and prompt list for all four orders;
- compact traces and result summaries, not raw images/checkpoints in this repo.

The only intervention should be replacing visual-token reveal order.
