# DPRM-LLaDA-V Patch Map

`DPRM-LLaDA-V` applies DPRM to image-conditioned text generation. The vision
encoder, multimodal projector, language denoiser, tokenizer, and VQA evaluation
protocol stay fixed. DPRM changes only which masked answer tokens are revealed
at each denoising step.

Upstream resources:

- Model card: <https://huggingface.co/GSAI-ML/LLaDA-V>
- Code release used by the host project: the LLaDA-V training/evaluation stack
  around `generate_with_embeds` and `lmms-eval`.

## Host Mapping

- `confidence`: probability assigned to the host model's proposed answer token.
- `candidate_mask`: currently masked answer positions that can be revealed.
- `phase_ids`: denoising step bucket across the answer-generation schedule.
- `aux_bin_ids`: relative answer-position bin; task or answer-type bins can be
  added for heterogeneous VQA.
- `rewards`: target-normalized VQA correctness or another task-level utility
  computed after the answer is decoded.

## Ordering Variants

- `random`: reveal eligible answer positions uniformly at random.
- `progressive_confidence`: reveal highest-confidence proposed answer tokens.
- `dprm_confidence_warmup`: confidence warmup, then DPRM table guidance.
- `dprm_random_warmup`: random warmup, then DPRM table guidance.
- `entropy`: reward-blind uncertainty control using `1 - confidence`.
- `eot_suppression` / SACM-style controls: diagnostic baselines for early
  end-of-turn and suffix-anchor overconfidence.

The token values still come from LLaDA-V. DPRM only changes reveal order.

## Overlay Files

- `overlay/dprm_generation.py`: host-facing helpers for table lookup, DPRM score
  mixing, trace logging, and EOT diagnostics.

Use this overlay by importing the helper in the host `modeling_llada.py` file and
calling it where the host currently computes the remasking score before `topk`.

## Public Result Summary

The compact four-order VQA summary is in
`statistics_outputs/multimodal_order_results.csv`.

- AI2D: DPRM-confidence reaches `0.692` target-normalized accuracy, above
  confidence-progressive `0.658`.
- RealWorldQA: confidence-progressive remains strongest at `0.46013`; the coarse
  DPRM table transfers negatively on this broader distribution.

The mechanism controls in `statistics_outputs/mechanism_controls.csv` include
entropy-only, EOT suppression, and SACM-style rows. They should be interpreted as
boundary diagnostics: DPRM helps on short structured AI2D answers, while direct
EOT handling is better on RealWorldQA.

## Reproduction Sketch

1. Clone and set up the upstream LLaDA-V host.
2. Add `overlay/dprm_generation.py` to the host import path.
3. Run `random` and `progressive_confidence` with `trace_order_stats` enabled.
4. Build a DPRM table with `examples/build_bucket_table_from_traces.py`.
5. Re-run evaluation with `dprm_table=<table.json>` and the same generation
   length, block length, task split, and seed set.

## Codex / Claude Guidance

Ask the assistant to preserve:

- the image encoder, projector, language model, tokenizer, and lmms-eval task
  definitions;
- the generation budget (`gen_length`, `block_length`, `gen_steps`) and seed set;
- the original `random` and `progressive_confidence` remasking modes;
- per-example outputs and order traces for paired or target-normalized analysis.

The only intervention should be replacing the answer-token reveal order.
