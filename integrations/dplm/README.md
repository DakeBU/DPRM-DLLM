# DPRM-DPLM Overlay

`DPRM-DPLM` keeps the DPLM / DPLM-2 Bit architecture, multimodal conditioning, structure tokenizer, and denoising losses fixed. The intervention is ordering-only.

The main fair experiment is the `DPLM-2 Bit` path: same backbone, same data, same training budget, only token ordering changes.

## Host Mapping

- `confidence`: amino-acid token confidence from the current diffusion denoiser.
- `candidate_mask`: masked residue positions eligible for reveal.
- `phase_ids`: progressive masking phase during training or decode-step bucket during skeptical decoding.
- `aux_bin_ids`: optional structure bucket, contact-density bucket, or other protein-specific state bucket.
- `rewards`: self-supervised terminal utility such as amino-acid recovery.

## What Changed In The Local Fork

- `src/byprot/models/dplm/dprm_order.py` provides a host-specific controller with optional structure bins;
- `src/byprot/models/dplm/modules/dplm_adapter.py` and `src/byprot/models/dplm2/dplm2.py` call the controller during training and decoding;
- `configs/experiment/dplm2/progressive_dplm2_bit_650m.yaml` is the confidence-only progressive baseline;
- `configs/experiment/dplm2/dprm_random_dplm2_bit_650m.yaml` is the random-to-DPRM warmup variant;
- `configs/experiment/dplm2/dprm_dplm_650m.yaml` is the confidence-to-DPRM fair `DPLM-2 Bit` run;
- `configs/experiment/dplm/cond_dprm_dplm_650m.yaml` is a conditional inverse-folding auxiliary experiment.

## Current DPLM-2 Bit Result

The matched ordering comparison is summarized in `statistics_outputs/latest/`.

- Forward-folding RMSD decreases from `35.47` for DPLM-2 Bit to `29.43` for the best ordering-aware variant.
- Forward-folding TM-score improves from `0.3071` to `0.3321`.
- Co-generation remains multi-objective: the confidence-progressive variant is strongest on TM-score, pLDDT, and designable rate, while DPRM-DPLM has the smallest CoGen RMSD penalty among ordering-aware variants.

## Codex / Claude Guidance

Tell the assistant to:

- leave the protein model and tokenizer untouched;
- preserve the original max-step budget and datamodule;
- use protein-specific auxiliary buckets only if the host already computes them cheaply;
- keep the original low-confidence decoder as a baseline option.
