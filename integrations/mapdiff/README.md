# DPRM-MapDiff Patch Map

This integration targets [MapDiff](https://github.com/peizhenbai/MapDiff), the
mask-prior-guided discrete denoising diffusion model for inverse protein folding
described in the [Nature Machine Intelligence paper](https://www.nature.com/articles/s42256-025-01042-6).

The intended intervention is narrow:

- keep MapDiff's graph/backbone conditioning, mask prior, denoising network, loss, CATH4.2 split, and evaluation metrics fixed;
- replace only the residue reveal / update ordering policy;
- preserve the original MapDiff ordering as a baseline flag;
- save per-protein recovery and NSSR metrics so bootstrap intervals can be computed after evaluation.

Recommended controller variants:

- `baseline`: original MapDiff reveal order.
- `progressive`: current-model confidence-ranked reveal order.
- `dprm`: confidence warmup followed by online DPRM Soft-BoN.
- `dprm_random`: random warmup followed by online DPRM Soft-BoN.

## Host Mapping

- `confidence`: amino-acid probability assigned by the current MapDiff denoiser.
- `candidate_mask`: currently masked residue positions eligible for reveal.
- `phase_ids`: reveal-step bucket derived from the current mask fraction.
- `aux_bin_ids`: optional structural bucket if the host already computes one cheaply.
- `rewards`: self-supervised sequence recovery on the masked residue positions.

## Reported CATH4.2 Diagnostic

The lightweight ordering diagnostic evaluates `1,120` CATH4.2 test proteins.
The best point estimates come from `DPRM(random)-MapDiff`, but intervals overlap:

- recovery: `0.5928` to `0.5934`;
- BLOSUM90 / NSSR90: `0.7542` to `0.7554`.

These results are best read as evidence that the DPRM plug-in can be inserted
into a second protein diffusion architecture without changing the host model.
They are not a strong claim of statistically separated MapDiff improvement.

## Codex / Claude Guidance

Ask the assistant to preserve:

- the original MapDiff graph denoiser and mask-prior mechanism;
- the CATH4.2 preprocessing and train/test split;
- the original baseline ordering mode;
- per-protein logging for sequence recovery and NSSR bootstrap intervals.

The only intervention should be replacing the residue reveal order with either
confidence ordering or the scheduled DPRM controller.
