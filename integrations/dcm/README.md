# DPRM-DCM Overlay

`DPRM-DCM` applies the DPRM ordering controller to Discrete Cell Models
(DCM), a masked discrete diffusion framework for single-cell gene expression
modeling.

Upstream resources:

- Paper: [Discrete Diffusion for Single-Cell Gene Expression Modeling](https://www.biorxiv.org/content/10.64898/2026.02.19.705033v1.full.pdf)
- Code: [sanjukta7/aivc-dcm](https://github.com/sanjukta7/aivc-dcm)

## Host Mapping

- `confidence`: per-gene expression-bin probability from the current DCM denoiser.
- `candidate_mask`: currently masked gene positions eligible for reveal.
- `phase_ids`: reveal-step bucket during blockwise decoding.
- `aux_bin_ids`: unused in the Dentate Gyrus unconditional experiment.
- `rewards`: self-supervised reconstruction utility, implemented as token recovery on the selected gene bins.

## What Changed In The Local Fork

- The DCM model, count-bin vocabulary, data split, optimizer, and denoising loss are kept fixed.
- The training loop exposes `order_policy` with `random_ordered`, `confidence`, `dprm_random`, and `dprm_confidence`.
- DPRM state is checkpointed with the model and loaded by the evaluator.
- The evaluation script decodes from the all-mask state with the same ordering family used at training time, then reports per-cell token recovery, mean absolute error, and zero-expression accuracy with bootstrap confidence intervals.

## Current Dentate Gyrus Result

The matched four-way evaluation is summarized in `statistics_outputs/dcm/` and mirrored in `statistics_outputs/latest/`.

- Token recovery improves from `66.76%` for DCM-random to `76.07%` for Progressive-DCM and `76.00%` for DPRM(conf.)-DCM.
- MAE decreases from `0.758` for DCM-random to `0.628` for Progressive-DCM and `0.642` for DPRM(conf.)-DCM.
- Zero-expression accuracy improves from `82.83%` for DCM-random to `99.86%` for DPRM(random)-DCM.

## Reproduction Sketch

Clone the upstream host repository, then apply the DCM overlay pattern from the research fork. The Dentate Gyrus run used the small SEDD-style DCM configuration:

```bash
python scripts/preprocess_dentate_rnaseq.py \
  --input datasets/dentate/DentateGyrus.h5ad \
  --output datasets/dentate/dentate_5000_bins32.h5ad \
  --top-genes 5000 \
  --num-bins 32

python scripts/train_rnaseq.py CONFIG=configs/rnaseq_dcm_dentate.yaml
python scripts/train_rnaseq.py CONFIG=configs/rnaseq_progressive_dentate.yaml
python scripts/train_rnaseq.py CONFIG=configs/rnaseq_dprm_dentate.yaml
python scripts/train_rnaseq.py CONFIG=configs/rnaseq_dprm_confidence_dentate.yaml
```

After training, evaluate all checkpoints with paired per-cell logging:

```bash
python scripts/eval_dcm_ordering_bootstrap.py \
  --series DCM-random=configs/rnaseq_dcm_dentate.yaml=experiments/dcm_single_cell_real_random/best.pt=random_ordered \
  --series Progressive-DCM=configs/rnaseq_progressive_dentate.yaml=experiments/dcm_single_cell_real_progressive/best.pt=confidence \
  --series DPRM-random=configs/rnaseq_dprm_dentate.yaml=experiments/dcm_single_cell_real_dprm/best.pt=dprm_random \
  --series DPRM-confidence=configs/rnaseq_dprm_confidence_dentate.yaml=experiments/dcm_single_cell_real_dprm_confidence/best.pt=dprm_confidence \
  --output-dir eval_outputs/dentate_real_bootstrap_4way \
  --data-path datasets/dentate/dentate_5000_bins32.h5ad \
  --num-steps 32 \
  --num-samples 4 \
  --bootstrap 5000
```

## Codex / Claude Guidance

Ask the assistant to preserve:

- the original DCM architecture and SEDD-style denoising loss;
- the count-bin preprocessing and train/validation split;
- the original random-order baseline mode;
- per-cell evaluation logging for paired bootstrap.

The only intervention should be replacing the gene-position reveal order with
either confidence ordering or the scheduled DPRM controller.
