# DPRM-SDPO Overlay

`DPRM-SDPO` applies the DPRM ordering controller to SDPO (Stepwise Decomposition
Preference Optimization), a reward-guided discrete diffusion framework for DNA
sequence design.

Upstream resources:

- Paper: [Discrete Diffusion Trajectory Alignment via Stepwise Decomposition](https://arxiv.org/pdf/2507.04832)
- Code: [hanjq17/discrete-diffusion-sdpo](https://github.com/hanjq17/discrete-diffusion-sdpo)

The upstream repository is released under the **MIT License**.

## Host Mapping

- `confidence`: per-token proposal probability from the discrete diffusion denoiser (CNN backbone with substitution parameterization).
- `candidate_mask`: currently masked DNA positions eligible for reveal at each diffusion step.
- `phase_ids`: diffusion-phase bucket derived from the current mask fraction (`(1 - mask_frac) * num_phase_bins`).
- `rewards`: terminal oracle reward from an Enformer-based regulatory expression predictor (HepG2 expression level), scaled by `exp(beta * reward)`.

## What Changed In The Local Fork

- The diffusion model architecture (CNN backbone), substitution parameterization, noise schedule (loglinear), and SDPO reward-weighted training objective are kept fixed.
- The `_ddpm_update` method in `diffusion_gosai_update.py` exposes `order_policy` with four modes: `baseline` (standard SDPO sampling), `progressive` (confidence-ranked reveal), `dprm` (DPRM Soft-BoN with confidence shortlist), and `dprm_random` (DPRM with random warmup then confidence shortlist).
- DPRM statistics (phase x confidence bin counts and reward-weighted sums) are maintained as registered buffers and updated online during training via `update_dprm_stats_from_batch`.
- A three-stage schedule controls the DPRM gate: random/confidence ordering before `warmup_steps`, linear ramp from `warmup_steps` to `switch_steps`, then full DPRM guidance gated by per-bin readiness.
- The evaluation script (`eval_dna_bootstrap.py`) generates DNA sequences from the all-mask state with the same ordering family used at training time, then reports HepG2 predicted expression, ATAC accessibility accuracy, 3-mer Pearson correlation with high-expression sequences, log-likelihood under the reference model, and a composite total metric with bootstrap confidence intervals.

## Reproduction Sketch

Clone the upstream host repository, then apply the SDPO overlay pattern from the research fork. The Gosai enhancer dataset experiment used the absorbing-state discrete diffusion configuration:

```bash
# Prepare data and reward oracle checkpoints (see upstream README)
# Ensure data_and_model/.ready exists

# Train four ordering variants
ORDER_POLICY=baseline  RUN_NAME=sdpo-dna-baseline  bash scripts/run_sdpo_dna_variant.sh
ORDER_POLICY=progressive RUN_NAME=sdpo-dna-progressive bash scripts/run_sdpo_dna_variant.sh
ORDER_POLICY=dprm        RUN_NAME=sdpo-dna-dprm-confidence bash scripts/run_sdpo_dna_variant.sh
ORDER_POLICY=dprm_random RUN_NAME=sdpo-dna-dprm-random bash scripts/run_sdpo_dna_variant.sh

# Evaluate all variants with paired bootstrap
bash scripts/run_sdpo_dna_eval_compare.sh

# Generate summary CSV, LaTeX table, and bar plots
python scripts/summarize_sdpo_dna_results.py \
  --output-root data_and_model/dprm_sdpo_outputs \
  --summary-dir eval_outputs/sdpo_dna_ordering
```

Key hyperparameters (defaults in `run_sdpo_dna_variant.sh`):
- SDPO beta: `0.5`, learning rate: `1e-5`, epochs: `2`, batch size K: `2000`
- DPRM beta: `1.0`, warmup steps: `100`, switch steps: `400`, ready count: `64`, shortlist size: `64`
- Evaluation: 10 batches x 64 samples = 640 total, 1000 bootstrap iterations

## Codex / Claude Guidance

Ask the assistant to preserve:

- the original discrete diffusion architecture and substitution parameterization;
- the SDPO reward-weighted training objective and reference model;
- the Enformer-based oracle reward computation;
- the original baseline (standard SDPO) ordering mode;
- per-sample evaluation logging for bootstrap confidence intervals.

The only intervention should be replacing the token reveal order during the DDPM
sampling loop with either confidence ordering or the scheduled DPRM controller.
