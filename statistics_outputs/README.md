# Statistical Outputs

This directory contains compact result summaries and plots used by the paper draft and repository README.

## Current headline exports

The most recent PUMA and DPLM rerun artifacts are under [`latest/`](./latest):

- `puma_1530k_tinygsm_comparison_results.json`
- `puma_accuracy_ci_1530k.png`
- `puma_vs_dprm_puma_1530k_gsm8k.png`
- `forward_folding_summary.csv`
- `forward_folding_deltas.csv`
- `co_generation_overall_summary.csv`
- `co_generation_deltas.csv`
- `co_generation_lengthwise_summary.csv`
- `dplm2_order_forward_ci.png`
- `dplm2_order_cogen_ci.png`
- `dcm_dentate_metrics.csv`
- `dcm_dentate_paired_deltas.csv`
- `dcm_dentate_metrics.png`
- `genmol_denovo_bootstrap_summary.csv`
- `genmol_fragment_bootstrap_summary.csv`
- `genmol_denovo_bootstrap_metrics.png`
- `genmol_fragment_quality_bootstrap.png`
- `genmol_fragment_validity_bootstrap.png`

These are the preferred artifacts for README-level reporting.

## Legacy aggregate exports

The root-level CSVs and PNGs are earlier journal-style exports for DMPO, Prism, and preliminary PUMA/DPLM summaries. They are retained because they document the uncertainty policy and contain useful baselines:

- `dmpo_summary.csv`
- `prism_summary.csv`
- `puma_summary.csv`
- `dplm_forward_summary.csv`
- `dplm_cogen_summary.csv`

## SDPO-DNA DNA sequence design

The artifacts under [`sdpo/`](./sdpo) report an ordering ablation for reward-guided discrete diffusion DNA design (SDPO). Four ordering variants (baseline SDPO, progressive, DPRM-confidence, DPRM-random) are compared on HepG2 expression prediction, ATAC accessibility accuracy, 3-mer Pearson correlation, log-likelihood, and a composite total metric. Each method generates 640 DNA sequences; 1000 bootstrap iterations produce 95% confidence intervals.

When a newer file exists under `latest/`, use the `latest/` file.

## GenMol V2 pilot

The GenMol V2 artifacts under [`genmol/`](./genmol) and `latest/` report a pilot ordering ablation for molecular SAFE diffusion. De novo generation uses `1,000` samples per method. Fragment-constrained evaluation uses the same stable subset for every method: seven fragment examples across five tasks with one generated sample per example and task. Three fragment rows from the upstream demo CSV are skipped because GenMol's native `fragment_linking` sampler exits at the native-library level for at least one ordering checkpoint. This filtering is method-independent and is documented so the pilot is interpretable rather than presented as a full GenMol benchmark reproduction.

## Uncertainty policy

- Use paired bootstrap whenever the same evaluation units are observed under two methods.
- Use ordinary bootstrap for single-model summaries.
- Use Wilson intervals only when the host did not save per-example outcomes.

For new integrations, save per-example evaluation artifacts by default so paired bootstrap can be computed later.
