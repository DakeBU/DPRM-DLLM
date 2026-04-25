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

These are the preferred artifacts for README-level reporting.

## Legacy aggregate exports

The root-level CSVs and PNGs are earlier journal-style exports for DMPO, Prism, and preliminary PUMA/DPLM summaries. They are retained because they document the uncertainty policy and contain useful baselines:

- `dmpo_summary.csv`
- `prism_summary.csv`
- `puma_summary.csv`
- `dplm_forward_summary.csv`
- `dplm_cogen_summary.csv`

When a newer file exists under `latest/`, use the `latest/` file.

## Uncertainty policy

- Use paired bootstrap whenever the same evaluation units are observed under two methods.
- Use ordinary bootstrap for single-model summaries.
- Use Wilson intervals only when the host did not save per-example outcomes.

For new integrations, save per-example evaluation artifacts by default so paired bootstrap can be computed later.
