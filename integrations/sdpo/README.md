# DPRM-SDPO

DPRM-SDPO preserves the DNA diffusion backbone, substitution parameterization,
noise schedule, SDPO objective, oracle suite, data, and optimizer. The action is
a masked DNA position. The bucket utility is the sample-level HepG2 expression
score stored in `batch['clss'][:, 0]`. ATAC, high-expression k-mer alignment,
and reference log-likelihood are evaluation metrics.

## Order Policies

- `baseline`: native SDPO sampling;
- `progressive`: confidence order;
- `dprm`: confidence-warmup DPRM;
- `dprm_random`: random-warmup DPRM.

## Paper Configuration

SDPO temperature `0.5`, learning rate `1e-5`, two epochs, and `K=2000`. DPRM
uses one phase, `10` confidence bins, `beta=1`, warmup `100`, switch `400`,
readiness `64`, and shortlist size `64`. Evaluation generates `640` sequences
and uses `1000` bootstrap draws. Reference log-likelihood remains a separate
distributional-quality metric.

## Reproduction

Set `SDPO_DNA_ROOT`, copy the overlay files into the upstream checkout, and run
`scripts/run_sdpo_dna_variant.sh` for each registry policy. Use
`run_sdpo_dna_eval_compare.sh` and `summarize_sdpo_dna_results.py` to regenerate
the table.
