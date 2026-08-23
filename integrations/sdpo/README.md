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

The tested upstream commit and all four executable commands are recorded under
`sdpo` in
[`../../reproducibility/experiments.json`](../../reproducibility/experiments.json).
Set `SDPO_DNA_ROOT`, copy the overlay files into the upstream checkout, and run
`scripts/run_sdpo_dna_variant.sh` for each registry policy. Use
`run_sdpo_dna_eval_compare.sh` and `summarize_sdpo_dna_results.py` to regenerate
the table. `eval_dna_bootstrap.py` writes the sample count, controller coverage,
and bootstrap confidence interval for every reported DNA metric. It also writes
`eval_records.npz`, containing all 640 sequences, per-sequence oracle values,
the bootstrap indices, and the nonlinear k-mer bootstrap values. Verify a
released record file with:

```bash
python integrations/sdpo/scripts/summarize_sdpo_dna_records.py eval_records.npz \
  --reference-summary eval_bootstrap.json
```

The release bundle stores the confidence, DPRM-confidence, and DPRM-random
records in `sdpo/records/formal_640_raw_records.tar.zst`. After extraction, run
the command above in each method directory. The saved arrays reproduce the
reported means and intervals; differences from the original evaluator JSON are
limited to floating-point serialization in the last decimal place.
