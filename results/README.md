# Canonical Paper Results

`paper_results.csv` is the machine-readable source for the numerical results
reported by this release. Each row records one metric, its direction, evaluation
unit, sample count when available, and a protocol tag. Blank confidence limits
mean that the corresponding compact paper table reports a point estimate only.

The protocol tags have the following meanings:

- `shared_checkpoint`: methods use the same training step and evaluation set.
- `matched_pipeline`: the host model, objective, data, and budget are fixed; the
  ordering pipeline changes.
- `archived_diagnostic`: a mechanism or compute diagnostic that is not a paper
  endpoint.
- `heldout_after_dev`: hyperparameters are selected on a disjoint development
  interval and evaluated on the listed held-out interval.
- `frozen_transfer`: the controller is fixed before opening the target dataset.
- `development_gate`: a predeclared diagnostic gate; it is not promoted to a
  full benchmark claim.

Compact mechanism artifacts in `artifacts/` contain no model weights, raw
datasets, generated images, or private filesystem paths. Full regeneration uses
the commands in `reproducibility/experiments.json` and the upstream host assets.
`scripts/sync_scientific_results.py` rebuilds the DCM and GenMol rows from the
same native-value artifact used by the paper radar and supplementary table.

The completed-path Omni action-search artifact is retained separately as a
diagnostic. It is excluded from `paper_results.csv` because it uses terminal
continuations rather than the train--test-matched one-path controller evaluated
in the paper.

No row should be interpreted as a uniform-dominance claim across metrics.
DPLM-2 Bit, DCM, and GenMol explicitly expose terminal-invariance or
multi-objective trade-offs.
