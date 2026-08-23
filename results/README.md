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
`artifacts/puma_reveal_order_summary.json` is regenerated from paired PUMA trace
files by `integrations/puma/scripts/analyze_reveal_order.py`; it contains the
content-only order diagnostics and their paired intervals.

Every `variant` in `paper_results.csv` is registered in
`reproducibility/experiments.json`. The four entries under `variants` are the
host-level order families. DCM and GenMol additionally declare
`result_variants` mappings for the fixed terminal-utility preferences reported
in their multi-objective sweep. Each preference label maps to its executable
parent order family. `scripts/verify_release.py` rejects unknown labels and
unknown parents so that a scalarization endpoint cannot be silently attributed
to a different order family.

The registered Omni protocol evaluates five position actions from a shared
step-96 canvas. Guidance is selected on 128 development prompts and fixed
before 512 disjoint confirmation prompts are opened. CLIP-L/14 supplies the
action value; CLIP-B/32 is an independent check. The report retains all five
paths, action metadata, shared-canvas hashes, and paired intervals.

The two paper examples form a separate checkpoint-500 mechanism diagnostic.
Their prompt, seed, candidate ranks, selected visual index, records, and
intermediate canvases are retained with the release artifact. They are not
included in development selection or the 512-prompt confirmation mean.

No row should be interpreted as a uniform-dominance claim across metrics.
DPLM-2 Bit, DCM, and GenMol explicitly expose terminal-invariance or
multi-objective trade-offs.
