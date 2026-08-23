# DPRM-GenMol V2

DPRM-GenMol preserves GenMol V2's bracket-SAFE tokenizer, model, denoising
objective, checkpoint format, sampling temperatures, and RDKit evaluation. The
action is a masked SAFE-token position. The controller supports selected-token
confidence and multi-objective molecular reconstruction utilities.

## Order Policies

- `random_mdlm`: native GenMol V2 order;
- `confidence`: confidence-progressive order;
- `dprm_confidence`: weighted multi-objective DPRM;
- `dprm_random`: random-warmup DPRM.

The controller combines sample-level QED and normalized
synthetic-accessibility benefit with either a weighted sum or augmented smooth
Tchebycheff scalarization. Set `SCALARIZATION=weighted_sum` or
`SCALARIZATION=smooth_tchebycheff` when running
`scripts/run_preference_sweep.sh`; each run trains the declared extreme QED and
SA preferences together with the balanced preference. The same manifest declares the bidirectional response
criterion. Validity, uniqueness, and diversity remain evaluation axes because
they are not all meaningful as per-sample training rewards.
Each candidate also uses an eight-way auxiliary class computed from its
provisional SAFE token: separator/special, topology/bond, carbon, nitrogen,
oxygen, other heteroatom, halogen, or other. The tokenizer-derived map is the
same during training and decoding.

## Paper Configuration

Eight phases, `16` confidence bins, eight provisional token-class bins,
`beta=1`, guidance `1`, warmup `500`, switch `2000`, readiness `128`, shortlist
`min(64, max(8, 4*m_t))`, `5000` training updates,
and `1000` de novo generation attempts per method. Reported metrics are QED,
normalized synthetic-accessibility benefit, validity, uniqueness, and diversity
on native scales. The release reports preference-response intervals and Pareto
relations without collapsing these axes into a post-hoc geometric mean.
The reported preference checkpoints use random warmup during training. At
evaluation, populated cells use the learned reward correction and under-ready
cells use confidence as the safe fallback; the decode configuration records
this fallback policy separately from the training warmup.

## Reproduction

The tested upstream commit and all four executable commands are recorded under
`genmol` in
[`../../reproducibility/experiments.json`](../../reproducibility/experiments.json).
Set `GENMOL_ROOT`, copy `overlay/src/` and `overlay/configs/` into the upstream
checkout, then run
`SCALARIZATION=smooth_tchebycheff GPU_LIST=0,1,2 bash integrations/genmol/scripts/run_preference_sweep.sh`.
Evaluate each endpoint with
`evaluate_ordering.py` and aggregate with `aggregate_ordering_eval.py`.
Molecular objectives remain separate in the output. Preference vectors are
declared before training rather than fitted to the evaluation set.
`evaluate_ordering.py` saves one record per generated molecule and
`aggregate_ordering_eval.py` computes bootstrap confidence intervals over
those records.

The release archive `genmol/records/denovo_1000.tar.zst` contains the confidence,
QED, balanced, and SA `denovo_raw.csv` files together with their evaluator
summaries. The confidence checkpoint and the three preference checkpoints are
listed separately in `reproducibility/release_artifacts.json`.
