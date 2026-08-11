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

The Tchebycheff utility is available through
`DPRM_REWARD_MODE=reconstruction_tchebycheff` for the scalarization audit.

## Paper Configuration

Eight phases, `16` confidence bins, `beta=1`, warmup `500`, switch `2000`,
readiness `128`, shortlist `min(64, max(8, 4*m_t))`, `5000` training updates,
and `1000` de novo generation attempts per method. Reported metrics are
validity, QED/SA quality, uniqueness, diversity, and their geometric mean on
native scales.

## Reproduction

Set `GENMOL_ROOT`, copy `overlay/src/` and `overlay/configs/` into the upstream
checkout, and run `overlay/scripts/run_ordering_train.sh`. Evaluate with
`evaluate_ordering.py` and aggregate with `aggregate_ordering_eval.py`.
Molecular objectives remain separate in the output; the evaluator does not
declare a uniform winner from a post-hoc scalar score.
