# DPRM-DCM

DPRM-DCM preserves DCM's SEDD-style transformer, count-bin preprocessing,
denoising loss, optimizer, and Dentate Gyrus split. The action is a masked gene
position. Confidence is the provisional expression-bin probability. The
multi-objective utility separates nonzero-bin recovery, nonzero-bin error, and
zero-expression accuracy so sparse zeros are not counted in all three terms.

## Order Policies

- `random_ordered`;
- `confidence`;
- `dprm_confidence` or `dprm_random` with `reconstruction_weighted_sum`;
- `dprm_confidence` or `dprm_random` with `reconstruction_tchebycheff`.

`scripts/run_preference_sweep.sh` provides a training-time scalarization
control with nonzero-recovery, nonzero-MAE, balanced, and zero preferences.
Candidates are additionally partitioned by whether the denoiser's provisional
expression token is zero or nonzero. This inference-observable auxiliary bin
prevents the abundant zero-expression decisions from dominating the reward
statistics of rare nonzero decisions.
The preference vectors and axis-wise response criteria are declared in
`reproducibility/scientific_preference_sweeps.json` before evaluation.

The paper configuration uses
`overlay/scripts/calibrate_dcm_terminal_order.py`. It freezes the confidence model,
branches from shared training states with confidence, predicted-nonzero,
predicted-zero, and random reveal sets, completes each branch with the same
decoder, and fits the phase/confidence/predicted-zero table from terminal
smooth-Tchebycheff utility. Model parameters and validation cells are not used
by this calibration.

## Paper Configuration

Top `5000` variable genes, `32` requested quantile bins (eight populated bins
in the evaluated Dentate artifact), hidden size `128`, `4` layers, and `4`
heads. The fixed Progressive-DCM checkpoint is shared by confidence and every
DPRM preference. Calibration uses the first `256` training cells, branch steps
`0,8,16,24`, four decode phases, `16` confidence bins, two provisional
zero/nonzero bins, `beta=1`, and readiness `64`. It updates only the controller
table.

Guidance is selected from `0.5,1,2,4` on the next `96` training cells by mean
declared utility gain, with activity and positive-endpoint gates. Evaluation
then decodes each of `293` held-out cells from all masks for `32` steps, with
four samples per cell and `5000` paired-bootstrap draws. Two additional decode
seeds check directional replication without participating in selection; their
cell bootstraps remain separate rather than being pooled.
The evaluator fixes the data partition with `--split-seed 42`; `--seed`
controls sampling and bootstrap randomness only. Development sweeps use
`--split train --cell-offset 256` and formal evaluation uses `--split val`, so
the formal cells are opened once after the development gate.

## Reproduction

Copy `overlay/` into the upstream DCM checkout, run
`scripts/calibrate_dcm_terminal_order.py` on the training split, select guidance
on a disjoint training subset, and pass the four calibrated controller
checkpoints to `scripts/eval_dcm_ordering_bootstrap.py`. The evaluator writes
per-cell paired summaries and controller-activity diagnostics.
