# DPRM-DCM

DPRM-DCM preserves DCM's SEDD-style transformer, count-bin preprocessing,
denoising loss, optimizer, and Dentate Gyrus split. The action is a masked gene
position. Confidence is the provisional expression-bin probability. Utilities
combine token recovery, expression-bin error, and zero-expression behavior.

## Order Policies

- `random_ordered`;
- `confidence`;
- `dprm_confidence` with `reconstruction_weighted_sum`;
- `dprm_random` with `reconstruction_tchebycheff`.

## Paper Configuration

Top `5000` variable genes, `32` requested quantile bins (eight populated bins
in the evaluated Dentate artifact), hidden size `128`, `4` layers, `4` heads,
batch size `8`, learning rate `1e-4`, weight decay `0.01`, and `50` epochs.
DPRM uses one phase, `16` confidence bins, `beta=1`, warmup `100`, switch
`500`, readiness `32`, and shortlist `min(64, max(8, 4*m_t))`.
Evaluation decodes each of `293` held-out cells from all masks for `32` steps,
with four samples per cell and `5000` bootstrap draws.

## Reproduction

Copy `overlay/` into the upstream DCM checkout and run the four registry
commands. `scripts/eval_dcm_ordering_bootstrap.py` accepts all four checkpoint
specifications and writes per-cell paired summaries.
