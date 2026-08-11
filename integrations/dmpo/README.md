# DPRM-DMPO

DPRM-DMPO preserves the LLaDA-8B-Instruct base, DMPO reward-tilted clean target,
WDCE loss, replay reuse, optimizer, rollout reward, and decode budget. DPRM
orders masked response positions during teacher-forced progressive training and
uses the saved bucket estimator for aligned decoding.

## Order Policies

- `LOSS_MASK_SAMPLER=random`: DMPO random masking.
- `LOSS_MASK_SAMPLER=progressive` and `LOSS_PROGRESSIVE_ORDER_POLICY=confidence`.
- `dprm_soft_bon` with `DPRM_WARMUP_POLICY=confidence`.
- `dprm_soft_bon` with `DPRM_WARMUP_POLICY=random`.

## Paper Configuration

Eight progressive phases, `16` confidence bins, `beta=1`, warmup `500`, switch
`2000`, readiness `128`, and shortlist `min(32, max(8, 4*m_t))`. The reasoning
runs use `5000` updates, `128` diffusion steps, generation length `256`, block
length `32`, and temperature `0.2`. Evaluation reports pass@K for
`K={1,2,4,8,16,32}`.

## Overlay

Copy `overlay/DMPO/` and `overlay/dprm_guidance.py` into the upstream checkout.
`run_dmpo.sh` trains a named policy. The scripts in `overlay/eval/` require
explicit `RANDOM_RUN_DIR` and `PROGRESSIVE_RUN_DIR`; no machine-local checkpoint
path is assumed.

The exact four commands are listed under `dmpo` in
[`../../reproducibility/experiments.json`](../../reproducibility/experiments.json).
