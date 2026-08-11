# DPRM-PUMA

DPRM-PUMA preserves PUMA's TinyGSM model, teacher-forced progressive denoising
loss, optimizer, data, and reveal budget. The ordered action is a masked response
position. Confidence is the model probability of its provisional token, and the
bucket utility is the mean teacher-forced log-probability on newly revealed
ground-truth tokens.

## Order Policies

- `random`: standard random masking.
- `confidence`: confidence-progressive PUMA.
- `dprm_soft_bon` with `warmup_policy=confidence`.
- `dprm_soft_bon` with `warmup_policy=random`.

## Paper Configuration

Hidden size `512`, `14` layers, `8` heads, batch size `32`, learning rate
`3e-4`, weight decay `0.01`, EMA `0.9999`, and `20` epochs. The progressive
horizon increases from `K=12` to `K=42` by step `330k`. DPRM uses `16`
confidence bins, `beta=1`, warmup `2k`, switch `60k`, readiness `128`, and a
shortlist of `min(64, max(8, 4*m_t))`. Evaluation uses the shared `1.53M` EMA
checkpoint, temperature `0`, and `unmasking_num` in `{2,3}`.

## Overlay

Copy `overlay/` into a prepared PUMA checkout, preserving relative paths. The
training entry points are `train.py` and `train_block.py`; validation uses
`sampling.py`. The saved checkpoint includes the DPRM bucket state. PUMA parses
a single `--cfg` YAML argument, so the random-warmup policy has its own
`tinygsm_puma_dprm_random.yaml` configuration.

The exact four commands are listed under `puma` in
[`../../reproducibility/experiments.json`](../../reproducibility/experiments.json).
