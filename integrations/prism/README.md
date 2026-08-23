# DPRM-Prism

DPRM-Prism preserves Prism's hierarchical trajectory search, branching width,
pruning cadence, verifier, and survivor budget. The controller orders reveal
and remask positions inside HTS. Terminal utility is the verifier score already
computed by the search.

## Order Policies

- `ORDER_POLICY=random`.
- `ORDER_POLICY=confidence`.
- `ORDER_POLICY=dprm_soft_bon`, using confidence warmup.
- `ORDER_POLICY=dprm_random`, using random warmup.

## Paper Configuration

LLaDA-2.0-mini on GSM8K; initial width `16`, final survivors `4`, survivor count
`2`, decay `1.8`, pruning interval `3`, block length `32`, `32` generation steps,
length `256`, and temperature `0.7`. DPRM uses `8` phases, `16` confidence bins,
`beta=1`, warmup `0.2T`, switch `0.7T`, readiness `64`, and shortlist
`min(64, max(8, 4*m_t))`.

## Overlay

The tested upstream commit and all four executable commands are recorded under
`prism` in
[`../../reproducibility/experiments.json`](../../reproducibility/experiments.json).
The release includes Dream, LLaDA, and LLaDA-2.0-mini HTS overlays. The paper
command is `overlay/LLaDA2mini/LLaDA2mini_Prism/scripts/run_gsm8k.sh`. Set
`PRISM_ROOT` to the corresponding upstream checkout; Dream also requires
`DREAM_MODEL_PATH`. `GPU_IDS`, `MODEL_PATH`, and `BASE_OUTPUT_PATH` are optional
overrides. Run the variants in the registry and report accuracy together with
mean NFE and verifier calls.
Retain the question-level vote correctness, survivor correctness, and NFE
records. `scripts/paired_bootstrap.py` recomputes each confidence--DPRM paired
interval after choosing the corresponding scalar field with `--value`.

The released raw records can be reduced to the paper table and paired intervals
without the upstream plotting stack:

```bash
python integrations/prism/scripts/summarize_prism_records.py \
  --confidence "$ARTIFACT_ROOT/prism/gsm8k/confidence_res.jsonl" \
  --dprm "$ARTIFACT_ROOT/prism/gsm8k/dprm_res.jsonl" \
  --output-dir results/prism
```
