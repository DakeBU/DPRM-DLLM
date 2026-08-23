# DPRM-DPLM-2 Bit

DPRM-DPLM preserves the DPLM-2 Bit backbone, sequence/structure tokenizers,
multimodal losses, data, optimizer, and 500-step sampler. The action is a masked
residue position. The two multi-objective utilities combine provisional
amino-acid recovery and structure-token recovery by a normalized weighted sum
or smooth augmented Tchebycheff scalarization.

## Four Registered Orders

- host random/default masking (`model.order.enable=false`);
- confidence-progressive training and confidence decoding;
- weighted-sum DPRM;
- Tchebycheff DPRM.

## Paper Configuration

The matched runs use `5000` updates, eight phases, `16` confidence bins, one
modality-specific sequence/structure stream, reward temperature `8`, guidance
`1`, warmup `500`, switch `2000`, readiness `128`, and shortlist
`min(32, max(8, 4*m_t))`. Multi-objective weights are `(0.5,0.5)`;
Tchebycheff temperature and augmentation are `0.05`.

## Overlay and Evaluation

The tested upstream commit and all four executable commands are recorded under
`dplm` in
[`../../reproducibility/experiments.json`](../../reproducibility/experiments.json).
Set `DPLM_ROOT` to the upstream checkout and `DPLM2_BIT_CHECKPOINT` to the
pretrained DPLM-2 Bit checkpoint. Copy `overlay/src/` and `overlay/configs/` into
the upstream checkout, then use `overlay/run/train_dprm_dplm2_bit_650m_fair.sh`
or the Hydra commands in the registry. Set
`DPLM_CONFIG=dprm_joint_tcheby_dplm_650m` for the Tchebycheff controller; the
weighted controller is used otherwise. CAMEO uses all `163` targets. CoGen-200
is a predeclared 10-sample development gate; the release reports it as a diagnostic
because neither scalarization passed the bootstrap threshold for full
five-length confirmation.
For CAMEO, save one metric row per shared target and use
`scripts/paired_bootstrap.py --direction lower` for RMSD or the default
`--direction higher` for TM-score. Co-generation samples are unpaired and are
therefore bootstrapped by the host evaluator rather than this paired utility.

The released CoGen-200 records reproduce the bounded development gate with:

```bash
python integrations/dplm/scripts/summarize_dplm_cogen_records.py \
  --baseline confidence.csv --candidate ws=weighted_sum.csv \
  --candidate tcheby=tchebycheff.csv --bootstrap 5000 --seed 20260813 \
  --reference-summary development_gate.json
```
