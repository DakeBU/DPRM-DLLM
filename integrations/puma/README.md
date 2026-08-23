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
shortlist of `min(64, max(8, 4*m_t))`. Evaluation uses the retained `2.00M` EMA
pair, temperature `0`, and `unmasking_num=3`. The released per-question records
and reveal-order analysis cover all `1,319` GSM8K questions.

## Evaluation

The paper comparison uses the two separately trained EMA endpoints at the same
2.00M step. After copying `overlay/` into the pinned PUMA checkout, run:

```bash
PUMA_CONFIDENCE_CHECKPOINT=ckpts/puma_confidence/ema_step=2000000.pt
PUMA_DPRM_CHECKPOINT=ckpts/puma_dprm/ema_step=2000000.pt

python eval_dprm_checkpoint.py \
  --cfg yaml_files/tinygsm_puma.yaml \
  --ckpt "$PUMA_CONFIDENCE_CHECKPOINT" \
  --confidence top_k --unmasking-num 3 \
  --output-dir outputs/puma_confidence_unmask3

python eval_dprm_checkpoint.py \
  --cfg yaml_files/tinygsm_puma_dprm.yaml \
  --ckpt "$PUMA_DPRM_CHECKPOINT" \
  --confidence dprm_soft_bon --unmasking-num 3 \
  --output-dir outputs/puma_dprm_unmask3
```

`eval_dprm_checkpoint.py` rejects a DPRM run if the checkpoint has no
`dprm_order_state`. The reported checkpoint stores a `39 x 16 x 1` table. Its
39 decode phases are restored from that table; the evaluator does not replace
it with a newly initialized table from the validation YAML. Each output summary
records the checkpoint SHA-256, table shape, and whether the DPRM state was
loaded.

## Overlay

Copy `overlay/` into a prepared PUMA checkout, preserving relative paths. The
training entry points are `train.py` and `train_block.py`; the paper evaluation
uses `eval_dprm_checkpoint.py` and `sampling.py`. The saved checkpoint includes
the DPRM bucket state. PUMA parses
a single `--cfg` YAML argument, so the random-warmup policy has its own
`tinygsm_puma_dprm_random.yaml` configuration.

The exact four commands are listed under `puma` in
[`../../reproducibility/experiments.json`](../../reproducibility/experiments.json).
Keep the per-question JSONL produced by validation and recompute the reported
paired interval with `scripts/paired_bootstrap.py`, using the question index as
`--key` and the final correctness bit as `--value`.

For the content-only reveal analysis used in the paper, save `trace_steps` with
the selected positions and decoded token text, then run:

```bash
python integrations/puma/scripts/analyze_reveal_order.py \
  --confidence outputs/topk/puma_topk_trace_*.jsonl \
  --dprm outputs/dprm/puma_dprm_trace_*.jsonl \
  --output outputs/puma_reveal_order_summary.json \
  --case-output outputs/puma_dprm_only_cases.jsonl
```

The script excludes EOT-only actions and reports paired accuracy, same-step
span, nonlocal-step rate, adjacency, centroid backfill, first numeric reveal,
and 5,000-resample paired intervals.

The artifact bundle stores these 1,319 paired traces as
`puma/traces/unmask3_all_1319.tar.zst`. Extract it with
`tar -I zstd -xf unmask3_all_1319.tar.zst`, then pass the two 21-file glob
expansions to the command above. This reproduces both the accuracy row and the
content-only reveal diagnostics from one retained endpoint.
