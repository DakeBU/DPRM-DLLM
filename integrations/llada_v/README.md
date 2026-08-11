# DPRM-LLaDA-V

LLaDA-V conditions a diffusion language model on an encoded image. DPRM orders
masked answer-token positions; token identities still come from the same
image-conditioned logits. The bucket key contains decode phase, confidence,
prompt-only answer format, candidate EOT status, and relative answer position.
Terminal utility is task-normalized VQA correctness.

## Order Policies

- `random`;
- `progressive_confidence`;
- `dprm_confidence_warmup`;
- `dprm_random_warmup`.

Entropy and EOT/suffix-anchor policies are mechanism controls, not DPRM
variants.

## Paper Configuration

AI2D uses `500` documents. RealWorldQA uses documents `0:128` for table fitting,
`128:256` for controller selection, and `256:765` for the strict held-out
interval. The selected table uses one phase, `8` confidence bins, four relative
position bins, prompt-format and candidate-EOT state, guidance `4`, and the
symmetric zero-count normalization. The table and guidance are frozen before
evaluation on `500` ChartQA documents.

## Reproduction

1. Copy `overlay/dprm_generation.py` and `overlay/host/` into the LLaDA-V
   checkout.
2. Run random and confidence policies with order tracing.
3. Build the table with `scripts/build_dprm_table.py`.
4. Select only on the development interval with `scripts/select_controller.py`.
5. Evaluate `dprm_confidence_warmup` on the held-out interval and use
   `audit_chartqa_transfer.py` for frozen transfer.

Set `DPRM_LLADAV_TABLE` to the selected JSON table. A DPRM policy without a
table raises an error unless explicit diagnostic fallback is enabled. The
compact task and class-level statistics are in
[`../../results/artifacts/multimodal_summary.json`](../../results/artifacts/multimodal_summary.json).
