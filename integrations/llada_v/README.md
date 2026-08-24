# DPRM-LLaDA-V

LLaDA-V conditions a discrete diffusion model on an encoded image. DPRM orders
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

RealWorldQA uses documents `0:128` for table fitting, `128:256` for controller
selection, and `256:765` for confirmation. Its selected controller has one
phase, eight confidence bins, guidance `4`, prompt-format state, candidate-EOT
state, and four relative-position bins. The strict numeric/count subset contains
`78` held-out documents and is defined from prompt text before scoring.

The preregistered AI2D protocol uses `0:128`, `128:256`, and `256:500` for the
same three roles. It selected one phase, eight confidence bins, guidance `8`,
candidate-EOT state, and four relative-position bins. The independent interval
did not improve over confidence, so AI2D is retained as a non-promoted
diagnostic rather than a paper result.

## Reproduction

The tested upstream commit and all four executable commands are recorded under
`llada_v` in
[`../../reproducibility/experiments.json`](../../reproducibility/experiments.json).
Set `LLADA_V_LMMS_ROOT`, `LLADA_V_MODEL_PATH`, and
`LLADA_V_OUTPUT_ROOT`. The evaluator verifies the pinned commit and applies
`overlay/` with `apply_overlay.sh` before launch. The complete RealWorldQA
fit/development/confirmation protocol is:

```bash
bash integrations/llada_v/run_realworldqa_protocol.sh
```

The disjoint AI2D diagnostic protocol is:

```bash
bash integrations/llada_v/run_ai2d_protocol.sh
```

`run_lmms_eval.sh` is the lower-level frozen-model entrypoint. It writes the
task, order, table, gate, and document limit to a run
manifest before invoking lmms-eval.

Set `DPRM_LLADAV_TABLE` to the selected JSON table. A DPRM policy without a
table raises an error unless explicit diagnostic fallback is enabled. The
compact task and class-level statistics are in
[`../../results/artifacts/multimodal_summary.json`](../../results/artifacts/multimodal_summary.json).

Rebuild task accuracies, prompt-format slices, and `5000`-draw paired-bootstrap
intervals directly from the saved lmms-eval samples with:

```bash
python integrations/llada_v/scripts/summarize_multimodal_results.py \
  --rwqa-confidence outputs/rwqa/confidence.jsonl \
  --rwqa-dprm outputs/rwqa/dprm_confidence.jsonl \
  --rwqa-doc-min 256 --rwqa-doc-max 765 \
  --base-summary results/artifacts/multimodal_summary.json \
  --output outputs/llada_v_multimodal_summary.json
```

The script pairs by `doc_id`, rejects incomplete declared intervals, and uses
only prompt text to define choice, numeric, and open-answer subsets.

## Qualitative Gallery

The gallery contains all seven DPRM-only wins in the strict held-out
numeric/count class. The document ids are fixed in
[`../../reproducibility/llada_v_qualitative_gallery.json`](../../reproducibility/llada_v_qualitative_gallery.json).
After downloading the official RealWorldQA evaluation split, rebuild both
gallery pages with:

```bash
python integrations/llada_v/scripts/render_qualitative_gallery.py \
  --confidence-records outputs/rwqa/confidence.jsonl \
  --dprm-records outputs/rwqa/dprm_confidence.jsonl \
  --manifest reproducibility/llada_v_qualitative_gallery.json \
  --output-dir "$HOME/outputs/llada_v_qualitative_gallery"
```

The renderer checks that every listed document is incorrect under confidence
order and correct under DPRM using the same target-normalized evaluator as the
reported paired audit.
