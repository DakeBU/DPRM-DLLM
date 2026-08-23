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
The formal Countdown command must set `DATASET_JSONL` to the complete
5,120-row test JSONL. When `TEST_SIZE=5120`, the evaluator rejects the 256-row
`countdown_cd3_test` fallback so a single-difficulty diagnostic cannot be
reported as the full benchmark.

The Hugging Face bundle keeps one reconstructed Countdown DPRM adapter and the
complete paired confidence/DPRM success matrices retained for MATH and
Countdown:

```bash
python integrations/dmpo/scripts/package_release.py \
  --repro-root "$DMPO_REPRO_ROOT" \
  --release-root "$DPRM_ARTIFACT_ROOT"
```

The packager rejects incomplete success matrices, strips machine-local paths,
and writes byte counts and SHA-256 digests. The archived GSM8K per-example DPRM
matrix was not retained, so the release makes no GSM8K paper-result claim.
GSM8K remains runnable from the command registry below.

When retained paper matrices were produced by an earlier checkpoint copy, pass
an explicit source map instead of placing them under the reconstruction root:

```bash
python integrations/dmpo/scripts/package_release.py \
  --repro-root "$DMPO_REPRO_ROOT" \
  --release-root "$DPRM_ARTIFACT_ROOT" \
  --record-source-map "$DMPO_RECORD_SOURCE_MAP"
```

Each map entry declares its record directory, provenance label, and whether the
packaged adapter is verified as the direct source. Unverified archived records
do not receive the rebuilt adapter hash in their metadata. This preserves the
paper evidence without asserting a checkpoint-to-record link that was not
retained.

After downloading the artifact bundle, regenerate the paper table and curves
directly from the retained matrices:

```bash
python integrations/dmpo/scripts/render_matched_results.py \
  --repro-root /tmp/unused \
  --record-source-map reproducibility/dmpo_record_sources.json \
  --artifact-root "$DPRM_ARTIFACT_ROOT" \
  --output results/artifacts/dmpo_matched_step5000_summary.json

python integrations/dmpo/scripts/plot_matched_passk.py \
  --source-map reproducibility/dmpo_record_sources.json \
  --artifact-root "$DPRM_ARTIFACT_ROOT" \
  --task math --output passk_math_all_levels.png
python integrations/dmpo/scripts/plot_matched_passk.py \
  --source-map reproducibility/dmpo_record_sources.json \
  --artifact-root "$DPRM_ARTIFACT_ROOT" \
  --task countdown --output passk_countdown_all_levels.png
```

Before evaluation, compare the two configurations:

```bash
python integrations/dmpo/scripts/verify_matched_training_args.py \
  --source-root "$DMPO_ROOT" \
  --confidence "$CONFIDENCE_CHECKPOINT" \
  --dprm "$DPRM_CHECKPOINT" \
  --output matched_training_args.json
```

The paper commands are task-specific:

```bash
bash DMPO/run_paper_dmpo.sh math dprm_confidence
bash DMPO/run_paper_dmpo.sh countdown dprm_confidence
```

The first argument is `gsm8k`, `math`, `countdown`, or `all`; the second is
`random`, `confidence`, `dprm_confidence`, or `dprm_random`. The wrapper fixes
eight rollout generations, per-device and generation batch size four, and
5,000 updates. GSM8K and Countdown use learning rate `1e-6` and gradient
accumulation two. MATH uses `3e-6` and gradient accumulation four. Progressive
policies use confidence-collapse threshold `0.9`.

`MAX_STEPS=5000` is explicit in the launcher. On a Slurm node the script uses
`srun accelerate`; on a workstation it invokes `accelerate` directly. Set
`NUM_PROCESSES` to the number of GPUs assigned to the run.

## Overlay

Synchronize the complete `overlay/` directory into the upstream checkout while
preserving its relative paths, for example with
`rsync -a integrations/dmpo/overlay/ "$DMPO_ROOT/"`. This installs the DMPO
trainer/controller files and the DPRM-aware Fast-dLLM sampler together.
`run_paper_dmpo.sh` fixes the paper configuration and calls `run_dmpo.sh`.
The launcher uses `dmpo_train_compat.py`, which treats a missing tensor-parallel
plan as an empty plan when recent Transformers versions warm up the distributed
CUDA allocator; it then executes the unchanged upstream training entry point.
The scripts in `overlay/eval/` require
explicit `RANDOM_RUN_DIR` and `PROGRESSIVE_RUN_DIR`; no machine-local checkpoint
path is assumed.

The exact four commands are listed under `dmpo` in
[`../../reproducibility/experiments.json`](../../reproducibility/experiments.json).
The evaluator should retain its Boolean example-by-sample success matrices.
`scripts/bootstrap_passk.py` recomputes the paired interval for the arithmetic
mean over `K={1,2,4,8,16,32}` directly from those matrices.

For multi-GPU evaluation, each task evaluator accepts disjoint
`--sample_idx_start`/`--sample_idx_end` ranges. Each range uses the same seed
formula as the serial run, so sharding changes only scheduling. Write each
range to a separate output directory, then construct the canonical matrix with:

```bash
python integrations/dmpo/scripts/merge_passk_sample_shards.py \
  --shards "$SHARD_ROOT"/shard_* \
  --output-dir "$CANONICAL_OUTPUT"
```

The merger rejects overlaps, missing columns, partial explicit shards, and any
metadata or example-order mismatch.
