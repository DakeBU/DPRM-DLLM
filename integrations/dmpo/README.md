# DPRM-DMPO Overlay

`DPRM-DMPO` keeps DMPO's reward-tilted clean target, WDCE loss, replay reuse, and LoRA fine-tuning setup fixed. The only algorithmic change is token ordering inside progressive teacher-forced masked states and aligned decode-time remasking.

## Host Mapping

- `confidence`: max token probability at each masked position.
- `candidate_mask`: masked non-prompt positions eligible for reveal.
- `phase_ids`: progressive unmasking phase.
- `aux_bin_ids`: unused in the current DMPO implementation.
- `rewards`: the sequence-level DMPO reward already computed by the trainer.

## What Changed In The Local Fork

- training-time progressive unmasking can use `dprm_soft_bon` instead of pure confidence;
- the trainer updates an online DPRM estimator during teacher-forced reveal;
- evaluation can decode with aligned `dprm_soft_bon` remasking;
- pass@K plotting scripts can merge DPRM curves into the baseline figures.

## Codex / Claude Guidance

When adapting another DMPO-like codebase, tell the assistant:

- keep the clean-sequence target distribution untouched;
- replace only the masked-state sampler and decode ordering;
- reuse the host reward already computed for WDCE or policy weighting;
- save the DPRM estimator next to checkpoints so evaluation can load it later.
