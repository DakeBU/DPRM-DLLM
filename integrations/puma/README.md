# DPRM-PUMA Overlay

`DPRM-PUMA` keeps PUMA's teacher-forced progressive unmasking objective and train-test alignment logic, but upgrades the reveal order from top-confidence to an online DPRM Soft-BoN policy.

## Host Mapping

- `confidence`: token confidence from the current denoiser.
- `candidate_mask`: currently masked positions.
- `phase_ids`: progressive unmasking phase or decode-step bucket.
- `aux_bin_ids`: unused in the current PUMA fork.
- `rewards`: teacher-forced self-supervised reconstruction payoff or a lightweight decode-time pseudo-reward.

## What Changed In The Local Fork

- `src/dprm/adapters/puma.py` is now the shared adapter that maps PUMA's legacy reveal-order API to the generic `OnlineDPRMController`;
- `progressive.py` and `progressive_block.py` can instantiate the adapter-backed DPRM controller;
- `sampling.py` uses the same controller at decode time for aligned ordering;
- YAML configs expose `order_policy=dprm_soft_bon` and the controller hyperparameters.

## Codex / Claude Guidance

Ask the assistant to preserve:

- PUMA's admissible teacher-forced forward process;
- the train-test occupancy matching argument;
- the existing reveal budget schedule.

Only the ordering score should change from confidence-only to confidence plus online continuation correction.
