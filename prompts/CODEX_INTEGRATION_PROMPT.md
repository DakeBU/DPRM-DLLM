# Codex Prompt Template

You are integrating the DPRM token-ordering module into an existing masked-generation or diffusion codebase.

Repository root:
- `<HOST_REPO_PATH>`

Task description:
- `<NATURAL_LANGUAGE_DESCRIPTION_OF_THE_TASK>`

Your job is to keep the host model, objective, evaluation protocol, and budget unchanged, and only replace the token-ordering policy with DPRM where appropriate.

Use the generic controller in `src/dprm/controller.py`.
Use `src/dprm/adapters/` when the host already follows a known pattern such as progressive teacher-forced unmasking or HTS-style test-time search.

Before editing code, inspect the repository and identify the exact ordering hook points.
You should explicitly locate:

- where the host computes confidence or proposal probabilities;
- where it picks high-confidence tokens, low-confidence tokens, or survivor trajectories;
- where a phase, decode step, or search progress counter already exists;
- where a utility signal already exists and can be reused for DPRM updates;
- where config flags or shell arguments should be added so the baseline remains runnable.

Map the host algorithm into the following DPRM contract:

1. `confidence`
   The host's current base proposal score per candidate token. Prefer token confidence or proposal probability unless the host already has a better proposal.
2. `candidate_mask`
   Which positions are eligible for reveal / remask / branch / verification at the current step.
3. `phase_ids`
   A progress bucket derived from the host schedule, such as progressive unmasking phase, decode step bucket, or search-stage bucket.
4. `aux_bin_ids`
   Optional task-specific bucket, such as structure bucket, verifier bucket, or difficulty bucket.
5. `rewards`
   A utility already produced by the host algorithm, such as teacher-forced reconstruction utility, reward-model score, amino-acid recovery, or verifier score.

Implementation requirements:

- keep the original host defaults available;
- add a switch so the user can choose between the original ordering and DPRM;
- preserve train-test alignment if the host already uses progressive teacher-forced masking;
- wire DPRM into both training-time masked-state construction and test-time ordering when the host supports both;
- avoid nested rollout or extra expensive oracle calls unless the host already pays for them.
- if the host is a test-time scaling method, keep search width, branching cadence, verifier logic, and pruning budget fixed;
- if the host is a training algorithm, keep the loss and clean target fixed and only modify masked-state occupancy through ordering.

Deliverables:

- the concrete code edits;
- one config or shell entry-point for the DPRM run;
- a short README note explaining which host files changed and why.
- the shortest baseline command and the shortest DPRM command.
