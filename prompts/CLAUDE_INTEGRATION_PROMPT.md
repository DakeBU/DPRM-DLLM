# Claude Prompt Template

Please adapt my masked-generation or diffusion codebase so that token ordering uses DPRM instead of the host's current random or confidence-only policy.

Codebase:
- `<HOST_REPO_PATH>`

Problem description:
- `<NATURAL_LANGUAGE_DESCRIPTION>`

Constraints:

- do not redesign the model architecture;
- do not change the denoising objective or task supervision;
- only change the ordering rule used to reveal, remask, branch on, or verify tokens;
- keep the original ordering mode as a baseline option.
- keep the host search budget, verifier, and branching logic fixed when the host is an inference-time scaling method.

Use the reusable controller in `src/dprm/controller.py` and map the host code to this interface:

- `confidence`: per-position proposal confidence;
- `candidate_mask`: positions eligible for action now;
- `phase_ids`: progress or schedule bucket;
- `aux_bin_ids`: optional task-specific bucket;
- `rewards`: utility already produced by the host pipeline.

The desired policy is:

1. warm up with the host's original confidence policy;
2. preserve or add progressive train-test alignment if the host is teacher-forced;
3. smoothly transition to online DPRM with bucketized continuation estimates and optional Soft-BoN shortlist selection.

Before writing code, inspect the repository and name the concrete hook points:

- confidence computation;
- token or trajectory selection;
- phase / decode-step / search-stage tracking;
- utility or reward computation;
- CLI or config entry points for preserving the baseline mode.

Please return:

- the patch;
- the new config flags;
- the exact files touched;
- the shortest baseline command;
- the shortest command needed to run the DPRM variant.
