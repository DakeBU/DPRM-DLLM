# DPRM-Prism Overlay

`DPRM-Prism` is a pure test-time scaling overlay. It leaves HTS, local branching, pruning cadence, verifier usage, and budget accounting fixed, and only changes the ranking heuristic used inside the search.

## Host Mapping

- `confidence`: token confidence from the host diffusion model.
- `candidate_mask`: tokens eligible for reveal or remask inside HTS.
- `phase_ids`: decode-step bucket or HTS progress bucket.
- `aux_bin_ids`: optional verifier or search-state bucket.
- `rewards`: verifier signal or terminal candidate quality already produced by Prism.

## What Changed In The Local Fork

- `src/dprm/adapters/prism.py` is the shared adapter that maps HTS-style search controllers to the generic DPRM core;
- the HTS samplers can instantiate an online DPRM Soft-BoN controller through that adapter;
- early search remains confidence-dominated;
- later search reranks candidates using online continuation estimates;
- low-score remasking uses DPRM scores instead of only low confidence.

## Codex / Claude Guidance

Ask the assistant to preserve:

- the exact search budget and branching schedule;
- the verifier and self-checking logic;
- the original confidence policy as a fallback mode.

The only module that should change is token ordering inside the HTS inner loop.

A concise instruction that works well is:

> Integrate DPRM into this Prism-style repository. Keep HTS, branching width, pruning cadence, verifier calls, and total budget fixed. Only replace the confidence-based token ranking and low-confidence remasking policy with a confidence-to-DPRM transition plus DPRM Soft-BoN shortlist selection. Preserve the original `confidence` mode.

The assistant should return:

- the exact HTS sampler file(s) changed;
- the config or shell argument that switches between `confidence` and `dprm_soft_bon`;
- one baseline command and one DPRM command with identical search hyperparameters.
