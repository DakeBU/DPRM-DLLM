# Host Integrations

Each directory contains the overlay files used to insert DPRM into one upstream
host. The upstream architecture, token-value sampler, objective, data pipeline,
and evaluator remain host-owned. The overlay controls only the order in which
eligible positions are revealed, retained, remasked, or branched.

| Integration | Ordered item | Primary utility |
|---|---|---|
| [`puma`](puma) | text token | teacher-forced token log-probability |
| [`dmpo`](dmpo) | text token | verified reasoning reward |
| [`prism`](prism) | text/search token | self-verification score |
| [`dplm`](dplm) | protein residue | amino-acid/structure recovery |
| [`dcm`](dcm) | gene-expression token | reconstruction utility |
| [`genmol`](genmol) | SAFE token | molecular reconstruction objectives |
| [`sdpo`](sdpo) | DNA token | GOSAI regulatory utility |
| [`omni_diffusion`](omni_diffusion) | visual codebook position | CLIP/aesthetic terminal utility |
| [`llada_v`](llada_v) | answer-token position | task-normalized VQA correctness |

The authoritative commands are in
[`../reproducibility/experiments.json`](../reproducibility/experiments.json).
Use `python scripts/reproduce.py --host HOST --variant VARIANT --dry-run` from
the repository root to inspect a command and its required upstream root.
