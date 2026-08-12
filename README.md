# DPRM: Token Ordering as a Control Axis for Diffusion Language Models

Official implementation of **DPRM**, a plug-in ordering controller for masked
discrete diffusion models. DPRM keeps the host denoiser, token sampler,
training labels, and task objective fixed. It changes which eligible token
positions are committed next.

## Method

For a partial state `s`, the host supplies a proposal `q0(a | s)` over reveal
actions and DPRM targets the Doob-transformed action law

```text
pi*(a | s) proportional to q0(a | s) exp(beta R*(a; s)),
R*(a; s) = beta^-1 log E[exp(beta R(X_T)) | s, a].
```

The repository implements two estimators of the conditional future utility:

- **Online bucket DPRM** maps each candidate to a generation `phase_id`, a
  `confidence_bin`, and an optional low-dimensional auxiliary bin. Counts and
  exponentiated terminal rewards provide a log-moment value for each cell.
  Readiness gates reduce unsupported cells to the host confidence order.
- **Action-conditioned DPRM-BoN** evaluates a fixed shortlist of actions from
  the same partial state, completes each branch with the same host decoder, and
  selects by terminal utility. This is used for Omni-Diffusion where a complete
  visual rollout supplies a direct action-value comparison.

`src/dprm/` contains the host-independent controller, multi-objective reward
scalarization, visual-order helpers, and host adapters. `integrations/` contains
the exact overlay files and commands used for the nine host settings.

For normalized maximization benefits `y_j` and a declared preference vector
`lambda`, the scientific integrations support the weighted utility
`sum_j lambda_j y_j` and an augmented smooth-Tchebycheff utility

```text
1 - mu * logsumexp(lambda_j * (1 - y_j) / mu)
    + rho * sum_j lambda_j y_j.
```

The latter is the maximization counterpart of [Smooth Tchebycheff
Scalarization](https://proceedings.mlr.press/v235/lin24y.html). The paper
configuration uses `mu=0.05` and `rho=0.05`. Preference vectors are declared in
[`reproducibility/scientific_preference_sweeps.json`](reproducibility/scientific_preference_sweeps.json)
before evaluation.

## Install

```bash
git clone https://github.com/DakeBU/DPRM-DLLM.git
cd DPRM-DLLM
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Host experiments additionally require the corresponding upstream repository,
model checkpoint, data, and evaluator. Each integration README identifies the
upstream revision surface and the files that must be overlaid.

## Minimal Use

```python
import torch
from dprm import DPRMConfig, HostDPRMBatch, OnlineDPRMController

controller = OnlineDPRMController(
    DPRMConfig(num_phases=8, confidence_bins=16, ready_count=64)
)
host = HostDPRMBatch(
    confidence=torch.tensor([[0.25, 0.80, 0.55]]),
    candidate_mask=torch.tensor([[True, True, True]]),
    phase_ids=torch.tensor([2]),
    global_step=100,
)
selection = controller.select(host, num_select=torch.tensor([1]))
```

The host remains responsible for predicting token values. DPRM returns only
the selected position mask and ordering diagnostics.

## Experiments

| Host | Domain | Ordered object | Integration |
|---|---|---|---|
| Omni-Diffusion | text-to-image | visual codebook position | [`integrations/omni_diffusion`](integrations/omni_diffusion) |
| LLaDA-V | image-conditioned VQA | answer-token position | [`integrations/llada_v`](integrations/llada_v) |
| PUMA | language pretraining | response-token position | [`integrations/puma`](integrations/puma) |
| DMPO | reasoning post-training | response-token position | [`integrations/dmpo`](integrations/dmpo) |
| Prism | test-time scaling | reveal/remask position | [`integrations/prism`](integrations/prism) |
| DPLM-2 Bit | protein diffusion | residue position | [`integrations/dplm`](integrations/dplm) |
| DCM | single-cell diffusion | gene-expression bin position | [`integrations/dcm`](integrations/dcm) |
| GenMol V2 | molecular diffusion | SAFE-token position | [`integrations/genmol`](integrations/genmol) |
| SDPO | DNA reward optimization | DNA-token position | [`integrations/sdpo`](integrations/sdpo) |

The machine-readable experiment registry is
[`reproducibility/experiments.json`](reproducibility/experiments.json). It lists
the tested upstream commit, four executable order settings per host, and whether each setting is a
paper result, a matched control, or an implemented reproduction mode. Numerical
paper results are stored in [`results/paper_results.csv`](results/paper_results.csv),
with protocol notes in [`results/README.md`](results/README.md).

Run the release audit before launching a host experiment:

```bash
python scripts/verify_release.py
python scripts/reproduce.py --list
python scripts/reproduce.py --host llada_v --variant dprm_confidence --dry-run
```

Use `--execute` only after setting the host-specific root and checkpoint
environment variables shown by the dry run.

## Result Highlights

- PUMA GSM8K mean accuracy at the shared `1.53M` EMA checkpoint is `29.34` for
  confidence and `34.27` for DPRM.
- DMPO-DPRM improves MATH Hard mean pass@K from `44.3` to `47.9` and Countdown
  Hard from `29.6` to `33.4`.
- Omni DPRM-BoN-2/4 reach CLIP-L/14 `0.28302/0.28708`, compared with `0.26791`
  for confidence, and beat uniform action selection at the same rollout budget.
- LLaDA-V reaches `0.692` on AI2D. On strict RealWorldQA, DPRM changes `0.4735`
  to `0.4892`; the prespecified numeric/count class improves by `8.97` points.
  The frozen controller transfers to ChartQA numeric questions with a `3.45`
  point gain.
- DPRM(random)-SDPO gives the highest DNA total metric (`2.119`), ATAC
  success (`0.754`), and k-mer Pearson (`0.842`).

The scientific-domain package reports declared preference vectors, native-scale
objective axes, bootstrap response intervals, and Pareto comparisons. DCM
separates expressed-gene fidelity from zero preservation; GenMol separates QED
and synthetic accessibility from set-level validity, uniqueness, and diversity.

## Repository Layout

```text
src/dprm/             host-independent DPRM implementation
integrations/         upstream overlays and experiment commands
reproducibility/      four-order registry for all nine hosts
results/              canonical paper tables and compact audit artifacts
tests/                controller and integration unit tests
scripts/              release verification and command launcher
```

## Citation

See [`CITATION.cff`](CITATION.cff). The paper title is *DPRM: Token Ordering as
a Control Axis for Diffusion Language Models*.

## License

The DPRM core is released under Apache-2.0. Host overlays remain subject to the
licenses of their upstream projects; see [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md).
