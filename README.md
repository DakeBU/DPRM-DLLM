# DPRM: Token Ordering as a Control Axis for Diffusion Language Models

Official implementation of [**DPRM: Token Ordering as a Control Axis for
Diffusion Language Models**](https://arxiv.org/abs/2604.24357), a plug-in
ordering controller for masked discrete diffusion models. DPRM keeps the host
denoiser, token sampler, training labels, and task objective fixed. It changes
which eligible token positions are committed next.

![DPRM overview](DPRM1.png)

## Results At A Glance

All entries compare DPRM with the matched confidence-order baseline while
holding the host model and evaluation protocol fixed.

| Setting | Confidence | DPRM | Relative improvement |
|---|---:|---:|---:|
| PUMA, GSM8K mean accuracy | 29.34 | 34.27 | **+16.8%** |
| DMPO, MATH-Hard mean pass@K | 44.3 | 47.9 | **+8.1%** |
| DMPO, Countdown-Hard mean pass@K | 29.6 | 33.4 | **+12.8%** |
| LLaDA-V, AI2D accuracy | 0.658 | 0.692 | **+5.2%** |

On strict RealWorldQA, the prompt-defined numeric/count class improves by
`8.97` percentage points and transfers to ChartQA numeric questions with a
`3.45` point gain. Scientific integrations expose declared preference vectors
for recovery--sparsity, molecular quality--diversity, and DNA
expression--accessibility Pareto control.

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
- **Matched visual DPRM** uses confidence and spatial cells learned from
  development rollouts. Random, confidence, and DPRM branches train from the
  same checkpoint on states induced by their deployed order. Each test prompt
  produces one image; no reward model or completed-image selection is used at
  inference.

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

| Host | Domain | Ordered object | DPRM integration | Upstream |
|---|---|---|---|---|
| Omni-Diffusion | text-to-image | visual codebook position | [`integrations/omni_diffusion`](integrations/omni_diffusion) | [VITA-MLLM/Omni-Diffusion](https://github.com/VITA-MLLM/Omni-Diffusion) |
| LLaDA-V | image-conditioned VQA | answer-token position | [`integrations/llada_v`](integrations/llada_v) | [ML-GSAI/LLaDA-V](https://github.com/ML-GSAI/LLaDA-V) |
| PUMA | language pretraining | response-token position | [`integrations/puma`](integrations/puma) | [JaeyeonKim01/PUMA](https://github.com/JaeyeonKim01/PUMA) |
| DMPO | reasoning post-training | response-token position | [`integrations/dmpo`](integrations/dmpo) | [yuchen-zhu-zyc/DMPO](https://github.com/yuchen-zhu-zyc/DMPO) |
| Prism | test-time scaling | reveal/remask position | [`integrations/prism`](integrations/prism) | [viiika/Prism](https://github.com/viiika/Prism) |
| DPLM-2 Bit | protein diffusion | residue position | [`integrations/dplm`](integrations/dplm) | [bytedance/dplm](https://github.com/bytedance/dplm) |
| DCM | single-cell diffusion | gene-expression bin position | [`integrations/dcm`](integrations/dcm) | [sanjukta7/aivc-dcm](https://github.com/sanjukta7/aivc-dcm) |
| GenMol V2 | molecular diffusion | SAFE-token position | [`integrations/genmol`](integrations/genmol) | [NVIDIA-Digital-Bio/genmol](https://github.com/NVIDIA-Digital-Bio/genmol) |
| SDPO | DNA reward optimization | DNA-token position | [`integrations/sdpo`](integrations/sdpo) | [hanjq17/discrete-diffusion-sdpo](https://github.com/hanjq17/discrete-diffusion-sdpo) |

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

## Result Files

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

```bibtex
@article{bu2026dprm,
  title   = {DPRM: Token Ordering as a Control Axis for Diffusion Language Models},
  author  = {Bu, Dake and Huang, Wei and Han, Andi and Wu, Si and Wong, Hau-San and Zhang, Qingfu and Suzuki, Taiji and Nitanda, Atsushi},
  journal = {arXiv preprint arXiv:2604.24357},
  year    = {2026}
}
```

Machine-readable citation metadata is in [`CITATION.cff`](CITATION.cff).

## License

The DPRM core is released under Apache-2.0. Host overlays remain subject to the
licenses of their upstream projects; see [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md).
