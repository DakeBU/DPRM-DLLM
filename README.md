# DPRM: A Plug-in Token-Ordering Module for Diffusion Language Models

Official implementation of [**DPRM: A Plug-in Token-Ordering Module for
Diffusion Language Models**](https://arxiv.org/abs/2604.24357), a plug-in
ordering controller for masked discrete diffusion models. DPRM keeps the host
architecture, token-value rule, training labels, and task objective fixed. It
changes which eligible token positions are committed next.

![DPRM overview](DPRM1.png)

**[Project page](https://dakebu.github.io/DPRM-DLLM/)** ·
**[Paper](https://arxiv.org/abs/2604.24357)** ·
**[Release artifacts](https://huggingface.co/DakeBU/DPRM-DLLM)**

## Results At A Glance

All entries compare DPRM with the matched confidence-order baseline while
holding the architecture, objective, token-value sampler, and evaluation
protocol fixed. Training comparisons use the same initialization and optimizer
settings, with order-induced partial states as the controlled difference.

| Setting | Confidence | DPRM | Relative improvement |
|---|---:|---:|---:|
| PUMA, GSM8K accuracy | 30.10 | 34.50 | **+14.6%** |
| Omni-Diffusion, CLIP-L/14 | 0.18661 | 0.21125 | **+13.2%** |
| Omni-Diffusion, CLIP-B/32 check | 0.23836 | 0.24854 | **+4.3%** |

Only results backed by retained states, per-example records, and the artifact
manifest are listed above. DMPO MATH and Countdown include the archived paired
pass@32 matrices used by the paper; their confidence-to-DPRM mean pass@K changes
are `+2.10` and `+1.67` percentage points, respectively. The release metadata
keeps these records separate from the reconstructed protocol checkpoint. The
public reducer intentionally omits tasks without retained paired matrices.

On strict RealWorldQA, the prompt-defined numeric/count class improves by
`8.97` percentage points, with a positive paired-bootstrap interval. The
preregistered AI2D confirmation is released as a non-promoted diagnostic.
Scientific integrations expose declared preferences for
single-cell reconstruction and molecular QED--synthetic-accessibility benefit;
the DNA integration separately measures how HepG2-guided order changes ATAC,
k-mer correlation, and reference likelihood.

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
- **Online visual DPRM** evaluates five predeclared position actions from one
  shared Omni-Diffusion canvas: the native confidence action and four
  confidence-rank strata. CLIP-L/14 supplies the terminal action value;
  CLIP-B/32 is held out as an independent semantic check. The selected path
  differs from the host path in one position action, while the checkpoint,
  provisional token values, seed, tokenizer, and continuation rule are fixed.

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
Each registry entry also declares the natural evaluation unit and the command
that rebuilds uncertainty from raw per-unit records.

Evaluators retain one record per prompt, question, target, cell, molecule, or
sequence. When an upstream evaluator does not compute uncertainty,
`scripts/paired_bootstrap.py` joins two CSV/JSON/JSONL files by shared unit id
and writes the paired mean difference, 95% percentile interval, and
win/tie/loss counts. Nested JSON fields use dot notation.
For DMPO success matrices, `scripts/bootstrap_passk.py` recomputes the paired
interval of the reported mean pass@K statistic.

```bash
python scripts/paired_bootstrap.py \
  --baseline confidence.jsonl --method dprm.jsonl \
  --key example_id --value correct --scale 100 \
  --output paired_accuracy.json
```

Run the release audit before launching a host experiment:

```bash
python scripts/verify_release.py
python scripts/reproduce.py --list
python scripts/reproduce.py --host llada_v --variant dprm_confidence --dry-run
```

The command registry checks implementation coverage. The separate artifact
manifest records the retained model/controller states and selected raw records
needed to audit the paper, including byte counts and SHA-256 digests:

```bash
python scripts/verify_artifact_manifest.py
python scripts/verify_artifact_manifest.py \
  --artifact-root "$DPRM_ARTIFACT_ROOT" --require-complete
python scripts/audit_artifact_semantics.py \
  --artifact-root "$DPRM_ARTIFACT_ROOT"
```

See [`reproducibility/release_artifacts.json`](reproducibility/release_artifacts.json).
An entry remains pending until the corresponding reported state is available;
the registry does not treat a runnable command as a substitute for that state.
The semantic audit opens the retained states and checks their phase, bucket,
gate, shortlist, frozen-model, and scientific-preference settings.
The complete code and artifact publication sequence is in
[`docs/RELEASE.md`](docs/RELEASE.md).

Use `--execute` only after setting the host-specific root and checkpoint
environment variables shown by the dry run. Execution is refused unless the
host checkout matches the 40-character commit recorded in the registry. Every
run writes `<HOST_ROOT>/dprm_run_manifests/<host>_<variant>.json` with the exact
command, upstream revision, registry hash, integration-code hash, UTC times,
and exit status. Use `--manifest-out` to place this record beside a particular
experiment output.

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
  title   = {DPRM: A Plug-in Token-Ordering Module for Diffusion Language Models},
  author  = {Bu, Dake and Huang, Wei and Han, Andi and Wu, Si and Wong, Hau-San and Zhang, Qingfu and Suzuki, Taiji and Nitanda, Atsushi},
  journal = {arXiv preprint arXiv:2604.24357},
  year    = {2026}
}
```

Machine-readable citation metadata is in [`CITATION.cff`](CITATION.cff).

## License

The DPRM core is released under Apache-2.0. Host overlays remain subject to the
licenses of their upstream projects; see [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md).
