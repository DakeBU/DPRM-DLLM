# DPRM-GenMol Patch Map

This integration targets [NVIDIA-Digital-Bio/genmol](https://github.com/NVIDIA-Digital-Bio/genmol), specifically the GenMol V2 SAFE / bracket-SAFE diffusion workflow described in the [GenMol paper](https://arxiv.org/abs/2501.06158).

The intended intervention is narrow:

- keep the GenMol V2 tokenizer, SAFE representation, model architecture, denoising objective, checkpoint format, and RDKit evaluation unchanged;
- replace only the ordering policy that chooses which masked molecular tokens are generated or committed next;
- preserve the original GenMol ordering as a baseline flag;
- log per-sample molecular validity, QED, SA, uniqueness, and fragment-task metrics so bootstrap intervals can be computed after generation.

Recommended controller variants:

- `baseline`: original GenMol V2 reveal order.
- `progressive`: confidence-ranked reveal order throughout training / decoding.
- `dprm_random`: random warmup followed by online DPRM Soft-BoN.
- `dprm_confidence`: confidence warmup followed by online DPRM Soft-BoN.

The current training integration uses a cheap self-supervised reconstruction-confidence utility. For property-targeted training or test-time constrained generation, the same controller can instead use molecular utilities already computed by the host, such as validity, QED/SA quality, fragment retention, or task-specific oracle score.

The pilot results in `statistics_outputs/genmol/` should be read as an ordering diagnostic, not a full reproduction of the GenMol V2 benchmark. They show task-dependent behavior: GenMol V2 remains strongest on de novo quality and uniqueness, while ordering-aware variants improve selected fragment-constrained metrics.

Current result snapshot:

- DPRM(random)-GenMol has the highest de novo validity, `0.997`.
- Progressive-GenMol has the highest de novo diversity, `0.853`.
- DPRM(random)-GenMol improves linker-design validity from `0.142` to `0.429` and linker-onestep validity from `0.430` to `0.573`.
- Progressive/DPRM-confidence improve motif-extension quality from `0.280` to `0.421` and scaffold-decoration quality from `0.429` to `0.712`.
