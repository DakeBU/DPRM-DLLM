# Integration Patch Maps

This directory contains host-specific patch maps extracted from the local research forks. It is documentation-first, not the main runtime entrypoint.

- [dmpo](./dmpo): reward-aware post-training overlay.
- [puma](./puma): pretraining overlay with progressive teacher-forced unmasking.
- [dplm](./dplm): protein diffusion overlay for DPLM / DPLM-2 Bit.
- [prism](./prism): test-time scaling overlay for HTS-based decoding.
- [dcm](./dcm): single-cell gene-expression discrete diffusion overlay for DCM.

Each folder contains:

- `README.md`: what DPRM changes in that host algorithm.
- `overlay/`: a minimal patch snapshot or bridge file, not a full standalone host implementation.
- adaptation notes for Codex or Claude.

If you want to reproduce a full experiment, clone the upstream host project and apply the corresponding overlay. If you want to port DPRM into a new codebase with Codex or Claude, start from the closest patch map here and keep the host's original baseline mode available.
