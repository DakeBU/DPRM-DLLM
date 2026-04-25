# Release Checklist

Before pushing this repository publicly:

- confirm the final author metadata in `CITATION.cff`;
- confirm the final repository URL in `README.md`;
- verify that no checkpoints, W&B runs, datasets, or generated sample dumps are tracked;
- confirm each integration overlay is compatible with the upstream host license;
- run `python examples/minimal_usage.py`;
- run `python -m compileall src examples`;
- optionally add CI for linting and import tests.

Suggested GitHub repository name: `DPRM-DLLM`.
