# Release Checklist

Before pushing this repository publicly, run:

```bash
python scripts/verify_release.py
python -m pytest -q
python examples/minimal_usage.py
python -m compileall -q src examples integrations
```

Then confirm:

- confirm the author metadata in `CITATION.cff`;
- confirm the repository URL in `README.md`;
- verify that no checkpoints, W&B runs, datasets, or generated sample dumps are tracked;
- verify that no local absolute paths, private cache paths, or machine-specific launcher commands remain in public docs;
- verify that generated Python caches are absent with `find . -name __pycache__ -o -name '*.pyc'`;
- confirm each integration overlay is compatible with the upstream host license;
- the Omni formal rows come from the matched one-path promotion artifact, not
  the completed-path action-search diagnostic;
- every reported result has a registry command and immutable protocol tag;
- all tests and release checks above pass.

Suggested GitHub repository name: `DPRM-DLLM`.
