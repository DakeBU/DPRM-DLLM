# Release Checklist

Before pushing this repository publicly:

- confirm the final author metadata in `CITATION.cff`;
- confirm the final repository URL in `README.md`;
- verify that no checkpoints, W&B runs, datasets, or generated sample dumps are tracked;
- verify that no local absolute paths, private cache paths, or machine-specific launcher commands remain in public docs;
- verify that generated Python caches are absent with `find . -name __pycache__ -o -name '*.pyc'`;
- confirm each integration overlay is compatible with the upstream host license;
- run `python examples/minimal_usage.py`;
- run `python examples/build_bucket_table_from_traces.py --help`;
- run `python -m compileall src examples`;
- optionally add CI for linting and import tests.

Suggested GitHub repository name: `DPRM-DLLM`.
