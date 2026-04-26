# Anonymous Review Checklist

Before sharing this repository for anonymous review:

- keep author, affiliation, email, homepage, and personal repository metadata out of tracked files;
- keep review-facing repository URLs generic until an anonymous mirror URL is assigned;
- verify that no checkpoints, W&B runs, datasets, or generated sample dumps are tracked;
- confirm each integration overlay is compatible with the upstream host license;
- run `python examples/minimal_usage.py`;
- run `python -m compileall src examples`;
- optionally add CI for linting and import tests in the non-anonymous development repository.
