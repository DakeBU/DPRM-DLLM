#!/usr/bin/env python3
"""Finalize the checksum-verified DPRM Hugging Face artifact bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_model_card(manifest: dict) -> str:
    rows = []
    for host, record in manifest["hosts"].items():
        artifacts = record.get("artifacts", [])
        links = ", ".join(
            f"[{artifact['id']}]({artifact['path']})" for artifact in artifacts
        )
        rows.append(f"| {host} | {links} |")
    table = "\n".join(rows)
    return f"""---
license: apache-2.0
library_name: pytorch
tags:
- diffusion-models
- masked-diffusion
- token-ordering
- process-reward-model
---

# DPRM Release Artifacts

This bundle contains the retained DPRM states and raw evaluation records for
**DPRM: A Plug-in Token-Ordering Module for Diffusion Language Models**.
The executable integrations, exact commands, and uncertainty reducers are in
[DakeBU/DPRM-DLLM](https://github.com/DakeBU/DPRM-DLLM). The paper is available
at [arXiv:2604.24357](https://arxiv.org/abs/2604.24357).

The bundle retains one deployable DPRM checkpoint or controller per host when
the host has trainable state. Baseline outputs are represented by raw matched
records and are regenerated with the public commands. `release_artifacts.json`
pins every reported file by byte size and SHA-256 digest.

| Host | Retained artifacts |
|---|---|
{table}

## Verification

```bash
git clone https://github.com/DakeBU/DPRM-DLLM.git
cd DPRM-DLLM
python scripts/verify_artifact_manifest.py \\
  --manifest "$DPRM_ARTIFACT_ROOT/release_artifacts.json" \\
  --artifact-root "$DPRM_ARTIFACT_ROOT" --require-complete
python scripts/audit_artifact_semantics.py \\
  --artifact-root "$DPRM_ARTIFACT_ROOT"
```

See each integration README in the code repository for upstream model and
dataset requirements. These artifacts do not replace the upstream licenses.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("reproducibility/release_artifacts.json"),
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    pending = [
        host
        for host, record in manifest["hosts"].items()
        if record.get("status") != "complete"
    ]
    if pending:
        raise SystemExit(f"refusing to prepare an incomplete bundle: {pending}")

    args.artifact_root.mkdir(parents=True, exist_ok=True)
    for host, record in manifest["hosts"].items():
        for artifact in record.get("artifacts", []):
            path = args.artifact_root / artifact["path"]
            if not path.is_file():
                raise SystemExit(f"missing {host} artifact: {path}")
            if path.stat().st_size != artifact["bytes"]:
                raise SystemExit(f"byte-size mismatch: {path}")
            if sha256(path) != artifact["sha256"]:
                raise SystemExit(f"SHA-256 mismatch: {path}")

    (args.artifact_root / "README.md").write_text(
        render_model_card(manifest), encoding="utf-8"
    )
    (args.artifact_root / "release_artifacts.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    for name in ("LICENSE", "NOTICE", "CITATION.cff"):
        source = args.repo_root / name
        if not source.is_file():
            raise SystemExit(f"missing repository release file: {source}")
        shutil.copy2(source, args.artifact_root / name)
    print(f"prepared complete Hugging Face bundle at {args.artifact_root}")


if __name__ == "__main__":
    main()
