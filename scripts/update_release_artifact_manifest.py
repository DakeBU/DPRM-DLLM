#!/usr/bin/env python3
"""Atomically merge one completed host fragment into the release manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--fragment", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-policy",
        type=Path,
        default=Path("reproducibility/hf_checkpoint_policy.json"),
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.host not in manifest.get("hosts", {}):
        raise SystemExit(f"unknown manifest host: {args.host}")
    fragment = json.loads(args.fragment.read_text(encoding="utf-8"))
    artifacts = fragment.get("artifacts", [])
    if not artifacts:
        raise SystemExit("host fragment has no artifacts")
    ids = [str(artifact.get("id", "")) for artifact in artifacts]
    if any(not artifact_id for artifact_id in ids) or len(ids) != len(set(ids)):
        raise SystemExit("host fragment has missing or duplicate artifact ids")
    paths = [str(artifact.get("path", "")) for artifact in artifacts]
    for relative in paths:
        pure = PurePosixPath(relative)
        if not relative or pure.is_absolute() or ".." in pure.parts:
            raise SystemExit(f"unsafe artifact path: {relative}")

    policy = json.loads(args.checkpoint_policy.read_text(encoding="utf-8"))
    primary = policy["hosts"][args.host].get("primary_artifact_id")
    if primary is not None and primary not in ids:
        raise SystemExit(f"fragment does not contain primary artifact {primary}")

    manifest["hosts"][args.host] = {
        "status": "complete",
        "artifacts": artifacts,
    }
    temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.manifest)


if __name__ == "__main__":
    main()
