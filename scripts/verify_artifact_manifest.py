#!/usr/bin/env python3
"""Validate or checksum the DPRM Hugging Face artifact bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("reproducibility/release_artifacts.json"),
    )
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--checkpoint-policy",
        type=Path,
        default=Path("reproducibility/hf_checkpoint_policy.json"),
    )
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    hosts = payload.get("hosts", {})
    if len(hosts) != 9:
        raise SystemExit(f"expected nine hosts, found {len(hosts)}")
    pending = []
    seen_paths: set[str] = set()
    checked = 0
    policy = json.loads(args.checkpoint_policy.read_text(encoding="utf-8"))
    policy_hosts = policy.get("hosts", {})
    if set(policy_hosts) != set(hosts):
        raise SystemExit("checkpoint policy hosts do not match artifact manifest")
    for host, entry in hosts.items():
        if entry.get("status") != "complete":
            pending.append(host)
        for artifact in entry.get("artifacts", []):
            relative = str(artifact["path"])
            pure = PurePosixPath(relative)
            if pure.is_absolute() or ".." in pure.parts:
                raise SystemExit(f"unsafe artifact path: {relative}")
            if relative in seen_paths:
                raise SystemExit(f"duplicate artifact path: {relative}")
            seen_paths.add(relative)
            digest = str(artifact.get("sha256", ""))
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise SystemExit(f"invalid SHA-256 for {host}/{artifact['id']}")
            if int(artifact.get("bytes", 0)) <= 0:
                raise SystemExit(f"invalid byte count for {host}/{artifact['id']}")
            if args.artifact_root is not None:
                path = args.artifact_root / pure
                if not path.is_file():
                    raise SystemExit(f"missing artifact: {path}")
                if path.stat().st_size != int(artifact["bytes"]):
                    raise SystemExit(f"size mismatch: {path}")
                if sha256(path) != digest:
                    raise SystemExit(f"checksum mismatch: {path}")
                checked += 1
        primary_id = policy_hosts[host].get("primary_artifact_id")
        artifact_ids = {artifact["id"] for artifact in entry.get("artifacts", [])}
        if entry.get("status") == "complete" and primary_id is not None:
            if primary_id not in artifact_ids:
                raise SystemExit(f"missing primary checkpoint for {host}: {primary_id}")
    if args.require_complete and pending:
        raise SystemExit("pending hosts: " + ", ".join(pending))
    print(
        json.dumps(
            {
                "hosts": len(hosts),
                "artifacts": len(seen_paths),
                "checked_files": checked,
                "pending_hosts": pending,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
