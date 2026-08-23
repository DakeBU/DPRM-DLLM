#!/usr/bin/env python3
"""Inspect or execute a registered DPRM host command."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = Path(
    os.environ.get("DPRM_EXPERIMENT_REGISTRY", ROOT / "reproducibility" / "experiments.json")
).expanduser().resolve()


def load_experiments() -> dict[str, dict]:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {row["id"]: row for row in payload["experiments"]}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        if "__pycache__" in file_path.parts or file_path.suffix in {".pyc", ".pyo"}:
            continue
        relative = file_path.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(file_path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--host")
    parser.add_argument("--variant")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--manifest-out",
        help="Run-manifest path. Defaults to <HOST_ROOT>/dprm_run_manifests/<host>_<variant>.json.",
    )
    args = parser.parse_args()
    experiments = load_experiments()

    if args.list:
        for experiment in experiments.values():
            variants = ", ".join(row["id"] for row in experiment["variants"])
            print(f"{experiment['id']}: {variants}")
        return
    if not args.host or not args.variant:
        parser.error("use --list or provide --host and --variant")
    if args.host not in experiments:
        parser.error(f"unknown host: {args.host}")

    experiment = experiments[args.host]
    variants = {row["id"]: row for row in experiment["variants"]}
    if args.variant not in variants:
        parser.error(f"unknown variant for {args.host}: {args.variant}")
    variant = variants[args.variant]
    root_env = experiment["root_env"]
    host_root = os.environ.get(root_env)
    execution_root = variant.get("execution_root", experiment["execution_root"])
    working_subdir = Path(
        variant.get("working_subdir", experiment.get("working_subdir", "."))
    )
    if execution_root == "release":
        working_directory = (ROOT / working_subdir).resolve()
    elif execution_root == "host" and host_root:
        working_directory = (Path(host_root).expanduser().resolve() / working_subdir).resolve()
    else:
        working_directory = None

    print(f"host: {experiment['host']}")
    print(f"variant: {variant['id']} ({variant['policy']})")
    print(f"result status: {variant['status']}")
    print(f"integration: {experiment['integration_readme']}")
    print(f"expected upstream commit: {experiment['upstream_commit']}")
    print(f"evaluation unit: {experiment['evaluation_unit']}")
    print(f"statistics command: {experiment['statistics_command']}")
    required_env = [root_env, *experiment.get("required_env", [])]
    required_env = list(dict.fromkeys(required_env))
    print(f"required environment: {', '.join(required_env)}")
    print(
        "working directory: "
        + (str(working_directory) if working_directory else f"${root_env}/{working_subdir}")
    )
    print(f"command: {variant['command']}")

    if args.execute:
        missing_env = [name for name in required_env if not os.environ.get(name)]
        if missing_env:
            parser.error(
                "set required environment variables before execution: "
                + ", ".join(missing_env)
            )
        host_path = Path(host_root).expanduser().resolve()
        try:
            actual_commit = git_commit(host_path)
        except (OSError, subprocess.CalledProcessError) as error:
            parser.error(f"cannot read git commit from {host_path}: {error}")
        expected_commit = experiment["upstream_commit"]
        if actual_commit != expected_commit:
            parser.error(
                f"upstream commit mismatch for {args.host}: "
                f"{actual_commit} != {expected_commit}"
            )

        manifest_path = (
            Path(args.manifest_out).expanduser().resolve()
            if args.manifest_out
            else host_path
            / "dprm_run_manifests"
            / f"{args.host}_{args.variant}.json"
        )
        integration_path = ROOT / Path(experiment["integration_readme"]).parent
        started_at = datetime.now(timezone.utc).isoformat()
        manifest = {
            "schema_version": 1,
            "host_id": args.host,
            "host": experiment["host"],
            "variant": variant["id"],
            "policy": variant["policy"],
            "status": "running",
            "started_at_utc": started_at,
            "upstream": experiment["upstream"],
            "upstream_commit": actual_commit,
            "command": variant["command"],
            "execution_root": execution_root,
            "working_subdir": working_subdir.as_posix(),
            "registry_sha256": sha256_file(REGISTRY),
            "integration_sha256": sha256_tree(integration_path),
            "required_environment_names": required_env,
        }
        write_json_atomic(manifest_path, manifest)
        result = subprocess.run(
            variant["command"],
            shell=True,
            check=False,
            cwd=working_directory,
            executable="/bin/bash",
        )
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["return_code"] = result.returncode
        manifest["status"] = "complete" if result.returncode == 0 else "failed"
        write_json_atomic(manifest_path, manifest)
        print(f"run manifest: {manifest_path}")
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    elif not args.dry_run:
        print("not executed; pass --execute after applying the integration overlay")


if __name__ == "__main__":
    main()
