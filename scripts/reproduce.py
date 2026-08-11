#!/usr/bin/env python3
"""Inspect or execute a registered DPRM host command."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "reproducibility" / "experiments.json"


def load_experiments() -> dict[str, dict]:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {row["id"]: row for row in payload["experiments"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--host")
    parser.add_argument("--variant")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
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

    print(f"host: {experiment['host']}")
    print(f"variant: {variant['id']} ({variant['policy']})")
    print(f"result status: {variant['status']}")
    print(f"integration: {experiment['integration_readme']}")
    print(f"required environment: {root_env}")
    print(f"command: {variant['command']}")

    if args.execute:
        if not host_root:
            parser.error(f"set {root_env} to the prepared upstream host checkout")
        subprocess.run(
            variant["command"],
            shell=True,
            check=True,
            cwd=Path(host_root).expanduser().resolve(),
            executable="/bin/bash",
        )
    elif not args.dry_run:
        print("not executed; pass --execute after applying the integration overlay")


if __name__ == "__main__":
    main()
