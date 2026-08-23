#!/usr/bin/env python3
"""Create deterministic, disjoint development and confirmation prompt files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(text: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}\0{text}".encode()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--development-output", required=True, type=Path)
    parser.add_argument("--confirmation-output", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    parser.add_argument("--development-count", type=int, default=64)
    parser.add_argument("--salt", default="dprm-omni-geneval-v1")
    args = parser.parse_args()

    prompts = [line.strip() for line in args.source.read_text(encoding="utf-8").splitlines()]
    prompts = [prompt for prompt in prompts if prompt]
    if len(prompts) != len(set(prompts)):
        raise SystemExit("prompt source must contain unique nonempty lines")
    if not 0 < args.development_count < len(prompts):
        raise SystemExit("development count must leave a nonempty confirmation split")

    ranked = sorted(prompts, key=lambda prompt: (digest(prompt, args.salt), prompt))
    development_set = set(ranked[: args.development_count])
    development = [prompt for prompt in prompts if prompt in development_set]
    confirmation = [prompt for prompt in prompts if prompt not in development_set]
    for path, values in (
        (args.development_output, development),
        (args.confirmation_output, confirmation),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(values) + "\n", encoding="utf-8")

    manifest = {
        "design": "salted-SHA256 prompt-text split before generation",
        "source": str(args.source),
        "source_sha256": file_sha256(args.source),
        "salt": args.salt,
        "source_count": len(prompts),
        "development_count": len(development),
        "confirmation_count": len(confirmation),
        "development_output": str(args.development_output),
        "development_sha256": file_sha256(args.development_output),
        "confirmation_output": str(args.confirmation_output),
        "confirmation_sha256": file_sha256(args.confirmation_output),
        "intersection_count": len(set(development) & set(confirmation)),
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
