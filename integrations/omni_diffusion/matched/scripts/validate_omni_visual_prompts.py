#!/usr/bin/env python3
"""Verify fixed Omni visual prompts before formal generation starts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def unique_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            content = str(row["messages"][0]["content"])
            prompt = "\n".join(content.split("\n")[1:]).strip()
            if prompt and prompt not in seen:
                seen.add(prompt)
                prompts.append(prompt)
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--eval-offset", type=int, required=True)
    parser.add_argument("--eval-count", type=int, required=True)
    parser.add_argument("--expected-prompt-ids", type=int, nargs=4, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source = unique_prompts(args.data)
    resolved = []
    for row in manifest.get("prompts", []):
        prompt_id = int(row["prompt_id"])
        if not args.eval_offset <= prompt_id < args.eval_offset + args.eval_count:
            raise ValueError(f"visual prompt {prompt_id} is outside the evaluation split")
        text = source[prompt_id]
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if text != row["text"] or digest != row["sha256"]:
            raise ValueError(f"visual prompt {prompt_id} differs from preregistration")
        resolved.append({"prompt_id": prompt_id, "sha256": digest, "text": text})
    if len(resolved) != 4 or len({row["prompt_id"] for row in resolved}) != 4:
        raise ValueError("formal visual preregistration must contain four unique prompts")
    if [row["prompt_id"] for row in resolved] != args.expected_prompt_ids:
        raise ValueError("runtime visual prompt ids differ from preregistration")
    result = {
        "passed": True,
        "selection_rule": manifest.get("selection_rule"),
        "eval_offset": args.eval_offset,
        "eval_count": args.eval_count,
        "prompts": resolved,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
