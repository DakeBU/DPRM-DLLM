#!/usr/bin/env python3
"""Merge matched-trajectory shards and create policy-specific Omni configs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def prompt_hash(row: dict) -> str:
    content = row["messages"][0]["content"].strip()
    lines = content.splitlines()
    prompt = "\n".join(lines[1:]).strip() if len(lines) > 1 else content
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def clean_target_hash(row: dict) -> str:
    """Hash the full clean conversation, including all visual target tokens."""
    payload = json.dumps(
        row["messages"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def merge(
    paths: list[Path],
    output: Path,
    *,
    expected_policy: str,
) -> tuple[int, set[str], set[tuple[int, int]], dict[tuple[int, int], str]]:
    count = 0
    hashes: set[str] = set()
    state_keys: set[tuple[int, int]] = set()
    target_hashes: dict[tuple[int, int], str] = {}
    with output.open("w", encoding="utf-8") as target:
        for path in paths:
            with path.open(encoding="utf-8") as source:
                for line in source:
                    row = json.loads(line)
                    observed_policy = row.get("dprm_trajectory_policy")
                    if observed_policy != expected_policy:
                        raise RuntimeError(
                            f"trajectory-policy mismatch in {path}: expected "
                            f"{expected_policy}, found {observed_policy}"
                        )
                    if "dprm_revealed_visual_indices" not in row:
                        raise RuntimeError(f"trajectory state missing reveal indices in {path}")
                    trajectory_step = int(row["dprm_trajectory_step"])
                    revealed_count = len(row["dprm_revealed_visual_indices"])
                    if revealed_count != trajectory_step + 1:
                        raise RuntimeError(
                            f"trajectory state has {revealed_count} revealed positions at "
                            f"post-action step {trajectory_step}; expected {trajectory_step + 1}"
                        )
                    target.write(json.dumps(row) + "\n")
                    hashes.add(prompt_hash(row))
                    key = (int(row["dprm_source_index"]), trajectory_step)
                    if key in state_keys:
                        raise RuntimeError(f"duplicate trajectory state {key}")
                    state_keys.add(key)
                    target_hashes[key] = clean_target_hash(row)
                    count += 1
    return count, hashes, state_keys, target_hashes


def write_config(path: Path, dataset_name: str, json_path: Path) -> None:
    payload = {
        "xlsx_sample_num": 5,
        "dataset": {
            dataset_name: {
                "ratio": 1,
                "json_paths": [str(json_path.resolve())],
                "prefix_path": "datasets/BLIP3o/",
            }
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confidence-shards", type=Path, nargs="+", required=True)
    parser.add_argument("--dprm-shards", type=Path, nargs="+", required=True)
    parser.add_argument("--random-shards", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--forbidden-prompts", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    confidence_json = args.output_dir / "confidence_matched.jsonl"
    dprm_json = args.output_dir / "dprm_matched.jsonl"
    random_json = args.output_dir / "random_matched.jsonl"
    confidence_count, confidence_hashes, confidence_keys, confidence_targets = merge(
        args.confidence_shards,
        confidence_json,
        expected_policy="progressive_confidence",
    )
    dprm_count, dprm_hashes, dprm_keys, dprm_targets = merge(
        args.dprm_shards,
        dprm_json,
        expected_policy="dprm_confidence_warmup",
    )
    random_count, random_hashes, random_keys, random_targets = merge(
        args.random_shards,
        random_json,
        expected_policy="random",
    )
    if not (
        confidence_count == dprm_count == random_count
        and confidence_hashes == dprm_hashes == random_hashes
        and confidence_keys == dprm_keys == random_keys
        and confidence_targets == dprm_targets == random_targets
    ):
        raise RuntimeError(
            "random, confidence, and DPRM trajectory shards do not share paired "
            "prompts, states, and clean targets"
        )
    overlap: set[str] = set()
    if args.forbidden_prompts:
        forbidden = {
            hashlib.sha256(line.strip().encode("utf-8")).hexdigest()
            for line in args.forbidden_prompts.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        overlap = confidence_hashes & forbidden
        if overlap:
            raise RuntimeError(f"training/evaluation prompt leakage: {len(overlap)} prompts")
    confidence_config = args.output_dir / "confidence_matched.yaml"
    dprm_config = args.output_dir / "dprm_matched.yaml"
    random_config = args.output_dir / "random_matched.yaml"
    write_config(confidence_config, "DPRM_Omni_confidence_matched", confidence_json)
    write_config(dprm_config, "DPRM_Omni_DPRM_matched", dprm_json)
    write_config(random_config, "DPRM_Omni_random_matched", random_json)
    manifest = {
        "format": "omni_matched_training_data_v1",
        "rows_per_policy": confidence_count,
        "unique_prompts": len(confidence_hashes),
        "prompt_sha256": sorted(confidence_hashes),
        "paired_state_key_count": len(confidence_keys),
        "paired_state_key_sha256": hashlib.sha256(
            "\n".join(
                f"{source_index}:{trajectory_step}"
                for source_index, trajectory_step in sorted(confidence_keys)
            ).encode("utf-8")
        ).hexdigest(),
        "policy_pairing_verified": True,
        "clean_target_pairing_verified": True,
        "paired_clean_target_sha256": hashlib.sha256(
            "\n".join(
                f"{source_index}:{trajectory_step}:{confidence_targets[(source_index, trajectory_step)]}"
                for source_index, trajectory_step in sorted(confidence_keys)
            ).encode("utf-8")
        ).hexdigest(),
        "forbidden_prompt_overlap": len(overlap),
        "confidence_json": str(confidence_json),
        "dprm_json": str(dprm_json),
        "confidence_config": str(confidence_config),
        "dprm_config": str(dprm_config),
        "random_json": str(random_json),
        "random_config": str(random_config),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
