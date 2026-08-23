#!/usr/bin/env python3
"""Merge deterministic pass@K sample-index shards into one canonical record."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


SHARD_KEYS = {"sample_idx_start", "sample_idx_end", "shard_merge"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def canonical_metadata(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if key not in SHARD_KEYS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    reference_metadata = None
    reference_levels = None
    merged = None
    owners: dict[int, str] = {}
    provenance = []

    for shard_dir in args.shards:
        metadata_path = shard_dir / "metadata.json"
        success_path = shard_dir / "success_matrix.npy"
        progress_path = shard_dir / "sample_progress.npy"
        levels_path = shard_dir / "levels.npy"
        for path in (metadata_path, success_path, progress_path, levels_path):
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"missing pass@K shard input: {path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        comparable = canonical_metadata(metadata)
        if reference_metadata is None:
            reference_metadata = comparable
        elif comparable != reference_metadata:
            raise ValueError(f"metadata mismatch in sample shard: {shard_dir}")

        num_examples = int(metadata["num_examples"])
        max_k = max(map(int, metadata["ks"]))
        success = np.load(success_path).astype(np.bool_)
        progress = np.load(progress_path).astype(np.int64)
        levels = np.load(levels_path)
        if success.shape != (num_examples, max_k):
            raise ValueError(f"invalid success matrix in {shard_dir}: {success.shape}")
        if progress.shape != (max_k,):
            raise ValueError(f"invalid sample progress in {shard_dir}: {progress.shape}")
        if levels.shape != (num_examples,):
            raise ValueError(f"invalid levels in {shard_dir}: {levels.shape}")
        if reference_levels is None:
            reference_levels = levels
            merged = np.zeros_like(success)
        elif not np.array_equal(levels, reference_levels):
            raise ValueError(f"level ordering mismatch in sample shard: {shard_dir}")

        start = max(0, int(metadata.get("sample_idx_start", 0)))
        raw_end = int(metadata.get("sample_idx_end", -1))
        end = max_k if raw_end < 0 else min(max_k, raw_end)
        explicit_shard = start != 0 or raw_end >= 0
        completed = np.flatnonzero(progress == num_examples).tolist()
        partial = np.flatnonzero((progress > 0) & (progress < num_examples)).tolist()
        invalid = np.flatnonzero((progress < 0) | (progress > num_examples)).tolist()
        if invalid:
            raise ValueError(f"invalid progress values in {shard_dir}: {invalid}")
        if explicit_shard:
            expected = list(range(start, end))
            if completed != expected or partial:
                raise ValueError(
                    f"incomplete explicit shard {shard_dir}: expected {expected}, "
                    f"completed {completed}, partial {partial}"
                )

        for sample_idx in completed:
            if sample_idx in owners:
                raise ValueError(
                    f"sample column {sample_idx} is complete in both {owners[sample_idx]} "
                    f"and {shard_dir}"
                )
            merged[:, sample_idx] = success[:, sample_idx]
            owners[sample_idx] = str(shard_dir)
        provenance.append(
            {
                "path": str(shard_dir),
                "completed_columns": completed,
                "ignored_partial_columns": partial,
                "metadata_sha256": sha256(metadata_path),
                "success_matrix_sha256": sha256(success_path),
            }
        )

    assert reference_metadata is not None and reference_levels is not None and merged is not None
    max_k = max(map(int, reference_metadata["ks"]))
    missing = sorted(set(range(max_k)) - set(owners))
    if missing:
        raise ValueError(f"sample shard coverage is incomplete: {missing}")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    num_examples = int(reference_metadata["num_examples"])
    progress = np.full(max_k, num_examples, dtype=np.int64)
    metadata = dict(reference_metadata)
    metadata.update(
        {
            "sample_idx_start": 0,
            "sample_idx_end": -1,
            "shard_merge": {
                "seed_contract": "seed + sample_idx * 100000 + batch_id",
                "sources": provenance,
            },
        }
    )
    atomic_write_json(output / "metadata.json", metadata)
    atomic_save_npy(output / "success_matrix.npy", merged)
    atomic_save_npy(output / "sample_progress.npy", progress)
    atomic_save_npy(output / "levels.npy", reference_levels)
    atomic_write_json(
        output / "status.json",
        {
            "model_label": metadata["model_label"],
            "completed_samples": max_k,
            "max_k": max_k,
            "sample_progress": progress.tolist(),
            "num_examples": num_examples,
        },
    )
    print(json.dumps({"examples": num_examples, "samples": max_k, "shards": len(args.shards)}))


if __name__ == "__main__":
    main()
