#!/usr/bin/env python3
"""Validate and package DMPO checkpoints and paper evaluation records."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

import numpy as np


TASK_SIZES = {"gsm8k": 1319, "math": 500, "countdown": 5120}
POLICIES = ("confidence", "dprm_confidence")
CHECKPOINT_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "dprm_estimator.json",
    "trainer_state.json",
    "training_args.bin",
)
CORE_RECORD_FILES = (
    "success_matrix.npy",
    "sample_progress.npy",
    "levels.npy",
    "status.json",
)
OPTIONAL_RECORD_FILES = (
    "per_example_summary.jsonl",
    "endpoint.sha256",
    "FORMAL_EVALUATION_COMPLETE",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing release input: {path}")
    return path


def validate_records(eval_dir: Path, task: str) -> dict:
    expected_examples = TASK_SIZES[task]
    success = np.load(require_file(eval_dir / "success_matrix.npy"))
    progress = np.load(require_file(eval_dir / "sample_progress.npy"))
    levels = np.load(require_file(eval_dir / "levels.npy"))
    status = json.loads(require_file(eval_dir / "status.json").read_text())

    if success.shape != (expected_examples, 32):
        raise ValueError(
            f"{task} success shape is {success.shape}, expected {(expected_examples, 32)}"
        )
    if progress.shape != (32,) or not np.all(progress == expected_examples):
        raise ValueError(f"{task} evaluation is incomplete: {progress.tolist()}")
    if levels.shape != (expected_examples,):
        raise ValueError(f"{task} level shape is {levels.shape}")
    if status.get("completed_samples") != 32 or status.get("num_examples") != expected_examples:
        raise ValueError(f"{task} status does not describe a complete pass@32 run")
    return {
        "examples": expected_examples,
        "samples_per_example": 32,
        "successes": int(success.sum()),
    }


def canonical_metadata(
    source: Path,
    checkpoint_hash: str | None,
    estimator_hash: str | None,
    *,
    result_provenance: str,
    checkpoint_binding_verified: bool,
    record_hashes: dict[str, str],
) -> dict:
    metadata = json.loads(require_file(source).read_text())
    for key in ("checkpoint", "dprm_estimator_path", "resolved_dprm_estimator_path"):
        metadata.pop(key, None)
    for key, value in list(metadata.items()):
        if isinstance(value, str) and value.startswith("/"):
            metadata.pop(key)
            metadata[f"{key}_basename"] = Path(value).name
    if checkpoint_hash is not None:
        metadata["checkpoint_sha256"] = checkpoint_hash
    if estimator_hash is not None:
        metadata["dprm_estimator_sha256"] = estimator_hash
    metadata["release_provenance"] = {
        "result_provenance": result_provenance,
        "checkpoint_binding_verified": checkpoint_binding_verified,
        "record_sha256": record_hashes,
    }
    return metadata


def make_archive(eval_dir: Path, output: Path, metadata: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dprm_dmpo_records_") as temporary:
        staging = Path(temporary)
        for name in CORE_RECORD_FILES:
            shutil.copy2(require_file(eval_dir / name), staging / name)
        for name in OPTIONAL_RECORD_FILES:
            source = eval_dir / name
            if source.is_file() and source.stat().st_size > 0:
                shutil.copy2(source, staging / name)
        (staging / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with tarfile.open(output, "w:gz") as archive:
            for path in sorted(staging.iterdir()):
                archive.add(path, arcname=path.name, recursive=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repro-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--record-source-map",
        type=Path,
        help=(
            "Optional JSON map from task/policy to retained paper-record paths. "
            "Use this when result records predate the packaged reproduction adapter."
        ),
    )
    parser.add_argument(
        "--primary-checkpoint-task",
        choices=tuple(TASK_SIZES),
        default="countdown",
        help="Only this task's DPRM adapter is included in the HF bundle.",
    )
    args = parser.parse_args()

    source_map = None
    if args.record_source_map is not None:
        source_map = json.loads(require_file(args.record_source_map).read_text())
        if source_map.get("schema_version") != 1:
            raise ValueError("record source map must use schema_version 1")
        source_map = source_map.get("sources")
        if not isinstance(source_map, dict):
            raise ValueError("record source map must contain an object named sources")

    artifacts: list[tuple[str, Path]] = []
    summaries = {}
    for task in TASK_SIZES:
        checkpoint = args.repro_root / "outputs" / task / "dprm_confidence" / "checkpoint-5000"
        summaries[task] = {}

        adapter_hash = sha256(require_file(checkpoint / "adapter_model.safetensors"))
        estimator_hash = sha256(require_file(checkpoint / "dprm_estimator.json"))
        if task == args.primary_checkpoint_task:
            checkpoint_archive = (
                args.release_root
                / "dmpo"
                / task
                / "dprm_checkpoint_step5000.tar.gz"
            )
            checkpoint_archive.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(checkpoint_archive, "w:gz") as archive:
                for name in CHECKPOINT_FILES:
                    archive.add(require_file(checkpoint / name), arcname=name)
            artifacts.append((f"{task}_dprm_checkpoint", checkpoint_archive))
        for policy in POLICIES:
            source_spec = None if source_map is None else source_map.get(task, {}).get(policy)
            if source_map is not None and source_spec is None:
                summaries[task][policy] = {"status": "not_packaged"}
                continue
            if source_spec is None:
                eval_dir = args.repro_root / "evaluations" / task / f"{policy}_step5000"
                result_provenance = "direct_reproduction_evaluation"
                checkpoint_binding_verified = policy == "dprm_confidence"
            else:
                if not isinstance(source_spec, dict) or "path" not in source_spec:
                    raise ValueError(f"invalid record source for {task}/{policy}")
                eval_dir = Path(source_spec["path"])
                result_provenance = str(
                    source_spec.get("result_provenance", "archived_paper_result")
                )
                checkpoint_binding_verified = bool(
                    source_spec.get("checkpoint_binding_verified", False)
                )
            summaries[task][policy] = validate_records(eval_dir, task)
            record_hashes = {
                name: sha256(require_file(eval_dir / name)) for name in CORE_RECORD_FILES
            }
            metadata = canonical_metadata(
                eval_dir / "metadata.json",
                adapter_hash
                if policy == "dprm_confidence" and checkpoint_binding_verified
                else None,
                estimator_hash
                if policy == "dprm_confidence" and checkpoint_binding_verified
                else None,
                result_provenance=result_provenance,
                checkpoint_binding_verified=checkpoint_binding_verified,
                record_hashes=record_hashes,
            )
            summaries[task][policy]["result_provenance"] = result_provenance
            summaries[task][policy][
                "checkpoint_binding_verified"
            ] = checkpoint_binding_verified
            record_name = "dprm" if policy == "dprm_confidence" else policy
            archive = (
                args.release_root
                / "dmpo"
                / "records"
                / f"{task}_{record_name}_step5000.tar.gz"
            )
            make_archive(eval_dir, archive, metadata)
            artifacts.append((f"{task}_{record_name}_records", archive))

    payload = {
        "schema_version": 2,
        "host": "DMPO",
        "endpoint_step": 5000,
        "primary_checkpoint_task": args.primary_checkpoint_task,
        "tasks": summaries,
        "artifacts": [
            {
                "id": artifact_id,
                "path": path.relative_to(args.release_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for artifact_id, path in artifacts
        ],
    }
    manifest = args.manifest or args.release_root / "dmpo" / "release_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
