import importlib.util
import json
import tarfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "integrations" / "dmpo" / "scripts" / "package_release.py"
SPEC = importlib.util.spec_from_file_location("dmpo_package_release", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CHECKPOINT_FILES = MODULE.CHECKPOINT_FILES
TASK_SIZES = MODULE.TASK_SIZES
POLICIES = MODULE.POLICIES
main = MODULE.main


def test_package_release_keeps_only_endpoint_files_and_canonical_metadata(
    tmp_path: Path, monkeypatch,
) -> None:
    repro = tmp_path / "repro"
    release = tmp_path / "release"
    for task, examples in TASK_SIZES.items():
        checkpoint = repro / "outputs" / task / "dprm_confidence" / "checkpoint-5000"
        checkpoint.mkdir(parents=True)
        for name in CHECKPOINT_FILES:
            (checkpoint / name).write_bytes(b"checkpoint\n")
        for policy in POLICIES:
            evaluation = repro / "evaluations" / task / f"{policy}_step5000"
            evaluation.mkdir(parents=True)
            np.save(evaluation / "success_matrix.npy", np.zeros((examples, 32), dtype=bool))
            np.save(evaluation / "sample_progress.npy", np.full(32, examples))
            np.save(evaluation / "levels.npy", np.zeros(examples, dtype=np.int64))
            (evaluation / "status.json").write_text(
                json.dumps({"completed_samples": 32, "num_examples": examples})
            )
            (evaluation / "metadata.json").write_text(
                json.dumps(
                    {
                        "checkpoint": "/private/checkpoint",
                        "dataset_jsonl": "/private/test.jsonl",
                        "ks": [1, 2, 4, 8, 16, 32],
                    }
                )
            )
            (evaluation / "endpoint.sha256").write_text("hash  endpoint\n")
            (evaluation / "FORMAL_EVALUATION_COMPLETE").write_text("complete\n")

    monkeypatch.setattr(
        "sys.argv",
        ["package_release.py", "--repro-root", str(repro), "--release-root", str(release)],
    )
    main()

    manifest = json.loads((release / "dmpo" / "release_manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["endpoint_step"] == 5000
    assert manifest["primary_checkpoint_task"] == "countdown"
    assert len(manifest["artifacts"]) == 7
    checkpoint_paths = [
        row["path"] for row in manifest["artifacts"] if "dprm_checkpoint" in row["path"]
    ]
    assert checkpoint_paths == ["dmpo/countdown/dprm_checkpoint_step5000.tar.gz"]
    assert not (release / "dmpo" / "gsm8k" / "checkpoint-5000").exists()
    assert not (release / "dmpo" / "math" / "checkpoint-5000").exists()
    assert not (release / "dmpo" / "math" / "checkpoint-5000" / "optimizer.pt").exists()
    archive = release / "dmpo" / "records" / "countdown_dprm_step5000.tar.gz"
    with tarfile.open(archive, "r:gz") as handle:
        metadata = json.load(handle.extractfile("metadata.json"))
    assert metadata["dataset_jsonl_basename"] == "test.jsonl"
    assert "/private" not in json.dumps(metadata)
    assert (release / "dmpo" / "records" / "countdown_confidence_step5000.tar.gz").is_file()
    with tarfile.open(release / checkpoint_paths[0], "r:gz") as handle:
        assert sorted(handle.getnames()) == sorted(CHECKPOINT_FILES)


def test_package_release_separates_archived_results_from_rebuilt_checkpoint(
    tmp_path: Path, monkeypatch,
) -> None:
    repro = tmp_path / "repro"
    release = tmp_path / "release"
    archive_root = tmp_path / "archived"
    for task in TASK_SIZES:
        checkpoint = repro / "outputs" / task / "dprm_confidence" / "checkpoint-5000"
        checkpoint.mkdir(parents=True)
        for name in CHECKPOINT_FILES:
            (checkpoint / name).write_bytes(b"rebuilt checkpoint\n")

    sources = {"schema_version": 1, "sources": {"math": {}}}
    for policy in POLICIES:
        evaluation = archive_root / policy
        evaluation.mkdir(parents=True)
        np.save(evaluation / "success_matrix.npy", np.zeros((500, 32), dtype=bool))
        np.save(evaluation / "sample_progress.npy", np.full(32, 500))
        np.save(evaluation / "levels.npy", np.zeros(500, dtype=np.int64))
        (evaluation / "status.json").write_text(
            json.dumps({"completed_samples": 32, "num_examples": 500})
        )
        (evaluation / "metadata.json").write_text(
            json.dumps({"checkpoint": "/deleted/original/checkpoint"})
        )
        sources["sources"]["math"][policy] = {
            "path": str(evaluation),
            "result_provenance": "archived_paper_result",
            "checkpoint_binding_verified": False,
        }
    source_map = tmp_path / "record_sources.json"
    source_map.write_text(json.dumps(sources))

    monkeypatch.setattr(
        "sys.argv",
        [
            "package_release.py",
            "--repro-root",
            str(repro),
            "--release-root",
            str(release),
            "--record-source-map",
            str(source_map),
        ],
    )
    main()

    manifest = json.loads((release / "dmpo" / "release_manifest.json").read_text())
    assert manifest["tasks"]["gsm8k"]["confidence"]["status"] == "not_packaged"
    assert manifest["tasks"]["math"]["dprm_confidence"] == {
        "checkpoint_binding_verified": False,
        "examples": 500,
        "result_provenance": "archived_paper_result",
        "samples_per_example": 32,
        "successes": 0,
    }
    archive = release / "dmpo" / "records" / "math_dprm_step5000.tar.gz"
    with tarfile.open(archive, "r:gz") as handle:
        metadata = json.load(handle.extractfile("metadata.json"))
    provenance = metadata["release_provenance"]
    assert provenance["checkpoint_binding_verified"] is False
    assert provenance["result_provenance"] == "archived_paper_result"
    assert "checkpoint_sha256" not in metadata
    assert "dprm_estimator_sha256" not in metadata
