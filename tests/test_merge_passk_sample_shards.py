import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "integrations/dmpo/scripts/merge_passk_sample_shards.py"


def write_shard(root: Path, start: int, end: int, value: bool) -> Path:
    root.mkdir()
    metadata = {
        "model_label": "DMPO",
        "ks": [1, 2, 4],
        "num_examples": 3,
        "selected_indices": [0, 1, 2],
        "sample_idx_start": start,
        "sample_idx_end": end,
    }
    matrix = np.zeros((3, 4), dtype=bool)
    matrix[:, start:end] = value
    progress = np.zeros(4, dtype=np.int64)
    progress[start:end] = 3
    (root / "metadata.json").write_text(json.dumps(metadata))
    np.save(root / "success_matrix.npy", matrix)
    np.save(root / "sample_progress.npy", progress)
    np.save(root / "levels.npy", np.array([0, 1, 2]))
    return root


def test_merges_disjoint_sample_columns(tmp_path: Path) -> None:
    first = write_shard(tmp_path / "first", 0, 2, True)
    second = write_shard(tmp_path / "second", 2, 4, False)
    output = tmp_path / "merged"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--shards",
            str(first),
            str(second),
            "--output-dir",
            str(output),
        ],
        check=True,
    )
    matrix = np.load(output / "success_matrix.npy")
    assert matrix[:, :2].all()
    assert not matrix[:, 2:].any()
    assert np.array_equal(np.load(output / "sample_progress.npy"), np.full(4, 3))


def test_rejects_overlapping_complete_columns(tmp_path: Path) -> None:
    first = write_shard(tmp_path / "first", 0, 3, True)
    second = write_shard(tmp_path / "second", 2, 4, False)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--shards",
            str(first),
            str(second),
            "--output-dir",
            str(tmp_path / "merged"),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "complete in both" in result.stderr


def test_keeps_complete_prefix_and_replaces_partial_column(tmp_path: Path) -> None:
    serial = write_shard(tmp_path / "serial", 0, 1, True)
    metadata_path = serial / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.pop("sample_idx_start")
    metadata.pop("sample_idx_end")
    metadata_path.write_text(json.dumps(metadata))
    progress = np.array([3, 1, 0, 0])
    np.save(serial / "sample_progress.npy", progress)

    remainder = write_shard(tmp_path / "remainder", 1, 4, False)
    output = tmp_path / "merged"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--shards",
            str(serial),
            str(remainder),
            "--output-dir",
            str(output),
        ],
        check=True,
    )
    matrix = np.load(output / "success_matrix.npy")
    assert matrix[:, 0].all()
    assert not matrix[:, 1:].any()
    merge = json.loads((output / "metadata.json").read_text())["shard_merge"]
    assert merge["sources"][0]["ignored_partial_columns"] == [1]
