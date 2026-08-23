from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bootstrap_passk.py"


def test_bootstrap_passk_reads_paired_success_matrices(tmp_path: Path) -> None:
    baseline = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]], dtype=bool)
    method = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=bool)
    np.save(tmp_path / "baseline.npy", baseline)
    np.save(tmp_path / "method.npy", method)
    output = tmp_path / "paired.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline",
            str(tmp_path / "baseline.npy"),
            "--method",
            str(tmp_path / "method.npy"),
            "--ks",
            "1",
            "2",
            "4",
            "--bootstrap",
            "200",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["n"] == 3
    assert payload["k_values"] == [1, 2, 4]
    assert payload["paired_delta"] > 0


def test_bootstrap_passk_rejects_unpaired_shapes(tmp_path: Path) -> None:
    np.save(tmp_path / "baseline.npy", np.zeros((2, 4), dtype=bool))
    np.save(tmp_path / "method.npy", np.zeros((3, 4), dtype=bool))
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline",
            str(tmp_path / "baseline.npy"),
            "--method",
            str(tmp_path / "method.npy"),
            "--ks",
            "1",
            "2",
            "4",
            "--output",
            str(tmp_path / "out.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "differ in shape" in result.stderr
