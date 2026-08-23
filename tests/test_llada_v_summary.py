from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "integrations" / "llada_v" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "summarize_multimodal_results", SCRIPT_DIR / "summarize_multimodal_results.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_paired_interval_is_reproducible() -> None:
    baseline = np.asarray([0.0, 1.0, 0.0, 1.0])
    method = np.asarray([1.0, 1.0, 0.0, 1.0])
    first = MODULE.paired_interval(baseline, method, draws=200, seed=7)
    second = MODULE.paired_interval(baseline, method, draws=200, seed=7)
    assert first == second
    assert first["mean"] == 0.25


def test_read_rows_rejects_duplicate_doc_ids(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    row = {"doc_id": 1, "target": "A", "filtered_resps": ["A"]}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    try:
        MODULE.read_rows(path)
    except ValueError as error:
        assert "duplicate doc_id" in str(error)
    else:
        raise AssertionError("duplicate document ids were accepted")


def test_ai2d_summary_accepts_declared_confidence_dprm_pair(tmp_path: Path) -> None:
    confidence = tmp_path / "confidence.jsonl"
    dprm = tmp_path / "dprm.jsonl"
    confidence.write_text(
        "\n".join(
            json.dumps({"doc_id": i, "target": "A", "filtered_resps": [value]})
            for i, value in enumerate(("B", "A"))
        )
        + "\n"
    )
    dprm.write_text(
        "\n".join(
            json.dumps({"doc_id": i, "target": "A", "filtered_resps": ["A"]})
            for i in range(2)
        )
        + "\n"
    )
    args = type(
        "Args",
        (),
        {
            "ai2d_random": None,
            "ai2d_confidence": confidence,
            "ai2d_dprm_confidence": dprm,
            "ai2d_dprm_random": None,
            "ai2d_expected": 2,
            "bootstrap": 200,
            "seed": 7,
        },
    )()
    summary = MODULE.optional_ai2d_orders(args)
    assert summary["confidence"] == 0.5
    assert summary["dprm_confidence"] == 1.0
    assert "random" not in summary
