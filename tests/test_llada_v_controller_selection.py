import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


SCRIPT = (
    Path(__file__).parents[1]
    / "integrations"
    / "llada_v"
    / "scripts"
    / "select_controller.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("select_controller", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_bootstrap_delta_uses_paired_examples() -> None:
    baseline = np.asarray([0.0, 1.0, 0.0, 1.0])
    candidate = np.asarray([1.0, 1.0, 1.0, 0.0])
    result = MODULE.bootstrap_delta(baseline, candidate, seed=7, draws=1000)
    assert result["delta"] == 0.25
    assert result["ci_low"] <= result["delta"] <= result["ci_high"]


def test_load_samples_and_trace_summary(tmp_path: Path) -> None:
    root = tmp_path / "run"
    for order in ("progressive_confidence", "dprm_confidence_warmup"):
        task_root = root / order / "ai2d_lite"
        task_root.mkdir(parents=True)
        sample_path = task_root / f"x_samples_ai2d_lite.jsonl"
        sample_path.write_text(
            json.dumps({"doc_id": 3, "target": "A", "filtered_resps": ["A"]})
            + "\n",
            encoding="utf-8",
        )

    baseline_trace = root / "progressive_confidence" / "ai2d_lite" / "order_trace.jsonl"
    candidate_trace = root / "dprm_confidence_warmup" / "ai2d_lite" / "order_trace.jsonl"
    baseline_trace.write_text(
        "\n".join(
            json.dumps({"doc_id": 3, "step": step, "selected_positions": [position]})
            for step, position in enumerate((10, 11))
        )
        + "\n",
        encoding="utf-8",
    )
    candidate_trace.write_text(
        "\n".join(
            json.dumps(
                {
                    "doc_id": 3,
                    "step": step,
                    "selected_positions": [position],
                    "dprm_selected_gate_mean": 1.0,
                    "dprm_selected_score_mean": 0.2,
                    "dprm_selected_base_log_score_mean": 0.1,
                }
            )
            for step, position in enumerate((11, 10))
        )
        + "\n",
        encoding="utf-8",
    )

    samples = MODULE.load_samples(root, "progressive_confidence", "ai2d_lite")
    assert set(samples) == {3}
    summary = MODULE.trace_summary(
        MODULE.load_trace(root, "dprm_confidence_warmup", "ai2d_lite"),
        MODULE.load_trace(root, "progressive_confidence", "ai2d_lite"),
        [3],
    )
    assert summary["order_changed_rate"] == 1.0
    assert summary["selected_ready_rate"] == 1.0
    assert np.isclose(summary["mean_abs_score_correction"], 0.1)
