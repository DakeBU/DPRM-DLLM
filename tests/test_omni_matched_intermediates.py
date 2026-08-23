from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGER = (
    REPO_ROOT
    / "integrations"
    / "omni_diffusion"
    / "matched"
    / "scripts"
    / "package_omni_matched_intermediates.py"
)


def load_packager():
    spec = importlib.util.spec_from_file_location("omni_matched_packager", PACKAGER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_same_model_divergence_keeps_scores_and_canvas_coordinates(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "dprm_trace.jsonl"
    rows = [
        {
            "step": 0,
            "selected_visual_indices": [0],
            "confidence_default_candidate_index": 0,
        },
        {
            "step": 32,
            "selected_visual_indices": [18],
            # Position zero is already committed, so compressed candidate 15
            # maps to visual index 16 on the fixed 256-position canvas.
            "confidence_default_candidate_index": 15,
            "selected_base_order_scores": [0.0],
            "selected_dprm_values": [0.16],
            "selected_adjusted_order_scores": [0.012],
            "confidence_default_base_order_score": 0.0,
            "confidence_default_dprm_value": 0.11,
            "confidence_default_adjusted_order_score": 0.008,
        },
    ]
    trace.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    module = load_packager()
    result = module.first_policy_divergence(
        {"dprm_confidence_warmup": {"order_trace_path": str(trace)}}
    )

    assert result is not None
    assert result["step"] == 32
    assert result["confidence_default"] == (1, 0)
    assert result["dprm"] == (1, 2)
    assert result["base_order_score"] == {
        "confidence_default": 0.0,
        "dprm": 0.0,
    }
    assert result["gated_process_value"] == {
        "confidence_default": 0.11,
        "dprm": 0.16,
    }
