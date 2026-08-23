from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "integrations/omni_diffusion/matched/scripts/select_omni_training_endpoint.py"


def write_candidate(root: Path, step: int, clip_l: float, clip_b: float) -> tuple[Path, Path]:
    paired = root / f"paired_{step}.json"
    order = root / f"order_{step}.json"
    paired.write_text(
        json.dumps(
            {
                "comparisons_by_metric": {
                    "clip_cosine": [{"baseline": "progressive_confidence", "method": "dprm_confidence_warmup", "matched_prompts": 128, "mean_delta": clip_l}],
                    "clip_b32_cosine": [{"baseline": "progressive_confidence", "method": "dprm_confidence_warmup", "matched_prompts": 128, "mean_delta": clip_b}],
                }
            }
        ),
        encoding="utf-8",
    )
    order.write_text(
        json.dumps(
            {
                "comparisons": [{"reference": "progressive_confidence", "method": "dprm_confidence_warmup", "direct_override_fraction": 0.01, "has_direct_override": 0.5}]
            }
        ),
        encoding="utf-8",
    )
    return paired, order


def test_selector_uses_positive_dual_clip_development_endpoint(tmp_path: Path) -> None:
    paired_500, order_500 = write_candidate(tmp_path, 500, 0.002, 0.001)
    paired_1000, order_1000 = write_candidate(tmp_path, 1000, 0.004, -0.001)
    output = tmp_path / "selection.json"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--candidate", "500", str(paired_500), str(order_500), "--candidate", "1000", str(paired_1000), str(order_1000), "--expected-prompts", "128", "--output", str(output)],
        check=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["selected_step"] == 500
    assert payload["confirmation_data_read"] is False


def test_selector_refuses_endpoints_without_two_positive_means(tmp_path: Path) -> None:
    paired_500, order_500 = write_candidate(tmp_path, 500, 0.002, -0.001)
    output = tmp_path / "selection.json"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--candidate", "500", str(paired_500), str(order_500), "--expected-prompts", "128", "--output", str(output)],
        check=True,
    )
    assert json.loads(output.read_text(encoding="utf-8"))["selected_step"] is None
