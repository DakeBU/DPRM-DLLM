from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGER = REPO_ROOT / "integrations" / "omni_diffusion" / "matched" / "scripts" / "package_omni_formal_visual_audit.py"
ORDERS = ("random", "progressive_confidence", "dprm_confidence_warmup")


def test_formal_visual_audit_uses_fixed_and_blinded_prompts_only(tmp_path: Path) -> None:
    records: dict[str, list[dict]] = {}
    prompt_ids = ("prompt_2302", "prompt_2300", "prompt_2301")
    for order_idx, order in enumerate(ORDERS):
        rows = []
        for prompt_idx, prompt_id in enumerate(prompt_ids):
            image_path = tmp_path / f"{order}_{prompt_id}.png"
            Image.new(
                "RGB",
                (16, 16),
                color=(40 * order_idx, 40 * prompt_idx, 80),
            ).save(image_path)
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "prompt": f"fixed prompt {prompt_id}",
                    "image_path": str(image_path),
                    "has_image": True,
                    # Make prompt_2302 the largest DPRM gain. It must not move
                    # into the fixed first-example set.
                    "clip_cosine": 0.1 + order_idx * prompt_idx,
                }
            )
        records[order] = rows

    records_path = tmp_path / "records.json"
    summary_path = tmp_path / "summary.json"
    out_dir = tmp_path / "audit"
    records_path.write_text(json.dumps(records), encoding="utf-8")
    summary_path.write_text(json.dumps({}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(PACKAGER),
            "--records",
            str(records_path),
            "--summary",
            str(summary_path),
            "--out-dir",
            str(out_dir),
            "--orders",
            *ORDERS,
            "--num-examples",
            "2",
            "--fixed-prompt-ids",
            "prompt_2301",
            "prompt_2302",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((out_dir / "selection_manifest.json").read_text())
    assert manifest["paper_role"] == "supplementary fixed-index and blinded visual audit"
    assert manifest["main_text_figure_prompt_id"] is None
    assert manifest["supplement_fixed_prompt_ids"] == ["prompt_2301", "prompt_2302"]
    assert manifest["first_prompt_ids"] == ["prompt_2300", "prompt_2301"]
    assert manifest["blind_prompt_ids"] == [
        "prompt_2300",
        "prompt_2301",
        "prompt_2302",
    ]
    assert manifest["outcome_ranked_selection"] is False
    assert manifest["clip_used_for_selection"] is False
    assert not list(out_dir.glob("*top*dprm*"))
    assert "Largest" not in (out_dir / "visual_audit_index.md").read_text()
    assert (out_dir / "omni_formal_preregistered_examples.png").is_file()
    header = (out_dir / "human_rating_template.tsv").read_text().splitlines()[0]
    assert "A_recognizable_yes_no" in header
