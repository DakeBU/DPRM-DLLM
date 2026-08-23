from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKER = (
    ROOT
    / "integrations"
    / "omni_diffusion"
    / "matched"
    / "scripts"
    / "check_omni_fixed_visual_review.py"
)


def write_fixture(tmp_path: Path, recognizable: str = "yes") -> tuple[Path, Path]:
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps({"supplement_fixed_prompt_ids": ["prompt_2501"]}),
        encoding="utf-8",
    )
    ratings = tmp_path / "ratings.tsv"
    fields = [
        "prompt_id",
        "sheet",
        "A_recognizable_yes_no",
        "A_semantic_alignment_1to5",
        "A_visual_coherence_1to5",
        "A_artifact_free_1to5",
        "B_recognizable_yes_no",
        "B_semantic_alignment_1to5",
        "B_visual_coherence_1to5",
        "B_artifact_free_1to5",
        "preferred_column",
        "notes",
    ]
    with ratings.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "prompt_id": "prompt_2501",
                "sheet": "blind_sheet_00.png",
                "A_recognizable_yes_no": recognizable,
                "A_semantic_alignment_1to5": "4",
                "A_visual_coherence_1to5": "4",
                "A_artifact_free_1to5": "3",
                "B_recognizable_yes_no": "yes",
                "B_semantic_alignment_1to5": "3",
                "B_visual_coherence_1to5": "3",
                "B_artifact_free_1to5": "3",
            }
        )
    return selection, ratings


def test_fixed_visual_review_passes_complete_blinded_ratings(tmp_path: Path) -> None:
    selection, ratings = write_fixture(tmp_path)
    output = tmp_path / "review" / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--ratings",
            str(ratings),
            "--selection-manifest",
            str(selection),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["passed"] is True
    assert (output.parent / "SUPPLEMENT_VISUAL_READY").is_file()


def test_fixed_visual_review_rejects_unrecognizable_output(tmp_path: Path) -> None:
    selection, ratings = write_fixture(tmp_path, recognizable="no")
    output = tmp_path / "review" / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--ratings",
            str(ratings),
            "--selection-manifest",
            str(selection),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert json.loads(output.read_text())["passed"] is False
    assert (output.parent / "SUPPLEMENT_VISUAL_FAILED").is_file()
