from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "integrations/llada_v/scripts/package_ai2d_diagnostic.py"


def write_rows(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps({"doc_id": index}) + "\n" for index in range(500)),
        encoding="utf-8",
    )


def test_packages_non_promoted_ai2d_confirmation(tmp_path: Path) -> None:
    ai2d = tmp_path / "ai2d"
    release = tmp_path / "release"
    (ai2d / "EVALUATION_COMPLETE").parent.mkdir(parents=True)
    (ai2d / "EVALUATION_COMPLETE").write_text("done\n")
    (ai2d / "PROTOCOL.txt").write_text("fit 0:128; dev 128:256; confirm 256:500\n")
    (ai2d / "development_selection.json").write_text("{}\n")
    (ai2d / "confirmation_audit.json").write_text(
        json.dumps(
            {
                "selected": "p1_g8",
                "candidates": {
                    "p1_g8": {
                        "active_controller": True,
                        "positive_point_delta": False,
                        "documents": 244,
                    }
                },
            }
        )
    )
    table = ai2d / "tables/p1.json"
    table.parent.mkdir(parents=True)
    table.write_text("{}\n")
    (ai2d / "frozen_controller.json").write_text(
        json.dumps({"label": "p1_g8", "table": str(table)})
    )
    for branch in ("baseline", "confirmation"):
        write_rows(ai2d / branch / "000_samples_ai2d_lite.jsonl")
        (ai2d / branch / "order_trace.jsonl").write_text("{}\n")

    controller = release / "llada_v/controllers/p1_b8_pos4.json"
    records = release / "llada_v/records/realworldqa_split_records.tar.zst"
    controller.parent.mkdir(parents=True)
    records.parent.mkdir(parents=True)
    controller.write_text("{}\n")
    records.write_bytes(b"records")
    fragment = tmp_path / "fragment.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--ai2d-root",
            str(ai2d),
            "--release-root",
            str(release),
            "--fragment",
            str(fragment),
        ],
        check=True,
    )

    payload = json.loads(fragment.read_text())
    assert payload["status"] == "complete"
    assert [item["id"] for item in payload["artifacts"]] == [
        "rwqa_controller",
        "rwqa_records",
        "ai2d_preregistered_diagnostic",
    ]
    archive = release / "llada_v/diagnostics/ai2d_preregistered_confirmation.tar.gz"
    with tarfile.open(archive, "r:gz") as handle:
        assert "confirmation_audit.json" in handle.getnames()
        assert "dprm_order_trace.jsonl" in handle.getnames()
