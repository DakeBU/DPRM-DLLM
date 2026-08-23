import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "integrations" / "dmpo" / "scripts" / "render_matched_results.py"


def test_renderer_reduces_paired_success_matrices(tmp_path):
    levels_by_task = {
        "gsm8k": np.array([3, 2, 3, 1]),
        "math": np.array([4, 2, 4, 5]),
        "countdown": np.array([3, 2, 3, 4]),
    }
    for task, levels in levels_by_task.items():
        confidence = np.zeros((4, 32), dtype=bool)
        dprm = confidence.copy()
        confidence[1, 0] = True
        dprm[0, 0] = True
        dprm[1, 0] = True
        for policy, success in (
            ("confidence", confidence),
            ("dprm_confidence", dprm),
        ):
            directory = tmp_path / "evaluations" / task / f"{policy}_step5000"
            directory.mkdir(parents=True)
            np.save(directory / "success_matrix.npy", success)
            np.save(directory / "levels.npy", levels)

    output = tmp_path / "summary.json"
    latex = tmp_path / "rows.tex"
    csv_output = tmp_path / "rows.csv"
    paper_results = tmp_path / "paper_results.csv"
    paper_results.write_text(
        "host,task,method,variant,metric,direction,value,ci95_low,ci95_high,n,protocol\n"
        "PUMA,GSM8K,DPRM-PUMA,dprm_confidence,accuracy,higher,1,,,4,test\n"
        "DMPO,old,old,confidence,mean_pass_at_k,higher,0,,,4,stale\n"
    )
    registry = tmp_path / "experiments.json"
    registry.write_text(
        json.dumps(
            {
                "experiments": [
                    {
                        "id": "dmpo",
                        "variants": [
                            {"id": "random", "status": "formal_pending"},
                            {"id": "confidence", "status": "formal_pending"},
                            {"id": "dprm_confidence", "status": "formal_pending"},
                            {"id": "dprm_random", "status": "implemented_control"},
                        ],
                    }
                ]
            }
        )
    )
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repro-root",
            str(tmp_path),
            "--output",
            str(output),
            "--latex-output",
            str(latex),
            "--csv-output",
            str(csv_output),
            "--paper-results-csv",
            str(paper_results),
            "--experiment-registry",
            str(registry),
            "--bootstrap-iters",
            "100",
        ],
        check=True,
    )

    payload = json.loads(output.read_text())
    assert payload["tasks"]["gsm8k"]["all"]["paired_delta"] == 0.25
    assert payload["tasks"]["gsm8k"]["hard"]["paired_delta"] == 0.5
    assert "Progressive DMPO & 25.0 & 0.0" in latex.read_text()
    assert "DMPO-DPRM & \\bestcell{50.0} & \\bestcell{50.0}" in latex.read_text()
    assert csv_output.read_text().count("matched_step5000") == 18
    merged = paper_results.read_text()
    assert "PUMA,GSM8K" in merged
    assert "DMPO,old" not in merged
    assert merged.count("matched_step5000") == 18
    statuses = {
        row["id"]: row["status"]
        for row in json.loads(registry.read_text())["experiments"][0]["variants"]
    }
    assert statuses == {
        "random": "formal_pending",
        "confidence": "reported",
        "dprm_confidence": "reported",
        "dprm_random": "implemented_control",
    }


def test_renderer_accepts_paired_archived_record_map(tmp_path):
    sources = {"schema_version": 1, "sources": {}}
    for task, hard_level in (("math", 4), ("countdown", 3)):
        sources["sources"][task] = {}
        for policy in ("confidence", "dprm_confidence"):
            directory = tmp_path / "archive" / task / policy
            directory.mkdir(parents=True)
            success = np.zeros((2, 32), dtype=bool)
            if policy == "dprm_confidence":
                success[0, 0] = True
            np.save(directory / "success_matrix.npy", success)
            np.save(directory / "levels.npy", np.array([hard_level, hard_level + 1]))
            sources["sources"][task][policy] = {
                "path": str(directory),
                "result_provenance": "archived_paper_result",
                "checkpoint_binding_verified": False,
            }
    source_map = tmp_path / "sources.json"
    source_map.write_text(json.dumps(sources), encoding="utf-8")
    output = tmp_path / "summary.json"
    latex = tmp_path / "rows.tex"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repro-root",
            str(tmp_path),
            "--record-source-map",
            str(source_map),
            "--output",
            str(output),
            "--latex-output",
            str(latex),
            "--bootstrap-iters",
            "100",
        ],
        check=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert set(payload["tasks"]) == {"math", "countdown"}
    assert payload["tasks"]["math"]["record_provenance"]["dprm_confidence"][
        "checkpoint_binding_verified"
    ] is False
    assert "Progressive DMPO & 0.0 & 0.0 & 0.0 & 0.0" in latex.read_text()
