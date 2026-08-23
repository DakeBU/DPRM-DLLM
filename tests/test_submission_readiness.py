import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_submission_ready.py"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_submission_audit_reports_pending_hosts_and_placeholders(tmp_path):
    experiments = []
    artifact_hosts = {}
    for index in range(9):
        host = f"Host-{index}"
        experiments.append(
            {
                "id": f"host_{index}",
                "host": host,
                "variants": [
                    {"id": "confidence", "status": "reported"},
                    {"id": "dprm_confidence", "status": "reported"},
                    {"id": "random", "status": "implemented_control"},
                    {"id": "dprm_random", "status": "implemented_control"},
                ],
            }
        )
        artifact_hosts[f"host_{index}"] = {"status": "complete", "artifacts": []}
    experiments[3]["variants"][1]["status"] = "formal_pending"
    artifact_hosts["host_4"]["status"] = "pending"

    registry = tmp_path / "experiments.json"
    results = tmp_path / "results.csv"
    artifacts = tmp_path / "artifacts.json"
    paper = tmp_path / "paper"
    generated = paper / "generated"
    generated.mkdir(parents=True)
    write_json(registry, {"experiments": experiments})
    write_json(artifacts, {"hosts": artifact_hosts})
    with results.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["host"])
        writer.writeheader()
        for experiment in experiments:
            writer.writerow({"host": experiment["host"]})
    (paper / "main_tpami.tex").write_text("paper", encoding="utf-8")
    (generated / "rows.tex").write_text("evaluation in progress", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--registry",
            str(registry),
            "--results",
            str(results),
            "--artifact-manifest",
            str(artifacts),
            "--paper-root",
            str(paper),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert completed.returncode == 1
    assert not result["ready"]
    assert any("host_3 pending variants" in error for error in result["errors"])
    assert any("host_3 has no reported DPRM" in error for error in result["errors"])
    assert any("incomplete artifact hosts: host_4" in error for error in result["errors"])
    assert any("unresolved token: in progress" in error for error in result["errors"])
