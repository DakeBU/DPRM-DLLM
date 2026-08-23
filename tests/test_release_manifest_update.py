import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/update_release_artifact_manifest.py"


def test_merges_host_fragment_and_requires_primary_checkpoint(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    fragment = tmp_path / "fragment.json"
    policy = tmp_path / "policy.json"
    manifest.write_text(json.dumps({"hosts": {"omni": {"status": "pending"}}}))
    policy.write_text(
        json.dumps({"hosts": {"omni": {"primary_artifact_id": "selected"}}})
    )
    fragment.write_text(
        json.dumps(
            {
                "artifacts": [
                    {"id": "selected", "path": "omni/model.tar.zst", "bytes": 4, "sha256": "a" * 64}
                ]
            }
        )
    )
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--host",
            "omni",
            "--fragment",
            str(fragment),
            "--manifest",
            str(manifest),
            "--checkpoint-policy",
            str(policy),
        ],
        check=True,
    )
    merged = json.loads(manifest.read_text())
    assert merged["hosts"]["omni"]["status"] == "complete"
    assert merged["hosts"]["omni"]["artifacts"][0]["id"] == "selected"
