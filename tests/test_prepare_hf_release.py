import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_hf_release.py"
SPEC = importlib.util.spec_from_file_location("prepare_hf_release", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
main = MODULE.main
render_model_card = MODULE.render_model_card


def test_model_card_lists_artifact_links() -> None:
    manifest = {
        "hosts": {
            "puma": {
                "status": "complete",
                "artifacts": [{"id": "state", "path": "puma/state.pt"}],
            }
        }
    }
    card = render_model_card(manifest)
    assert "| puma | [state](puma/state.pt) |" in card
    assert "DakeBU/DPRM-DLLM" in card


def test_incomplete_bundle_is_rejected(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"hosts": {"omni_diffusion": {"status": "pending"}}})
    )
    monkeypatch.setattr(
        "sys.argv",
        ["prepare_hf_release.py", "--manifest", str(manifest), "--artifact-root", str(tmp_path / "out")],
    )
    with pytest.raises(SystemExit, match="incomplete bundle"):
        main()


def test_complete_bundle_writes_release_files(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    artifacts = tmp_path / "artifacts"
    payload = artifacts / "puma" / "state.pt"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"checkpoint")
    for name in ("LICENSE", "NOTICE", "CITATION.cff"):
        repo.mkdir(exist_ok=True)
        (repo / name).write_text(name)
    record = {
        "id": "state",
        "path": "puma/state.pt",
        "bytes": payload.stat().st_size,
        "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
    }
    manifest = repo / "manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "hosts": {"puma": {"status": "complete", "artifacts": [record]}}})
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_hf_release.py",
            "--manifest",
            str(manifest),
            "--artifact-root",
            str(artifacts),
            "--repo-root",
            str(repo),
        ],
    )
    main()
    assert (artifacts / "README.md").is_file()
    assert (artifacts / "release_artifacts.json").is_file()
    assert (artifacts / "LICENSE").read_text() == "LICENSE"
