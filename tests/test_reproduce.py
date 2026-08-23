import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "reproduce.py"


def dry_run(host: str, variant: str, **environment: str) -> str:
    env = os.environ.copy()
    env.update(environment)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--host",
            host,
            "--variant",
            variant,
            "--dry-run",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_dcm_uses_variant_specific_execution_roots(tmp_path: Path) -> None:
    random_output = dry_run("dcm", "random", DCM_ROOT=str(tmp_path))
    calibrated_output = dry_run("dcm", "dprm_confidence", DCM_ROOT=str(tmp_path))

    assert f"working directory: {tmp_path.resolve()}" in random_output
    assert f"working directory: {ROOT.resolve()}" in calibrated_output


def test_genmol_uses_variant_specific_execution_roots(tmp_path: Path) -> None:
    trained_output = dry_run("genmol", "dprm_confidence", GENMOL_ROOT=str(tmp_path))
    sweep_output = dry_run("genmol", "dprm_random", GENMOL_ROOT=str(tmp_path))

    assert f"working directory: {tmp_path.resolve()}" in trained_output
    assert f"working directory: {ROOT.resolve()}" in sweep_output


def test_dry_run_lists_all_omni_requirements() -> None:
    output = dry_run("omni_diffusion", "dprm_confidence")
    for name in (
        "OMNI_ROOT",
        "OMNI_MODEL_PATH",
        "OMNI_IMAGE_TOKENIZER_PATH",
        "OMNI_ONLINE_PROMPT_FILE",
        "OMNI_ONLINE_ROOT",
        "VIRTUAL_ENV",
    ):
        assert name in output


def test_lladav_runs_release_entrypoint_against_pinned_lmms_checkout(
    tmp_path: Path,
) -> None:
    output = dry_run(
        "llada_v",
        "dprm_confidence",
        LLADA_V_LMMS_ROOT=str(tmp_path),
    )
    assert f"working directory: {ROOT.resolve()}" in output
    for name in (
        "LLADA_V_LMMS_ROOT",
        "LLADA_V_MODEL_PATH",
        "LLADA_V_OUTPUT_ROOT",
    ):
        assert name in output
    assert "bash integrations/llada_v/run_realworldqa_protocol.sh" in output


def test_lladav_order_variants_use_the_same_formal_task() -> None:
    registry = json.loads(
        (ROOT / "reproducibility/experiments.json").read_text(encoding="utf-8")
    )
    experiment = next(row for row in registry["experiments"] if row["id"] == "llada_v")
    commands = {row["id"]: row["command"] for row in experiment["variants"]}

    assert "LLADA_V_TASK=realworldqa" in commands["random"]
    assert "LLADA_V_TASK=realworldqa" in commands["confidence"]
    assert "run_realworldqa_protocol.sh" in commands["dprm_confidence"]
    assert "LLADA_V_TASK=realworldqa" in commands["dprm_random"]
    assert "DPRM_LLADAV_TABLE" in commands["dprm_random"]


def init_host_checkout(path: Path, commit: str | None = None) -> str:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "DPRM Test"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "dprm-test@example.invalid"],
        check=True,
    )
    (path / "marker.txt").write_text("test\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "marker.txt"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "fixture"], check=True)
    actual = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit is not None:
        subprocess.run(["git", "-C", str(path), "update-ref", "HEAD", commit], check=True)
        actual = commit
    return actual


def test_execute_rejects_wrong_upstream_commit(tmp_path: Path) -> None:
    host = tmp_path / "host"
    host.mkdir()
    init_host_checkout(host)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--host", "puma", "--variant", "random", "--execute"],
        cwd=ROOT,
        env={**os.environ, "PUMA_ROOT": str(host)},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "upstream commit mismatch" in result.stderr


def test_execute_writes_completed_manifest(tmp_path: Path) -> None:
    host = tmp_path / "host"
    host.mkdir()
    actual_commit = init_host_checkout(host)
    registry = json.loads((ROOT / "reproducibility" / "experiments.json").read_text())
    experiment = next(row for row in registry["experiments"] if row["id"] == "puma")
    experiment["upstream_commit"] = actual_commit
    experiment["variants"][0]["command"] = "true"
    registry_path = tmp_path / "experiments.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--host",
            "puma",
            "--variant",
            "random",
            "--execute",
            "--manifest-out",
            str(manifest),
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "PUMA_ROOT": str(host),
            "DPRM_EXPERIMENT_REGISTRY": str(registry_path),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert payload["return_code"] == 0
    assert payload["upstream_commit"] == actual_commit
    assert len(payload["registry_sha256"]) == 64
    assert len(payload["integration_sha256"]) == 64
