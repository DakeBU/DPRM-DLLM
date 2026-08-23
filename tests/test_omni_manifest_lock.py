from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    REPO_ROOT
    / "integrations"
    / "omni_diffusion"
    / "matched"
    / "scripts"
    / "run_omni_t2i_manifest.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("omni_manifest_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def age_lock(lock: Path) -> None:
    old = time.time() - 600
    os.utime(lock, (old, old))


def test_live_manifest_lock_is_not_reclaimed(tmp_path: Path) -> None:
    lock = tmp_path / ".job.running"
    lock.mkdir()
    (lock / "owner_pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    age_lock(lock)

    assert not load_runner().reclaim_stale_lock(lock, 120)
    assert lock.is_dir()


def test_abandoned_manifest_lock_is_reclaimed(tmp_path: Path) -> None:
    lock = tmp_path / ".job.running"
    lock.mkdir()
    (lock / "owner_pid").write_text("999999999\n", encoding="utf-8")
    age_lock(lock)

    assert load_runner().reclaim_stale_lock(lock, 120)
    assert not lock.exists()
