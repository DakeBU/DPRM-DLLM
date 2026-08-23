#!/usr/bin/env python3
"""Package the preregistered AI2D confirmation as a diagnostic artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path


def require(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing diagnostic input: {path}")
    return path


def one(root: Path, pattern: str) -> Path:
    paths = sorted(root.rglob(pattern))
    if len(paths) != 1:
        raise ValueError(f"expected one {pattern} below {root}, found {len(paths)}")
    return require(paths[0])


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def artifact(artifact_id: str, path: Path, release_root: Path) -> dict:
    return {
        "id": artifact_id,
        "path": path.relative_to(release_root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai2d-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--fragment", type=Path, required=True)
    args = parser.parse_args()

    require(args.ai2d_root / "EVALUATION_COMPLETE")
    confirmation_path = require(args.ai2d_root / "confirmation_audit.json")
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    label = confirmation.get("selected")
    selected = confirmation.get("candidates", {}).get(label, {})
    if label != "p1_g8" or not selected.get("active_controller"):
        raise ValueError("AI2D diagnostic does not contain the frozen active controller")
    if selected.get("positive_point_delta") is not False:
        raise ValueError("AI2D diagnostic is expected to retain the non-promoted confirmation")
    if int(selected.get("documents", -1)) != 244:
        raise ValueError("AI2D diagnostic confirmation interval is incomplete")

    confidence = one(args.ai2d_root / "baseline", "*_samples_ai2d_lite.jsonl")
    dprm = one(args.ai2d_root / "confirmation", "*_samples_ai2d_lite.jsonl")
    for path in (confidence, dprm):
        rows = read_jsonl(path)
        if len(rows) != 500 or {int(row["doc_id"]) for row in rows} != set(range(500)):
            raise ValueError(f"AI2D diagnostic needs document ids 0:500: {path}")

    frozen = require(args.ai2d_root / "frozen_controller.json")
    frozen_payload = json.loads(frozen.read_text(encoding="utf-8"))
    table = require(Path(frozen_payload["table"]))
    target = args.release_root / "llada_v" / "diagnostics"
    target.mkdir(parents=True, exist_ok=True)
    archive = target / "ai2d_preregistered_confirmation.tar.gz"

    with tempfile.TemporaryDirectory(prefix="dprm_lladav_ai2d_") as temporary:
        stage = Path(temporary)
        inputs = {
            "confidence_samples.jsonl": confidence,
            "dprm_samples.jsonl": dprm,
            "confidence_order_trace.jsonl": one(args.ai2d_root / "baseline", "order_trace.jsonl"),
            "dprm_order_trace.jsonl": one(args.ai2d_root / "confirmation", "order_trace.jsonl"),
            "protocol.txt": require(args.ai2d_root / "PROTOCOL.txt"),
            "development_selection.json": require(args.ai2d_root / "development_selection.json"),
            "confirmation_audit.json": confirmation_path,
            "frozen_controller.json": frozen,
            "selected_table.json": table,
        }
        for name, source in inputs.items():
            shutil.copy2(source, stage / name)
        with tarfile.open(archive, "w:gz") as output:
            for path in sorted(stage.iterdir()):
                output.add(path, arcname=path.name, recursive=False)

    rwqa_controller = require(args.release_root / "llada_v/controllers/p1_b8_pos4.json")
    rwqa_records = require(args.release_root / "llada_v/records/realworldqa_split_records.tar.zst")
    artifacts = [
        artifact("rwqa_controller", rwqa_controller, args.release_root),
        artifact("rwqa_records", rwqa_records, args.release_root),
        artifact("ai2d_preregistered_diagnostic", archive, args.release_root),
    ]
    args.fragment.parent.mkdir(parents=True, exist_ok=True)
    args.fragment.write_text(
        json.dumps({"status": "complete", "artifacts": artifacts}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifacts, indent=2))


if __name__ == "__main__":
    main()
