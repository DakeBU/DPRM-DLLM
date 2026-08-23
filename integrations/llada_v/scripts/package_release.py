#!/usr/bin/env python3
"""Package the declared LLaDA-V protocols without model-weight duplication."""

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
        raise FileNotFoundError(f"missing release input: {path}")
    return path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def one(root: Path, pattern: str) -> Path:
    paths = sorted(root.rglob(pattern))
    if len(paths) != 1:
        raise ValueError(f"expected one {pattern} below {root}, found {len(paths)}")
    return require(paths[0])


def count_rows(path: Path) -> int:
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return len(rows)


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
    parser.add_argument("--chartqa-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--fragment", type=Path, required=True)
    args = parser.parse_args()

    require(args.ai2d_root / "EVALUATION_COMPLETE")
    require(args.chartqa_root / "EVALUATION_COMPLETE")
    confirmation_audit = json.loads(
        require(args.ai2d_root / "confirmation_audit.json").read_text(encoding="utf-8")
    )
    selected_label = confirmation_audit.get("selected")
    selected_audit = confirmation_audit.get("candidates", {}).get(selected_label, {})
    if not selected_label or not selected_audit.get("active_controller"):
        raise ValueError("AI2D confirmation does not contain an active DPRM controller")
    if not selected_audit.get("positive_point_delta"):
        raise ValueError("AI2D confirmation does not improve over confidence")
    controller = require(args.ai2d_root / "frozen_controller.json")
    controller_payload = json.loads(controller.read_text(encoding="utf-8"))
    table = require(Path(controller_payload["table"]))

    ai2d_conf = one(args.ai2d_root / "baseline", "*_samples_ai2d_lite.jsonl")
    ai2d_dprm = one(args.ai2d_root / "confirmation", "*_samples_ai2d_lite.jsonl")
    chart_conf = one(args.chartqa_root / "confidence", "*_samples_chartqa_lite.jsonl")
    chart_dprm = one(args.chartqa_root / "dprm", "*_samples_chartqa_lite.jsonl")
    for path, expected in ((ai2d_conf, 500), (ai2d_dprm, 500), (chart_conf, 500), (chart_dprm, 500)):
        if count_rows(path) != expected:
            raise ValueError(f"{path} does not contain {expected} records")

    target = args.release_root / "llada_v"
    config_dir = target / "configurations"
    records_dir = target / "records"
    config_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)
    copied_controller = config_dir / "ai2d_frozen_controller.json"
    copied_table = config_dir / "ai2d_selected_table.json"
    shutil.copy2(controller, copied_controller)
    shutil.copy2(table, copied_table)

    archive = records_dir / "ai2d_chartqa_protocol_records.tar.gz"
    with tempfile.TemporaryDirectory(prefix="dprm_lladav_") as temporary:
        stage = Path(temporary)
        inputs = {
            "ai2d_confidence_samples.jsonl": ai2d_conf,
            "ai2d_dprm_samples.jsonl": ai2d_dprm,
            "chartqa_confidence_samples.jsonl": chart_conf,
            "chartqa_dprm_samples.jsonl": chart_dprm,
            "ai2d_protocol.txt": require(args.ai2d_root / "PROTOCOL.txt"),
            "ai2d_development_selection.json": require(args.ai2d_root / "development_selection.json"),
            "ai2d_confirmation_audit.json": require(args.ai2d_root / "confirmation_audit.json"),
            "chartqa_protocol.txt": require(args.chartqa_root / "PROTOCOL.txt"),
            "chartqa_summary.json": require(args.chartqa_root / "summary.json"),
        }
        for name, source in inputs.items():
            shutil.copy2(source, stage / name)
        for label, root in (("ai2d_confidence", args.ai2d_root / "baseline"), ("ai2d_dprm", args.ai2d_root / "confirmation"), ("chartqa_confidence", args.chartqa_root / "confidence"), ("chartqa_dprm", args.chartqa_root / "dprm")):
            trace = one(root, "order_trace.jsonl")
            shutil.copy2(trace, stage / f"{label}_order_trace.jsonl")
        with tarfile.open(archive, "w:gz") as output:
            for path in sorted(stage.iterdir()):
                output.add(path, arcname=path.name, recursive=False)

    rwqa_controller = require(target / "controllers" / "p1_b8_pos4.json")
    rwqa_records = require(target / "records" / "realworldqa_split_records.tar.zst")
    artifacts = [
        artifact("rwqa_controller", rwqa_controller, args.release_root),
        artifact("rwqa_records", rwqa_records, args.release_root),
        artifact("ai2d_controller_configuration", copied_controller, args.release_root),
        artifact("ai2d_selected_table", copied_table, args.release_root),
        artifact("ai2d_chartqa_records", archive, args.release_root),
    ]
    args.fragment.parent.mkdir(parents=True, exist_ok=True)
    args.fragment.write_text(
        json.dumps({"status": "complete", "artifacts": artifacts}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifacts, indent=2))


if __name__ == "__main__":
    main()
