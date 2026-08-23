#!/usr/bin/env python3
"""Stage an Omni Hugging Face model with the released ordering hooks.

The model weights remain symlinked to the downloaded checkpoint. Only the
small generation module is copied from this repository so diagnostics can use
the same observer and one-action override hooks as the released host code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_model(model_path: Path, output_path: Path, generation_utils: Path) -> Path:
    model_path = model_path.resolve()
    generation_utils = generation_utils.resolve()
    if output_path.resolve() == model_path:
        raise ValueError("output path must differ from the source model path")
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"missing model config under {model_path}")
    if not generation_utils.is_file():
        raise FileNotFoundError(generation_utils)

    output_path.mkdir(parents=True, exist_ok=True)
    for source in sorted(model_path.iterdir()):
        if source.name == "generation_utils.py":
            continue
        target = output_path / source.name
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise ValueError(f"stale staging link: {target}")
            continue
        if target.exists():
            raise FileExistsError(f"refusing to replace staged entry: {target}")
        os.symlink(source, target, target_is_directory=source.is_dir())

    staged_generation = output_path / "generation_utils.py"
    shutil.copy2(generation_utils, staged_generation)
    manifest = {
        "source_model_path": str(model_path),
        "source_config_sha256": sha256(model_path / "config.json"),
        "generation_utils_source": str(generation_utils),
        "generation_utils_sha256": sha256(generation_utils),
        "weights": "symlinked without modification",
    }
    manifest_path = output_path / "dprm_model_staging.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument(
        "--generation-utils",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "matched/overlay/omni_diffusion/models/dream/generation_utils.py"
        ),
    )
    args = parser.parse_args()
    print(stage_model(args.model_path, args.output_path, args.generation_utils))


if __name__ == "__main__":
    main()
