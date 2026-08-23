#!/usr/bin/env python3
"""Run predeclared Omni T2I jobs while loading the host model only once."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer, GenerationConfig


def reclaim_stale_lock(lock: Path, stale_lock_seconds: float) -> bool:
    """Remove an abandoned job lock after a grace period."""
    owner_path = lock / "owner_pid"
    try:
        owner_pid = int(owner_path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        owner_pid = None
    if owner_pid is not None:
        try:
            os.kill(owner_pid, 0)
            return False
        except PermissionError:
            return False
        except ProcessLookupError:
            pass
    try:
        age = time.time() - lock.stat().st_mtime
    except FileNotFoundError:
        return False
    if age < stale_lock_seconds:
        return False
    shutil.rmtree(lock, ignore_errors=True)
    return not lock.exists()


def load_smoke(path: Path):
    spec = importlib.util.spec_from_file_location("omni_t2i_smoke_cached", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-script", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-tokenizer-path", required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--stale-lock-seconds", type=float, default=120.0)
    args = parser.parse_args()

    smoke = load_smoke(args.smoke_script)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).eval()
    generation_config = GenerationConfig.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    model.generation_config = generation_config
    image_processor = smoke.ImageProcessor(
        args.image_tokenizer_path,
        "dynamic",
        image_size=512,
        normalize_type="imagenet",
        min_patch_grid=1,
        max_patch_grid=12,
    )
    image_processor.image_tokenizer.rank = 0
    image_processor.load_model()

    class CachedTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return tokenizer

    class CachedModel:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return model

    class CachedGenerationConfig:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return generation_config

    smoke.AutoTokenizer = CachedTokenizer
    smoke.AutoModel = CachedModel
    smoke.GenerationConfig = CachedGenerationConfig
    smoke.ImageProcessor = lambda *_args, **_kwargs: image_processor
    image_processor.load_model = lambda: None

    jobs = [
        json.loads(line)
        for line in args.jobs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    original_argv = sys.argv
    try:
        while True:
            incomplete = 0
            made_progress = False
            for job in jobs:
                output_dir = Path(job["output_dir"])
                order = str(job["order_policy"])
                result = output_dir / f"omni_t2i_{order}.json"
                if result.is_file() and result.stat().st_size > 0:
                    (output_dir / "COMPLETE").touch()
                    continue
                incomplete += 1
                output_dir.mkdir(parents=True, exist_ok=True)
                lock = output_dir / f".{order}.running"
                try:
                    lock.mkdir()
                except FileExistsError:
                    reclaim_stale_lock(lock, args.stale_lock_seconds)
                    continue
                (lock / "owner_pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
                made_progress = True
                argv = [
                    str(args.smoke_script),
                    "--model-path", args.model_path,
                    "--image-tokenizer-path", args.image_tokenizer_path,
                    "--output-dir", str(output_dir),
                    "--prompt", str(job["prompt"]),
                    "--order-policy", order,
                    "--steps", str(job.get("steps", 260)),
                    "--max-tokens", str(job.get("max_tokens", 260)),
                    "--seed", str(job["seed"]),
                ]
                argv.extend(str(value) for value in job.get("extra_args", []))
                try:
                    for attempt in range(1, 4):
                        try:
                            sys.argv = argv
                            smoke.main()
                            if not result.is_file() or result.stat().st_size == 0:
                                raise RuntimeError(
                                    f"Omni job returned without a result artifact: {result}"
                                )
                            (output_dir / "COMPLETE").touch()
                            print(f"completed {output_dir}", flush=True)
                            break
                        except Exception:
                            traceback.print_exc()
                            gc.collect()
                            torch.cuda.empty_cache()
                            if attempt == 3:
                                raise
                finally:
                    shutil.rmtree(lock, ignore_errors=True)
            if incomplete == 0:
                break
            if not made_progress:
                # Another manifest worker owns every incomplete job.
                time.sleep(2)
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
