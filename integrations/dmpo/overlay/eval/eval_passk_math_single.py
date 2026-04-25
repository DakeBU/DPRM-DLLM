#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DMPO_ROOT = os.path.join(REPO_ROOT, "DMPO")
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
if DMPO_ROOT not in sys.path:
    sys.path.append(DMPO_ROOT)

from generate import generate
from fast_samplers.fast_dllm.generate import (
    generate_pd,
    generate_with_dual_cache,
    generate_with_prefix_cache,
)
from fast_samplers.fast_dllm.modeling_llada import LLaDAModelLM as AutoModelFastdLLM
from fast_samplers.wino.generate import generate_wino
from fast_samplers.wino.modeling_llada import LLaDAModelLM as AutoModelWino
from dprm_guidance import load_dprm_estimator, resolve_dprm_estimator_path
from math500 import MATH500_SYSTEM_PROMPT
from parser_helper import is_equiv, last_boxed_only_string, remove_boxed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--model_label", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--test_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--diffusion_steps", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--save_every_batches", type=int, default=1)
    parser.add_argument("--sample_idx_start", type=int, default=0)
    parser.add_argument("--sample_idx_end", type=int, default=-1)
    parser.add_argument(
        "--use_fast_sampler",
        type=str,
        default="fast_dllm",
        choices=["no", "fast_dllm", "wino"],
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="pd_cache_prefix",
        choices=["llada", "pd", "pd_cache_prefix", "pd_cache_dual", "wino"],
    )
    parser.add_argument(
        "--remasking",
        type=str,
        default="low_confidence",
        choices=["low_confidence", "random", "dprm_soft_bon"],
    )
    parser.add_argument("--dprm_estimator_path", type=str, default="")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.save(handle, array)
    os.replace(tmp_path, path)


def atomic_write_json(path: Path, payload: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, path)


def load_eval_dataset(test_size, seed):
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    original_size = len(dataset)
    selected_indices = np.arange(original_size, dtype=np.int64)
    if test_size is not None and test_size < original_size:
        shuffled = np.random.default_rng(seed).permutation(original_size)[:test_size]
        selected_indices = shuffled.astype(np.int64)
        dataset = dataset.select(selected_indices.tolist())
    return dataset, selected_indices


def build_prompt(tokenizer, problem):
    messages = [{"role": "user", "content": MATH500_SYSTEM_PROMPT.strip() + "\n\n" + problem}]
    user_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return user_input + "<reasoning>"


def extract_math_answer(text):
    try:
        boxed = last_boxed_only_string(text)
        if boxed:
            parsed = remove_boxed(boxed)
            if parsed:
                return parsed
    except Exception:
        pass

    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_match:
        parsed = answer_match.group(1).strip()
        if parsed:
            return parsed

    return None


def is_correct(prediction, answer):
    if prediction is None or answer is None:
        return False
    return bool(is_equiv(prediction, answer))


def get_generate_fn(sampler):
    return {
        "llada": generate,
        "pd": generate_pd,
        "pd_cache_prefix": generate_with_prefix_cache,
        "pd_cache_dual": generate_with_dual_cache,
        "wino": generate_wino,
    }.get(sampler, generate)


def get_model_class(use_fast_sampler):
    return {
        "fast_dllm": AutoModelFastdLLM,
        "wino": AutoModelWino,
    }.get(use_fast_sampler, AutoModel)


def load_model(base_model_path, checkpoint_path, use_fast_sampler):
    model_class = get_model_class(use_fast_sampler)
    model = model_class.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    if checkpoint_path:
        model = PeftModel.from_pretrained(
            model,
            checkpoint_path,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
    model.eval()
    return model


def load_or_init_state(output_dir: Path, metadata: dict, num_examples: int, max_k: int):
    metadata_path = output_dir / "metadata.json"
    success_path = output_dir / "success_matrix.npy"
    progress_path = output_dir / "sample_progress.npy"

    if metadata_path.exists() and success_path.exists() and progress_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        legacy_defaults = {
            "remasking": "low_confidence",
            "dprm_estimator_path": "",
            "resolved_dprm_estimator_path": "",
        }
        required_matches = [
            "base_model_path",
            "checkpoint",
            "ks",
            "test_size",
            "seed",
            "batch_size",
            "gen_length",
            "block_length",
            "diffusion_steps",
            "temperature",
            "sampler",
            "use_fast_sampler",
            "remasking",
            "dprm_estimator_path",
            "resolved_dprm_estimator_path",
            "sample_idx_start",
            "sample_idx_end",
            "selected_indices",
        ]
        for key in required_matches:
            existing_value = existing.get(key, legacy_defaults.get(key))
            if existing_value != metadata.get(key):
                raise RuntimeError(
                    f"Existing cache in {output_dir} is incompatible for key '{key}'. "
                    f"Delete the cache directory or use a new output_dir."
                )
        success_matrix = np.load(success_path)
        sample_progress = np.load(progress_path)
        if success_matrix.shape != (num_examples, max_k):
            raise RuntimeError(f"Cached success_matrix shape mismatch: {success_matrix.shape}")
        if sample_progress.shape != (max_k,):
            raise RuntimeError(f"Cached sample_progress shape mismatch: {sample_progress.shape}")
        return success_matrix.astype(np.bool_), sample_progress.astype(np.int64), True

    success_matrix = np.zeros((num_examples, max_k), dtype=np.bool_)
    sample_progress = np.zeros((max_k,), dtype=np.int64)
    atomic_write_json(metadata_path, metadata)
    atomic_save_npy(success_path, success_matrix)
    atomic_save_npy(progress_path, sample_progress)
    return success_matrix, sample_progress, False


def persist_state(output_dir: Path, success_matrix: np.ndarray, sample_progress: np.ndarray) -> None:
    atomic_save_npy(output_dir / "success_matrix.npy", success_matrix)
    atomic_save_npy(output_dir / "sample_progress.npy", sample_progress)


def save_status(output_dir: Path, sample_progress: np.ndarray, num_examples: int, label: str) -> None:
    payload = {
        "model_label": label,
        "completed_samples": int((sample_progress >= num_examples).sum()),
        "max_k": int(sample_progress.shape[0]),
        "sample_progress": sample_progress.tolist(),
        "num_examples": int(num_examples),
    }
    atomic_write_json(output_dir / "status.json", payload)


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    output_dir = Path(args.output_dir)

    ks = sorted(args.ks)
    max_k = max(ks)

    set_seed(args.seed)
    dataset, selected_indices = load_eval_dataset(args.test_size, args.seed)
    num_examples = len(dataset)
    levels = np.array(dataset["level"], dtype=np.int64)
    answers = dataset["answer"]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    prompts = [build_prompt(tokenizer, problem) for problem in dataset["problem"]]
    metadata = {
        "model_label": args.model_label,
        "base_model_path": args.base_model_path,
        "checkpoint": args.checkpoint,
        "resolved_dprm_estimator_path": resolve_dprm_estimator_path(
            checkpoint_path=args.checkpoint,
            explicit_path=args.dprm_estimator_path,
        ),
        "ks": ks,
        "test_size": args.test_size,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "gen_length": args.gen_length,
        "block_length": args.block_length,
        "diffusion_steps": args.diffusion_steps,
        "temperature": args.temperature,
        "cfg_scale": args.cfg_scale,
        "sampler": args.sampler,
        "use_fast_sampler": args.use_fast_sampler,
        "remasking": args.remasking,
        "dprm_estimator_path": args.dprm_estimator_path,
        "sample_idx_start": args.sample_idx_start,
        "sample_idx_end": args.sample_idx_end,
        "num_examples": num_examples,
        "selected_indices": selected_indices.tolist(),
        "levels": levels.tolist(),
    }

    success_matrix, sample_progress, resumed = load_or_init_state(output_dir, metadata, num_examples, max_k)
    atomic_save_npy(output_dir / "levels.npy", levels)
    save_status(output_dir, sample_progress, num_examples, args.model_label)

    if resumed:
        print(f"Resuming {args.model_label} from cached state in {output_dir}")
    else:
        print(f"Starting fresh evaluation for {args.model_label} in {output_dir}")

    model = load_model(args.base_model_path, args.checkpoint, args.use_fast_sampler)
    generate_fn = get_generate_fn(args.sampler)
    dprm_estimator = None
    if args.remasking == "dprm_soft_bon":
        dprm_estimator = load_dprm_estimator(
            checkpoint_path=args.checkpoint,
            explicit_path=args.dprm_estimator_path,
        )
        if dprm_estimator is None:
            raise FileNotFoundError("dprm_soft_bon remasking requires a DPRM estimator file")

    try:
        num_batches = math.ceil(num_examples / args.batch_size)
        sample_idx_start = max(0, int(args.sample_idx_start))
        sample_idx_end = max_k if int(args.sample_idx_end) < 0 else min(max_k, int(args.sample_idx_end))
        if sample_idx_start >= sample_idx_end:
            raise ValueError(
                f"Invalid sample shard range: start={sample_idx_start}, end={sample_idx_end}, max_k={max_k}"
            )

        for sample_idx in range(sample_idx_start, sample_idx_end):
            resume_start = int(sample_progress[sample_idx])
            if resume_start >= num_examples:
                continue

            desc = f"{args.model_label} sample {sample_idx + 1}/{max_k}"
            start_iter = range(resume_start, num_examples, args.batch_size)
            progress_bar = tqdm(
                start_iter,
                desc=desc,
                initial=resume_start // args.batch_size,
                total=num_batches,
            )
            for batch_start in progress_bar:
                batch_id = batch_start // args.batch_size
                set_seed(args.seed + sample_idx * 100000 + batch_id)

                batch_end = min(batch_start + args.batch_size, num_examples)
                batch_prompts = prompts[batch_start:batch_end]
                batch_answers = answers[batch_start:batch_end]

                inputs = tokenizer(
                    batch_prompts,
                    padding="longest",
                    return_tensors="pt",
                )
                input_ids = inputs.input_ids.to("cuda")

                with torch.no_grad():
                    outputs = generate_fn(
                        model,
                        input_ids,
                        steps=args.diffusion_steps,
                        gen_length=args.gen_length,
                        block_length=args.block_length,
                        temperature=args.temperature,
                        cfg_scale=args.cfg_scale,
                        remasking=args.remasking,
                        dprm_estimator=dprm_estimator,
                        dprm_global_step=dprm_estimator.global_updates if dprm_estimator is not None else 0,
                        dprm_force_full=args.remasking == "dprm_soft_bon",
                    )

                generations = tokenizer.batch_decode(outputs[:, -args.gen_length:], skip_special_tokens=False)
                for offset, generation in enumerate(generations):
                    idx = batch_start + offset
                    prediction = extract_math_answer(generation)
                    success_matrix[idx, sample_idx] = is_correct(prediction, batch_answers[offset])

                sample_progress[sample_idx] = batch_end
                if ((batch_id + 1) % args.save_every_batches == 0) or batch_end == num_examples:
                    persist_state(output_dir, success_matrix, sample_progress)
                    save_status(output_dir, sample_progress, num_examples, args.model_label)

        persist_state(output_dir, success_matrix, sample_progress)
        save_status(output_dir, sample_progress, num_examples, args.model_label)
        print(f"Completed evaluation for {args.model_label}")
        print(f"State saved under {output_dir}")
    finally:
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
