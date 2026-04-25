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
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_ROOT = os.path.abspath(os.path.dirname(__file__))
DMPO_ROOT = os.path.join(REPO_ROOT, "DMPO")
if EVAL_ROOT not in sys.path:
    sys.path.append(EVAL_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
if DMPO_ROOT not in sys.path:
    sys.path.append(DMPO_ROOT)

from data_utils import SYSTEM_PROMPT
from dprm_guidance import load_dprm_estimator, resolve_dprm_estimator_path
from fast_samplers.fast_dllm.generate import (
    generate_pd,
    generate_with_dual_cache,
    generate_with_prefix_cache,
)
from fast_samplers.fast_dllm.modeling_llada import LLaDAModelLM as AutoModelFastdLLM
from fast_samplers.wino.generate import generate_wino
from fast_samplers.wino.modeling_llada import LLaDAModelLM as AutoModelWino
from generate import generate
from parsers import last_boxed_only_string, remove_boxed


LEVEL_LABELS = {
    0: "trivial_2",
    1: "easy_3",
    2: "medium_4",
    3: "hard_5",
    4: "ood_6",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--model_label", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_jsonl", type=str, default="")
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--test_size", type=int, default=5120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--diffusion_steps", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--success_threshold", type=float, default=0.1)
    parser.add_argument("--save_every_batches", type=int, default=1)
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


def resolve_dataset_jsonl(explicit_path: str) -> str:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Countdown dataset jsonl not found: {path}")
        return str(path)

    snapshot_root = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--divelab--countdown" / "snapshots"
    candidates = sorted(snapshot_root.glob("*/test.jsonl"))
    if candidates:
        return str(candidates[-1].resolve())

    fallback = Path(REPO_ROOT) / "dataset" / "countdown_cd3_test.jsonl"
    if fallback.exists():
        return str(fallback.resolve())

    raise FileNotFoundError(
        "Unable to resolve countdown test jsonl. Pass --dataset_jsonl explicitly or cache divelab/countdown."
    )


def parse_countdown_row(row: dict) -> dict:
    if "question" in row and "answer" in row:
        numbers = [int(x) for x in row["question"]]
        target = int(row["answer"])
        level = int(row.get("level", len(numbers) - 2))
        solution = row.get("solution", "")
        return {
            "numbers": numbers,
            "target": target,
            "level": level,
            "solution": solution,
        }

    if "input" in row and "output" in row:
        numbers = [int(x) for x in str(row["input"]).split(",")]
        target = int(row["output"])
        level = int(row.get("level", len(numbers) - 2))
        solution = row.get("solution", "")
        return {
            "numbers": numbers,
            "target": target,
            "level": level,
            "solution": solution,
        }

    raise KeyError(f"Unsupported countdown row schema: keys={sorted(row.keys())}")


def balanced_subsample_indices(levels: np.ndarray, test_size: int, seed: int) -> np.ndarray:
    total = len(levels)
    if test_size is None or test_size >= total:
        return np.arange(total, dtype=np.int64)

    rng = np.random.default_rng(seed)
    unique_levels = sorted(np.unique(levels).tolist())
    base_take = test_size // len(unique_levels)
    remainder = test_size % len(unique_levels)

    selected = []
    for offset, level in enumerate(unique_levels):
        level_indices = np.flatnonzero(levels == level)
        rng.shuffle(level_indices)
        take = min(len(level_indices), base_take + (1 if offset < remainder else 0))
        selected.extend(level_indices[:take].tolist())

    selected = np.array(selected, dtype=np.int64)
    rng.shuffle(selected)
    return selected


def load_eval_dataset(dataset_jsonl: str, test_size: int, seed: int):
    path = Path(dataset_jsonl)
    rows = [parse_countdown_row(json.loads(line)) for line in path.open("r", encoding="utf-8")]
    levels = np.array([row["level"] for row in rows], dtype=np.int64)
    selected_indices = balanced_subsample_indices(levels, test_size, seed)
    rows = [rows[idx] for idx in selected_indices.tolist()]
    levels = levels[selected_indices]
    return rows, selected_indices.astype(np.int64), levels


def build_prompt(tokenizer, numbers, target):
    user_prompt = (
        f"{SYSTEM_PROMPT}\nUsing only the numbers {numbers}, create an arithmetic expression that "
        f"evaluates to exactly {target}. You must use all numbers from the list, and each number "
        "must be used exactly once. You may use the operations +, -, *, and / as needed. After "
        "reasoning, provide only your final expression inside <answer></answer> tags without "
        "including an equals sign or the target number. For example, if the numbers are [2, 3, 4] "
        "and the target is 5, a valid answer is: <answer>\n2*4-3\n</answer>"
    )
    messages = [{"role": "user", "content": user_prompt}]
    user_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return user_input + "<reasoning>"


def extract_equation(text):
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_match is None:
        return None

    equation = answer_match.group(1).strip()
    if not equation:
        return None

    try:
        equation = remove_boxed(last_boxed_only_string(equation))
    except Exception:
        pass

    equation = (
        equation.replace(r"\div", "/")
        .replace(r"\times", "*")
        .replace(r"\cdot", "*")
        .replace("×", "*")
        .replace("÷", "/")
    )
    return equation.strip()


def validate_equation(equation, numbers):
    try:
        numbers_in_eq = sorted(int(match) for match in re.findall(r"\d+", equation))
        return numbers_in_eq == sorted(int(x) for x in numbers)
    except Exception:
        return False


def evaluate_equation(equation):
    try:
        if not re.match(r"^[\d+\-*/().\s]+$", equation):
            return None
        return eval(equation, {"__builtins__": None}, {})
    except Exception:
        return None


def is_correct(generation, numbers, target, success_threshold):
    equation = extract_equation(generation)
    if equation is None:
        return False
    if not validate_equation(equation, numbers):
        return False
    result = evaluate_equation(equation)
    if result is None:
        return False
    score = 1.0 if abs(float(result) - float(target)) < 1e-5 else success_threshold
    return bool(score > success_threshold)


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
            "dataset_jsonl",
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
    dataset_jsonl = resolve_dataset_jsonl(args.dataset_jsonl)
    rows, selected_indices, levels = load_eval_dataset(dataset_jsonl, args.test_size, args.seed)
    num_examples = len(rows)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    prompts = [build_prompt(tokenizer, row["numbers"], row["target"]) for row in rows]
    metadata = {
        "model_label": args.model_label,
        "base_model_path": args.base_model_path,
        "checkpoint": args.checkpoint,
        "dataset_jsonl": dataset_jsonl,
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
        "num_examples": num_examples,
        "selected_indices": selected_indices.tolist(),
        "levels": levels.tolist(),
        "level_labels": LEVEL_LABELS,
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
        for sample_idx in range(max_k):
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
                batch_rows = rows[batch_start:batch_end]

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
                    row = batch_rows[offset]
                    success_matrix[idx, sample_idx] = is_correct(
                        generation,
                        row["numbers"],
                        row["target"],
                        args.success_threshold,
                    )

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
